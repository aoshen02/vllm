//! Tiny blocking HTTP/1.1 client: one request per connection, bodies read
//! with read(2) straight into the destination buffer.
//!
//! Request headers are byte-for-byte those httpx 0.28.1 sends for the v1/v2
//! clients (captured on the wire): same names, case, order and values,
//! including `Accept-Encoding: gzip, deflate` and `Connection: keep-alive`.
//! A response with any Content-Encoding is rejected (v3 never decodes), so a
//! compressing server makes v3 FAIL loudly instead of measuring other bytes.
//!
//! Framing is strict (fail closed): CRLF line endings only; status line
//! `HTTP/1.0|1.1 DDD reason`; header names must be tokens, no obs-fold;
//! repeated / list Content-Length values must all agree; Transfer-Encoding
//! must be exactly `chunked` (one coding, any number of fields totalling one);
//! Transfer-Encoding together with Content-Length is rejected; chunk sizes are
//! hex (overflow-checked) with optional `;ext`; chunk data must be followed by
//! CRLF; trailers must be valid header lines. Sizes are checked against the
//! caller's maximum before any allocation; allocations use try_reserve.

use crate::json::R;
use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::os::fd::AsRawFd;
use std::time::Duration;

pub const USER_AGENT: &str = "python-httpx/0.28.1";

pub struct Url {
    pub hostport: String,
}

pub fn parse_url(url: &str) -> R<Url> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("only http:// urls supported: {url}"))?;
    let hostport = rest.trim_end_matches('/').to_string();
    if hostport.contains('/') || !hostport.contains(':') {
        return Err(format!("url must be http://host:port: {url}"));
    }
    Ok(Url { hostport })
}

pub struct Conn {
    stream: TcpStream,
    stage: Vec<u8>,
    pos: usize,
    end: usize,
}

#[derive(Debug)]
pub struct Head {
    pub status: u16,
    pub content_length: Option<u64>,
    pub chunked: bool,
}

fn read_fd(fd: i32, dst: *mut u8, cap: usize) -> R<usize> {
    loop {
        let r = unsafe { libc::read(fd, dst as *mut libc::c_void, cap) };
        if r >= 0 {
            return Ok(r as usize);
        }
        let err = std::io::Error::last_os_error();
        match err.kind() {
            std::io::ErrorKind::Interrupted => continue,
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                return Err("read timeout (http-timeout)".into())
            }
            _ => return Err(format!("read: {err}")),
        }
    }
}

const MAX_READ: usize = 8 << 20;
const MAX_LINE: usize = 64 << 10;

fn is_tchar(c: u8) -> bool {
    c.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&c)
}

/// Parses one header line `name: value` (strict); returns (lowercase name, trimmed value).
fn header_line(line: &[u8]) -> R<(String, String)> {
    if line.first().map_or(false, |&c| c == b' ' || c == b'\t') {
        return Err("obsolete header line folding".into());
    }
    let colon = line.iter().position(|&c| c == b':').ok_or("header line without ':'")?;
    let name = &line[..colon];
    if name.is_empty() || !name.iter().all(|&c| is_tchar(c)) {
        return Err(format!("invalid header name {:?}", String::from_utf8_lossy(name)));
    }
    let value = &line[colon + 1..];
    if value.iter().any(|&c| (c < 0x20 && c != b'\t') || c == 0x7f) {
        return Err("control character in header value".into());
    }
    let value = String::from_utf8_lossy(value).trim_matches(|c| c == ' ' || c == '\t').to_string();
    Ok((String::from_utf8_lossy(name).to_ascii_lowercase(), value))
}

impl Conn {
    /// Connects with ONE deadline (`timeout_s`) covering name resolution and
    /// all connection attempts together; `timeout_s` then also bounds every
    /// later read/write (idle timeout, like httpx).
    pub fn connect(url: &Url, timeout_s: f64) -> R<Conn> {
        let t = Duration::from_secs_f64(timeout_s);
        let deadline = std::time::Instant::now() + t;
        let addrs: Vec<std::net::SocketAddr> = match url.hostport.parse::<std::net::SocketAddr>() {
            Ok(a) => vec![a], // numeric host: no resolver involved
            Err(_) => {
                let (tx, rx) = std::sync::mpsc::channel();
                let hp = url.hostport.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(hp.to_socket_addrs().map(|a| a.collect::<Vec<_>>()));
                });
                match rx.recv_timeout(t) {
                    Ok(r) => r.map_err(|e| format!("resolve {}: {e}", url.hostport))?,
                    Err(_) => return Err(format!("resolve {}: timed out (http-timeout)", url.hostport)),
                }
            }
        };
        let mut last = format!("no addresses for {}", url.hostport);
        let mut stream = None;
        for addr in addrs {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                last = format!("connect {}: timed out (http-timeout)", url.hostport);
                break;
            }
            match TcpStream::connect_timeout(&addr, left) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => last = format!("connect {addr}: {e}"),
            }
        }
        let stream = stream.ok_or(last)?;
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(t)).map_err(|e| e.to_string())?;
        stream.set_write_timeout(Some(t)).map_err(|e| e.to_string())?;
        Ok(Conn { stream, stage: vec![0u8; 64 << 10], pos: 0, end: 0 })
    }

    /// The generate request exactly as httpx sends it for v1/v2.
    pub fn send_generate(&mut self, url: &Url, body: &[u8]) -> R<()> {
        let head = format!(
            "POST /inference/v1/generate HTTP/1.1\r\nHost: {}\r\nAccept: */*\r\nAccept-Encoding: gzip, deflate\r\nConnection: keep-alive\r\nUser-Agent: {USER_AGENT}\r\ncontent-type: application/json\r\nContent-Length: {}\r\n\r\n",
            url.hostport,
            body.len()
        );
        let mut msg = Vec::with_capacity(head.len() + body.len());
        msg.extend_from_slice(head.as_bytes());
        msg.extend_from_slice(body);
        self.stream.write_all(&msg).map_err(|e| format!("send: {e}"))
    }

    /// The pause request exactly as `httpx.post(url)` sends it for v2.
    pub fn send_pause(&mut self, url: &Url) -> R<()> {
        let head = format!(
            "POST /pause?mode=abort HTTP/1.1\r\nHost: {}\r\nContent-Length: 0\r\nAccept: */*\r\nAccept-Encoding: gzip, deflate\r\nConnection: keep-alive\r\nUser-Agent: {USER_AGENT}\r\n\r\n",
            url.hostport
        );
        self.stream.write_all(head.as_bytes()).map_err(|e| format!("send: {e}"))
    }

    fn fill(&mut self) -> R<usize> {
        if self.pos == self.end {
            self.pos = 0;
            self.end = 0;
        }
        if self.end == self.stage.len() {
            self.stage.copy_within(self.pos..self.end, 0);
            self.end -= self.pos;
            self.pos = 0;
            if self.end == self.stage.len() {
                return Err("line too long".into());
            }
        }
        let fd = self.stream.as_raw_fd();
        let cap = self.stage.len() - self.end;
        let got = read_fd(fd, unsafe { self.stage.as_mut_ptr().add(self.end) }, cap)?;
        self.end += got;
        Ok(got)
    }

    /// One CRLF-terminated line (without the CRLF). A bare LF is an error.
    fn read_line(&mut self) -> R<Vec<u8>> {
        loop {
            if let Some(off) = memchr::memchr(b'\n', &self.stage[self.pos..self.end]) {
                let line = &self.stage[self.pos..self.pos + off];
                let Some(line) = line.strip_suffix(b"\r") else {
                    return Err("line terminated by bare LF (CRLF required)".into());
                };
                if line.contains(&b'\r') {
                    return Err("bare CR inside a line".into());
                }
                let line = line.to_vec();
                self.pos += off + 1;
                return Ok(line);
            }
            if self.end - self.pos > MAX_LINE {
                return Err("header/chunk line too long".into());
            }
            if self.fill()? == 0 {
                return Err("connection closed while reading a line".into());
            }
        }
    }

    pub fn read_head(&mut self) -> R<Head> {
        loop {
            let status_line = self.read_line()?;
            // status-line = HTTP-version SP 3DIGIT [ SP reason-phrase ]
            // reason-phrase = *( HTAB / SP / VCHAR / obs-text )  (h11 grammar)
            let l = &status_line;
            let bad = || format!("bad status line: {:?}", String::from_utf8_lossy(l));
            if l.len() < 12 || !(l.starts_with(b"HTTP/1.1 ") || l.starts_with(b"HTTP/1.0 ")) {
                return Err(bad());
            }
            let code = &l[9..12];
            if !code.iter().all(|c| c.is_ascii_digit()) {
                return Err(bad());
            }
            if l.len() > 12 {
                if l[12] != b' ' || l[13..].iter().any(|&c| c != b'\t' && (c < 0x20 || c == 0x7f)) {
                    return Err(bad());
                }
            }
            let status: u16 = std::str::from_utf8(code).unwrap().parse().unwrap();
            let mut cl: Option<u64> = None;
            let mut codings: Vec<String> = vec![];
            let mut header_count = 0;
            loop {
                let line = self.read_line()?;
                if line.is_empty() {
                    break;
                }
                header_count += 1;
                if header_count > 256 {
                    return Err("too many header lines".into());
                }
                let (name, value) = header_line(&line)?;
                match name.as_str() {
                    "content-length" => {
                        for v in value.split(',') {
                            let v = v.trim();
                            if v.is_empty() || v.len() > 19 || !v.bytes().all(|c| c.is_ascii_digit()) {
                                return Err(format!("bad Content-Length {value:?}"));
                            }
                            let n: u64 = v.parse().map_err(|_| format!("bad Content-Length {value:?}"))?;
                            if cl.map_or(false, |old| old != n) {
                                return Err("conflicting Content-Length values".into());
                            }
                            cl = Some(n);
                        }
                    }
                    "transfer-encoding" => {
                        for c in value.split(',') {
                            let c = c.trim_matches(|x| x == ' ' || x == '\t').to_ascii_lowercase();
                            if c.is_empty() {
                                return Err(format!("empty transfer-coding element in {value:?}"));
                            }
                            codings.push(c);
                        }
                    }
                    "content-encoding" if !value.eq_ignore_ascii_case("identity") => {
                        return Err(format!("unsupported content-encoding {value}"))
                    }
                    _ => {}
                }
            }
            if (100..200).contains(&status) {
                continue; // interim response
            }
            let chunked = match codings.as_slice() {
                [] => false,
                [c] if c == "chunked" => true,
                other => return Err(format!("unsupported transfer-coding {other:?}")),
            };
            if chunked && cl.is_some() {
                return Err("both Transfer-Encoding and Content-Length".into());
            }
            return Ok(Head { status, content_length: cl, chunked });
        }
    }

    /// Ensures `out` can grow to `want` bytes (<= max), fallibly.
    fn grow(out: &mut Vec<u8>, want: usize, max: usize) -> R<()> {
        if want > max {
            return Err(format!("response body exceeds max-body-bytes {max}"));
        }
        if out.capacity() < want {
            let target = want.max(out.capacity().saturating_mul(2)).max(1 << 20).min(max);
            out.try_reserve_exact(target - out.len()).map_err(|e| format!("body allocation of {target} bytes failed: {e}"))?;
        }
        Ok(())
    }

    /// Appends exactly `n` bytes to `out`, stage first, then direct reads.
    fn read_exact_into(&mut self, out: &mut Vec<u8>, mut n: usize, max: usize) -> R<()> {
        Self::grow(out, out.len().checked_add(n).ok_or("size overflow")?, max)?;
        let take = n.min(self.end - self.pos);
        out.extend_from_slice(&self.stage[self.pos..self.pos + take]);
        self.pos += take;
        n -= take;
        let fd = self.stream.as_raw_fd();
        while n > 0 {
            let want = n.min(MAX_READ);
            let len = out.len();
            debug_assert!(out.capacity() - len >= want);
            let got = read_fd(fd, unsafe { out.as_mut_ptr().add(len) }, want)?;
            if got == 0 {
                return Err(format!("connection closed with {n} body bytes missing"));
            }
            unsafe { out.set_len(len + got) };
            n -= got;
        }
        Ok(())
    }

    /// Reads the body; never allocates more than `max` bytes of capacity.
    pub fn read_body(&mut self, head: &Head, max: usize) -> R<Vec<u8>> {
        let mut out = Vec::new();
        if head.chunked {
            loop {
                let line = self.read_line()?;
                let (size_text, ext) = match line.iter().position(|&c| c == b';') {
                    Some(p) => (&line[..p], Some(&line[p + 1..])),
                    None => (&line[..], None),
                };
                let size_text = {
                    // optional BWS before ';'
                    let mut s = size_text;
                    while ext.is_some() && matches!(s.last(), Some(b' ' | b'\t')) {
                        s = &s[..s.len() - 1];
                    }
                    s
                };
                if size_text.is_empty() || size_text.len() > 16 || !size_text.iter().all(|c| c.is_ascii_hexdigit()) {
                    return Err(format!("bad chunk size line {:?}", String::from_utf8_lossy(&line)));
                }
                if let Some(e) = ext {
                    if e.iter().any(|&c| (c < 0x20 && c != b'\t') || c == 0x7f) {
                        return Err("control character in chunk extension".into());
                    }
                }
                let size = u64::from_str_radix(std::str::from_utf8(size_text).unwrap(), 16).unwrap();
                let size = usize::try_from(size).map_err(|_| "chunk size overflow")?;
                if size == 0 {
                    loop {
                        let t = self.read_line()?;
                        if t.is_empty() {
                            break;
                        }
                        header_line(&t).map_err(|e| format!("bad trailer: {e}"))?;
                    }
                    return Ok(out);
                }
                if size > max || out.len() + size > max {
                    return Err(format!("response body exceeds max-body-bytes {max}"));
                }
                self.read_exact_into(&mut out, size, max)?;
                let crlf = self.read_line()?;
                if !crlf.is_empty() {
                    return Err("missing CRLF after chunk data".into());
                }
            }
        } else if let Some(cl) = head.content_length {
            let cl = usize::try_from(cl).map_err(|_| "Content-Length overflow")?;
            if cl > max {
                return Err(format!("Content-Length {cl} exceeds max-body-bytes {max}"));
            }
            out.try_reserve_exact(cl).map_err(|e| format!("body allocation of {cl} bytes failed: {e}"))?;
            self.read_exact_into(&mut out, cl, max)?;
            Ok(out)
        } else {
            // close-delimited
            let rest = self.end - self.pos;
            Self::grow(&mut out, rest, max)?;
            out.extend_from_slice(&self.stage[self.pos..self.end]);
            self.pos = self.end;
            let fd = self.stream.as_raw_fd();
            loop {
                if out.len() == max {
                    // exactly at the limit: probe for EOF without growing
                    let mut probe = [0u8; 1];
                    return match read_fd(fd, probe.as_mut_ptr(), 1)? {
                        0 => Ok(out),
                        _ => Err(format!("response body exceeds max-body-bytes {max}")),
                    };
                }
                if out.len() == out.capacity() {
                    let want = out.len() + 1;
                    Self::grow(&mut out, want, max)?;
                }
                let len = out.len();
                let cap = (out.capacity() - len).min(MAX_READ);
                let got = read_fd(fd, unsafe { out.as_mut_ptr().add(len) }, cap)?;
                if got == 0 {
                    return Ok(out);
                }
                unsafe { out.set_len(len + got) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::header_line;

    #[test]
    fn header_lines() {
        assert!(header_line(b"Content-Length: 5").is_ok());
        assert!(header_line(b" folded").is_err());
        assert!(header_line(b"Bad Name: x").is_err());
        assert!(header_line(b"NoColon").is_err());
        assert!(header_line(b": empty").is_err());
        assert_eq!(header_line(b"X-A:\t v \t").unwrap(), ("x-a".to_string(), "v".to_string()));
    }
}
