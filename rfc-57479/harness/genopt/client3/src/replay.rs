//! `replay-server`: test-only HTTP server that answers every generate request
//! with one recorded response body (request_id digits patched to the
//! request's index), honouring the harness protocol: once all N requests
//! arrived it writes a consumption log with N records, holds every response
//! until `POST /pause`, then sends them all at once (Content-Length framing).
//! Used to measure client v3 receive+parse throughput without a real cohort.
//!
//!   genopt-client3 replay-server --port P --body FILE --requests N
//!       --output-tokens T --consumption-log DIR

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

struct State {
    arrived: usize,
    paused: bool,
}

fn read_request(s: &mut TcpStream) -> Result<(String, Vec<u8>), String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 65536];
    let head_end = loop {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("eof".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
    let cl: usize = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok()).flatten()
        })
        .unwrap_or(0);
    let mut body = buf[head_end..].to_vec();
    while body.len() < cl {
        let n = s.read(&mut tmp).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("eof in body".into());
        }
        body.extend_from_slice(&tmp[..n]);
    }
    Ok((path, body))
}

pub fn serve(argv: &[String]) {
    let mut port = 8399u16;
    let mut body_path = PathBuf::new();
    let mut requests = 1usize;
    let mut tokens = 0usize;
    let mut log = PathBuf::new();
    let mut i = 0;
    while i < argv.len() {
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--port" => port = v.parse().unwrap(),
            "--body" => body_path = v.into(),
            "--requests" => requests = v.parse().unwrap(),
            "--output-tokens" => tokens = v.parse().unwrap(),
            "--consumption-log" => log = v.into(),
            o => panic!("replay-server: unknown {o}"),
        }
        i += 2;
    }
    let body = Arc::new(std::fs::read(&body_path).expect("read body"));
    let id_at = body[..body.len().min(4096)]
        .windows(5)
        .position(|w| w == b"mock-")
        .expect("request_id mock-NNNN not found in body head")
        + 5;
    std::fs::create_dir_all(&log).unwrap();
    let state = Arc::new((Mutex::new(State { arrived: 0, paused: false }), Condvar::new()));
    let listener = TcpListener::bind(("0.0.0.0", port)).expect("bind");
    eprintln!("replay-server: listening on {port}, body {} bytes", body.len());
    for conn in listener.incoming() {
        let Ok(mut s) = conn else { continue };
        let (body, state, log) = (body.clone(), state.clone(), log.clone());
        std::thread::spawn(move || {
            let Ok((path, req)) = read_request(&mut s) else { return };
            let (m, cv) = &*state;
            if path.starts_with("/pause") {
                m.lock().unwrap().paused = true;
                cv.notify_all();
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
                return;
            }
            let key = b"\"request_id\": \"mock-";
            let p = req.windows(key.len()).position(|w| w == key).expect("request_id in request") + key.len();
            let digits: String = req[p..].iter().take_while(|c| c.is_ascii_digit()).map(|&c| c as char).collect();
            let index: usize = digits.parse().unwrap();
            {
                let mut g = m.lock().unwrap();
                g.arrived += 1;
                if g.arrived == requests {
                    let mut text = String::new();
                    for k in 0..requests {
                        text.push_str(&format!(
                            "{{\"request_id\": \"mock-{k:04}\", \"tokens\": {tokens}, \"logprob_positions\": {tokens}}}\n"
                        ));
                    }
                    let tmp = log.join(".consumed-replay.tmp");
                    std::fs::write(&tmp, text).unwrap();
                    std::fs::rename(&tmp, log.join("consumed-replay.jsonl")).unwrap();
                }
                while !g.paused {
                    g = cv.wait(g).unwrap();
                }
            }
            let mut head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .into_bytes();
            let split = (id_at + 4096).min(body.len());
            let mut prefix = body[..split].to_vec();
            prefix[id_at..id_at + 4].copy_from_slice(format!("{index:04}").as_bytes());
            head.extend_from_slice(&prefix);
            if s.write_all(&head).is_err() {
                return;
            }
            let _ = s.write_all(&body[split..]);
        });
    }
}
