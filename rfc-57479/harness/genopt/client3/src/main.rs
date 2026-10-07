//! genopt client v3: native measurement client for the /inference/v1/generate
//! pause(mode=abort) cohort. Same CLI and request bytes as genopt-client2.py;
//! a separate instrument (its receive policy differs from v2: compare
//! servers only within the same client version and arguments).
//!
//! Receive policy ("v3-global-permits"): one blocking OS thread per request
//! (all requests in flight before the barrier). A thread reads the response
//! head immediately (headers_t), then takes one of workers*max_reading global
//! body permits plus a byte reservation from --memory-budget-gib (body bytes
//! = Content-Length, or --max-body-bytes when the length is unknown, plus a
//! format-aware parse-scratch estimate), reads the body with read(2) into one
//! buffer (received_t), takes one of --parse-threads parse permits, runs
//! sha256 + full JSON parse + validation (parsed_t), frees the buffer, and
//! only then returns the body permit and bytes.
//!
//! Subcommands: `bench-body` (offline parse of a recorded body), `replay-server`
//! (test-only loopback server), `build-info` (target/runtime CPU features).

#![recursion_limit = "512"]

mod b64;
mod fast;
mod http;
mod json;
mod replay;
mod validate;

use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use validate::{check_body, Cfg, Scratch, Stats, TOP_K};

const GIB: u64 = 1 << 30;
const RECEIVE_POLICY: &str = "v3-global-permits: one blocking thread per request; head read immediately; \
then a global body permit (workers*max_reading) + byte reservation (memory budget) is taken before the body \
is read and held through sha256+parse+validation until the buffer is freed; the response connection stays open \
until its parse completes; parse concurrency = parse_threads. \
Not v2's policy (v2: request i on worker i%workers, per-worker read slots released after aread, parse blocks \
that worker's event loop); compare only within the same client version and arguments.";

fn clock(id: libc::clockid_t) -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(id, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}
fn mono() -> f64 {
    clock(libc::CLOCK_MONOTONIC)
}
fn wall() -> f64 {
    clock(libc::CLOCK_REALTIME)
}
fn thread_cpu() -> f64 {
    clock(libc::CLOCK_THREAD_CPUTIME_ID)
}

fn die(msg: &str) -> ! {
    eprintln!("genopt-client3: FAIL: {msg}");
    std::process::exit(1);
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------- CPU features

/// Compile-time target features and the running CPU's features; refuses to
/// run (clear message, no SIGILL) if the binary needs a feature the CPU lacks.
fn build_info() -> serde_json::Value {
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! feats {
            ($($f:tt),*) => {{
                let mut compiled = vec![];
                let mut runtime = vec![];
                let mut missing = vec![];
                $(
                    let c = cfg!(target_feature = $f);
                    let r = std::arch::is_aarch64_feature_detected!($f);
                    if c { compiled.push($f); }
                    if r { runtime.push($f); }
                    if c && !r { missing.push($f); }
                )*
                (compiled, runtime, missing)
            }};
        }
        let (compiled, runtime, missing) =
            feats!("neon", "aes", "sha2", "sha3", "crc", "lse", "rdm", "dotprod", "fp16", "sve", "sve2");
        json!({"arch": "aarch64", "target_cpu": "generic (no -C target-cpu)",
               "compiled_target_features": compiled, "runtime_cpu_features": runtime,
               "missing_on_this_cpu": missing,
               "sha256_backend": "ring (runtime-dispatched ARMv8 SHA2 asm if the CPU has sha2, else portable)"})
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        json!({"arch": std::env::consts::ARCH, "missing_on_this_cpu": Vec::<&str>::new()})
    }
}

fn check_cpu() -> serde_json::Value {
    let info = build_info();
    let missing = info["missing_on_this_cpu"].as_array().map(|a| a.len()).unwrap_or(0);
    if missing > 0 {
        die(&format!("this binary was compiled for CPU features this CPU lacks: {}", info["missing_on_this_cpu"]));
    }
    info
}

// ---------------------------------------------------------------- arguments

#[derive(Clone)]
struct Args {
    urls: Vec<String>,
    workers: usize,
    max_reading: usize,
    parse_threads: usize,
    barrier_timeout: f64,
    http_timeout: f64,
    completion_timeout: f64,
    validate_requests: usize,
    requests: usize,
    input_tokens: usize,
    output_tokens: usize,
    logprobs_format: Option<String>,
    token_pool: PathBuf,
    consumption_log: PathBuf,
    result: PathBuf,
    max_body_bytes: u64,
    memory_budget_bytes: u64,
    debug_hooks: bool,
    compact_include_sampled: bool,
    compact_include_ranks: bool,
    routed_layers: usize,
}

impl Args {
    fn routed_rows(&self) -> u64 {
        if self.routed_layers == 0 {
            0
        } else {
            (self.input_tokens + self.output_tokens - 1) as u64
        }
    }
    /// Decoded size of the R3 .npy (data + generous header allowance).
    fn routed_npy_bytes(&self) -> u64 {
        if self.routed_layers == 0 {
            0
        } else {
            self.routed_rows() * self.routed_layers as u64 * 8 + 4096
        }
    }
}

fn int_arg(flag: &str, s: &str, min: u64) -> u64 {
    let v: u64 = if !s.is_empty() && s.len() <= 19 && s.bytes().all(|c| c.is_ascii_digit()) {
        s.parse().unwrap()
    } else {
        die(&format!("{flag}: expected a non-negative integer, got {s:?}"))
    };
    if v < min {
        die(&format!("{flag} must be >= {min}, got {v}"));
    }
    v
}

fn secs_arg(flag: &str, s: &str) -> f64 {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 && v < 1e9 => v,
        _ => die(&format!("{flag}: expected a positive number of seconds, got {s:?}")),
    }
}

fn b64_len(x: u64) -> u64 {
    (x + 2) / 3 * 4
}

/// Format-aware upper bound for one response body (generous, ~1.25-2x the
/// largest bodies seen: compact 341 MB, Python openai 2.2 GB, Rust 3.4 GB
/// at n=245760), plus the R3 routed_experts string when enabled. Bodies
/// above it fail before allocation.
fn default_max_body(n: u64, compact: bool, routed_npy: u64) -> u64 {
    let routed = if routed_npy > 0 { b64_len(routed_npy) + b64_len(routed_npy) / 4 } else { 0 };
    if compact {
        let exact = 2 * b64_len(n * 129 * 4) + b64_len(n * 4) + n * 12;
        exact + exact / 4 + (1 << 20) + routed
    } else {
        n * 129 * 256 + n * 12 + (1 << 20) + routed
    }
}

/// Parse-time memory beyond the body buffer: compact decodes one array at a
/// time into a scratch buffer (< 3/4 of the body); validated openai bodies
/// keep min(1024, n) parsed template rows.
fn scratch_estimate(body: u64, compact: bool, validated: bool, routed_npy: u64) -> u64 {
    routed_npy
        + if compact {
            body / 4 * 3 + (1 << 20)
        } else if validated {
            512 << 20
        } else {
            1 << 20
        }
}

fn parse_args(argv: &[String]) -> Args {
    let mut a = Args {
        urls: vec![],
        workers: 8,
        max_reading: 4,
        parse_threads: 0,
        barrier_timeout: 3600.0,
        http_timeout: 7200.0,
        completion_timeout: 0.0,
        validate_requests: 1 << 30,
        requests: 1,
        input_tokens: 16384,
        output_tokens: 256,
        logprobs_format: None,
        token_pool: PathBuf::new(),
        consumption_log: PathBuf::new(),
        result: PathBuf::new(),
        max_body_bytes: 0,
        memory_budget_bytes: 256 * GIB,
        debug_hooks: false,
        compact_include_sampled: true,
        compact_include_ranks: true,
        routed_layers: 0,
    };
    let mut i = 0;
    let mut have = BTreeSet::new();
    while i < argv.len() {
        let raw = argv[i].as_str();
        let (flag, inline) = match raw.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (raw.to_string(), None),
        };
        i += 1;
        let mut val = || -> String {
            if let Some(v) = inline.clone() {
                return v;
            }
            let v = argv.get(i).cloned().unwrap_or_else(|| die(&format!("{flag} needs a value")));
            i += 1;
            v
        };
        match flag.as_str() {
            "--urls" => {
                if let Some(v) = inline.clone() {
                    a.urls.push(v);
                }
                while i < argv.len() && !argv[i].starts_with("--") {
                    a.urls.push(argv[i].clone());
                    i += 1;
                }
            }
            "--workers" => a.workers = int_arg(&flag, &val(), 1) as usize,
            "--max-reading" => a.max_reading = int_arg(&flag, &val(), 1) as usize,
            "--parse-threads" => a.parse_threads = int_arg(&flag, &val(), 1) as usize,
            "--barrier-timeout" => a.barrier_timeout = secs_arg(&flag, &val()),
            "--http-timeout" => a.http_timeout = secs_arg(&flag, &val()),
            "--completion-timeout" => a.completion_timeout = secs_arg(&flag, &val()),
            "--validate-requests" => a.validate_requests = int_arg(&flag, &val(), 0) as usize,
            "--requests" => a.requests = int_arg(&flag, &val(), 1) as usize,
            "--input-tokens" => a.input_tokens = int_arg(&flag, &val(), 0) as usize,
            "--output-tokens" => a.output_tokens = int_arg(&flag, &val(), 1) as usize,
            "--max-body-bytes" => a.max_body_bytes = int_arg(&flag, &val(), 1),
            "--memory-budget-gib" => a.memory_budget_bytes = int_arg(&flag, &val(), 1) * GIB,
            "--debug-hooks" => a.debug_hooks = true,
            "--compact-no-sampled" => a.compact_include_sampled = false,
            "--compact-no-ranks" => a.compact_include_ranks = false,
            "--routed-experts-layers" => a.routed_layers = int_arg(&flag, &val(), 0) as usize,
            "--logprobs-format" => {
                let v = val();
                if v != "openai" && v != "compact" {
                    die("--logprobs-format must be openai or compact");
                }
                a.logprobs_format = Some(v)
            }
            "--token-pool" => a.token_pool = val().into(),
            "--consumption-log" => a.consumption_log = val().into(),
            "--result" => a.result = val().into(),
            "--parse-workers" => {
                int_arg(&flag, &val(), 1); // v1 flag: accepted and ignored
            }
            other => die(&format!("unknown argument {other}")),
        }
        have.insert(flag);
    }
    for req in ["--urls", "--token-pool", "--consumption-log", "--result"] {
        if !have.contains(req) {
            die(&format!("{req} is required"));
        }
    }
    if a.urls.is_empty() {
        die("--urls needs at least one url");
    }
    if a.requests > 9999 {
        die("--requests > 9999 not supported (request ids are mock-%04d)");
    }
    if a.parse_threads == 0 {
        a.parse_threads = a.workers;
    }
    if a.completion_timeout == 0.0 {
        a.completion_timeout = a.http_timeout;
    }
    let compact = a.logprobs_format.as_deref() == Some("compact");
    if a.max_body_bytes == 0 {
        a.max_body_bytes = default_max_body(a.output_tokens as u64, compact, a.routed_npy_bytes());
    }
    let worst = a.max_body_bytes + scratch_estimate(a.max_body_bytes, compact, true, a.routed_npy_bytes());
    if worst > a.memory_budget_bytes {
        die(&format!(
            "one body reservation ({worst} B = max-body-bytes + parse scratch) exceeds --memory-budget-gib ({} B)",
            a.memory_budget_bytes
        ));
    }
    a
}

struct Hooks {
    dump_dir: Option<PathBuf>,
    corrupt: Option<usize>,
}

impl Hooks {
    fn from_env(debug: bool) -> Hooks {
        let dump = std::env::var_os("GENOPT_CLIENT3_DUMP_DIR");
        let corrupt = std::env::var("GENOPT_CLIENT3_CORRUPT_EXPECTED_POSITION").ok();
        if (dump.is_some() || corrupt.is_some()) && !debug {
            die("GENOPT_CLIENT3_* debug hook variables are set but --debug-hooks was not given; \
                 refusing to run (would contaminate a measurement)");
        }
        Hooks {
            dump_dir: dump.map(PathBuf::from),
            corrupt: corrupt.map(|v| int_arg("GENOPT_CLIENT3_CORRUPT_EXPECTED_POSITION", &v, 0) as usize),
        }
    }
    fn active(&self) -> bool {
        self.dump_dir.is_some() || self.corrupt.is_some()
    }
    fn json(&self) -> serde_json::Value {
        if !self.active() {
            return serde_json::Value::Null;
        }
        json!({"dump_dir": self.dump_dir.as_ref().map(|p| p.display().to_string()),
               "corrupt_expected_position": self.corrupt})
    }
}

fn load_pool(path: &Path) -> Vec<i64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("token pool {path:?}: {e}")));
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| die(&format!("token pool json: {e}")));
    v.as_array()
        .unwrap_or_else(|| die("token pool is not a list"))
        .iter()
        .map(|e| e["id"].as_i64().unwrap_or_else(|| die("token pool entry without integer id")))
        .collect()
}

/// Exactly genopt-client.py: json.dumps(body)[:-1] (default separators).
fn payload_prefix(a: &Args, pool: &[i64]) -> String {
    let prompt: Vec<String> = (0..a.input_tokens).map(|i| pool[i % pool.len()].to_string()).collect();
    let mut s = format!(
        "{{\"model\": \"hy4-mock\", \"token_ids\": [{}], \"stream\": false, \"sampling_params\": {{\"max_tokens\": {}, \"logprobs\": {}, \"ignore_eos\": true}}",
        prompt.join(", "),
        a.output_tokens + 1,
        TOP_K
    );
    if let Some(f) = &a.logprobs_format {
        s.push_str(&format!(", \"logprobs_format\": \"{f}\""));
    }
    // same keys, order and values as genopt-client.py (amendment v3)
    if !a.compact_include_sampled {
        s.push_str(", \"compact_include_sampled\": false");
    }
    if !a.compact_include_ranks {
        s.push_str(", \"compact_include_ranks\": false");
    }
    s
}

// ---------------------------------------------------------------- permits

struct Sem {
    m: Mutex<u64>,
    c: Condvar,
    total: u64,
}
impl Sem {
    fn new(n: u64) -> Self {
        assert!(n > 0);
        Sem { m: Mutex::new(n), c: Condvar::new(), total: n }
    }
    /// Blocks until `n` units are free. `n` > total is an error (would never succeed).
    fn acquire(&self, n: u64) -> Result<Permit<'_>, String> {
        if n > self.total {
            return Err(format!("reservation {n} exceeds total {}", self.total));
        }
        let mut g = lock(&self.m);
        while *g < n {
            g = self.c.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        *g -= n;
        Ok(Permit { sem: self, n })
    }
}
struct Permit<'a> {
    sem: &'a Sem,
    n: u64,
}
impl Permit<'_> {
    /// Takes `n` more units if they are free right now (never blocks).
    fn try_grow(&mut self, n: u64) -> bool {
        let mut g = lock(&self.sem.m);
        if *g >= n {
            *g -= n;
            self.n += n;
            true
        } else {
            false
        }
    }
    /// Returns the part of the reservation above `keep`.
    fn shrink(&mut self, keep: u64) {
        if keep < self.n {
            *lock(&self.sem.m) += self.n - keep;
            self.n = keep;
            self.sem.c.notify_all();
        }
    }
}
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *lock(&self.sem.m) += self.n;
        self.sem.c.notify_all();
    }
}

// ---------------------------------------------------------------- requests

struct Rec {
    headers_t: f64,
    received_t: f64,
    parsed_t: f64,
    parse_validate_s: f64,
    cpu_s: f64,
    sha_s: f64,
    bookkeeping_s: f64,
    validated: bool,
    top_entries: BTreeSet<usize>,
    bytes: usize,
    sha: String,
    scratch_peak: u64,
}

struct Shared {
    args: Args,
    cfg: Cfg,
    compact: bool,
    prefix: String,
    urls: Vec<http::Url>,
    hooks: Hooks,
    go: AtomicBool,
    finished: AtomicUsize,
    records: Mutex<Vec<Option<Rec>>>,
    errors: Mutex<Vec<String>>,
    done_cv: Condvar,
    done_m: Mutex<()>,
    bodies: Sem,
    bytes: Sem,
    parsers: Sem,
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// sha256 via ring (runtime-dispatched ARMv8 SHA2 assembly, ~2.0 GB/s/core
/// on Neoverse-V2 vs ~1.74 GB/s for the sha2 crate's intrinsics backend).
fn sha256_hex(body: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, body).as_ref())
}

/// Startup cross-check of the two independent sha256 implementations.
fn sha_selfcheck() {
    let mut buf = vec![0u8; (1 << 20) + 37];
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    for v in buf.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = x as u8;
    }
    for len in [0, 1, 55, 56, 63, 64, 65, 1000, buf.len()] {
        if sha256_hex(&buf[..len]) != hex(&Sha256::digest(&buf[..len])) {
            die("sha256 self-check failed (ring vs sha2)");
        }
    }
}

/// sha256 + parse/validate of one body; returns (sha, sha_s, parse_validate_s, cpu_s, stats).
fn process_body<'a>(
    body: &'a [u8],
    index: usize,
    validated: bool,
    cfg: &Cfg,
    sc: &mut Scratch,
    mem: &'a json::Mem<'a>,
) -> Result<(String, f64, f64, f64, Stats), String> {
    let cpu0 = thread_cpu();
    let t0 = mono();
    let sha = sha256_hex(body);
    let t1 = mono();
    let mut st = Stats::default();
    check_body(body, index, validated, cfg, &mut st, sc, mem)?;
    let t2 = mono();
    Ok((sha, t1 - t0, t2 - t1, thread_cpu() - cpu0, st))
}

fn one_request(i: usize, sh: &Shared) -> Result<Rec, String> {
    let url = &sh.urls[i % sh.urls.len()];
    let mut conn = http::Conn::connect(url, sh.args.http_timeout)?;
    let body = format!("{}, \"request_id\": \"mock-{i:04}\"}}", sh.prefix);
    conn.send_generate(url, body.as_bytes())?;
    let head = conn.read_head()?;
    let headers_t = mono();
    let max = sh.args.max_body_bytes;
    if let Some(cl) = head.content_length {
        if cl > max {
            return Err(format!("Content-Length {cl} exceeds max-body-bytes {max}"));
        }
    }
    let validated = i < sh.args.validate_requests;
    let body_res = head.content_length.unwrap_or(max);
    let _body_permit = sh.bodies.acquire(1)?;
    let routed_npy = sh.args.routed_npy_bytes();
    let mut byte_permit = sh.bytes.acquire(body_res + scratch_estimate(body_res, sh.compact, validated, routed_npy))?;
    let data = conn.read_body(&head, max as usize)?;
    let received_t = mono();
    // The connection stays open until this response is parsed (closer to v2,
    // whose httpx response stream lives through parsing); dropped below.
    // unknown-length body: keep only what it actually needs
    byte_permit.shrink(data.capacity() as u64 + scratch_estimate(data.len() as u64, sh.compact, validated, routed_npy));
    if !sh.go.load(Ordering::SeqCst) {
        return Err(format!(
            "returned before the consumption barrier: status={} body={}",
            head.status,
            String::from_utf8_lossy(&data[..data.len().min(500)])
        ));
    }
    if head.status != 200 {
        return Err(format!("HTTP {}: {}", head.status, String::from_utf8_lossy(&data[..data.len().min(300)])));
    }
    if let Some(dir) = &sh.hooks.dump_dir {
        // debug hook (--debug-hooks only): record bodies for offline tests
        std::fs::write(dir.join(format!("body-{i:04}.json")), &data).map_err(|e| format!("dump: {e}"))?;
    }
    let _parse_permit = sh.parsers.acquire(1)?;
    let mut sc = Scratch::new();
    // Parse-time allocations are charged to this request's scratch
    // reservation; beyond it they may take more from the global byte budget
    // without blocking, else the request fails (no abort, no deadlock).
    let scratch_reserved = scratch_estimate(data.len() as u64, sh.compact, validated, routed_npy);
    let permit_cell = std::cell::RefCell::new(byte_permit);
    let grow = |n: u64| permit_cell.borrow_mut().try_grow(n);
    let mem = json::Mem::new(scratch_reserved, Some(&grow));
    let (sha, sha_s, parse_s, cpu_s, st) = process_body(&data, i, validated, &sh.cfg, &mut sc, &mem)?;
    let scratch_peak = mem.peak();
    let v0 = mono();
    let bytes = data.len();
    drop(data);
    drop(sc);
    drop(conn);
    let parsed_t = mono();
    Ok(Rec {
        headers_t,
        received_t,
        parsed_t,
        parse_validate_s: parse_s,
        cpu_s,
        sha_s,
        bookkeeping_s: parsed_t - v0,
        validated: st.validated,
        top_entries: st.top_entries,
        bytes,
        sha,
        scratch_peak,
    })
}

/// Records the outcome of request `i` exactly once, whatever happens
/// (normal return, error, panic or an unexpected unwind past catch_unwind).
struct Completion {
    sh: Arc<Shared>,
    i: usize,
    done: bool,
}
impl Completion {
    fn finish(&mut self, r: Result<Rec, String>) {
        match r {
            Ok(rec) => lock(&self.sh.records)[self.i] = Some(rec),
            Err(e) => lock(&self.sh.errors).push(format!("request {}: {e}", self.i)),
        }
        self.done = true;
        self.sh.finished.fetch_add(1, Ordering::SeqCst);
        let _g = lock(&self.sh.done_m);
        self.sh.done_cv.notify_all();
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if !self.done {
            self.finish(Err("request thread exited without a result".into()));
        }
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

// ---------------------------------------------------------------- barrier

/// Collects "consumed" records from consumed-*.jsonl under `dir` (recursive).
/// A missing root directory means no records yet; every other error fails.
fn read_consumed(dir: &Path, root: bool, out: &mut Vec<serde_json::Value>) -> Result<(), String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if root && e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("consumption log dir {dir:?}: {e}")),
    };
    for e in entries {
        let e = e.map_err(|e| format!("consumption log dir {dir:?}: {e}"))?;
        let path = e.path();
        let ft = e.file_type().map_err(|e| format!("{path:?}: {e}"))?;
        if ft.is_dir() {
            read_consumed(&path, false, out)?;
            continue;
        }
        let name = e.file_name().to_string_lossy().to_string();
        if !(name.starts_with("consumed-") && name.ends_with(".jsonl")) {
            continue;
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("consumption log {path:?}: {e}"))?;
        for line in text.split_inclusive('\n') {
            if !line.ends_with('\n') {
                continue; // partial last line: being written
            }
            let rec: serde_json::Value =
                serde_json::from_str(line).map_err(|e| format!("bad consumption record in {path:?}: {e}"))?;
            if !rec.is_object() {
                return Err(format!("consumption record is not an object in {path:?}"));
            }
            if rec.get("event").map_or(true, |v| v.as_str() == Some("consumed")) {
                out.push(rec);
            }
        }
    }
    Ok(())
}

/// Index of a frontend request id: mock-%04d or generate-tokens-mock-%04d.
fn id_index(id: &str) -> Option<usize> {
    let rest = id.strip_prefix("generate-tokens-").unwrap_or(id).strip_prefix("mock-")?;
    if rest.len() == 4 && rest.bytes().all(|c| c.is_ascii_digit()) {
        rest.parse().ok()
    } else {
        None
    }
}

/// Ok(true) when the barrier is complete and correct; Ok(false) = keep waiting.
fn barrier_check(consumed: &[serde_json::Value], n: usize, tokens: usize) -> Result<bool, String> {
    let mut ids = Vec::with_capacity(consumed.len());
    for r in consumed {
        match r.get("request_id") {
            Some(serde_json::Value::String(s)) => ids.push(s.clone()),
            other => return Err(format!("consumption record without a string request_id: {other:?}")),
        }
    }
    if consumed.len() > n {
        return Err(format!("{} consumption records for {n} requests", consumed.len()));
    }
    if consumed.len() < n {
        return Ok(false);
    }
    let unique: BTreeSet<&String> = ids.iter().collect();
    if unique.len() != n {
        return Err("Duplicate barrier request IDs".into());
    }
    let idx: BTreeSet<usize> = ids.iter().filter_map(|s| id_index(s)).collect();
    if idx.len() != n || idx.iter().any(|&k| k >= n) {
        return Err(format!("barrier request IDs are not exactly the {n} requests sent: {:?}", unique));
    }
    let t = tokens as u64;
    if consumed.iter().any(|r| r["tokens"].as_u64() != Some(t) || r["logprob_positions"].as_u64() != Some(t)) {
        return Err("Barrier counts differ from target".into());
    }
    Ok(true)
}

fn rusage_cpu() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6 + ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 * 1e-6
}

fn vm_hwm_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))
        })
        .map_or(f64::NAN, |k| k / 1024.0 / 1024.0)
}

fn run(args: Args, build: serde_json::Value) {
    let hooks = Hooks::from_env(args.debug_hooks);
    let pool = load_pool(&args.token_pool);
    if pool.is_empty() {
        die("empty token pool");
    }
    let compact = args.logprobs_format.as_deref() == Some("compact");
    let mut cfg = Cfg::new(args.output_tokens, compact, pool.clone());
    cfg.include_sampled = args.compact_include_sampled;
    cfg.include_ranks = args.compact_include_ranks;
    cfg.routed_layers = args.routed_layers;
    cfg.routed_rows = args.routed_rows() as usize;
    if let Some(p) = hooks.corrupt {
        cfg.corrupt_expected_position = Some(p);
        eprintln!("genopt-client3: DEBUG NEGATIVE CONTROL: expectation corrupted at position {p}");
    }
    if let Some(d) = &hooks.dump_dir {
        eprintln!("genopt-client3: DEBUG: dumping bodies to {d:?} (timings contaminated)");
    }
    let urls: Vec<http::Url> = args.urls.iter().map(|u| http::parse_url(u).unwrap_or_else(|e| die(&e))).collect();
    let prefix = payload_prefix(&args, &pool);
    let sh = Arc::new(Shared {
        cfg,
        compact,
        prefix,
        urls,
        hooks,
        go: AtomicBool::new(false),
        finished: AtomicUsize::new(0),
        records: Mutex::new((0..args.requests).map(|_| None).collect()),
        errors: Mutex::new(vec![]),
        done_cv: Condvar::new(),
        done_m: Mutex::new(()),
        bodies: Sem::new((args.workers * args.max_reading) as u64),
        bytes: Sem::new(args.memory_budget_bytes),
        parsers: Sem::new(args.parse_threads as u64),
        args: args.clone(),
    });
    let mut handles = Vec::with_capacity(args.requests);
    for i in 0..args.requests {
        let sh2 = sh.clone();
        let h = std::thread::Builder::new()
            .name(format!("req-{i}"))
            .stack_size(8 << 20)
            .spawn(move || {
                let mut c = Completion { sh: sh2.clone(), i, done: false };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one_request(i, &sh2)));
                c.finish(r.unwrap_or_else(|p| Err(format!("request thread panicked: {}", panic_text(&*p)))));
            })
            .unwrap_or_else(|e| die(&format!("spawn: {e}")));
        handles.push(h);
    }
    // Barrier: every request consumed by the frontends, none returned yet.
    let deadline = mono() + args.barrier_timeout;
    loop {
        if sh.finished.load(Ordering::SeqCst) > 0 {
            let errs = lock(&sh.errors).clone();
            die(&format!("A request returned before the consumption barrier: {errs:?}"));
        }
        let mut consumed = vec![];
        read_consumed(&args.consumption_log, true, &mut consumed).unwrap_or_else(|e| die(&e));
        match barrier_check(&consumed, args.requests, args.output_tokens) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => die(&e),
        }
        if mono() > deadline {
            die(&format!("Frontend consumption barrier not reached: {}", consumed.len()));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let barrier_wall = wall();
    sh.go.store(true, Ordering::SeqCst);
    let started = mono();
    let started_wall = wall();
    let pause_start = json!({"monotonic": started, "wall": started_wall, "barrier_wall": barrier_wall});
    std::fs::write(args.result.with_file_name("pause-start.json"), pause_start.to_string())
        .unwrap_or_else(|e| die(&format!("write pause-start.json: {e}")));
    let completion_deadline = started + args.completion_timeout;
    // The pause request runs on its own thread so the completion deadline
    // also bounds a pause response that trickles forever.
    let (ptx, prx) = std::sync::mpsc::channel();
    {
        let sh = sh.clone();
        let t = args.http_timeout;
        std::thread::spawn(move || {
            let r = (|| -> Result<u16, String> {
                let mut c = http::Conn::connect(&sh.urls[0], t)?;
                c.send_pause(&sh.urls[0])?;
                let head = c.read_head()?;
                c.read_body(&head, 64 << 20)?;
                Ok(head.status)
            })();
            let _ = ptx.send(r);
        });
    }
    let left = (completion_deadline - mono()).max(0.0);
    let pause_status = match prx.recv_timeout(std::time::Duration::from_secs_f64(left)) {
        Ok(r) => r.unwrap_or_else(|e| die(&format!("pause request: {e}"))),
        Err(_) => die(&format!("completion deadline ({} s after pause start) exceeded while the pause request was outstanding", args.completion_timeout)),
    };
    let pause_elapsed = mono() - started;
    if !(200..300).contains(&pause_status) {
        die(&format!("pause returned HTTP {pause_status}"));
    }
    {
        let mut g = lock(&sh.done_m);
        loop {
            if let Some(e) = lock(&sh.errors).first() {
                die(e);
            }
            if sh.finished.load(Ordering::SeqCst) == args.requests {
                break;
            }
            if mono() > completion_deadline {
                die(&format!(
                    "completion deadline ({} s after pause) exceeded with {}/{} requests finished",
                    args.completion_timeout,
                    sh.finished.load(Ordering::SeqCst),
                    args.requests
                ));
            }
            g = sh.done_cv.wait_timeout(g, std::time::Duration::from_millis(500)).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
    for h in handles {
        h.join().ok();
    }
    let recs = lock(&sh.records);
    let rows: Vec<&Rec> = recs.iter().map(|r| r.as_ref().unwrap_or_else(|| die("missing record"))).collect();
    let fmax = |f: &dyn Fn(&Rec) -> f64| rows.iter().map(|r| f(r)).fold(f64::NEG_INFINITY, f64::max);
    let fsum = |f: &dyn Fn(&Rec) -> f64| rows.iter().map(|r| f(r)).sum::<f64>();
    let top: BTreeSet<usize> = rows.iter().flat_map(|r| r.top_entries.iter().copied()).collect();
    let debug = sh.hooks.active();
    let status = if debug { "DEBUG" } else { "PASS" };
    let report = json!({
        "client": "v3-native",
        "status": status,
        "receive_policy": RECEIVE_POLICY,
        "workers": args.workers,
        "max_reading_per_worker": args.max_reading,
        "max_bodies_in_memory": args.workers * args.max_reading,
        "parse_threads": args.parse_threads,
        "memory_budget_bytes": args.memory_budget_bytes,
        "max_body_bytes": args.max_body_bytes,
        "http_timeout_s": args.http_timeout,
        "completion_timeout_s": args.completion_timeout,
        "endpoint": "/inference/v1/generate",
        "logprobs_format": args.logprobs_format.clone().unwrap_or_else(|| "openai".into()),
        "requests": args.requests,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "top_logprobs": TOP_K,
        "compact_include_sampled": args.compact_include_sampled,
        "compact_include_ranks": args.compact_include_ranks,
        "routed_experts_layers": args.routed_layers,
        "pause_s": pause_elapsed,
        "all_headers_s": fmax(&|r| r.headers_t) - started,
        "all_bodies_received_s": fmax(&|r| r.received_t) - started,
        "all_parsed_s": fmax(&|r| r.parsed_t) - started,
        "v3_sha_parse_validate_cpu_s_sum": fsum(&|r| r.cpu_s),
        "v3_sha256_s_sum": fsum(&|r| r.sha_s),
        "v3_parse_validate_s_sum": fsum(&|r| r.parse_validate_s),
        "v3_post_parse_bookkeeping_s_sum": fsum(&|r| r.bookkeeping_s),
        "v3_parse_scratch_peak_bytes_max": rows.iter().map(|r| r.scratch_peak).max().unwrap_or(0),
        "client_process_cpu_s": rusage_cpu(),
        "client_max_rss_gib": vm_hwm_gib(),
        "semantically_validated_requests": rows.iter().filter(|r| r.validated).count(),
        "top_entries_per_position": top.into_iter().collect::<Vec<_>>(),
        "response_bytes": rows.iter().map(|r| r.bytes).collect::<Vec<_>>(),
        "response_sha256": rows.iter().map(|r| r.sha.clone()).collect::<Vec<_>>(),
        "per_request": rows.iter().enumerate().map(|(i, r)| json!({
            "index": i, "headers_t": r.headers_t, "received_t": r.received_t, "parsed_t": r.parsed_t,
        })).collect::<Vec<_>>(),
        "pause_monotonic": started,
        "debug_hooks": sh.hooks.json(),
        "build": build,
        "field_semantics": {
            "headers_t": "CLOCK_MONOTONIC s when the response head was parsed (before any permit)",
            "received_t": "CLOCK_MONOTONIC s when the last body byte was in the client buffer (includes waiting for a body permit/byte reservation)",
            "parsed_t": "CLOCK_MONOTONIC s after sha256 + parse + validation + freeing the buffer",
            "all_*_s": "max over requests of the event minus pause start; overlapping end points, never add them",
            "v3_sha_parse_validate_cpu_s_sum": "sum of per-request thread CPU for sha256 + parse + validation (NOT comparable to v1/v2 post_receive_parse_cpu_s_sum, which excludes sha256)",
            "v3_sha256_s_sum": "sum of per-request sha256 wall time",
            "v3_parse_validate_s_sum": "sum of per-request single-pass JSON parse + base64 + structural + semantic validation wall time",
            "v3_post_parse_bookkeeping_s_sum": "sum of buffer/scratch/connection release time after validation",
            "v3_parse_scratch_peak_bytes_max": "max over requests of charged parse memory (value trees, decoded strings, base64/npy scratch, key sets)",
            "client_process_cpu_s": "whole-process user+sys CPU (getrusage)",
            "client_max_rss_gib": "VmHWM of the client process",
            "per_request": "no worker field: v3 threads have no worker affinity"
        },
        "note": "v3-native (Rust). Every body: full strict JSON parse (Python json.loads acceptance incl. 4300-digit int \
limit and nesting <= 9990 containers), duplicate keys rejected in checked objects, request_id, one choice, finish=abort, \
token_ids length and values; compact: strict base64 decode + sizes of all arrays, num_slots/sampled_slot/ranks per \
the include_sampled/include_ranks switches; openai: len(content); R3 (routed_experts_layers > 0): base64 + strict .npy \
header (version 1.0/2.0, descr |u1, C order, shape (P+N-1, L, 8)) for every body, else routed_experts must be absent/null. \
Indices < validate_requests: v1 full per-position semantics incl. every R3 value. Strict HTTP framing; request headers identical to httpx 0.28.1.",
    });
    std::fs::write(&args.result, serde_json::to_string_pretty(&report).unwrap() + "\n")
        .unwrap_or_else(|e| die(&format!("write result: {e}")));
    let mut short = report.clone();
    for k in ["response_sha256", "response_bytes", "per_request", "note", "field_semantics", "receive_policy"] {
        short.as_object_mut().unwrap().remove(k);
    }
    println!("{short}");
    if debug {
        eprintln!("genopt-client3: status DEBUG (debug hooks active): exit 2, not a measurement");
        std::process::exit(2);
    }
}

/// bench-body --body FILE --index I --output-tokens N --token-pool P
///   [--logprobs-format compact] [--validated 0|1] [--repeat K] [--threads T]
/// Parses a recorded body K times on each of T threads; prints throughput.
fn bench_body(argv: &[String]) {
    let mut body = PathBuf::new();
    let mut index = 0usize;
    let mut n = 0usize;
    let mut pool_path = PathBuf::new();
    let mut compact = false;
    let mut validated = true;
    let mut repeat = 1usize;
    let mut threads = 1usize;
    let mut include_sampled = true;
    let mut include_ranks = true;
    let mut routed_layers = 0usize;
    let mut input_tokens = 16384usize;
    let mut i = 0;
    while i < argv.len() {
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        let f = argv[i].clone();
        match f.as_str() {
            "--body" => body = v.into(),
            "--index" => index = int_arg(&f, &v, 0) as usize,
            "--output-tokens" => n = int_arg(&f, &v, 1) as usize,
            "--token-pool" => pool_path = v.into(),
            "--logprobs-format" => compact = v == "compact",
            "--validated" => validated = v == "1",
            "--repeat" => repeat = int_arg(&f, &v, 1) as usize,
            "--threads" => threads = int_arg(&f, &v, 1) as usize,
            "--input-tokens" => input_tokens = int_arg(&f, &v, 0) as usize,
            "--routed-experts-layers" => routed_layers = int_arg(&f, &v, 0) as usize,
            "--compact-no-sampled" => {
                include_sampled = false;
                i += 1;
                continue;
            }
            "--compact-no-ranks" => {
                include_ranks = false;
                i += 1;
                continue;
            }
            o => die(&format!("bench-body: unknown {o}")),
        }
        i += 2;
    }
    let data = Arc::new(std::fs::read(&body).unwrap_or_else(|e| die(&format!("{body:?}: {e}"))));
    let mut cfg0 = Cfg::new(n, compact, load_pool(&pool_path));
    cfg0.include_sampled = include_sampled;
    cfg0.include_ranks = include_ranks;
    cfg0.routed_layers = routed_layers;
    cfg0.routed_rows = if routed_layers > 0 { input_tokens + n - 1 } else { 0 };
    let cfg = Arc::new(cfg0);
    let t0 = mono();
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let data = data.clone();
            let cfg = cfg.clone();
            std::thread::spawn(move || {
                let mut sc = Scratch::new();
                let mut out = vec![];
                for _ in 0..repeat {
                    let mem = json::Mem::unlimited();
                    let r = process_body(&data, index, validated, &cfg, &mut sc, &mem);
                    out.push(r.map(|x| (x, mem.peak())));
                }
                out
            })
        })
        .collect();
    let mut sha_s = 0.0;
    let mut parse_s = 0.0;
    let mut cpu_s = 0.0;
    let mut result = json!(null);
    let mut ok = true;
    for h in hs {
        for r in h.join().unwrap() {
            match r {
                Ok(((sha, a, b, c, st), peak)) => {
                    sha_s += a;
                    parse_s += b;
                    cpu_s += c;
                    result = json!({"sha256": sha, "validated": st.validated, "parse_scratch_peak_bytes": peak,
                                    "top_entries_per_position": st.top_entries.iter().collect::<Vec<_>>()});
                }
                Err(e) => {
                    ok = false;
                    result = json!({"error": e});
                }
            }
        }
    }
    let elapsed = mono() - t0;
    let k = (repeat * threads) as f64;
    let gb = data.len() as f64 / 1e9;
    println!(
        "{}",
        json!({
            "status": if ok {"PASS"} else {"FAIL"}, "result": result, "bytes": data.len(),
            "threads": threads, "repeat": repeat, "wall_s": elapsed,
            "aggregate_GBps": gb * k / elapsed,
            "per_body_sha256_s": sha_s / k, "per_body_parse_validate_s": parse_s / k,
            "per_body_cpu_s": cpu_s / k,
            "per_core_GBps_sha_plus_parse": gb / (cpu_s / k),
            "sha256_GBps": gb / (sha_s / k), "parse_validate_GBps": gb / (parse_s / k),
        })
    );
    if !ok {
        std::process::exit(1);
    }
}

fn main() {
    let build = check_cpu();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    sha_selfcheck();
    let bad = validate::scores_selfcheck();
    if bad != 0 {
        eprintln!("genopt-client3: note: Rust recomputation of expected_scores differs in {bad} entries; using numpy-derived constants");
    }
    match argv.first().map(String::as_str) {
        Some("bench-body") => bench_body(&argv[1..]),
        Some("replay-server") => replay::serve(&argv[1..]),
        Some("build-info") => println!("{}", serde_json::to_string_pretty(&build).unwrap()),
        _ => run(parse_args(&argv), build),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: serde_json::Value, t: u64) -> serde_json::Value {
        json!({"request_id": id, "tokens": t, "logprob_positions": t})
    }

    #[test]
    fn barrier_rules() {
        let ok = vec![rec(json!("mock-0000"), 5), rec(json!("generate-tokens-mock-0001"), 5)];
        assert_eq!(barrier_check(&ok, 2, 5), Ok(true));
        assert_eq!(barrier_check(&ok[..1], 2, 5), Ok(false));
        assert!(barrier_check(&[json!({"tokens": 5, "logprob_positions": 5})], 2, 5).is_err()); // no id
        assert!(barrier_check(&[rec(json!(7), 5)], 2, 5).is_err()); // non-string id
        assert!(barrier_check(&[rec(json!(["mock-0000"]), 5)], 2, 5).is_err());
        assert!(barrier_check(&[rec(json!("mock-0000"), 5), rec(json!("mock-0000"), 5)], 2, 5).is_err()); // dup
        assert!(barrier_check(&[rec(json!("mock-0000"), 5), rec(json!("mock-0002"), 5)], 2, 5).is_err()); // wrong set
        assert!(barrier_check(&[rec(json!("mock-0000"), 5), rec(json!("other-1"), 5)], 2, 5).is_err());
        assert!(barrier_check(&[rec(json!("mock-0000"), 5), rec(json!("mock-0001"), 4)], 2, 5).is_err()); // counts
        assert!(barrier_check(&[ok[0].clone(), ok[1].clone(), rec(json!("mock-0002"), 5)], 2, 5).is_err()); // too many
    }

    #[test]
    fn permits() {
        let s = Sem::new(10);
        assert!(s.acquire(11).is_err());
        let mut p = s.acquire(10).unwrap();
        p.shrink(4);
        let q = s.acquire(6).unwrap();
        drop(p);
        drop(q);
        assert_eq!(*s.m.lock().unwrap(), 10);
    }

    #[test]
    fn max_body_defaults() {
        // observed bodies at n=245760 must fit with margin
        assert!(default_max_body(245_760, true, 0) > 340_975_635 * 12 / 10);
        assert!(default_max_body(245_760, false, 0) > 3_408_730_748 * 2);
        let npy = (16_384 + 245_760 - 1) * 4 * 8 + 4096;
        assert!(default_max_body(245_760, true, npy) - default_max_body(245_760, true, 0) > b64_len(npy));
    }
}
