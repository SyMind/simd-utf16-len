//! Times every kernel this CPU supports, and the `utf16_len` dispatch, on
//! non-ASCII inputs of increasing size, to show where a wider kernel starts
//! to pay off.
//!
//! Run `cargo run --release --manifest-path perf/kernels/Cargo.toml`, with
//! `--features avx512` to include the AVX-512 kernel, or the Kernel sweep
//! workflow, which does both on every runner.

use std::fmt::Write as _;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use simd_utf16_len::__kernels;

/// A kernel, or the dispatch, under its display name.
type Column = (&'static str, fn(&str) -> usize);

const SIZES: &[usize] = &[
    13, 16, 24, 32, 48, 64, 96, 128, 144, 160, 192, 208, 256, 320, 384, 512, 768, 1024, 2048,
    4096, 16384,
];
/// Target duration of one timed batch.
const BATCH: Duration = Duration::from_millis(1);
const ROUNDS: usize = 21;

/// CJK text, three bytes per character, padded with ASCII to exactly `size`
/// bytes, so every input starts with a non-ASCII byte.
fn input(size: usize) -> String {
    let mut s = "中".repeat(size / 3);
    while s.len() < size {
        s.push('a');
    }
    s
}

fn main() {
    let mut columns: Vec<Column> = __kernels::available()
        .into_iter()
        .map(|kernel| (kernel.name, kernel.utf16_len))
        .collect();
    columns.push(("utf16_len", simd_utf16_len::utf16_len));
    let inputs: Vec<String> = SIZES.iter().map(|&size| input(size)).collect();

    for input in &inputs {
        let expected = input.encode_utf16().count();
        for (name, f) in &columns {
            assert_eq!(f(input), expected, "{name} miscounts {} bytes", input.len());
        }
    }

    let mut out = String::new();
    writeln!(
        out,
        "### Kernel sweep on {} {}\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
    .unwrap();
    writeln!(
        out,
        "{}, {}. Median ns per call over {ROUNDS} rounds; every input starts with a non-ASCII byte, and `utf16_len` is the dispatch.\n",
        cpu_model().unwrap_or_else(|| "unknown CPU".to_owned()),
        command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown rustc".to_owned()),
    )
    .unwrap();
    write!(out, "| Bytes |").unwrap();
    for (name, _) in &columns {
        write!(out, " {name} |").unwrap();
    }
    out.push('\n');
    write!(out, "|------:|").unwrap();
    for _ in &columns {
        write!(out, "------:|").unwrap();
    }
    out.push('\n');

    for input in &inputs {
        let iters: Vec<u64> = columns.iter().map(|(_, f)| calibrate(f, input)).collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ROUNDS); columns.len()];
        for _ in 0..ROUNDS {
            // Interleave the columns, so drift within a round affects them equally.
            for (i, (_, f)) in columns.iter().enumerate() {
                let elapsed = time_batch(f, input, iters[i]);
                samples[i].push(elapsed.as_secs_f64() * 1e9 / iters[i] as f64);
            }
        }
        write!(out, "| {} |", input.len()).unwrap();
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
            write!(out, " {:.1} |", values[values.len() / 2]).unwrap();
        }
        out.push('\n');
    }

    print!("{out}");
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write as _;
        let written = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .and_then(|mut file| file.write_all(out.as_bytes()));
        if let Err(error) = written {
            eprintln!("failed to append to {path}: {error}");
        }
    }
}

/// The iteration count that makes one batch take at least `BATCH`.
fn calibrate(f: &fn(&str) -> usize, input: &str) -> u64 {
    let mut iters = 1;
    while time_batch(f, input, iters) < BATCH {
        iters *= 2;
    }
    iters
}

#[inline(never)]
fn time_batch(f: &fn(&str) -> usize, input: &str, iters: u64) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f(black_box(input)));
    }
    start.elapsed()
}

fn cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let field = |text: String, key: &str| {
            text.lines().find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name.trim() == key).then(|| value.trim().to_owned())
            })
        };
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| field(info, "model name"))
            .or_else(|| command_output("lscpu", &[]).and_then(|info| field(info, "Model name")))
    }
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(target_os = "windows")]
    {
        command_output(
            "powershell",
            &[
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ],
        )
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (output.status.success() && !text.is_empty()).then(|| text.to_owned())
}
