use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use simd_utf16_len::utf16_len;

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";
const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";
const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨\u{200d}👩\u{200d}👧\u{200d}👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";
const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

const WARMUP_ITERS: u32 = 500;
const BENCH_ITERS: u32 = 10_000;

fn bench<F: Fn() -> usize>(f: F) -> Duration {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        black_box(f());
    }

    // Measure
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        black_box(f());
    }
    start.elapsed()
}

fn print_env_info() {
    println!("## Environment\n");
    println!("| Item | Value |");
    println!("|------|-------|");

    // OS
    println!("| OS | {} {} |", std::env::consts::OS, std::env::consts::ARCH);

    // CPU model
    let cpu = get_cpu_model().unwrap_or_else(|| "unknown".into());
    println!("| CPU | {} |", cpu.trim());

    // Rust version
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());
    println!("| Rust | {} |", rustc.trim());

    println!();
}

fn get_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let output = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in output.lines() {
            if line.starts_with("model name") {
                return line.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        String::from_utf8(output.stdout).ok()
    }
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("wmic")
            .args(["cpu", "get", "name"])
            .output()
            .ok()?;
        let text = String::from_utf8(output.stdout).ok()?;
        text.lines().nth(1).map(|s| s.trim().to_string())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

fn main() {
    print_env_info();

    let inputs: &[(&str, &str)] = &[
        ("ascii", ASCII),
        ("cjk", CJK),
        ("emoji", EMOJI),
        ("mixed", MIXED),
    ];

    println!("## Benchmark: SIMD vs std `encode_utf16().count()`\n");
    println!("| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |");
    println!("|-------|------:|---------------:|--------------:|--------:|");

    let mut all_passed = true;

    for &(name, input) in inputs {
        let simd_dur = bench(|| utf16_len(input));
        let std_dur = bench(|| input.encode_utf16().count());

        let simd_ns = simd_dur.as_nanos() as f64 / BENCH_ITERS as f64;
        let std_ns = std_dur.as_nanos() as f64 / BENCH_ITERS as f64;
        let speedup = std_ns / simd_ns;

        println!(
            "| {:<5} | {:>5} | {:>14.1} | {:>13.1} | {:>5.1}x |",
            name,
            input.len(),
            simd_ns,
            std_ns,
            speedup,
        );

        if speedup < 1.0 {
            all_passed = false;
            eprintln!(
                "WARNING: SIMD is slower than std for '{}' ({:.2}x)",
                name, speedup
            );
        }
    }

    println!();

    if !all_passed {
        eprintln!("Some benchmarks showed SIMD slower than std!");
        std::process::exit(1);
    }
}
