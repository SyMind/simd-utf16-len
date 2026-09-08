use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::utf16_len;
use std::time::Duration;

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";

const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";

const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨‍👩‍👧‍👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";

const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

#[inline]
fn ascii_guard_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}

fn bench_inputs(c: &mut Criterion) {
    let ascii_long = ASCII.repeat(64);
    let cjk_long = CJK.repeat(64);
    let emoji_long = EMOJI.repeat(64);
    let mixed_long = MIXED.repeat(64);
    let inputs: &[(&str, &str)] = &[
        ("ascii", ASCII),
        ("ascii_long", ascii_long.as_str()),
        ("cjk", CJK),
        ("emoji", EMOJI),
        ("mixed", MIXED),
        ("cjk_long", cjk_long.as_str()),
        ("emoji_long", emoji_long.as_str()),
        ("mixed_long", mixed_long.as_str()),
    ];

    let mut group = c.benchmark_group("utf16_len");
    for &(name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "encode_utf16"), |b| {
            b.iter(|| black_box(input).encode_utf16().count());
        });
        if name.starts_with("ascii") {
            group.bench_function(BenchmarkId::new(name, "is_ascii"), |b| {
                b.iter(|| ascii_guard_len(black_box(input)));
            });
        }
    }
    group.finish();
}

fn bench_ascii_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ascii_sizes");
    for len in [0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 256, 4096] {
        let input = "a".repeat(len);
        group.bench_function(BenchmarkId::new("simd", len), |b| {
            b.iter(|| utf16_len(black_box(input.as_str())));
        });
        group.bench_function(BenchmarkId::new("is_ascii", len), |b| {
            b.iter(|| ascii_guard_len(black_box(input.as_str())));
        });
    }
    group.finish();
}

fn bench_non_ascii_position(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_ascii_position");
    for prefix_len in [0, 2048, 4092] {
        let input = "a".repeat(prefix_len) + "🦀" + &"a".repeat(4092 - prefix_len);
        group.bench_function(BenchmarkId::new("simd", prefix_len), |b| {
            b.iter(|| utf16_len(black_box(input.as_str())));
        });
        group.bench_function(BenchmarkId::new("is_ascii", prefix_len), |b| {
            b.iter(|| ascii_guard_len(black_box(input.as_str())));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = bench_inputs, bench_ascii_sizes, bench_non_ascii_position
}
criterion_main!(benches);
