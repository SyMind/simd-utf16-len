use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::utf16_len;

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
    let ascii = ASCII.repeat(64);
    let inputs: &[(&str, &str)] = &[
        ("ascii", ascii.as_str()),
        ("cjk", CJK),
        ("emoji", EMOJI),
        ("mixed", MIXED),
    ];

    let mut group = c.benchmark_group("utf16_len");
    for &(name, input) in inputs {
        group.bench_function(BenchmarkId::new(name, "baseline"), |b| {
            b.iter(|| simd_utf16_baseline::utf16_len(black_box(input)));
        });
        group.bench_function(BenchmarkId::new(name, "simd"), |b| {
            b.iter(|| utf16_len(black_box(input)));
        });
        if name == "ascii" {
            group.bench_function(BenchmarkId::new(name, "is_ascii"), |b| {
                b.iter(|| ascii_guard_len(black_box(input)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_inputs);
criterion_main!(benches);
