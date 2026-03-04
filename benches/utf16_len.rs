use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use simd_utf16_len::utf16_len;

const ASCII: &str = "The quick brown fox jumps over the lazy dog. This is a longer sentence to provide more data for benchmarking purposes, with various words and punctuation marks included.";

const CJK: &str = "这是一段中文测试文本，用于测试UTF-8编码中多字节字符的处理性能。日本語のテキストも含まれています。한국어 텍스트도 포함되어 있습니다。";

const EMOJI: &str = "Hello 🌍🌎🌏! Flags: 🇺🇸🇬🇧🇯🇵🇨🇳 Family: 👨‍👩‍👧‍👦 Skin: 👋🏻👋🏼👋🏽👋🏾👋🏿 Fun: 🎉🎊🎈🎁🎄🎃";

const MIXED: &str = "Hello, 世界! 🌍 Привет мир! こんにちは世界！Héllo wörld! 你好世界！안녕하세요 세계! مرحبا بالعالم";

fn bench_inputs(c: &mut Criterion) {
    let inputs: &[(&str, &str)] = &[
        ("ascii", ASCII),
        ("cjk", CJK),
        ("emoji", EMOJI),
        ("mixed", MIXED),
    ];

    let mut group = c.benchmark_group("utf16_len");
    for &(name, input) in inputs {
        group.bench_function(BenchmarkId::new("simd", name), |b| {
            b.iter(|| black_box(utf16_len(input)));
        });
        group.bench_function(BenchmarkId::new("std", name), |b| {
            b.iter(|| black_box(input.encode_utf16().count()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_inputs);
criterion_main!(benches);
