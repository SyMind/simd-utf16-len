# simd-utf16-len

SIMD-accelerated UTF-16 length calculation from UTF-8 strings, with dedicated ASCII fast paths and no runtime dependencies.

## Usage

```rust
use simd_utf16_len::utf16_len;

assert_eq!(utf16_len("Hello"), 5);
assert_eq!(utf16_len("Hello, 世界! 🌍"), 13);
```

The result counts UTF-16 code units. Characters outside the Basic Multilingual Plane, such as `🌍`, contribute two units.

## How it works

Computing the UTF-16 length of a UTF-8 string doesn't require actually encoding it. The length can be derived directly from byte patterns:

```text
utf16_len = byte_length - continuation_bytes + four_byte_leaders
```

Where:

- **Continuation bytes** (`(byte & 0xC0) == 0x80`) don't produce UTF-16 code units
- **Four-byte leaders** (`byte >= 0xF0`) produce surrogate pairs (2 UTF-16 code units instead of 1)

The SIMD implementations first scan for an ASCII prefix. Entirely ASCII strings return their byte length; otherwise, the verified prefix contributes its byte length and the remaining bytes are counted using 16-byte SIMD vectors. The ASCII scans follow Rust 1.98.0's standard-library strategy: x86_64 uses 64-byte SSE2 blocks with a word-at-a-time path below 64 bytes, while aarch64 and wasm32 use aligned `usize` loads between unaligned first and last words.

Call `utf16_len(s)` directly when the ASCII status is unknown. If the caller already guarantees or caches that a string is ASCII, `s.len()` remains an O(1) operation and avoids scanning altogether.

## Platform support

| Architecture | SIMD | Instruction set |
|-------------|------|-----------------|
| x86_64 | SSE2 | Available by default on this architecture |
| aarch64 | NEON | Available by default on this architecture |
| wasm32 | simd128 | Requires `target_feature = "simd128"` |
| Other | — | Falls back to `encode_utf16().count()` |

## Benchmarks

The [Benchmark workflow](.github/workflows/bench.yml) runs [bench_compare](examples/bench_compare.rs) in release mode on Linux, macOS, and Windows. It compares `utf16_len` with this standard-library baseline, including an ASCII fast path:

```rust
fn std_guard_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.encode_utf16().count()
    }
}
```

Results below are from [run 34220758753](https://github.com/SyMind/simd-utf16-len/actions/runs/34220758753) on **2026-09-08**, at commit [`a241509`](https://github.com/SyMind/simd-utf16-len/commit/a2415096978d0d3923d16fdf0711963a079f7506), using Rust **1.98.1**. Each measurement uses 1,000 warmup iterations followed by 10 samples of 10,000 iterations; the example reports the middle sorted sample as nanoseconds per iteration. Speedup is the baseline time divided by the SIMD time, as reported by the example.

For this run, the 169-byte ASCII input was **1.5–2.4x faster** than the ASCII guard, and the non-ASCII inputs were **8.8–13.1x faster** than the same baseline. Results depend on input length, character distribution, CPU, and compiler; these ratios do not establish a speedup for every string or platform.

### Linux x86_64

CPU: AMD EPYC 7763 64-Core Processor. [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/34220758753/job/102043054326).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            6.8 |                16.4 |    2.4x |
| cjk   |   194 |            9.9 |                91.0 |    9.2x |
| emoji |   170 |            9.9 |               105.1 |   10.6x |
| mixed |   144 |            8.0 |               101.7 |   12.7x |

### macOS aarch64

CPU: Apple M1 (Virtual). [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/34220758753/job/102043053876).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            6.1 |                 9.2 |    1.5x |
| cjk   |   194 |            8.5 |                92.0 |   10.8x |
| emoji |   170 |           11.5 |               101.1 |    8.8x |
| mixed |   144 |            7.3 |                96.1 |   13.1x |

### Windows x86_64

CPU: AMD EPYC 7763 64-Core Processor. [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/34220758753/job/102043054189).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            8.4 |                16.4 |    2.0x |
| cjk   |   194 |           10.8 |               111.8 |   10.3x |
| emoji |   170 |           10.8 |               113.3 |   10.5x |
| mixed |   144 |            9.0 |               106.6 |   11.9x |

### Reproduce locally

```sh
cargo run --release --example bench_compare
```

Use `--release`: a plain `cargo run --example bench_compare` builds unoptimized code and does not reproduce the workflow's performance measurements. The example prints the build mode, OS, CPU, and Rust version alongside the results, and exits with an error if any SIMD result is slower than its baseline.

### CodSpeed regression tracking

The separate [CodSpeed workflow](.github/workflows/codspeed.yml) runs the [benchmark suite](benches/utf16_len.rs) in **Simulation** mode by default for pushes, pull requests, and manual runs. Its 9 cases cover long ASCII (10,816 bytes), CJK, emoji, and mixed text; they compare SIMD with `encode_utf16().count()` and include the ASCII guard for the ASCII input.

Use the [CodSpeed dashboard](https://app.codspeed.io/SyMind/simd-utf16-len) to track changes across commits and inspect flamegraphs. Simulation results represent modeled execution costs and are distinct from the native timings above. The workflow also supports **Walltime** mode through its manual `mode` input to measure actual elapsed time.

The [CI workflow](.github/workflows/ci.yml) runs `cargo test` on Linux, macOS, and Windows to check correctness.

## License

MIT
