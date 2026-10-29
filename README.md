# simd-utf16-len

SIMD-accelerated UTF-16 length calculation from UTF-8 strings, with dedicated ASCII fast paths and no runtime dependencies.

## Usage

```rust
use simd_utf16_len::utf16_len;

assert_eq!(utf16_len("Hello"), 5);
assert_eq!(utf16_len("Hello, 世界! 🌍"), 13);
```

The result counts UTF-16 code units. Characters outside the Basic Multilingual Plane, such as `🌍`, contribute two units.

For UTF-8 held as bytes rather than a `str`, `utf16_len_bytes` counts a `&[u8]` without validating it:

```rust
use simd_utf16_len::utf16_len_bytes;

assert_eq!(utf16_len_bytes("Hello, 世界! 🌍".as_bytes()), 13);
```

Its result is only accurate for valid UTF-8; for other bytes it is unspecified, though the call is still safe.

## How it works

Computing the UTF-16 length of a UTF-8 string doesn't require actually encoding it. The length can be derived directly from byte patterns:

```text
utf16_len = byte_length - continuation_bytes + four_byte_leaders
```

Where:

- **Continuation bytes** (`(byte & 0xC0) == 0x80`) don't produce UTF-16 code units
- **Four-byte leaders** (`byte >= 0xF0`) produce surrogate pairs (2 UTF-16 code units instead of 1)

The SIMD implementations first scan for an ASCII prefix. Entirely ASCII strings return their byte length; otherwise, the verified prefix contributes its byte length and the remaining bytes are counted using 16-byte SIMD vectors. The ASCII scans follow Rust's standard-library strategy at commit [`4aa1fbc`](https://github.com/rust-lang/rust/blob/4aa1fbcf467cf38ce58abfa8eb9213a789c5381c/library/core/src/slice/ascii.rs): x86_64 uses 64-byte SSE2 blocks, and aarch64 uses 64-byte NEON blocks with a 16-byte vector tail. Both use word-sized checks below 64 bytes. wasm32 uses aligned `usize` loads between unaligned first and last words. The adaptation returns the verified prefix length instead of a boolean; it uses the same block sizes, loads, and tail checks.

Call `utf16_len(s)` directly when the ASCII status is unknown. If the caller already guarantees or caches that a string is ASCII, `s.len()` remains an O(1) operation and avoids scanning altogether.

## Platform support

| Architecture | SIMD | Instruction set |
|-------------|------|-----------------|
| x86_64 | AVX2 / SSE2, or AVX-512BW with the `avx512` feature (not on Zen 4) | Runtime dispatch; SSE2 available by default on this architecture |
| aarch64 | NEON | Available by default on this architecture |
| wasm32 | simd128 | Requires `target_feature = "simd128"` |
| Other | — | Falls back to the same formula, one byte at a time |

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

Results below are from [run 39534553692](https://github.com/SyMind/simd-utf16-len/actions/runs/39534553692) on **2026-10-21**, at commit [`ae1e4ff`](https://github.com/SyMind/simd-utf16-len/commit/ae1e4ffdc0a49e439440f32987d07a430d41f07d), the 0.3.0 release, using Rust **1.99.1** and default features. Each measurement uses 1,000 warmup iterations followed by 10 samples of 10,000 iterations; the example reports the middle sorted sample as nanoseconds per iteration. Speedup is the baseline time divided by the SIMD time, as reported by the example.

For this run, the 169-byte ASCII input was **1.6–4.4x faster** than the ASCII guard, and the non-ASCII inputs were **15.8–24.0x faster** than the same baseline. Results depend on input length, character distribution, CPU, and compiler; these ratios do not establish a speedup for every string or platform. Hosted runners also vary from run to run: the Linux and Windows jobs land on different AMD EPYC models, and on macOS the first inputs measured, `ascii` and `cjk`, have taken up to 3x longer in other runs of the same code.

### Linux x86_64

CPU: AMD EPYC 9V74 80-Core Processor. [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/39534553692/job/119008064449).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            3.0 |                13.3 |    4.4x |
| cjk   |   194 |            6.0 |               110.5 |   18.5x |
| emoji |   170 |            5.7 |                91.6 |   16.1x |
| mixed |   144 |            5.1 |                81.3 |   15.8x |

### macOS aarch64

CPU: Apple M1 (Virtual). [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/39534553692/job/119008064417).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            3.1 |                 5.1 |    1.6x |
| cjk   |   194 |            5.7 |               137.5 |   24.0x |
| emoji |   170 |            6.2 |               121.1 |   19.6x |
| mixed |   144 |            4.7 |               101.2 |   21.4x |

### Windows x86_64

CPU: AMD EPYC 9V74 80-Core Processor. [Job output](https://github.com/SyMind/simd-utf16-len/actions/runs/39534553692/job/119008064256).

| Input | Bytes | SIMD (ns/iter) | std guard (ns/iter) | Speedup |
|-------|------:|---------------:|--------------------:|--------:|
| ascii |   169 |            4.8 |                15.3 |    3.2x |
| cjk   |   194 |            7.0 |               140.3 |   19.9x |
| emoji |   170 |            6.8 |               111.1 |   16.4x |
| mixed |   144 |            6.0 |                94.4 |   15.8x |

### Reproduce locally

```sh
cargo run --release --example bench_compare
```

Use `--release`: a plain `cargo run --example bench_compare` builds unoptimized code and does not reproduce the workflow's performance measurements. The example prints the build mode, OS, CPU, and Rust version alongside the results, and exits with an error if any SIMD result is slower than its baseline.

### Base-vs-PR check

The [Perf A/B workflow](.github/workflows/perf-ab.yml) builds `utf16_len` from the base commit and from the change into two binaries, each by the same harness from the same directory, so identical code sits at identical addresses in both. It times each input's two sides back to back in fresh processes, alternating which goes first, repeats this 7 times, and uses the median. It runs for pull requests and pushes to `main` on Linux x86_64, Windows x86_64, and macOS aarch64, and writes the median change per input to the job summary. It fails when an input's median is more than 5% slower and every run is more than 1% slower. To accept an intended slowdown, label the pull request `perf-regression-accepted`.

Run the same comparison locally against any ref:

```sh
scripts/perf-ab.sh main
```

### CodSpeed regression tracking

The separate [CodSpeed workflow](.github/workflows/codspeed.yml) runs the [benchmark suite](benches/utf16_len.rs) in **Simulation** mode by default for pushes, pull requests, and manual runs. Its 9 cases cover long ASCII (10,816 bytes), CJK, emoji, and mixed text; they compare SIMD with `encode_utf16().count()` and include the ASCII guard for the ASCII input. The long ASCII fixture has a separate benchmark identity from the historical 169-byte fixture, so changing the input size is not reported as a code regression; Unicode benchmark identities remain unchanged.

Use the [CodSpeed dashboard](https://app.codspeed.io/SyMind/simd-utf16-len) to track changes across commits and inspect flamegraphs. Simulation results represent modeled execution costs and are distinct from the native timings above. The workflow also supports **Walltime** mode through its manual `mode` input to measure actual elapsed time.

The [CI workflow](.github/workflows/ci.yml) runs `cargo test` on Linux, macOS, and Windows to check correctness.

## License

MIT
