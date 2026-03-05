# simd-utf16-len

SIMD-accelerated UTF-16 length calculation from UTF-8 strings — **3x\~11x faster** than `str::encode_utf16().count()`.

## How it works

Computing the UTF-16 length of a UTF-8 string doesn't require actually encoding it. The length can be derived directly from byte patterns:

```
utf16_len = byte_length - continuation_bytes + four_byte_leaders
```

Where:
- **Continuation bytes** (`(byte & 0xC0) == 0x80`) don't produce UTF-16 code units
- **Four-byte leaders** (`byte >= 0xF0`) produce surrogate pairs (2 UTF-16 code units instead of 1)

This library vectorizes both counts using SIMD, processing **16 bytes per iteration**.

## Platform support

| Architecture | SIMD | Instruction set |
|-------------|------|-----------------|
| x86_64 | SSE2 | Always available |
| aarch64 | NEON | Always available |
| wasm32 | simd128 | Requires `target_feature = "simd128"` |
| Other | — | Falls back to `encode_utf16().count()` |

## Usage

```rust
use simd_utf16_len::utf16_len;

let len = utf16_len("Hello, 世界! 🌍");
assert_eq!(len, "Hello, 世界! 🌍".encode_utf16().count());
```

## Benchmark

Benchmarks run in CI on real hardware across three platforms. See [workflow](.github/workflows/bench.yml).

### Linux x86_64

CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           14.3 |         107.4 |   7.5x |
| cjk   |   194 |           10.9 |          96.0 |   8.8x |
| emoji |   170 |           15.2 |         100.4 |   6.6x |
| mixed |   144 |            8.6 |         102.8 |  12.0x |

### macOS aarch64

CPU: Apple M1 (Virtual)

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           11.9 |          85.8 |   7.2x |
| cjk   |   194 |            8.0 |          73.5 |   9.2x |
| emoji |   170 |           12.2 |          83.0 |   6.8x |
| mixed |   144 |            6.9 |          77.8 |  11.3x |

### Windows x86_64

CPU: AMD EPYC 7763 64-Core Processor

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           14.2 |          86.5 |   6.1x |
| cjk   |   194 |           10.5 |         106.1 |  10.1x |
| emoji |   170 |           14.2 |          91.0 |   6.4x |
| mixed |   144 |            8.7 |          80.7 |   9.3x |

## License

MIT
