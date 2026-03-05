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

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           31.7 |         117.8 |   3.7x |
| cjk   |   194 |           30.7 |          99.0 |   3.2x |
| emoji |   170 |           29.0 |         102.7 |   3.5x |
| mixed |   144 |           33.5 |         107.8 |   3.2x |

### macOS aarch64

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           11.4 |          91.0 |   8.0x |
| cjk   |   194 |            8.4 |          83.7 |  10.0x |
| emoji |   170 |           13.1 |          87.3 |   6.7x |
| mixed |   144 |            7.4 |          82.3 |  11.2x |

### Windows x86_64

| Input | Bytes | SIMD (ns/iter) | std (ns/iter) | Speedup |
|-------|------:|---------------:|--------------:|--------:|
| ascii |   169 |           28.3 |          90.9 |   3.2x |
| cjk   |   194 |           12.2 |         107.1 |   8.8x |
| emoji |   170 |           13.9 |          95.9 |   6.9x |
| mixed |   144 |            8.6 |          85.2 |   9.9x |

## License

MIT
