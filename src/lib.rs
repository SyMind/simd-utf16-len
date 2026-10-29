//! SIMD-accelerated UTF-16 length calculation from UTF-8 bytes.
//!
//! Formula: `utf16_len = byte_length - continuation_bytes + four_byte_leaders`
//!
//! Where:
//! - continuation bytes: `(byte & 0xC0) == 0x80`
//! - four-byte leaders: `byte >= 0xF0`
//!
//! The SIMD kernels follow napi-rs/json-escape-simd's `src/simd`: a pointer
//! cursor with a remaining-byte count, unrolled vectors per iteration, and an
//! in-register tail.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
mod ascii;

/// The 16-byte vector holding an input's last `nb` bytes, with zeros after
/// them, and the mask of those `nb` lanes, as `$load` produces them.
///
/// Must be expanded inside an `unsafe` block, with `$sptr` pointing at the
/// input's last `nb` bytes and `0 < nb < 16`.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
macro_rules! short_vector {
    ($sptr:expr, $nb:expr, $load:expr) => {{
        // Without the hint, LLVM lays this block out on the path of longer
        // inputs, which measured them 10 to 18% slower on an M3 Max.
        crate::cold();
        let (sptr, nb): (*const u8, usize) = ($sptr, $nb);
        let halves = crate::short_halves(sptr, nb);
        (
            $load(halves.as_ptr().cast::<u8>()),
            $load(crate::keep_first(nb)),
        )
    }};
}

/// The `nb` bytes at `p`, `0 < nb < 16`, as the two little-endian halves of
/// a 16-byte vector, with zeros after the last byte. No read leaves the
/// bytes: two overlapping reads of 8 or 4 bytes, with the overlap shifted
/// out, or the first, middle, and last byte.
///
/// # Safety
/// `p` must point at `nb` readable bytes.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
unsafe fn short_halves(p: *const u8, nb: usize) -> [u64; 2] {
    debug_assert!(0 < nb && nb < 16);
    // SAFETY: each read below stays within the `nb` bytes at `p`.
    let [lo, hi] = unsafe {
        if nb >= 8 {
            let first = u64::from_le(p.cast::<u64>().read_unaligned());
            let last = u64::from_le(p.add(nb - 8).cast::<u64>().read_unaligned());
            // Two shifts, since one shift by 64 would be out of range.
            [first, last >> (8 * (15 - nb)) >> 8]
        } else if nb >= 4 {
            let first = u32::from_le(p.cast::<u32>().read_unaligned()) as u64;
            let last = u32::from_le(p.add(nb - 4).cast::<u32>().read_unaligned()) as u64;
            [first | ((last >> (8 * (8 - nb))) << 32), 0]
        } else {
            let first = *p as u64;
            let middle = (*p.add(nb / 2) as u64) << (8 * (nb / 2));
            let last = (*p.add(nb - 1) as u64) << (8 * (nb - 1));
            [first | middle | last, 0]
        }
    };
    [lo.to_le(), hi.to_le()]
}

/// Marks the block that calls it as unlikely, so LLVM lays it out after the
/// others. The empty call itself is dropped.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[cold]
#[inline(never)]
fn cold() {}

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
mod scalar;

/// UTF-16 code units each byte contributes, by its high nibble: ASCII bytes
/// and two- or three-byte leaders count 1, continuation bytes (`0x80..=0xBF`)
/// count 0, and four-byte leaders (`0xF0..`) count 2 for their surrogate
/// pair. The kernels shuffle this by the high nibble, the way
/// json-escape-simd's nibble-table classifier does.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
static UNITS_BY_HIGH_NIBBLE: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2];

/// Lane masks for the last vector: 64 zero bytes, 64 `0xFF` bytes, 64 zero
/// bytes. `keep_last` and `keep_first` load a window of it, so the tail
/// needs no runtime broadcast and compare.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
static KEEP: [u8; 192] = {
    let mut keep = [0u8; 192];
    let mut i = 64;
    while i < 128 {
        keep[i] = 0xFF;
        i += 1;
    }
    keep
};

/// A `lanes`-byte mask that is `0xFF` in only its last `nb` lanes, for the
/// overlapping load of the last `lanes` bytes when only `nb` are uncounted.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
fn keep_last(lanes: usize, nb: usize) -> *const u8 {
    debug_assert!(lanes <= 64 && 0 < nb && nb < lanes);
    KEEP.as_ptr().wrapping_add(64 - lanes + nb)
}

/// A mask that is `0xFF` in only its first `nb` lanes, for the vector that
/// holds the last `nb` bytes in its first lanes.
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
fn keep_first(nb: usize) -> *const u8 {
    debug_assert!(0 < nb && nb < 64);
    KEEP.as_ptr().wrapping_add(128 - nb)
}

#[cfg(target_arch = "x86_64")]
use x86_64 as arch;

#[cfg(target_arch = "aarch64")]
use aarch64 as arch;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm32 as arch;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
use scalar as arch;

/// Compute the number of UTF-16 code units in a UTF-8 string, the same
/// count as `s.encode_utf16().count()`.
///
/// ```
/// use simd_utf16_len::utf16_len;
///
/// assert_eq!(utf16_len("Hello, 世界! 🌍"), 13);
/// ```
#[inline]
pub fn utf16_len(s: &str) -> usize {
    arch::utf16_len(s.as_bytes())
}

/// Compute the number of UTF-16 code units in UTF-8 `bytes`, without
/// checking that they are UTF-8.
///
/// The result is only accurate for valid UTF-8: then it equals
/// [`utf16_len`] of the same bytes as a `str`. For any other bytes it is
/// unspecified, and in particular is not the length of a lossy decoding
/// such as [`String::from_utf8_lossy`], but the call is still safe and
/// never panics or reads outside `bytes`.
///
/// Use it when the bytes are known to be UTF-8 but aren't held as a `str`,
/// and checking them again would cost more than counting.
///
/// ```
/// use simd_utf16_len::utf16_len_bytes;
///
/// assert_eq!(utf16_len_bytes("Hello, 世界! 🌍".as_bytes()), 13);
/// ```
#[inline]
pub fn utf16_len_bytes(bytes: &[u8]) -> usize {
    arch::utf16_len(bytes)
}

/// The kernels behind `utf16_len`, for this crate's tests and benchmarks.
/// Not part of the public API.
#[doc(hidden)]
pub mod __kernels {
    /// One kernel, including the ASCII prefix scan that runs before it.
    pub struct Kernel {
        pub name: &'static str,
        pub utf16_len: fn(&[u8]) -> usize,
    }

    /// Every kernel this CPU supports, ending with the one `utf16_len` runs.
    /// Each counts any bytes, so tests can check that they agree off UTF-8.
    pub fn available() -> Vec<Kernel> {
        #[cfg(target_arch = "x86_64")]
        {
            crate::x86_64::kernels()
        }
        #[cfg(target_arch = "aarch64")]
        {
            vec![Kernel {
                name: "neon",
                utf16_len: crate::aarch64::utf16_len,
            }]
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            vec![Kernel {
                name: "simd128",
                utf16_len: crate::wasm32::utf16_len,
            }]
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128"),
        )))]
        {
            vec![Kernel {
                name: "scalar",
                utf16_len: crate::scalar::utf16_len,
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    /// `crate::utf16_len`, after checking that `utf16_len_bytes` and every
    /// kernel this CPU supports agree with it, so each test below covers all
    /// of them.
    #[track_caller]
    fn utf16_len(s: &str) -> usize {
        let result = super::utf16_len(s);
        assert_eq!(utf16_len_bytes(s.as_bytes()), result);
        result
    }

    /// `crate::utf16_len_bytes`, after checking that every kernel this CPU
    /// supports agrees with it.
    #[track_caller]
    fn utf16_len_bytes(bytes: &[u8]) -> usize {
        let result = super::utf16_len_bytes(bytes);
        for kernel in super::__kernels::available() {
            assert_eq!(
                (kernel.utf16_len)(bytes),
                result,
                "{} kernel disagrees on {} bytes",
                kernel.name,
                bytes.len()
            );
        }
        result
    }

    // CI sets this where a kernel must run, so an emulator or runner that hides
    // a CPU feature fails here instead of silently skipping that kernel.
    #[test]
    fn expected_kernels_are_available() {
        let Ok(expected) = std::env::var("SIMD_UTF16_LEN_EXPECT_KERNELS") else {
            return;
        };
        let available: Vec<_> = super::__kernels::available()
            .iter()
            .map(|kernel| kernel.name)
            .collect();
        for name in expected.split(',') {
            assert!(
                available.contains(&name),
                "{name} kernel is not available; found {available:?}"
            );
        }
    }

    /// Reference implementation using the standard library.
    fn reference(s: &str) -> usize {
        s.encode_utf16().count()
    }

    #[test]
    fn empty() {
        assert_eq!(utf16_len(""), reference(""));
    }

    #[test]
    fn ascii_only() {
        assert_eq!(utf16_len("hello"), reference("hello"));
        // Include both ends of the ASCII range and unaligned slice starts.
        let bytes: Vec<u8> = (0..272).map(|i| (i % 128) as u8).collect();
        let input = String::from_utf8(bytes).unwrap();
        for offset in 0..16 {
            for len in 0..=256 {
                let s = &input[offset..offset + len];
                assert_eq!(utf16_len(s), len, "offset: {offset}, len: {len}");
            }
        }
    }

    #[test]
    fn two_byte_chars() {
        // Latin, Cyrillic, etc.
        let s = "café résumé";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn three_byte_chars() {
        // CJK characters (U+4E00..U+9FFF)
        let s = "你好世界";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn four_byte_chars() {
        // Emoji / supplementary plane (surrogate pairs in UTF-16)
        let s = "😀🎉🚀💯";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn mixed() {
        let s = "Hello, 世界! 🌍🌎🌏 café";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn single_char_boundaries() {
        // One character of each UTF-8 width
        for c in ['a', 'é', '中', '🦀'] {
            let s = String::from(c);
            assert_eq!(utf16_len(&s), reference(&s), "char: {c}");
        }
    }

    #[test]
    fn every_leader_and_continuation_byte() {
        // The first and last code point of each UTF-8 width, so every leader
        // byte value and every continuation byte value the kernels can see.
        let s = "\u{0}\u{7f}\u{80}\u{7ff}\u{800}\u{ffff}\u{10000}\u{10ffff}";
        for repeat in 1..=40 {
            let s = s.repeat(repeat);
            assert_eq!(utf16_len(&s), reference(&s), "repeat: {repeat}");
        }
    }

    #[test]
    fn longer_than_simd_width() {
        // Ensure the SIMD loop and the tail both work (> 16 bytes).
        let s = "abcdefghijklmnopqrstuvwxyz";
        assert_eq!(utf16_len(s), reference(s));

        let s = "αβγδεζηθικλμνξοπρστυφχψω";
        assert_eq!(utf16_len(s), reference(s));

        let s = "你好世界你好世界你好世界你好世界";
        assert_eq!(utf16_len(s), reference(s));

        let s = "🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀";
        assert_eq!(utf16_len(s), reference(s));
    }

    #[test]
    fn repeated_pattern_large() {
        // Stress test: exceed the accumulator batches of every kernel several
        // times, with the maximum 2 units per byte.
        let s = "a".repeat(5000);
        assert_eq!(utf16_len(&s), reference(&s));

        for chars in [1500, 1920, 1921, 3840, 3841, 5000, 7680, 7681, 10000] {
            let s = "🦀".repeat(chars);
            assert_eq!(utf16_len(&s), reference(&s), "chars: {chars}");
        }
    }

    #[test]
    fn all_byte_widths_interleaved() {
        // Repeating pattern of 1+2+3+4 byte chars to test alignment variations.
        let pattern = "aé中🦀";
        for repeat in [1, 2, 3, 5, 6, 7, 13, 25, 26, 100, 400, 1000] {
            let s = pattern.repeat(repeat);
            assert_eq!(utf16_len(&s), reference(&s), "repeat: {repeat}");
        }
    }

    #[test]
    fn every_length_of_cjk() {
        // Every tail length of every kernel, past the leftover-vector loop.
        let storage = "中".repeat(200);
        for end in storage.char_indices().map(|(i, _)| i) {
            let s = &storage[..end];
            assert_eq!(utf16_len(s), reference(s), "len: {}", s.len());
        }
    }

    #[test]
    fn short_inputs_at_page_edges() {
        // Surround short inputs with four-byte leaders, which would change
        // the count if a load counted bytes outside the input, at every
        // offset around a page boundary.
        use std::alloc::{Layout, alloc, dealloc};
        const PAGE: usize = 4096;
        let layout = Layout::from_size_align(3 * PAGE, PAGE).unwrap();
        // SAFETY: the layout has a non-zero size.
        let buf = unsafe { alloc(layout) };
        assert!(!buf.is_null());
        // SAFETY: buf points to 3 * PAGE writable bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buf, 3 * PAGE) };
        bytes.fill(0xF0);
        let inputs = [
            "a",
            "é",
            "中",
            "🦀",
            "héllo wörld",
            "héllo wörld 中文🦀!",
            "Привет, мир! 你好",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa中",
        ];
        for input in inputs {
            for offset in (PAGE - 80..PAGE + 16).chain(2 * PAGE - 80..2 * PAGE + 16) {
                let range = offset..offset + input.len();
                bytes[range.clone()].copy_from_slice(input.as_bytes());
                let s = std::str::from_utf8(&bytes[range.clone()]).unwrap();
                assert_eq!(
                    utf16_len(s),
                    reference(input),
                    "offset: {offset}, input: {input}"
                );
                bytes[range].fill(0xF0);
            }
        }
        // SAFETY: buf came from alloc with this layout.
        unsafe { dealloc(buf, layout) };
    }

    /// The review's Section-4.2 counterexample against the production
    /// kernels: a leading U+00E9, then ASCII up to 4,294,968,960 bytes
    /// (1,920 * 2,236,963, an exact NEON batch multiple, so no residual
    /// vectors). The former 32-bit NEON total returned 1,663 here.
    // 64-bit only: its byte length does not fit a 32-bit usize, and only
    // 64-bit targets can hand the kernels such an input anyway.
    #[cfg(target_pointer_width = "64")]
    #[test]
    #[ignore = "allocates a 4 GiB string"]
    fn huge_input_total_above_u32() {
        let mut s = String::with_capacity(4_294_968_960);
        s.push('\u{e9}');
        s.extend(std::iter::repeat_n('a', 4_294_968_958));
        assert_eq!(s.len(), 4_294_968_960);
        assert_eq!(utf16_len(&s), 4_294_968_959);
    }

    #[test]
    fn invalid_utf8_counts_the_same_everywhere() {
        // Bytes that are not UTF-8 have no specified count, but every kernel,
        // and the scalar formula they all compute, should agree on them: lone
        // continuation bytes, truncated and overlong sequences, surrogates,
        // and 0xF8..=0xFF, at every length around the vector and batch sizes.
        // Four offsets rather than sixteen keep this short under Miri.
        let formula = |bytes: &[u8]| -> usize {
            bytes
                .iter()
                .map(|&b| 1 + (b >= 0xF0) as usize - ((b & 0xC0) == 0x80) as usize)
                .sum()
        };
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let noise: Vec<u8> = (0..4200)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        for offset in [0, 1, 7, 15] {
            for len in (0..=300).chain([1919, 1920, 1921, 3840, 3841, 4096]) {
                let bytes = &noise[offset..offset + len];
                assert_eq!(
                    utf16_len_bytes(bytes),
                    formula(bytes),
                    "offset: {offset}, len: {len}"
                );
            }
        }
        for bytes in [
            &b"\x80"[..],
            b"\xff",
            b"\xc3",
            b"\xe4\xb8",
            b"\xf0\x9f\xa6",
            b"\xc0\xaf",
            b"\xed\xa0\x80",
            b"\xf8\x88\x80\x80\x80",
        ] {
            let padded = [&b"a".repeat(70)[..], bytes].concat();
            for input in [bytes, &padded] {
                assert_eq!(utf16_len_bytes(input), formula(input), "bytes: {input:x?}");
            }
        }
    }

    #[test]
    fn non_ascii_after_ascii_prefix() {
        for prefix_len in (0..=129).chain([2031, 2032, 2033, 4079, 4080, 4081, 4095, 4096, 4097]) {
            for suffix in [
                "é",
                "中",
                "🦀",
                "é中🦀",
                "\u{7ff}\u{800}\u{ffff}\u{10000}\u{10ffff}",
            ] {
                for tail_len in [0, 1, 15, 16, 63, 64, 65] {
                    // Exercise aligned word loads and overlapping tails from
                    // every possible 16-byte slice alignment.
                    for offset in 0..16 {
                        let storage =
                            "a".repeat(offset + prefix_len) + suffix + &"a".repeat(tail_len);
                        let s = &storage[offset..];
                        assert_eq!(
                            utf16_len(s),
                            reference(s),
                            "offset: {offset}, prefix_len: {prefix_len}, tail_len: {tail_len}, suffix: {suffix}"
                        );
                    }
                }
            }
        }
    }
}
