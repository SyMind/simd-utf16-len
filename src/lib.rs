//! SIMD-accelerated UTF-16 length calculation from UTF-8 bytes.
//!
//! Formula: `utf16_len = byte_length - continuation_bytes + four_byte_leaders`
//!
//! Where:
//! - continuation bytes: `(byte & 0xC0) == 0x80`
//! - four-byte leaders: `byte >= 0xF0`
//!
//! The NEON and simd128 kernels follow napi-rs/json-escape-simd's `src/simd`:
//! a pointer cursor with a remaining-byte count, unrolled vectors per
//! iteration, an in-register tail, and for inputs shorter than one vector a
//! full-vector load that stays within the memory page instead of a copy.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
mod ascii;

#[cfg(target_arch = "x86_64")]
/// Count the tail after skipping continuation bytes at `i`.
/// The caller has already counted each preceding leader's full UTF-16 contribution.
///
/// # Safety
/// `bytes` must be valid UTF-8, and `i <= bytes.len()`.
#[inline(always)]
unsafe fn utf16_len_tail(bytes: &[u8], i: usize) -> usize {
    let mut tail_start = i;
    // SAFETY: the length check guards each byte access.
    while tail_start < bytes.len() && (unsafe { *bytes.get_unchecked(tail_start) } & 0xC0) == 0x80 {
        tail_start += 1;
    }
    // SAFETY: bytes is valid UTF-8, and tail_start <= bytes.len() is a char boundary.
    let tail = unsafe { std::str::from_utf8_unchecked(bytes.get_unchecked(tail_start..)) };
    tail.encode_utf16().count()
}

/// The vector holding an input shorter than `$lanes` bytes from its last
/// `nb` bytes on, and the mask of those `nb` lanes, as `$load` produces them.
///
/// A full-vector load forward from the last `nb` bytes can't fault when it
/// stays within their page. Otherwise the vector that ends at the input's end
/// starts before the input but within the same page, so it can't fault either.
/// Debug builds, Miri, and wasm32 copy the bytes into a zeroed placeholder
/// instead.
///
/// Must be expanded inside an `unsafe` block, with `$sptr` pointing at the
/// input's last `nb` bytes, `0 < nb < $lanes`, and `$lanes <= 64`.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
macro_rules! short_vector {
    ($sptr:expr, $nb:expr, $lanes:expr, $load:expr) => {{
        let (sptr, nb): (*const u8, usize) = ($sptr, $nb);
        if crate::OVERREAD && crate::fits_in_page(sptr, $lanes) {
            ($load(sptr), $load(crate::keep_first(nb)))
        } else if crate::OVERREAD {
            // Rare: the bytes end within a vector of their page's end. A
            // branch, rather than selects, keeps the common case's registers
            // free.
            crate::cold();
            (
                $load(sptr.wrapping_add(nb).wrapping_sub($lanes)),
                $load(crate::keep_last($lanes, nb)),
            )
        } else {
            let mut placeholder = [0u8; $lanes];
            std::ptr::copy_nonoverlapping(sptr, placeholder.as_mut_ptr(), nb);
            ($load(placeholder.as_ptr()), $load(crate::keep_first(nb)))
        }
    }};
}

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
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
static UNITS_BY_HIGH_NIBBLE: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2];

/// Lane masks for the last vector: 64 zero bytes, 64 `0xFF` bytes, 64 zero
/// bytes. `keep_last` and `keep_first` load a window of it, so the tail
/// needs no runtime broadcast and compare.
#[cfg(any(
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
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
fn keep_last(lanes: usize, nb: usize) -> *const u8 {
    debug_assert!(lanes <= 64 && 0 < nb && nb < lanes);
    KEEP.as_ptr().wrapping_add(64 - lanes + nb)
}

/// A mask that is `0xFF` in only its first `nb` lanes, for a vector loaded
/// forward from the last `nb` bytes.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
fn keep_first(nb: usize) -> *const u8 {
    debug_assert!(0 < nb && nb < 64);
    KEEP.as_ptr().wrapping_add(128 - nb)
}

/// Whether an input shorter than one vector may be read with a full-vector
/// load that reaches past its end, as napi-rs/json-escape-simd does on Linux
/// and macOS, or past its start. Windows also protects memory in 4 KiB pages.
/// The kernels only do so when the load stays within the input's page, so it
/// can't fault. Debug builds and Miri copy into a buffer instead, since they
/// would flag the read, and so does wasm32, whose linear memory has no page
/// past its end.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
const OVERREAD: bool = cfg!(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    ),
    not(debug_assertions),
    not(miri)
));

/// Marks the branch that calls it as rarely taken: the empty cold function
/// keeps LLVM from turning the branch into selects, and the call itself is
/// dropped.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[cold]
#[inline(never)]
fn cold() {}

/// Whether a `lanes`-byte load at `ptr` stays within one 4 KiB page.
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
))]
#[inline(always)]
fn fits_in_page(ptr: *const u8, lanes: usize) -> bool {
    (ptr as usize & 4095) + lanes <= 4096
}

#[cfg(target_arch = "x86_64")]
pub use x86_64::utf16_len;

#[cfg(target_arch = "aarch64")]
pub use aarch64::utf16_len;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32::utf16_len;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
pub use scalar::utf16_len;

/// The kernels behind `utf16_len`, for this crate's tests and benchmarks.
/// Not part of the public API.
#[doc(hidden)]
pub mod __kernels {
    /// One kernel, including the ASCII prefix scan that runs before it.
    pub struct Kernel {
        pub name: &'static str,
        pub utf16_len: fn(&str) -> usize,
    }

    /// Every kernel this CPU supports, ending with the one `utf16_len` runs.
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
    /// `crate::utf16_len`, after checking that every kernel this CPU supports
    /// agrees with it, so each test below covers all of them.
    #[track_caller]
    fn utf16_len(s: &str) -> usize {
        let result = super::utf16_len(s);
        for kernel in super::__kernels::available() {
            assert_eq!(
                (kernel.utf16_len)(s),
                result,
                "{} kernel disagrees on {} bytes",
                kernel.name,
                s.len()
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
        // Inputs shorter than a vector load a whole vector forward from their
        // start when that stays within the page, otherwise backward from
        // their end, or are copied in debug builds. Surround them with
        // four-byte leaders, which would change the count if a load counted
        // bytes outside the input, at every offset around a page boundary.
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
