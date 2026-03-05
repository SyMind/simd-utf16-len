//! SIMD-accelerated UTF-16 length calculation from UTF-8 bytes.
//!
//! Formula: `utf16_len = byte_length - continuation_bytes + four_byte_leaders`
//!
//! Where:
//! - continuation bytes: `(byte & 0xC0) == 0x80`
//! - four-byte leaders: `byte >= 0xF0`

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

#[cfg(test)]
mod tests {
    use super::utf16_len;

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
    fn longer_than_simd_width() {
        // Ensure the SIMD loop and scalar tail both work (> 16 bytes).
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
        // Stress test: exceed the 255-iteration batch boundary (255 * 16 = 4080 bytes).
        let s = "a".repeat(5000);
        assert_eq!(utf16_len(&s), reference(&s));

        let s = "🦀".repeat(1500); // 1500 * 4 = 6000 bytes
        assert_eq!(utf16_len(&s), reference(&s));
    }

    #[test]
    fn all_byte_widths_interleaved() {
        // Repeating pattern of 1+2+3+4 byte chars to test alignment variations.
        let pattern = "aé中🦀";
        let s = pattern.repeat(100);
        assert_eq!(utf16_len(&s), reference(&s));
    }
}
