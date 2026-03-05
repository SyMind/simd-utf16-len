//! Scalar fallback for platforms without SIMD support.

/// Compute the number of UTF-16 code units for UTF-8 string using scalar code.
pub fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}
