//! Scalar fallback for platforms without SIMD support.

/// Compute the number of UTF-16 code units for UTF-8 string using scalar code.
// Keep the counting loop shared across call sites, including with LTO.
#[inline(never)]
pub fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}
