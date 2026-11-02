//! Scalar fallback for platforms without SIMD support.

/// Compute the number of UTF-16 code units for UTF-8 bytes using scalar code,
/// by the crate's byte formula, so any bytes count as the SIMD kernels count
/// them.
pub(crate) fn utf16_len(bytes: &[u8]) -> usize {
    bytes
        .iter()
        .map(|&b| 1 + (b >= 0xF0) as usize - ((b & 0xC0) == 0x80) as usize)
        .sum()
}
