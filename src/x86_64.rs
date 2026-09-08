//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
#[allow(unsafe_code)]
// Keep the SIMD loops shared across call sites, including with LTO.
#[inline(never)]
pub fn utf16_len(s: &str) -> usize {
    let len = s.len();
    if len < 16 {
        // At most 15 bytes, so this accumulator cannot overflow.
        return s.bytes().fold(0u8, |count, byte| {
            count + u8::from((byte as i8) > -65) + u8::from(byte >= 0xF0)
        }) as usize;
    }

    utf16_length_sse2(s)
}

/// SSE2 implementation: processes 16 bytes per iteration.
#[inline]
fn utf16_length_sse2(s: &str) -> usize {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i: usize = ascii_prefix_len_sse2(bytes);
    if i == len {
        return len;
    }

    let mut count = i;

    // SAFETY: SSE2 is always available on x86_64, and every load is guarded by
    // `i + 16 <= len`.
    unsafe {
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let zero = _mm_setzero_si128();

        // Each byte contributes 0, 1, or 2 UTF-16 code units. At most 127
        // iterations keep every u8 accumulator below 256.
        while i + 16 <= len {
            let batch = ((len - i) / 16).min(127);
            let batch_end = i + batch * 16;
            let mut acc = zero;
            while i < batch_end {
                let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
                let is_leader = _mm_cmpgt_epi8(chunk, cont_max);
                let is_four = _mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask);
                acc = _mm_sub_epi8(acc, is_leader);
                acc = _mm_sub_epi8(acc, is_four);
                i += 16;
            }
            let sad = _mm_sad_epu8(acc, zero);
            let sum = _mm_add_epi64(sad, _mm_srli_si128::<8>(sad));
            count += _mm_cvtsi128_si64(sum) as usize;
        }
    }

    if i < len {
        // SAFETY: the caller requires len >= 16. Reload the last full vector,
        // then discard mask bits for the bytes already counted by the loop.
        unsafe {
            let chunk = _mm_loadu_si128(bytes.as_ptr().add(len - 16) as *const __m128i);
            let leaders = _mm_movemask_epi8(_mm_cmpgt_epi8(chunk, _mm_set1_epi8(-65))) as u32;
            let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
            let fours =
                _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask))
                    as u32;
            let skip = 16 - (len - i);
            // Keep the two masks in separate halves so a four-byte leader
            // contributes twice, without requiring the POPCNT CPU feature.
            count += ((leaders >> skip) | ((fours >> skip) << 16)).count_ones() as usize;
        }
    }
    count
}

/// Return `bytes.len()` when all bytes are ASCII, otherwise return the start of
/// the first 16-byte block (or tail) that may contain a non-ASCII byte.
#[inline]
fn ascii_prefix_len_sse2(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let mut i = 0;

    while i + 16 <= len {
        // SAFETY: i + 16 <= len is guaranteed by the while condition, and
        // SSE2 is always available on x86_64.
        let high_bits =
            unsafe { _mm_movemask_epi8(_mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i)) };
        if high_bits != 0 {
            return i;
        }
        i += 16;
    }

    if bytes[i..].is_ascii() { len } else { i }
}
