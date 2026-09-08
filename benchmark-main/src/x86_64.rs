//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
#[allow(unsafe_code)]
pub fn utf16_len(s: &str) -> usize {
    let len = s.len();
    if len == 0 {
        return 0;
    }

    // SAFETY: Feature detection ensures the correct SIMD path is used.
    unsafe { utf16_length_sse2(s) }
}

/// SSE2 implementation: processes 16 bytes per iteration.
#[inline]
unsafe fn utf16_length_sse2(s: &str) -> usize {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut continuation_count: usize = 0;
    let mut four_byte_count: usize = 0;
    let mut i: usize = 0;

    let cont_mask = _mm_set1_epi8(0xC0_u8 as i8);
    let cont_val = _mm_set1_epi8(0x80_u8 as i8);
    let four_threshold = _mm_set1_epi8(0xEF_u8 as i8);
    let ones = _mm_set1_epi8(1);
    let zero = _mm_setzero_si128();

    // Process 16 bytes at a time, in batches of up to 255 iterations
    // to avoid u8 overflow in the per-lane accumulators.
    while i + 16 <= len {
        let batch = ((len - i) / 16).min(255);
        let mut cont_acc = zero;
        let mut four_acc = zero;

        for _ in 0..batch {
            let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

            let masked = _mm_and_si128(chunk, cont_mask);
            let is_cont = _mm_cmpeq_epi8(masked, cont_val);
            cont_acc = _mm_sub_epi8(cont_acc, is_cont);

            let sub = _mm_subs_epu8(chunk, four_threshold);
            let is_four = _mm_min_epu8(sub, ones);
            four_acc = _mm_add_epi8(four_acc, is_four);

            i += 16;
        }

        // Horizontal sum via SAD (Sum of Absolute Differences) against zero.
        let cont_sad = _mm_sad_epu8(cont_acc, zero);
        let high = _mm_srli_si128::<8>(cont_sad);
        let sum = _mm_add_epi64(cont_sad, high);
        continuation_count += _mm_cvtsi128_si64(sum) as usize;

        let four_sad = _mm_sad_epu8(four_acc, zero);
        let high = _mm_srli_si128::<8>(four_sad);
        let sum = _mm_add_epi64(four_sad, high);
        four_byte_count += _mm_cvtsi128_si64(sum) as usize;
    }

    // Tail: find the next char boundary and use encode_utf16().count().
    // Bytes between i and the char boundary are all continuation bytes,
    // contributing 0 to UTF-16 length, so we can skip them.
    let tail_start = crate::ceil_char_boundary(s, i);
    i - continuation_count + four_byte_count + s[tail_start..].encode_utf16().count()
}
