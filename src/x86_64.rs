//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
#[allow(unsafe_code)]
pub fn utf16_len(s: &str) -> usize {
    let len = s.len();
    if len < 16 {
        if s.is_ascii() {
            return len;
        }
        return crate::scalar::utf16_len(s);
    }

    utf16_length_sse2(s)
}

/// Process complete vectors, skipping classification for ASCII blocks.
#[inline]
fn utf16_length_sse2(s: &str) -> usize {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut count = 0;

    // SAFETY: SSE2 is always available on x86_64. Each vector load is guarded
    // by the length of its block; the caller handles strings below 16 bytes.
    unsafe {
        let cont_max = _mm_set1_epi8(-65);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let zero = _mm_setzero_si128();

        while i + 64 <= len {
            let ptr = bytes.as_ptr().add(i);
            let a = _mm_loadu_si128(ptr.cast());
            let b = _mm_loadu_si128(ptr.add(16).cast());
            let c = _mm_loadu_si128(ptr.add(32).cast());
            let d = _mm_loadu_si128(ptr.add(48).cast());
            let combined = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
            if _mm_movemask_epi8(combined) == 0 {
                count += 64;
            } else {
                // Reuse the vectors loaded by the ASCII check. Four vectors
                // contribute at most eight units per lane, so u8 cannot overflow.
                let mut units = zero;
                for chunk in [a, b, c, d] {
                    let leaders = _mm_cmpgt_epi8(chunk, cont_max);
                    let fours = _mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask);
                    units = _mm_sub_epi8(_mm_sub_epi8(units, leaders), fours);
                }
                let sad = _mm_sad_epu8(units, zero);
                count += _mm_cvtsi128_si64(_mm_add_epi64(sad, _mm_srli_si128::<8>(sad))) as usize;
            }
            i += 64;
        }

        while i + 16 <= len {
            let chunk = _mm_loadu_si128(bytes.as_ptr().add(i).cast());
            let leaders = _mm_cmpgt_epi8(chunk, cont_max);
            let fours = _mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask);
            let units = _mm_sub_epi8(_mm_sub_epi8(zero, leaders), fours);
            let sad = _mm_sad_epu8(units, zero);
            count += _mm_cvtsi128_si64(_mm_add_epi64(sad, _mm_srli_si128::<8>(sad))) as usize;
            i += 16;
        }

        if i < len {
            // Reload the final vector and exclude bytes already counted.
            let chunk = _mm_loadu_si128(bytes.as_ptr().add(len - 16).cast());
            let leaders = _mm_movemask_epi8(_mm_cmpgt_epi8(chunk, cont_max)) as u32;
            let fours =
                _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask))
                    as u32;
            let skip = 16 - (len - i);
            count += ((leaders >> skip) | ((fours >> skip) << 16)).count_ones() as usize;
        }
    }
    count
}
