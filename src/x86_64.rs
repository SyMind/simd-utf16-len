//! x86_64 SIMD UTF-16 length calculation.
//!
//! Uses SSE2 (16 bytes at a time, always available on x86_64).

use std::arch::x86_64::*;

/// Compute the number of UTF-16 code units for UTF-8 string.
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        utf16_length_sse2(bytes, start)
    }
}

/// Count UTF-8 bytes after a verified ASCII prefix, including short inputs.
#[inline(always)]
fn utf16_length_sse2(bytes: &[u8], mut i: usize) -> usize {
    let len = bytes.len();
    let mut count = i;

    // SAFETY: SSE2 is always available on x86_64, and every load is guarded by
    // `i + 16 <= len`.
    unsafe {
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let zero = _mm_setzero_si128();

        // ASCII bytes and UTF-8 leaders contribute one unit, with one extra
        // unit for four-byte leaders. Independent accumulators avoid a serial
        // dependency between the two additions and allow 255 iterations.
        while i + 16 <= len {
            let batch = ((len - i) / 16).min(255);
            let mut leader_acc = zero;
            let mut four_acc = zero;
            for _ in 0..batch {
                let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
                let is_leader = _mm_cmpgt_epi8(chunk, cont_max);
                let is_four = _mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask);
                leader_acc = _mm_sub_epi8(leader_acc, is_leader);
                four_acc = _mm_sub_epi8(four_acc, is_four);
                i += 16;
            }
            let sad = _mm_add_epi64(_mm_sad_epu8(leader_acc, zero), _mm_sad_epu8(four_acc, zero));
            let sum = _mm_add_epi64(sad, _mm_srli_si128::<8>(sad));
            count += _mm_cvtsi128_si64(sum) as usize;
        }
    }

    // SAFETY: i starts at the verified prefix and only advances across full
    // in-bounds vectors, so the remaining slice is valid.
    let tail = unsafe { bytes.get_unchecked(i..) };
    if tail.len() < 4 {
        // A complete four-byte character cannot start in the final three
        // bytes of valid UTF-8. Count only ASCII bytes and shorter leaders.
        count += tail.iter().filter(|&&byte| (byte as i8) > -65).count();
    } else if len >= 16 {
        // SAFETY: len >= 16 was checked above. Reload the last full vector,
        // then discard mask bits for the bytes already counted by the loop.
        unsafe {
            let chunk = _mm_loadu_si128(bytes.as_ptr().add(len - 16) as *const __m128i);
            let leaders = _mm_movemask_epi8(_mm_cmpgt_epi8(chunk, _mm_set1_epi8(-65))) as u32;
            let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
            let fours =
                _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(chunk, four_mask), four_mask))
                    as u32;
            let skip = 16 - tail.len();
            // Keep the two masks in separate halves so a four-byte leader
            // contributes twice, without requiring the POPCNT CPU feature.
            count += ((leaders >> skip) | ((fours >> skip) << 16)).count_ones() as usize;
        }
    } else {
        // The input is too short for a full vector load. Apply the same
        // leader + four-byte-leader formula directly to the remaining bytes.
        count = tail.iter().fold(count, |count, &byte| {
            count + usize::from((byte as i8) > -65) + usize::from(byte >= 0xF0)
        });
    }
    count
}
