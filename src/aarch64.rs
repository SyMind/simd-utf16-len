//! NEON-based UTF-16 length calculation (always available on aarch64).
//!
//! Shaped like napi-rs/json-escape-simd's kernels: a pointer cursor with a
//! remaining-byte count, four unrolled vectors per iteration counted into
//! four byte-lane accumulators, leftover vectors and an in-register tail
//! folded into the same accumulators, and one horizontal sum. Inputs of
//! fewer than four vectors skip the accumulators and count into one. Each
//! byte's units come from a `tbl` lookup of its high nibble, as in
//! json-escape-simd's nibble-table classifier; NEON shifts bytes directly,
//! so the index needs no masking.

use std::arch::aarch64::*;

/// Iterations of four vectors between two sums of the byte-lane
/// accumulators. A lane gains at most 2 per vector, so the merged
/// accumulators hold at most 4 * 2 * 30 = 240 after a batch, which leaves
/// room for 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

/// Sum of a vector's sixteen byte lanes.
///
/// # Safety
/// NEON is baseline on aarch64.
#[inline(always)]
unsafe fn sum_byte_lanes(v: uint8x16_t) -> usize {
    // `vaddlvq_u8` has no Miri shim (llvm.aarch64.neon.uaddlv); the
    // vpadalq/vaddvq formulation is bit-identical.
    #[cfg(not(miri))]
    let sum = unsafe { vaddlvq_u8(v) as usize };
    #[cfg(miri)]
    let sum = unsafe { vaddvq_u32(vpadalq_u16(vdupq_n_u32(0), vpaddlq_u8(v))) as usize };
    sum
}

/// Compute the number of UTF-16 code units for UTF-8 bytes using NEON.
#[inline]
pub(crate) fn utf16_len(bytes: &[u8]) -> usize {
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        // SAFETY: start is a verified ASCII prefix shorter than bytes.
        unsafe { utf16_len_neon(bytes, start) }
    }
}

/// One batch's merged byte lanes, widened pairwise into the running 64-bit
/// total. Each u64 lane counts only its own eight byte lanes' units, so it
/// stays below the input's length and cannot wrap for any input a slice
/// can hold. Factored out so the regression tests drive the production
/// accumulation, not a copy of it.
///
/// # Safety
/// NEON is baseline on aarch64.
#[inline(always)]
unsafe fn widen_into_total(total: uint64x2_t, merged: uint8x16_t) -> uint64x2_t {
    unsafe { vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(merged))) }
}

/// The full count: the ASCII prefix, the residual merged byte lanes, and
/// the batch total, reduced in 64 bits. Factored out with `widen_into_total`
/// so the regression tests drive the production reduction, not a copy of it.
///
/// # Safety
/// NEON is baseline on aarch64.
#[inline(always)]
unsafe fn combine(start: usize, merged: uint8x16_t, total: uint64x2_t) -> usize {
    unsafe { start + sum_byte_lanes(merged) + vaddvq_u64(total) as usize }
}

/// A function of its own, but always inlined: a caller that inlines
/// `utf16_len` gets the whole count without a call, and one the inliner
/// turns down calls `utf16_len` as before.
///
/// # Safety
/// `bytes` must have an ASCII prefix of `start` bytes, and `start <
/// bytes.len()`.
#[inline(always)]
unsafe fn utf16_len_neon(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    // SAFETY: NEON is baseline on aarch64. Each full-vector load stays within
    // the input, the mask table, or the short input's halves.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = vdupq_n_u8(0);
        let table = vld1q_u8(crate::UNITS_BY_HIGH_NIBBLE.as_ptr());

        macro_rules! units {
            ($v:expr) => {
                vqtbl1q_u8(table, vshrq_n_u8::<4>($v))
            };
        }
        // The units of the last nb bytes, from the vector that ends at the
        // input's end, which holds at least a vector: only its last nb lanes
        // are uncounted, and byte-wise counting needs no UTF-8 boundary care.
        macro_rules! tail_units {
            () => {{
                let v = vld1q_u8(sptr.add(nb).sub(LANES));
                vandq_u8(units!(v), vld1q_u8(crate::keep_last(LANES, nb)))
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, vld1q_u8);
            return start + sum_byte_lanes(vandq_u8(units!(v), keep));
        }

        if nb < CHUNK {
            // Fewer than four vectors: one accumulator, one sum.
            let mut acc = zero;
            while nb >= LANES {
                acc = vaddq_u8(acc, units!(vld1q_u8(sptr)));
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                acc = vaddq_u8(acc, tail_units!());
            }
            return start + sum_byte_lanes(acc);
        }

        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                vaddq_u8(vaddq_u8($a0, $a1), vaddq_u8($a2, $a3))
            };
        }

        // Sums of full batches, as two u64 lanes without a horizontal add:
        // u32 lanes and a u32 reduction returned the count modulo 2^32 for
        // inputs past 4,294,968,959 units.
        let mut total = vdupq_n_u64(0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        macro_rules! chunk {
            () => {
                a0 = vaddq_u8(a0, units!(vld1q_u8(sptr)));
                a1 = vaddq_u8(a1, units!(vld1q_u8(sptr.add(LANES))));
                a2 = vaddq_u8(a2, units!(vld1q_u8(sptr.add(LANES * 2))));
                a3 = vaddq_u8(a3, units!(vld1q_u8(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_BATCH;
            // More follows: widen now so the byte lanes can't overflow.
            total = widen_into_total(total, merge!(a0, a1, a2, a3));
            (a0, a1, a2, a3) = (zero, zero, zero, zero);
        }
        // Fewer than MAX_BATCH iterations remain.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = vaddq_u8(a0, units!(vld1q_u8(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            a1 = vaddq_u8(a1, tail_units!());
        }

        combine(start, merge!(a0, a1, a2, a3), total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The merged byte lanes of one full batch of the review's Section-4.2
    /// counterexample (a leading U+00E9, then ASCII): every lane of a real
    /// ASCII batch holds 30 * 4 = 120 units, and the first batch's first
    /// lane loses one unit to the two-byte leader, so the pairwise u32 sums
    /// are [479, 480, 480, 480] for the first batch and [480, 480, 480, 480]
    /// for every later one, as the exact arithmetic model computes.
    const FIRST_BATCH: [u8; 16] = {
        let mut lanes = [120; 16];
        lanes[0] = 119;
        lanes
    };
    const ASCII_BATCH: [u8; 16] = [120; 16];

    fn merged(lanes: &[u8; 16]) -> uint8x16_t {
        // SAFETY: the array holds 16 readable bytes, and NEON is baseline.
        unsafe { vld1q_u8(lanes.as_ptr()) }
    }

    fn widen(total: uint64x2_t, lanes: &[u8; 16]) -> uint64x2_t {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { widen_into_total(total, merged(lanes)) }
    }

    /// `FIRST_BATCH` once and `ASCII_BATCH` `batches - 1` times, counted
    /// through the production widening fold and final combine.
    fn counted(batches: usize) -> usize {
        // SAFETY: NEON is baseline on aarch64.
        let mut total = widen(unsafe { vdupq_n_u64(0) }, &FIRST_BATCH);
        for _ in 1..batches {
            total = widen(total, &ASCII_BATCH);
        }
        // SAFETY: NEON is baseline on aarch64.
        unsafe { combine(0, merged(&[0; 16]), total) }
    }

    /// The Section-4.2 counterexample without its 4 GiB allocation: the
    /// exact total exceeds u32::MAX, so a 32-bit final reduction of these
    /// batches returned 1,663.
    #[test]
    fn batch_total_survives_a_wrapping_u32_sum() {
        assert_eq!(counted(2_236_963), 4_294_968_959);
    }

    /// The model's second case: 8,947,849 batches push every former u32
    /// lane itself over u32::MAX, so even widening only the final reduction
    /// of 32-bit lanes returned 895.
    #[test]
    fn batch_total_survives_wrapping_u32_lanes() {
        assert_eq!(counted(8_947_849), 17_179_870_079);
    }

    /// The final combine still adds the ASCII prefix and the residual
    /// merged byte lanes, which the huge totals above leave at zero.
    #[test]
    fn combine_adds_prefix_and_residual() {
        let mut residual = [0u8; 16];
        residual[0] = 7;
        residual[15] = 3;
        // SAFETY: NEON is baseline on aarch64.
        let total = widen(unsafe { vdupq_n_u64(0) }, &ASCII_BATCH);
        // SAFETY: NEON is baseline on aarch64.
        assert_eq!(unsafe { combine(5, merged(&residual), total) }, 5 + 10 + 1920);
    }
}
