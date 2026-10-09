//! WASM SIMD128-based UTF-16 length calculation.
//!
//! Shaped like napi-rs/json-escape-simd's kernels: a pointer cursor with a
//! remaining-byte count, four unrolled vectors per iteration counted into
//! four byte-lane accumulators, leftover vectors and an in-register tail
//! folded into the same accumulators, and one horizontal sum. Inputs of
//! fewer than four vectors skip the accumulators and count into one. Each
//! byte's units come from an `i8x16.swizzle` lookup of its high nibble, as
//! in json-escape-simd's nibble-table classifier. Inputs shorter than a
//! vector are read with smaller loads that stay within them.

use std::arch::wasm32::*;

/// Iterations of four vectors between two sums of the byte-lane
/// accumulators. A lane gains at most 2 per vector, so the merged
/// accumulators hold at most 4 * 2 * 30 = 240 after a batch, which leaves
/// room for 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

/// Compute the number of UTF-16 code units for UTF-8 string using WASM SIMD128.
#[inline]
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        non_ascii(bytes, start)
    }
}

/// A function of its own, but always inlined: a caller that inlines
/// `utf16_len` gets the whole count without a call, and one the inliner
/// turns down calls `utf16_len` as before.
#[inline(always)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    const CHUNK: usize = LANES * 4;

    let len = bytes.len();
    // SAFETY: start < len is a verified ASCII prefix length, and each
    // full-vector load stays within the input, the short input's halves, or
    // the mask table.
    unsafe {
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = u8x16_splat(0);
        let table = v128_load(crate::UNITS_BY_HIGH_NIBBLE.as_ptr() as *const v128);

        macro_rules! load {
            ($p:expr) => {
                v128_load($p as *const v128)
            };
        }
        macro_rules! units {
            ($v:expr) => {
                i8x16_swizzle(table, u8x16_shr($v, 4))
            };
        }
        let load = |p: *const u8| v128_load(p as *const v128);
        // The units of the last nb bytes, from the vector that ends at the
        // input's end, which holds at least a vector: only its last nb lanes
        // are uncounted, and byte-wise counting needs no UTF-8 boundary care.
        macro_rules! tail_units {
            () => {{
                let v = load!(sptr.add(nb).sub(LANES));
                v128_and(units!(v), load!(crate::keep_last(LANES, nb)))
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, load);
            return start + horizontal_sum_u8(v128_and(units!(v), keep));
        }

        if nb < CHUNK {
            // Fewer than four vectors: one accumulator, one sum.
            let mut acc = zero;
            while nb >= LANES {
                acc = u8x16_add(acc, units!(load!(sptr)));
                sptr = sptr.add(LANES);
                nb -= LANES;
            }
            if nb > 0 {
                acc = u8x16_add(acc, tail_units!());
            }
            return start + horizontal_sum_u8(acc);
        }

        macro_rules! merge {
            ($a0:expr, $a1:expr, $a2:expr, $a3:expr) => {
                u8x16_add(u8x16_add($a0, $a1), u8x16_add($a2, $a3))
            };
        }
        // Byte lanes widened to four u32 lanes.
        macro_rules! widen {
            ($v:expr) => {
                u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16($v))
            };
        }

        // Sums of full batches, as four u32 lanes.
        let mut total = u32x4_splat(0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        macro_rules! chunk {
            () => {
                a0 = u8x16_add(a0, units!(load!(sptr)));
                a1 = u8x16_add(a1, units!(load!(sptr.add(LANES))));
                a2 = u8x16_add(a2, units!(load!(sptr.add(LANES * 2))));
                a3 = u8x16_add(a3, units!(load!(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_BATCH;
            // More follows: widen now so the lanes can't overflow.
            total = u32x4_add(total, widen!(merge!(a0, a1, a2, a3)));
            (a0, a1, a2, a3) = (zero, zero, zero, zero);
        }
        // Fewer than MAX_BATCH iterations remain.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = u8x16_add(a0, units!(load!(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            a1 = u8x16_add(a1, tail_units!());
        }

        total = u32x4_add(total, widen!(merge!(a0, a1, a2, a3)));
        start
            + (u32x4_extract_lane::<0>(total)
                + u32x4_extract_lane::<1>(total)
                + u32x4_extract_lane::<2>(total)
                + u32x4_extract_lane::<3>(total)) as usize
    }
}

/// Horizontal sum of all u8 lanes in a v128 register.
#[inline(always)]
fn horizontal_sum_u8(v: v128) -> usize {
    // u8x16 -> i16x8 (pairwise add adjacent u8 lanes)
    let pairs = i16x8_extadd_pairwise_u8x16(v);
    // i16x8 -> i32x4 (pairwise add adjacent i16 lanes)
    let quads = i32x4_extadd_pairwise_i16x8(pairs);
    // Sum the 4 i32 lanes.
    (i32x4_extract_lane::<0>(quads)
        + i32x4_extract_lane::<1>(quads)
        + i32x4_extract_lane::<2>(quads)
        + i32x4_extract_lane::<3>(quads)) as usize
}
