//! x86_64 SIMD UTF-16 length calculation.
//!
//! Runtime dispatch, as in napi-rs/json-escape-simd: AVX2 -> SSE2, which is
//! baseline on x86_64. Both kernels walk a pointer cursor with a
//! remaining-byte count, fold the tail into the same byte-lane accumulators
//! as the full vectors, and sum them once with `psadbw`. AVX2 counts four
//! unrolled vectors per iteration and looks up each byte's units by its high
//! nibble with `pshufb`; SSE2 has no byte shuffle, so it compares, one vector
//! per iteration. Inputs with fewer than `WIDE_MIN` bytes after their ASCII
//! prefix stay on the SSE2 kernel inlined into the dispatch: the AVX2
//! kernel's call, upper-register cleanup, and wider reduction cost more than
//! they save there.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Iterations of four vectors between two sums of the wide kernels' byte-lane
/// accumulators. A lane gains at most 2 per vector, so the merged
/// accumulators hold at most 4 * 2 * 30 = 240 after a batch, which leaves
/// room for 3 leftover vectors and the tail: 240 + 3 * 2 + 2 = 248 < 256.
const MAX_BATCH: usize = 30;

/// Below this many bytes after the ASCII prefix, the inlined SSE2 kernel
/// beats calling the AVX2 kernel, as measured by the kernel sweep on Zen 3,
/// Zen 4, and Granite Rapids.
const WIDE_MIN: usize = 64;

/// Compute the number of UTF-16 code units for UTF-8 string.
///
/// Not inlined into callers yet: inlining it, and the SSE2 kernel with it,
/// measured `mixed` and `cjk` 10 to 40% slower on Linux on Zen 3 with the
/// same instructions, so the kernel's placement decides its speed. The
/// kernels get restructured first; a later change inlines this. The ASCII
/// scan and its early return are all this function holds; the rest is a
/// tail call into `non_ascii`.
pub fn utf16_len(s: &str) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        non_ascii(bytes, start)
    }
}

/// Counts the bytes after the ASCII prefix: short inputs run the SSE2 kernel
/// here, longer ones call the widest kernel this CPU supports, as in
/// json-escape-simd's dispatch. Always inlined into `utf16_len`, so the call
/// from there is the only one; every call from here is a tail call, so this
/// saves no registers.
#[inline(never)]
fn non_ascii(bytes: &[u8], start: usize) -> usize {
    // SAFETY: bytes comes from a valid str, start is a verified ASCII prefix,
    // SSE2 is baseline on x86_64, and `detect_then_count` stores a kernel
    // only after detecting its features, so each wider kernel runs only when
    // the CPU supports them.
    unsafe {
        let nb = bytes.len() - start;
        if nb >= WIDE_MIN {
            return match WIDE_KERNEL.load(Ordering::Relaxed) {
                AVX2 => utf16_len_avx2(bytes, start),
                SSE2 => utf16_len_sse2_long(bytes, start),
                _ => detect_then_count(bytes, start),
            };
        }
        utf16_len_sse2(bytes, start)
    }
}

/// The widest kernel this CPU supports, once `detect_then_count` has looked:
/// `SSE2` or `AVX2`, and 0 before then. Keeping the selector here rather than
/// asking `is_x86_feature_detected!` in `non_ascii` keeps the detection's
/// initialization call, and the registers it would make `non_ascii` save on
/// every input, in that cold function.
static WIDE_KERNEL: AtomicU8 = AtomicU8::new(0);
const SSE2: u8 = 1;
const AVX2: u8 = 2;

/// Detects the CPU's features once, then counts with the kernel they select.
#[cold]
#[inline(never)]
fn detect_then_count(bytes: &[u8], start: usize) -> usize {
    let kernel = if is_x86_feature_detected!("avx2") {
        AVX2
    } else {
        SSE2
    };
    WIDE_KERNEL.store(kernel, Ordering::Relaxed);
    non_ascii(bytes, start)
}

/// The SSE2 kernel out of line, for long inputs on CPUs without AVX2, so
/// `non_ascii` saves no registers for it.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes.
#[inline(never)]
unsafe fn utf16_len_sse2_long(bytes: &[u8], start: usize) -> usize {
    // SAFETY: the caller's contract, and SSE2 is baseline on x86_64.
    unsafe { utf16_len_sse2(bytes, start) }
}

/// SSE2, then AVX2 when this CPU has it, in the order `utf16_len` prefers
/// them for long inputs.
pub(crate) fn kernels() -> Vec<crate::__kernels::Kernel> {
    use crate::__kernels::Kernel;
    let mut kernels = vec![Kernel {
        name: "sse2",
        utf16_len: sse2,
    }];
    if is_x86_feature_detected!("avx2") {
        kernels.push(Kernel {
            name: "avx2",
            utf16_len: avx2,
        });
    }
    kernels
}

/// The ASCII prefix scan, then `kernel` on the rest, like `utf16_len`.
#[inline(always)]
fn with_kernel(s: &str, kernel: impl FnOnce(&[u8], usize) -> usize) -> usize {
    let bytes = s.as_bytes();
    let start = crate::ascii::ascii_prefix_len(bytes);
    if start == bytes.len() {
        start
    } else {
        kernel(bytes, start)
    }
}

fn sse2(s: &str) -> usize {
    // SAFETY: SSE2 is baseline on x86_64, and with_kernel passes a valid str's
    // bytes with a verified ASCII prefix.
    with_kernel(s, |bytes, start| unsafe { utf16_len_sse2(bytes, start) })
}

fn avx2(s: &str) -> usize {
    // SAFETY: `kernels` only lists this after detecting AVX2.
    with_kernel(s, |bytes, start| unsafe { utf16_len_avx2(bytes, start) })
}

/// SSE2 kernel, `#[inline(always)]` so short inputs run it inside `non_ascii`
/// with no further call.
///
/// # Safety
/// `bytes` must be valid UTF-8 with an ASCII prefix of `start` bytes, and
/// `start < bytes.len()`.
#[inline(always)]
unsafe fn utf16_len_sse2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 16;
    /// A lane of either accumulator gains at most 1 per vector, so this many
    /// vectors and the tail fit before a sum.
    const SSE2_BATCH: usize = 254;

    // SAFETY: SSE2 is baseline on x86_64. Each full-vector load stays within
    // the input, the mask table, or the short input's halves.
    unsafe {
        let len = bytes.len();
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = len - start;

        let zero = _mm_setzero_si128();
        let cont_max = _mm_set1_epi8(0xBF_u8 as i8);
        let four_mask = _mm_set1_epi8(0xF0_u8 as i8);
        let load = |p: *const u8| _mm_loadu_si128(p as *const __m128i);

        // ASCII bytes and UTF-8 leaders contribute one unit, with one extra
        // unit for four-byte leaders. Independent accumulators avoid a serial
        // dependency between the two subtractions.
        let (mut leader_acc, mut four_acc) = (zero, zero);
        macro_rules! count {
            ($v:expr, $keep:expr) => {{
                let v = $v;
                let is_leader = _mm_cmpgt_epi8(v, cont_max);
                let is_four = _mm_cmpeq_epi8(_mm_and_si128(v, four_mask), four_mask);
                leader_acc = _mm_sub_epi8(leader_acc, $keep(is_leader));
                four_acc = _mm_sub_epi8(four_acc, $keep(is_four));
            }};
        }
        macro_rules! sum {
            () => {{
                let sad =
                    _mm_add_epi64(_mm_sad_epu8(leader_acc, zero), _mm_sad_epu8(four_acc, zero));
                _mm_cvtsi128_si64(_mm_add_epi64(sad, _mm_srli_si128::<8>(sad))) as usize
            }};
        }

        if len < LANES {
            // The whole input is shorter than a vector.
            let (v, keep) = short_vector!(sptr, nb, load);
            count!(v, |mask| _mm_and_si128(mask, keep));
            return start + sum!();
        }

        // Besides the accumulators, only the cursor, the remaining count, and
        // the running total stay live across the loop, so `utf16_len` needs
        // no callee-saved registers, which every call would pay for, ASCII
        // input included.
        let all = |mask| mask;
        let mut total = start;
        loop {
            let batch = (nb / LANES).min(SSE2_BATCH);
            for _ in 0..batch {
                count!(load(sptr), all);
                sptr = sptr.add(LANES);
            }
            nb -= batch * LANES;
            if nb < LANES {
                break;
            }
            // More full vectors follow: sum now so the lanes can't overflow.
            total += sum!();
            (leader_acc, four_acc) = (zero, zero);
        }
        if nb > 0 {
            // The vector that ends at the input's end, which holds at least a
            // vector: only its last nb lanes are uncounted, and byte-wise
            // counting needs no UTF-8 boundary care.
            let v = load(sptr.add(nb).sub(LANES));
            let keep = load(crate::keep_last(LANES, nb));
            count!(v, |mask| _mm_and_si128(mask, keep));
        }
        total + sum!()
    }
}

/// AVX2 kernel. Each byte's units come from a `pshufb` lookup of its high
/// nibble. There is no byte shift, so it shifts 16-bit lanes and masks off
/// the bits that bled in from the neighbor byte, as json-escape-simd's
/// nibble-table classifier does.
///
/// # Safety
/// The CPU must support AVX2, and `bytes` must be valid UTF-8 with an ASCII
/// prefix of `start` bytes.
#[target_feature(enable = "avx2")]
unsafe fn utf16_len_avx2(bytes: &[u8], start: usize) -> usize {
    const LANES: usize = 32;
    const CHUNK: usize = LANES * 4;

    // SAFETY: the caller checked AVX2. Each full-vector load stays within the
    // input or the mask table.
    unsafe {
        if bytes.len() < LANES {
            // The whole input is shorter than a vector: `utf16_len` never
            // sends one here, but the kernel tests do. The SSE2 kernel
            // counts it without reading past it.
            return utf16_len_sse2(bytes, start);
        }
        let mut sptr = bytes.as_ptr().add(start);
        let mut nb = bytes.len() - start;

        let zero = _mm256_setzero_si256();
        let nibble = _mm256_set1_epi8(0x0F);
        let table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            crate::UNITS_BY_HIGH_NIBBLE.as_ptr() as *const __m128i,
        ));

        let load = |p: *const u8| _mm256_loadu_si256(p as *const __m256i);
        macro_rules! units {
            ($v:expr) => {
                _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16::<4>($v), nibble))
            };
        }

        // Sums of full batches, as four u64 lanes.
        let mut total = zero;
        let (mut a0, mut a1) = (zero, zero);
        // Two accumulators, each taking two of the four vectors per
        // iteration: the loop is bound by the shuffle port, not by their
        // dependency chains, and fewer live registers means fewer saves.
        macro_rules! chunk {
            () => {
                a0 = _mm256_add_epi8(a0, units!(load(sptr)));
                a1 = _mm256_add_epi8(a1, units!(load(sptr.add(LANES))));
                a0 = _mm256_add_epi8(a0, units!(load(sptr.add(LANES * 2))));
                a1 = _mm256_add_epi8(a1, units!(load(sptr.add(LANES * 3))));
                sptr = sptr.add(CHUNK);
            };
        }

        while nb >= CHUNK * MAX_BATCH {
            for _ in 0..MAX_BATCH {
                chunk!();
            }
            nb -= CHUNK * MAX_BATCH;
            // More follows: sum now so the lanes can't overflow.
            total = _mm256_add_epi64(total, _mm256_sad_epu8(_mm256_add_epi8(a0, a1), zero));
            (a0, a1) = (zero, zero);
        }
        // Fewer than MAX_BATCH iterations remain.
        while nb >= CHUNK {
            chunk!();
            nb -= CHUNK;
        }
        while nb >= LANES {
            a0 = _mm256_add_epi8(a0, units!(load(sptr)));
            sptr = sptr.add(LANES);
            nb -= LANES;
        }
        if nb > 0 {
            // The vector that ends at the input's end, which holds at least a
            // vector: only its last nb lanes are uncounted, and byte-wise
            // counting needs no UTF-8 boundary care.
            let v = load(sptr.add(nb).sub(LANES));
            let keep = load(crate::keep_last(LANES, nb));
            a1 = _mm256_add_epi8(a1, _mm256_and_si256(units!(v), keep));
        }

        total = _mm256_add_epi64(total, _mm256_sad_epu8(_mm256_add_epi8(a0, a1), zero));
        let sum = _mm_add_epi64(
            _mm256_castsi256_si128(total),
            _mm256_extracti128_si256::<1>(total),
        );
        let sum = _mm_add_epi64(sum, _mm_srli_si128::<8>(sum));
        start + _mm_cvtsi128_si64(sum) as usize
    }
}
