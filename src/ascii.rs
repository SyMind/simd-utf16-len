//! ASCII scanning adapted from Rust 1.98.0's `core::slice::ascii`:
//! https://github.com/rust-lang/rust/blob/d1fc603d1788cc3c0eebdb94a45a61c4f33b1674/library/core/src/slice/ascii.rs
//!
//! Keep the standard library's runtime scanning strategy, returning the length
//! on success and the already verified prefix on failure. This lets UTF-16
//! counting resume without scanning that prefix again.
//!
//! Copyright (c) The Rust Project Contributors. See LICENSE-MIT-RUST.

/// Return the length for ASCII, or a prefix known to contain only ASCII bytes.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn ascii_prefix_len(bytes: &[u8]) -> usize {
    const USIZE_SIZE: usize = size_of::<usize>();
    const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;

    // Match the standard library's word-at-a-time path for small inputs.
    if bytes.len() < 64 {
        let chunks = bytes.chunks_exact(USIZE_SIZE);
        let remainder = chunks.remainder();
        for chunk in chunks {
            let word = usize::from_ne_bytes(chunk.try_into().unwrap());
            if (word & NONASCII_MASK) != 0 {
                // SAFETY: chunk starts within the same allocation as bytes.
                return unsafe { chunk.as_ptr().offset_from_unsigned(bytes.as_ptr()) };
            }
        }
        return if remainder.iter().all(|b| b.is_ascii()) {
            bytes.len()
        } else {
            bytes.len() - remainder.len()
        };
    }

    ascii_prefix_len_sse2(bytes)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ascii_prefix_len_sse2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128};

    let (chunks, rest) = bytes.as_chunks::<64>();
    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes. SSE2 is baseline on x86_64.
        let mask = unsafe {
            let a1 = _mm_loadu_si128(ptr as *const __m128i);
            let a2 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
            let b1 = _mm_loadu_si128(ptr.add(32) as *const __m128i);
            let b2 = _mm_loadu_si128(ptr.add(48) as *const __m128i);
            let combined = _mm_or_si128(_mm_or_si128(a1, a2), _mm_or_si128(b1, b2));
            _mm_movemask_epi8(combined)
        };
        if mask != 0 {
            // SAFETY: chunk starts within the same allocation as bytes.
            return unsafe { ptr.offset_from_unsigned(bytes.as_ptr()) };
        }
    }

    if rest.iter().all(|b| b.is_ascii()) {
        bytes.len()
    } else {
        bytes.len() - rest.len()
    }
}

/// Match the standard library's word-at-a-time path on aarch64 and wasm32.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn ascii_prefix_len(bytes: &[u8]) -> usize {
    const USIZE_SIZE: usize = size_of::<usize>();

    const fn contains_nonascii(word: usize) -> bool {
        const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;
        (word & NONASCII_MASK) != 0
    }

    let len = bytes.len();
    let align_offset = bytes.as_ptr().align_offset(USIZE_SIZE);
    if len < USIZE_SIZE || len < align_offset || USIZE_SIZE < align_of::<usize>() {
        // Match is_ascii_simple's reverse byte scan. On failure no prefix has
        // been verified, so counting must start at zero.
        let mut remaining = bytes;
        while let [rest @ .., last] = remaining {
            if !last.is_ascii() {
                return 0;
            }
            remaining = rest;
        }
        return len;
    }

    let offset_to_aligned = if align_offset == 0 {
        USIZE_SIZE
    } else {
        align_offset
    };
    let start = bytes.as_ptr();
    // SAFETY: len >= USIZE_SIZE was checked above.
    let first_word = unsafe { start.cast::<usize>().read_unaligned() };
    if contains_nonascii(first_word) {
        return 0;
    }
    debug_assert!(offset_to_aligned <= len);

    // SAFETY: the offset is in bounds and aligns the pointer for usize loads.
    let mut word_ptr = unsafe { start.add(offset_to_aligned).cast::<usize>() };
    let mut byte_pos = offset_to_aligned;
    debug_assert!(word_ptr.is_aligned());

    // Leave the last word for an overlapping unaligned tail load, as in std.
    while byte_pos < len - USIZE_SIZE {
        debug_assert!(byte_pos + USIZE_SIZE <= len);
        debug_assert!(word_ptr.cast::<u8>() == start.wrapping_add(byte_pos));
        // SAFETY: word_ptr is aligned and the full word lies within bytes.
        let word = unsafe { word_ptr.read() };
        if contains_nonascii(word) {
            return byte_pos;
        }
        byte_pos += USIZE_SIZE;
        // SAFETY: byte_pos remains within the slice after this increment.
        word_ptr = unsafe { word_ptr.add(1) };
    }

    debug_assert!(byte_pos <= len && len - byte_pos <= USIZE_SIZE);
    // SAFETY: len >= USIZE_SIZE, so the last word is entirely within bytes.
    let last_word = unsafe { start.add(len - USIZE_SIZE).cast::<usize>().read_unaligned() };
    if contains_nonascii(last_word) {
        byte_pos
    } else {
        len
    }
}
