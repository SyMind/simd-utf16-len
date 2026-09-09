//! ASCII scanning adapted from Rust main's (1.100.0-dev) `core::slice::ascii`:
//! https://github.com/rust-lang/rust/blob/4aa1fbcf467cf38ce58abfa8eb9213a789c5381c/library/core/src/slice/ascii.rs
//!
//! Keep the standard library's runtime scanning strategy, returning the length
//! on success and the already verified prefix on failure. This lets UTF-16
//! counting resume without scanning that prefix again.
//!
//! Copyright (c) The Rust Project Contributors. See LICENSE-MIT-RUST.

/// Return the length for ASCII, or a prefix known to contain only ASCII bytes.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn ascii_prefix_len(bytes: &[u8]) -> usize {
    const USIZE_SIZE: usize = size_of::<usize>();
    const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;

    // Match the standard library's word-at-a-time path for small inputs.
    if bytes.len() < 64 {
        let (chunks, remainder) = bytes.as_chunks::<USIZE_SIZE>();
        for chunk in chunks {
            let word = usize::from_ne_bytes(*chunk);
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

    #[cfg(target_arch = "x86_64")]
    {
        ascii_prefix_len_sse2(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        ascii_prefix_len_neon(bytes)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ascii_prefix_len_sse2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128};

    let (chunks, rest) = bytes.as_chunks::<64>();
    let end = chunks.len() * 64;
    let mut offset = 0;
    while offset < end {
        // SAFETY: offset starts at zero and advances by one complete chunk.
        let ptr = unsafe { bytes.as_ptr().add(offset) };
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
            return offset;
        }
        offset += 64;
    }

    if rest.iter().all(|b| b.is_ascii()) {
        bytes.len()
    } else {
        bytes.len() - rest.len()
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ascii_prefix_len_neon(bytes: &[u8]) -> usize {
    use std::arch::aarch64::{vld1q_u8, vmaxvq_u8, vorrq_u8};

    let (chunks, rest) = bytes.as_chunks::<64>();
    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes. NEON is baseline on aarch64, and these
        // vector loads do not require alignment.
        let max = unsafe {
            let a1 = vld1q_u8(ptr);
            let a2 = vld1q_u8(ptr.add(16));
            let b1 = vld1q_u8(ptr.add(32));
            let b2 = vld1q_u8(ptr.add(48));
            let combined = vorrq_u8(vorrq_u8(a1, a2), vorrq_u8(b1, b2));
            // Match std: amortize the horizontal reduction over 64 bytes.
            vmaxvq_u8(combined)
        };
        if max >= 128 {
            // SAFETY: chunk starts within the same allocation as bytes.
            return unsafe { ptr.offset_from_unsigned(bytes.as_ptr()) };
        }
    }

    // Match std's NEON tail: full vectors, then fewer than 16 scalar bytes.
    let (vectors, rest) = rest.as_chunks::<16>();
    for vector in vectors {
        // SAFETY: vector contains 16 bytes, and the load is unaligned.
        let max = unsafe { vmaxvq_u8(vld1q_u8(vector.as_ptr())) };
        if max >= 128 {
            // SAFETY: vector starts within the same allocation as bytes.
            return unsafe { vector.as_ptr().offset_from_unsigned(bytes.as_ptr()) };
        }
    }

    if rest.iter().all(|b| b.is_ascii()) {
        bytes.len()
    } else {
        bytes.len() - rest.len()
    }
}

/// Match the standard library's word-at-a-time path on wasm32.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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
