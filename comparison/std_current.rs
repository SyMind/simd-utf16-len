// Rust runtime ASCII scan copied from 4aa1fbcf467cf38ce58abfa8eb9213a789c5381c.
// Copyright The Rust Project Contributors. See ../../LICENSE-MIT-RUST.

#[cfg(target_arch = "x86_64")]
const SSE2_CHUNK_SIZE: usize = 64;
#[cfg(target_arch = "aarch64")]
const NEON_CHUNK_SIZE: usize = 64;
#[cfg(target_arch = "aarch64")]
const NEON_VECTOR_SIZE: usize = 16;

#[cfg(target_arch = "x86_64")]
#[inline]
fn is_ascii_sse2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128};

    let (chunks, rest) = bytes.as_chunks::<SSE2_CHUNK_SIZE>();

    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes. SSE2 is baseline on x86_64.
        let mask = unsafe {
            let a1 = _mm_loadu_si128(ptr as *const __m128i);
            let a2 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
            let b1 = _mm_loadu_si128(ptr.add(32) as *const __m128i);
            let b2 = _mm_loadu_si128(ptr.add(48) as *const __m128i);
            // OR all chunks - if any byte has high bit set, combined will too.
            let combined = _mm_or_si128(_mm_or_si128(a1, a2), _mm_or_si128(b1, b2));
            // Create a mask from the MSBs of each byte.
            // If any byte is >= 128, its MSB is 1, so the mask will be non-zero.
            _mm_movemask_epi8(combined)
        };
        if mask != 0 {
            return false;
        }
    }

    // Handle remaining bytes
    rest.iter().all(|b| b.is_ascii())
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn is_ascii_neon(bytes: &[u8]) -> bool {
    use std::arch::aarch64::{vld1q_u8, vmaxvq_u8, vorrq_u8};

    let (chunks, rest) = bytes.as_chunks::<NEON_CHUNK_SIZE>();

    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes, and `vld1q_u8` has no alignment requirement.
        let max = unsafe {
            let a1 = vld1q_u8(ptr);
            let a2 = vld1q_u8(ptr.add(16));
            let b1 = vld1q_u8(ptr.add(32));
            let b2 = vld1q_u8(ptr.add(48));
            // OR all chunks - if any byte has high bit set, combined will too.
            let combined = vorrq_u8(vorrq_u8(a1, a2), vorrq_u8(b1, b2));
            // `vmaxvq_u8` is a horizontal reduction with a longer latency than
            // `vorrq_u8`, so it runs once per 64 bytes rather than once per load.
            vmaxvq_u8(combined)
        };
        if max >= 128 {
            return false;
        }
    }

    // The unrolled loop above leaves up to 63 bytes, so sweep those a vector at
    // a time before falling back to a byte-at-a-time check.
    let (vectors, rest) = rest.as_chunks::<NEON_VECTOR_SIZE>();

    for vector in vectors {
        // SAFETY: vector is 16 bytes, and `vld1q_u8` has no alignment requirement.
        let max = unsafe { vmaxvq_u8(vld1q_u8(vector.as_ptr())) };
        if max >= 128 {
            return false;
        }
    }

    // Handle remaining bytes
    rest.iter().all(|b| b.is_ascii())
}

#[inline]
pub fn is_ascii(bytes: &[u8]) -> bool {
    const USIZE_SIZE: usize = size_of::<usize>();
    const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;
    if bytes.len() < 64 {
        let (chunks, remainder) = bytes.as_chunks::<USIZE_SIZE>();
        for chunk in chunks {
            let word = usize::from_ne_bytes(*chunk);
            if (word & NONASCII_MASK) != 0 {
                return false;
            }
        }
        return remainder.iter().all(|b| b.is_ascii());
    }
    #[cfg(target_arch = "x86_64")]
    {
        is_ascii_sse2(bytes)
    }
    #[cfg(target_arch = "aarch64")]
    {
        is_ascii_neon(bytes)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        bytes.is_ascii()
    }
}
