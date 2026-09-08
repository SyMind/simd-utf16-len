//! NEON-based UTF-16 length calculation (always available on aarch64).

use std::arch::aarch64::*;

/// Compute the number of UTF-16 code units for UTF-8 string using NEON.
#[allow(unsafe_code)]
#[inline]
pub fn utf16_len(s: &str) -> usize {
    let start = ascii_prefix_len_neon(s.as_bytes());
    if start == s.len() {
        start
    } else {
        utf16_len_non_ascii(s, start)
    }
}

/// Count the remaining bytes after an already checked ASCII prefix.
/// Keeping the counting loop out of line lets callers inline the ASCII path.
#[inline(never)]
fn utf16_len_non_ascii(s: &str, mut i: usize) -> usize {
    let bytes = s.as_bytes();
    let len = bytes.len();

    let mut continuation_count: usize = 0;
    let mut four_byte_count: usize = 0;

    // SAFETY: NEON is always available on aarch64.
    unsafe {
        let cont_mask = vdupq_n_u8(0xC0);
        let cont_val = vdupq_n_u8(0x80);
        let four_threshold = vdupq_n_u8(0xEF);
        let one = vdupq_n_u8(1);

        // Process 16 bytes at a time, in batches of up to 255 iterations
        // to avoid u8 overflow in the per-lane accumulators.
        while i + 16 <= len {
            let batch = ((len - i) / 16).min(255);
            let mut cont_acc = vdupq_n_u8(0);
            let mut four_acc = vdupq_n_u8(0);

            for _ in 0..batch {
                let chunk = vld1q_u8(bytes.as_ptr().add(i));

                // Continuation bytes: (byte & 0xC0) == 0x80
                let masked = vandq_u8(chunk, cont_mask);
                let is_cont = vceqq_u8(masked, cont_val);
                // is_cont lanes are 0xFF (-1) for continuation bytes;
                // subtracting -1 is adding 1.
                cont_acc = vsubq_u8(cont_acc, is_cont);

                // Four-byte leaders (byte >= 0xF0):
                // saturating subtract 0xEF gives non-zero only for bytes >= 0xF0,
                // then clamp to 1 with min.
                let sub = vqsubq_u8(chunk, four_threshold);
                let is_four = vminq_u8(sub, one);
                four_acc = vaddq_u8(four_acc, is_four);

                i += 16;
            }

            // Horizontal sum across all lanes.
            continuation_count += vaddlvq_u8(cont_acc) as usize;
            four_byte_count += vaddlvq_u8(four_acc) as usize;
        }

        // Tail: find the next char boundary and use encode_utf16().count().
        // Bytes between i and the char boundary are all continuation bytes,
        // contributing 0 to UTF-16 length, so we can skip them.
        let tail_start = crate::ceil_char_boundary(s, i);
        i - continuation_count + four_byte_count + s[tail_start..].encode_utf16().count()
    }
}

/// Return the string length for ASCII, or the start of the first 64-byte
/// block (or tail) containing a non-ASCII byte. Earlier bytes need no counting.
#[inline]
fn ascii_prefix_len_neon(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let mut i = 0;

    while i + 64 <= len {
        // SAFETY: all four loads are within the slice, and NEON is available
        // on aarch64. OR preserves the presence of any byte's high bit.
        let non_ascii = unsafe {
            let ptr = bytes.as_ptr().add(i);
            let a = vld1q_u8(ptr);
            let b = vld1q_u8(ptr.add(16));
            let c = vld1q_u8(ptr.add(32));
            let d = vld1q_u8(ptr.add(48));
            let combined = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
            vmaxvq_u8(combined) >= 0x80
        };
        if non_ascii {
            return i;
        }
        i += 64;
    }

    if bytes[i..].is_ascii() { len } else { i }
}
