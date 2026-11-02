//! Every input is its own exactly-sized allocation: under ASan a one-byte
//! overread trips the redzone even within a page, and under Miri it is plain
//! out of bounds. This is their counterpart of tests/guard_pages.rs, which
//! needs real OS guard pages.

use simd_utf16_len::{__kernels, utf16_len};

/// A valid UTF-8 string of exactly `len` bytes: a rotation of 1-, 2-, 3-, and
/// 4-byte characters while at least four bytes remain, then one character of
/// exactly the remaining width. Built from `char`s, so always valid.
fn mixed_of_len(len: usize) -> String {
    let mut s = String::with_capacity(len);
    let mut widths = ['a', 'é', '中', '🦀'].into_iter().cycle();
    while len - s.len() >= 4 {
        s.push(widths.next().unwrap());
    }
    match len - s.len() {
        0 => {}
        1 => s.push('a'),
        2 => s.push('é'),
        3 => s.push('中'),
        _ => unreachable!(),
    }
    s
}

#[test]
fn utf16_len_never_reads_past_the_allocation() {
    // Every length the short-input paths can take, then the batch boundaries
    // of every kernel: 1920 = 30 * 64 (NEON), 3840 = 2 * 1920 (x86 AVX2),
    // 7680 = 4 * 1920 (x86 AVX-512), and 16384 for the long loops.
    let lengths = (0..=256).chain([1919, 1920, 1921, 3839, 3840, 3841, 7680, 16383, 16384, 16385]);
    for len in lengths {
        // into_boxed_slice drops any excess capacity: the slice covers the
        // whole allocation, so its end coincides with the redzone.
        let ascii: Box<[u8]> = vec![b'a'; len].into_boxed_slice();
        let mixed: Box<[u8]> = mixed_of_len(len).into_bytes().into_boxed_slice();
        for bytes in [&ascii, &mixed] {
            let s = std::str::from_utf8(bytes).unwrap();
            let expected = s.encode_utf16().count();
            assert_eq!(utf16_len(s), expected, "len: {len}");
            for kernel in __kernels::available() {
                assert_eq!(
                    (kernel.utf16_len)(bytes),
                    expected,
                    "{} kernel, len: {len}",
                    kernel.name,
                );
            }
        }
    }
}
