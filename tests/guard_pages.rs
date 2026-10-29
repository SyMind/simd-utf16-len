//! Over- and under-reads segfault here instead of passing silently: every
//! input is copied flush against an inaccessible guard page, ending exactly
//! at the upper guard and starting exactly after the lower guard in turn.
//!
//! Miri has no mmap shim; tests/exact_allocations.rs is the counterpart for
//! Miri and ASan.
#![cfg(all(unix, not(miri)))]

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

/// Four OS pages: guard, accessible, accessible, guard.
struct Guarded {
    base: *mut u8,
    page: usize,
}

impl Guarded {
    fn new() -> Self {
        // The real OS page size: 16 KiB on Apple Silicon, 4 KiB on most Linux.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = usize::try_from(page).unwrap();
        // The longest input below is 2048 bytes and must fit the two
        // accessible pages.
        assert!(2 * page >= 2048, "unexpected page size: {page}");
        let len = 4 * page;
        // SAFETY: a private anonymous mapping with a null hint and no fd.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(
            base,
            libc::MAP_FAILED,
            "mmap failed: {}",
            std::io::Error::last_os_error()
        );
        let base = base.cast::<u8>();
        for guard in [0_usize, 3] {
            // SAFETY: page `guard` is a page-aligned range within the mapping.
            let result =
                unsafe { libc::mprotect(base.add(guard * page).cast(), page, libc::PROT_NONE) };
            assert_eq!(
                result,
                0,
                "mprotect failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Self { base, page }
    }

    /// The first byte after the lower guard page.
    fn lower_edge(&self) -> *mut u8 {
        self.base.wrapping_add(self.page)
    }

    /// One past the last byte before the upper guard page.
    fn upper_edge(&self) -> *mut u8 {
        self.base.wrapping_add(3 * self.page)
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        // SAFETY: `base` is the start of the 4-page mapping made in `new`.
        let result = unsafe { libc::munmap(self.base.cast(), 4 * self.page) };
        assert_eq!(
            result,
            0,
            "munmap failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

#[test]
fn utf16_len_never_reads_past_the_input() {
    let guarded = Guarded::new();
    // Every short-input length, then around the NEON batch boundary
    // (30 * 64 = 1920 bytes) and the 2048 power of two.
    let lengths = (0..=130).chain([1919, 1920, 1921, 2047, 2048]);
    for len in lengths {
        let mixed = mixed_of_len(len);
        let ascii = "a".repeat(len);
        for input in [&mixed, &ascii] {
            assert_eq!(input.len(), len);
            // An empty input has no upper placement: a zero-length slice must
            // not point at the guard page, and nothing may read it.
            let placements = if len == 0 {
                vec![guarded.lower_edge()]
            } else {
                vec![guarded.upper_edge().wrapping_sub(len), guarded.lower_edge()]
            };
            for start in placements {
                // SAFETY: start..start+len lies in the two accessible pages,
                // which are writable for the test's duration.
                unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), start, len) };
                // SAFETY: start..start+len now holds a copy of the valid
                // UTF-8 `input`, so re-validating it cannot fail.
                let bytes = unsafe { std::slice::from_raw_parts(start, len) };
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
}
