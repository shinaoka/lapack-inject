//! LAPACK error handler (xerbla).

use crate::types::lapack_int;
use std::ffi::c_char;

/// LAPACKE_xerbla - LAPACKE error handler.
///
/// This function is called when an illegal parameter is detected.
/// It prints an error message to stderr.
///
/// Following OpenBLAS LAPACKE implementation.
///
/// # Safety
///
/// - `srname` must be a valid null-terminated C string or null
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_xerbla(srname: *const c_char, info: lapack_int) {
    let routine = if srname.is_null() {
        "<unknown>"
    } else {
        std::ffi::CStr::from_ptr(srname)
            .to_str()
            .unwrap_or("<invalid>")
    };
    eprintln!(
        "** On entry to {} parameter number {} had an illegal value",
        routine, info
    );
}

/// Internal helper for error reporting with static strings
#[inline]
pub fn lapacke_xerbla_internal(srname: &str, info: lapack_int) {
    eprintln!(
        "** On entry to {} parameter number {} had an illegal value",
        srname, info
    );
}
