//! LAPACK/LAPACKE type definitions.
//!
//! These types are compatible with the LAPACKE standard.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::ffi::c_char;

/// Integer type for LAPACK operations (LP64: 32-bit).
#[cfg(not(feature = "ilp64"))]
pub type lapack_int = i32;

/// Integer type for LAPACK operations (ILP64: 64-bit).
#[cfg(feature = "ilp64")]
pub type lapack_int = i64;

/// Alias for backward compatibility
pub type lapackint = lapack_int;

/// LAPACKE matrix layout constants
pub const LAPACK_ROW_MAJOR: i32 = 101;
pub const LAPACK_COL_MAJOR: i32 = 102;

/// LAPACKE error codes (same as OpenBLAS LAPACKE)
pub const LAPACK_WORK_MEMORY_ERROR: lapack_int = -1010;
pub const LAPACK_TRANSPOSE_MEMORY_ERROR: lapack_int = -1011;

/// Check if layout is row-major
#[inline]
pub fn is_row_major(layout: i32) -> bool {
    layout == LAPACK_ROW_MAJOR
}

/// Check if layout is column-major
#[inline]
pub fn is_col_major(layout: i32) -> bool {
    layout == LAPACK_COL_MAJOR
}

/// LAPACKE_lsame - Compare characters (case-insensitive)
/// Following OpenBLAS LAPACKE implementation
#[inline]
pub fn lapacke_lsame(ca: u8, cb: u8) -> bool {
    let ca_upper = if ca >= b'a' && ca <= b'z' { ca - 32 } else { ca };
    let cb_upper = if cb >= b'a' && cb <= b'z' { cb - 32 } else { cb };
    ca_upper == cb_upper
}

/// MAX macro equivalent
#[inline]
pub const fn lapack_max(a: lapack_int, b: lapack_int) -> lapack_int {
    if a > b { a } else { b }
}

/// MIN macro equivalent
#[inline]
pub const fn lapack_min(a: lapack_int, b: lapack_int) -> lapack_int {
    if a < b { a } else { b }
}

/// Job character constants for LAPACK functions.
pub mod job {
    use std::ffi::c_char;

    /// Compute no vectors
    pub const N: c_char = b'N' as c_char;
    /// Compute vectors
    pub const V: c_char = b'V' as c_char;
    /// Compute all vectors
    pub const A: c_char = b'A' as c_char;
    /// Compute some vectors
    pub const S: c_char = b'S' as c_char;
    /// Overwrite matrix
    pub const O: c_char = b'O' as c_char;
}

/// Upper/Lower triangle selector for symmetric/triangular operations.
pub mod uplo {
    use std::ffi::c_char;

    /// Upper triangle
    pub const U: c_char = b'U' as c_char;
    /// Lower triangle
    pub const L: c_char = b'L' as c_char;
}

/// Transpose/No transpose selector.
pub mod trans {
    use std::ffi::c_char;

    /// No transpose
    pub const N: c_char = b'N' as c_char;
    /// Transpose
    pub const T: c_char = b'T' as c_char;
    /// Conjugate transpose
    pub const C: c_char = b'C' as c_char;
}

/// Convert uppercase/lowercase to standard char.
#[inline]
pub fn normalize_char(c: c_char) -> c_char {
    let c_u8 = c as u8;
    if c_u8 >= b'a' && c_u8 <= b'z' {
        (c_u8 - b'a' + b'A') as c_char
    } else {
        c
    }
}

/// Convert char to uplo ('U' or 'L').
#[inline]
pub fn char_to_uplo(c: c_char) -> c_char {
    let c = normalize_char(c);
    match c as u8 {
        b'U' => uplo::U,
        b'L' => uplo::L,
        _ => uplo::U, // default
    }
}

/// Convert char to job ('N', 'V', 'A', 'S', 'O').
#[inline]
pub fn char_to_job(c: c_char) -> c_char {
    normalize_char(c)
}
