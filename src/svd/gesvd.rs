//! GESVD - SVD using standard algorithm.

use crate::backend::{get_cgesvd, get_dgesvd, get_sgesvd, get_zgesvd};
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

/// Computes SVD of a general M-by-N matrix A (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sgesvd(
    jobu: c_char,
    jobvt: c_char,
    m: lapackint,
    n: lapackint,
    a: *mut f32,
    lda: lapackint,
    s: *mut f32,
    u: *mut f32,
    ldu: lapackint,
    vt: *mut f32,
    ldvt: lapackint,
    work: *mut f32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let sgesvd = get_sgesvd();
    sgesvd(
        &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dgesvd(
    jobu: c_char,
    jobvt: c_char,
    m: lapackint,
    n: lapackint,
    a: *mut f64,
    lda: lapackint,
    s: *mut f64,
    u: *mut f64,
    ldu: lapackint,
    vt: *mut f64,
    ldvt: lapackint,
    work: *mut f64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let dgesvd = get_dgesvd();
    dgesvd(
        &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cgesvd(
    jobu: c_char,
    jobvt: c_char,
    m: lapackint,
    n: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    s: *mut f32,
    u: *mut Complex32,
    ldu: lapackint,
    vt: *mut Complex32,
    ldvt: lapackint,
    work: *mut Complex32,
    lwork: lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let cgesvd = get_cgesvd();
    cgesvd(
        &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zgesvd(
    jobu: c_char,
    jobvt: c_char,
    m: lapackint,
    n: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    s: *mut f64,
    u: *mut Complex64,
    ldu: lapackint,
    vt: *mut Complex64,
    ldvt: lapackint,
    work: *mut Complex64,
    lwork: lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let zgesvd = get_zgesvd();
    zgesvd(
        &jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, info,
    );
}
