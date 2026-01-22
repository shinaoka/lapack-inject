//! GESDD - SVD using divide and conquer algorithm.

use crate::backend::{get_cgesdd, get_dgesdd, get_sgesdd, get_zgesdd};
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

/// Computes SVD of a general M-by-N matrix A using divide and conquer (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sgesdd(
    jobz: c_char,
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
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let sgesdd = get_sgesdd();
    sgesdd(
        &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A using divide and conquer (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dgesdd(
    jobz: c_char,
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
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let dgesdd = get_dgesdd();
    dgesdd(
        &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A using divide and conquer (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cgesdd(
    jobz: c_char,
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
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let cgesdd = get_cgesdd();
    cgesdd(
        &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, info,
    );
}

/// Computes SVD of a general M-by-N matrix A using divide and conquer (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zgesdd(
    jobz: c_char,
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
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let zgesdd = get_zgesdd();
    zgesdd(
        &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, info,
    );
}
