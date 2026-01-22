//! GEEV - Computes eigenvalues and eigenvectors of a general matrix.

use crate::backend::{get_cgeev, get_dgeev, get_sgeev, get_zgeev};
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

/// Computes eigenvalues and eigenvectors of a general matrix (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sgeev(
    jobvl: c_char,
    jobvr: c_char,
    n: lapackint,
    a: *mut f32,
    lda: lapackint,
    wr: *mut f32,
    wi: *mut f32,
    vl: *mut f32,
    ldvl: lapackint,
    vr: *mut f32,
    ldvr: lapackint,
    work: *mut f32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let sgeev = get_sgeev();
    sgeev(
        &jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info,
    );
}

/// Computes eigenvalues and eigenvectors of a general matrix (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dgeev(
    jobvl: c_char,
    jobvr: c_char,
    n: lapackint,
    a: *mut f64,
    lda: lapackint,
    wr: *mut f64,
    wi: *mut f64,
    vl: *mut f64,
    ldvl: lapackint,
    vr: *mut f64,
    ldvr: lapackint,
    work: *mut f64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let dgeev = get_dgeev();
    dgeev(
        &jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info,
    );
}

/// Computes eigenvalues and eigenvectors of a general matrix (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cgeev(
    jobvl: c_char,
    jobvr: c_char,
    n: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    w: *mut Complex32,
    vl: *mut Complex32,
    ldvl: lapackint,
    vr: *mut Complex32,
    ldvr: lapackint,
    work: *mut Complex32,
    lwork: lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let cgeev = get_cgeev();
    cgeev(
        &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, info,
    );
}

/// Computes eigenvalues and eigenvectors of a general matrix (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zgeev(
    jobvl: c_char,
    jobvr: c_char,
    n: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    w: *mut Complex64,
    vl: *mut Complex64,
    ldvl: lapackint,
    vr: *mut Complex64,
    ldvr: lapackint,
    work: *mut Complex64,
    lwork: lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let zgeev = get_zgeev();
    zgeev(
        &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, info,
    );
}
