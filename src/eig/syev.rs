//! SYEV - Computes eigenvalues and eigenvectors of a symmetric matrix (real).

use crate::backend::{get_dsyev, get_ssyev};
use crate::lapackint;
use std::ffi::c_char;

/// Computes eigenvalues and eigenvectors of a symmetric matrix (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_ssyev(
    jobz: c_char,
    uplo: c_char,
    n: lapackint,
    a: *mut f32,
    lda: lapackint,
    w: *mut f32,
    work: *mut f32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let ssyev = get_ssyev();
    ssyev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
}

/// Computes eigenvalues and eigenvectors of a symmetric matrix (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dsyev(
    jobz: c_char,
    uplo: c_char,
    n: lapackint,
    a: *mut f64,
    lda: lapackint,
    w: *mut f64,
    work: *mut f64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let dsyev = get_dsyev();
    dsyev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
}
