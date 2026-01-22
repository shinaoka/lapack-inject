//! HEEV - Computes eigenvalues and eigenvectors of a Hermitian matrix (complex).

use crate::backend::{get_cheev, get_zheev};
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

/// Computes eigenvalues and eigenvectors of a Hermitian matrix (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cheev(
    jobz: c_char,
    uplo: c_char,
    n: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    w: *mut f32,
    work: *mut Complex32,
    lwork: lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let cheev = get_cheev();
    cheev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, info);
}

/// Computes eigenvalues and eigenvectors of a Hermitian matrix (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zheev(
    jobz: c_char,
    uplo: c_char,
    n: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    w: *mut f64,
    work: *mut Complex64,
    lwork: lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let zheev = get_zheev();
    zheev(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, info);
}
