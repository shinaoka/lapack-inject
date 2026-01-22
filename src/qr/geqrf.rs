//! GEQRF - QR factorization.

use crate::backend::{get_cgeqrf, get_dgeqrf, get_sgeqrf, get_zgeqrf};
use crate::lapackint;
use num_complex::{Complex32, Complex64};

/// Computes QR factorization of a general M-by-N matrix A (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sgeqrf(
    m: lapackint,
    n: lapackint,
    a: *mut f32,
    lda: lapackint,
    tau: *mut f32,
    work: *mut f32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let sgeqrf = get_sgeqrf();
    sgeqrf(&m, &n, a, &lda, tau, work, &lwork, info);
}

/// Computes QR factorization of a general M-by-N matrix A (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dgeqrf(
    m: lapackint,
    n: lapackint,
    a: *mut f64,
    lda: lapackint,
    tau: *mut f64,
    work: *mut f64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let dgeqrf = get_dgeqrf();
    dgeqrf(&m, &n, a, &lda, tau, work, &lwork, info);
}

/// Computes QR factorization of a general M-by-N matrix A (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cgeqrf(
    m: lapackint,
    n: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    tau: *mut Complex32,
    work: *mut Complex32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let cgeqrf = get_cgeqrf();
    cgeqrf(&m, &n, a, &lda, tau, work, &lwork, info);
}

/// Computes QR factorization of a general M-by-N matrix A (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zgeqrf(
    m: lapackint,
    n: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    tau: *mut Complex64,
    work: *mut Complex64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let zgeqrf = get_zgeqrf();
    zgeqrf(&m, &n, a, &lda, tau, work, &lwork, info);
}
