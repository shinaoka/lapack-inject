//! ORGQR - Generates the orthogonal matrix Q from QR factorization (real).

use crate::backend::{get_dorgqr, get_sorgqr};
use crate::lapackint;

/// Generates the orthogonal matrix Q from QR factorization (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sorgqr(
    m: lapackint,
    n: lapackint,
    k: lapackint,
    a: *mut f32,
    lda: lapackint,
    tau: *const f32,
    work: *mut f32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let sorgqr = get_sorgqr();
    sorgqr(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

/// Generates the orthogonal matrix Q from QR factorization (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dorgqr(
    m: lapackint,
    n: lapackint,
    k: lapackint,
    a: *mut f64,
    lda: lapackint,
    tau: *const f64,
    work: *mut f64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let dorgqr = get_dorgqr();
    dorgqr(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}
