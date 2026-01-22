//! UNGQR - Generates the unitary matrix Q from QR factorization (complex).

use crate::backend::{get_cungqr, get_zungqr};
use crate::lapackint;
use num_complex::{Complex32, Complex64};

/// Generates the unitary matrix Q from QR factorization (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cungqr(
    m: lapackint,
    n: lapackint,
    k: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    tau: *const Complex32,
    work: *mut Complex32,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let cungqr = get_cungqr();
    cungqr(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

/// Generates the unitary matrix Q from QR factorization (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zungqr(
    m: lapackint,
    n: lapackint,
    k: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    tau: *const Complex64,
    work: *mut Complex64,
    lwork: lapackint,
    info: *mut lapackint,
) {
    let zungqr = get_zungqr();
    zungqr(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}
