//! GEES - Computes Schur decomposition of a general matrix.

use crate::backend::{
    get_cgees, get_dgees, get_sgees, get_zgees, CgeesSelectFn, DgeesSelectFn, SgeesSelectFn,
    ZgeesSelectFn,
};
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

/// Computes Schur decomposition of a general matrix (single precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_sgees(
    jobvs: c_char,
    sort: c_char,
    select: Option<SgeesSelectFn>,
    n: lapackint,
    a: *mut f32,
    lda: lapackint,
    sdim: *mut lapackint,
    wr: *mut f32,
    wi: *mut f32,
    vs: *mut f32,
    ldvs: lapackint,
    work: *mut f32,
    lwork: lapackint,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let sgees = get_sgees();
    sgees(
        &jobvs, &sort, select, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work, &lwork, bwork, info,
    );
}

/// Computes Schur decomposition of a general matrix (double precision).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_dgees(
    jobvs: c_char,
    sort: c_char,
    select: Option<DgeesSelectFn>,
    n: lapackint,
    a: *mut f64,
    lda: lapackint,
    sdim: *mut lapackint,
    wr: *mut f64,
    wi: *mut f64,
    vs: *mut f64,
    ldvs: lapackint,
    work: *mut f64,
    lwork: lapackint,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let dgees = get_dgees();
    dgees(
        &jobvs, &sort, select, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work, &lwork, bwork, info,
    );
}

/// Computes Schur decomposition of a general matrix (single precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_cgees(
    jobvs: c_char,
    sort: c_char,
    select: Option<CgeesSelectFn>,
    n: lapackint,
    a: *mut Complex32,
    lda: lapackint,
    sdim: *mut lapackint,
    w: *mut Complex32,
    vs: *mut Complex32,
    ldvs: lapackint,
    work: *mut Complex32,
    lwork: lapackint,
    rwork: *mut f32,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let cgees = get_cgees();
    cgees(
        &jobvs, &sort, select, &n, a, &lda, sdim, w, vs, &ldvs, work, &lwork, rwork, bwork, info,
    );
}

/// Computes Schur decomposition of a general matrix (double precision complex).
///
/// # Safety
///
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn lapack_zgees(
    jobvs: c_char,
    sort: c_char,
    select: Option<ZgeesSelectFn>,
    n: lapackint,
    a: *mut Complex64,
    lda: lapackint,
    sdim: *mut lapackint,
    w: *mut Complex64,
    vs: *mut Complex64,
    ldvs: lapackint,
    work: *mut Complex64,
    lwork: lapackint,
    rwork: *mut f64,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let zgees = get_zgees();
    zgees(
        &jobvs, &sort, select, &n, a, &lda, sdim, w, vs, &ldvs, work, &lwork, rwork, bwork, info,
    );
}
