//! Fortran LAPACK symbol exports.
//!
//! This module exports Fortran-style LAPACK symbols (e.g., `dgesv_`) that call
//! the registered function pointers. This allows lapack-inject to be a drop-in
//! replacement for lapack-src.

use crate::backend::*;
use crate::lapackint;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

// =============================================================================
// GESV - General linear solve
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut f32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgesv();
    f(n, nrhs, a, lda, ipiv, b, ldb, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgesv();
    f(n, nrhs, a, lda, ipiv, b, ldb, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut Complex32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cgesv();
    f(n, nrhs, a, lda, ipiv, b, ldb, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut Complex64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zgesv();
    f(n, nrhs, a, lda, ipiv, b, ldb, info);
}

// =============================================================================
// GELS - Least squares
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgels_(
    trans: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    b: *mut f32,
    ldb: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgels();
    f(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgels_(
    trans: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    b: *mut f64,
    ldb: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgels();
    f(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgels_(
    trans: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    b: *mut Complex32,
    ldb: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cgels();
    f(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgels_(
    trans: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    b: *mut Complex64,
    ldb: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zgels();
    f(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
}

// =============================================================================
// GETRF - LU factorization
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_sgetrf();
    f(m, n, a, lda, ipiv, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_dgetrf();
    f(m, n, a, lda, ipiv, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_cgetrf();
    f(m, n, a, lda, ipiv, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_zgetrf();
    f(m, n, a, lda, ipiv, info);
}

// =============================================================================
// GETRI - Matrix inverse from LU
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgetri_(
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgetri();
    f(n, a, lda, ipiv, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgetri_(
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgetri();
    f(n, a, lda, ipiv, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgetri_(
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cgetri();
    f(n, a, lda, ipiv, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgetri_(
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zgetri();
    f(n, a, lda, ipiv, work, lwork, info);
}

// =============================================================================
// POTRF - Cholesky factorization
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn spotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_spotrf();
    f(uplo, n, a, lda, info);
}

#[no_mangle]
pub unsafe extern "C" fn dpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dpotrf();
    f(uplo, n, a, lda, info);
}

#[no_mangle]
pub unsafe extern "C" fn cpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cpotrf();
    f(uplo, n, a, lda, info);
}

#[no_mangle]
pub unsafe extern "C" fn zpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zpotrf();
    f(uplo, n, a, lda, info);
}

// =============================================================================
// GEQRF - QR factorization
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    tau: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgeqrf();
    f(m, n, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgeqrf();
    f(m, n, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    tau: *mut Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cgeqrf();
    f(m, n, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    tau: *mut Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zgeqrf();
    f(m, n, a, lda, tau, work, lwork, info);
}

// =============================================================================
// ORGQR/UNGQR - Generate Q from QR
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sorgqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    tau: *const f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sorgqr();
    f(m, n, k, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dorgqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    tau: *const f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dorgqr();
    f(m, n, k, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cungqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    tau: *const Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_cungqr();
    f(m, n, k, a, lda, tau, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zungqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    tau: *const Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_zungqr();
    f(m, n, k, a, lda, tau, work, lwork, info);
}

// =============================================================================
// GESVD - SVD (standard)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    s: *mut f32,
    u: *mut f32,
    ldu: *const lapackint,
    vt: *mut f32,
    ldvt: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgesvd();
    f(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    s: *mut f64,
    u: *mut f64,
    ldu: *const lapackint,
    vt: *mut f64,
    ldvt: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgesvd();
    f(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    s: *mut f32,
    u: *mut Complex32,
    ldu: *const lapackint,
    vt: *mut Complex32,
    ldvt: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let f = get_cgesvd();
    f(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    s: *mut f64,
    u: *mut Complex64,
    ldu: *const lapackint,
    vt: *mut Complex64,
    ldvt: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let f = get_zgesvd();
    f(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
}

// =============================================================================
// GESDD - SVD (divide and conquer)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgesdd_(
    jobz: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    s: *mut f32,
    u: *mut f32,
    ldu: *const lapackint,
    vt: *mut f32,
    ldvt: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_sgesdd();
    f(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgesdd_(
    jobz: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    s: *mut f64,
    u: *mut f64,
    ldu: *const lapackint,
    vt: *mut f64,
    ldvt: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_dgesdd();
    f(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgesdd_(
    jobz: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    s: *mut f32,
    u: *mut Complex32,
    ldu: *const lapackint,
    vt: *mut Complex32,
    ldvt: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_cgesdd();
    f(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgesdd_(
    jobz: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    s: *mut f64,
    u: *mut Complex64,
    ldu: *const lapackint,
    vt: *mut Complex64,
    ldvt: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    iwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_zgesdd();
    f(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
}

// =============================================================================
// GEEV - General eigenvalue
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    wr: *mut f32,
    wi: *mut f32,
    vl: *mut f32,
    ldvl: *const lapackint,
    vr: *mut f32,
    ldvr: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_sgeev();
    f(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    wr: *mut f64,
    wi: *mut f64,
    vl: *mut f64,
    ldvl: *const lapackint,
    vr: *mut f64,
    ldvr: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dgeev();
    f(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    w: *mut Complex32,
    vl: *mut Complex32,
    ldvl: *const lapackint,
    vr: *mut Complex32,
    ldvr: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let f = get_cgeev();
    f(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    w: *mut Complex64,
    vl: *mut Complex64,
    ldvl: *const lapackint,
    vr: *mut Complex64,
    ldvr: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let f = get_zgeev();
    f(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

// =============================================================================
// SYEV - Symmetric eigenvalue (real)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn ssyev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    w: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_ssyev();
    f(jobz, uplo, n, a, lda, w, work, lwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dsyev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    w: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    let f = get_dsyev();
    f(jobz, uplo, n, a, lda, w, work, lwork, info);
}

// =============================================================================
// HEEV - Hermitian eigenvalue (complex)
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn cheev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    w: *mut f32,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    let f = get_cheev();
    f(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zheev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    w: *mut f64,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    let f = get_zheev();
    f(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

// =============================================================================
// GEES - Schur decomposition
// =============================================================================

#[no_mangle]
pub unsafe extern "C" fn sgees_(
    jobvs: *const c_char,
    sort: *const c_char,
    select: Option<SgeesSelectFn>,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    sdim: *mut lapackint,
    wr: *mut f32,
    wi: *mut f32,
    vs: *mut f32,
    ldvs: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_sgees();
    f(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn dgees_(
    jobvs: *const c_char,
    sort: *const c_char,
    select: Option<DgeesSelectFn>,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    sdim: *mut lapackint,
    wr: *mut f64,
    wi: *mut f64,
    vs: *mut f64,
    ldvs: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_dgees();
    f(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn cgees_(
    jobvs: *const c_char,
    sort: *const c_char,
    select: Option<CgeesSelectFn>,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    sdim: *mut lapackint,
    w: *mut Complex32,
    vs: *mut Complex32,
    ldvs: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_cgees();
    f(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
}

#[no_mangle]
pub unsafe extern "C" fn zgees_(
    jobvs: *const c_char,
    sort: *const c_char,
    select: Option<ZgeesSelectFn>,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    sdim: *mut lapackint,
    w: *mut Complex64,
    vs: *mut Complex64,
    ldvs: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    bwork: *mut lapackint,
    info: *mut lapackint,
) {
    let f = get_zgees();
    f(jobvs, sort, select, n, a, lda, sdim, w, vs, ldvs, work, lwork, rwork, bwork, info);
}
