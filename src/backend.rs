//! Fortran LAPACK function pointer registration.
//!
//! This module provides the infrastructure for registering Fortran LAPACK
//! function pointers at runtime. Each function has its own `OnceLock` to allow
//! partial registration (only register the functions you need).

use std::ffi::c_char;
use std::sync::OnceLock;

use num_complex::{Complex32, Complex64};

use crate::lapackint;

// =============================================================================
// Fortran LAPACK function pointer types
// =============================================================================

// -----------------------------------------------------------------------------
// GESV - General linear solve (Ax = B)
// -----------------------------------------------------------------------------

/// Fortran sgesv function pointer type
pub type SgesvFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut f32,
    ldb: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dgesv function pointer type
pub type DgesvFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
);

/// Fortran cgesv function pointer type
pub type CgesvFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut Complex32,
    ldb: *const lapackint,
    info: *mut lapackint,
);

/// Fortran zgesv function pointer type
pub type ZgesvFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    nrhs: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    b: *mut Complex64,
    ldb: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// GETRF - LU factorization
// -----------------------------------------------------------------------------

/// Fortran sgetrf function pointer type
pub type SgetrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
);

/// Fortran dgetrf function pointer type
pub type DgetrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
);

/// Fortran cgetrf function pointer type
pub type CgetrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
);

/// Fortran zgetrf function pointer type
pub type ZgetrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// GETRI - Matrix inverse from LU factorization
// -----------------------------------------------------------------------------

/// Fortran sgetri function pointer type
pub type SgetriFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dgetri function pointer type
pub type DgetriFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran cgetri function pointer type
pub type CgetriFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran zgetri function pointer type
pub type ZgetriFnPtr = unsafe extern "C" fn(
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// POTRF - Cholesky factorization
// -----------------------------------------------------------------------------

/// Fortran spotrf function pointer type
pub type SpotrfFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dpotrf function pointer type
pub type DpotrfFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    info: *mut lapackint,
);

/// Fortran cpotrf function pointer type
pub type CpotrfFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    info: *mut lapackint,
);

/// Fortran zpotrf function pointer type
pub type ZpotrfFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// GELS - Least squares (QR/LQ based)
// -----------------------------------------------------------------------------

/// Fortran sgels function pointer type
pub type SgelsFnPtr = unsafe extern "C" fn(
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
);

/// Fortran dgels function pointer type
pub type DgelsFnPtr = unsafe extern "C" fn(
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
);

/// Fortran cgels function pointer type
pub type CgelsFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zgels function pointer type
pub type ZgelsFnPtr = unsafe extern "C" fn(
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
);

// -----------------------------------------------------------------------------
// GEQRF - QR factorization
// -----------------------------------------------------------------------------

/// Fortran sgeqrf function pointer type
pub type SgeqrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    tau: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dgeqrf function pointer type
pub type DgeqrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran cgeqrf function pointer type
pub type CgeqrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    tau: *mut Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran zgeqrf function pointer type
pub type ZgeqrfFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    tau: *mut Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// ORGQR - Generate Q from QR factorization (real)
// -----------------------------------------------------------------------------

/// Fortran sorgqr function pointer type
pub type SorgqrFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    tau: *const f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dorgqr function pointer type
pub type DorgqrFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    tau: *const f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// UNGQR - Generate Q from QR factorization (complex)
// -----------------------------------------------------------------------------

/// Fortran cungqr function pointer type
pub type CungqrFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut Complex32,
    lda: *const lapackint,
    tau: *const Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran zungqr function pointer type
pub type ZungqrFnPtr = unsafe extern "C" fn(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    a: *mut Complex64,
    lda: *const lapackint,
    tau: *const Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// GESVD - SVD (standard algorithm)
// -----------------------------------------------------------------------------

/// Fortran sgesvd function pointer type
pub type SgesvdFnPtr = unsafe extern "C" fn(
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
);

/// Fortran dgesvd function pointer type
pub type DgesvdFnPtr = unsafe extern "C" fn(
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
);

/// Fortran cgesvd function pointer type (complex, has rwork)
pub type CgesvdFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zgesvd function pointer type (complex, has rwork)
pub type ZgesvdFnPtr = unsafe extern "C" fn(
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
);

// -----------------------------------------------------------------------------
// GESDD - SVD (divide and conquer)
// -----------------------------------------------------------------------------

/// Fortran sgesdd function pointer type
pub type SgesddFnPtr = unsafe extern "C" fn(
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
);

/// Fortran dgesdd function pointer type
pub type DgesddFnPtr = unsafe extern "C" fn(
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
);

/// Fortran cgesdd function pointer type (complex, has rwork)
pub type CgesddFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zgesdd function pointer type (complex, has rwork)
pub type ZgesddFnPtr = unsafe extern "C" fn(
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
);

// -----------------------------------------------------------------------------
// GEEV - General eigenvalue problem
// -----------------------------------------------------------------------------

/// Fortran sgeev function pointer type (real)
pub type SgeevFnPtr = unsafe extern "C" fn(
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
);

/// Fortran dgeev function pointer type (real)
pub type DgeevFnPtr = unsafe extern "C" fn(
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
);

/// Fortran cgeev function pointer type (complex, has rwork)
pub type CgeevFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zgeev function pointer type (complex, has rwork)
pub type ZgeevFnPtr = unsafe extern "C" fn(
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
);

// -----------------------------------------------------------------------------
// SYEV - Symmetric eigenvalue problem (real)
// -----------------------------------------------------------------------------

/// Fortran ssyev function pointer type
pub type SsyevFnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f32,
    lda: *const lapackint,
    w: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
);

/// Fortran dsyev function pointer type
pub type DsyevFnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    w: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
);

// -----------------------------------------------------------------------------
// HEEV - Hermitian eigenvalue problem (complex)
// -----------------------------------------------------------------------------

/// Fortran cheev function pointer type
pub type CheevFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zheev function pointer type
pub type ZheevFnPtr = unsafe extern "C" fn(
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
);

// -----------------------------------------------------------------------------
// GEES - Schur decomposition
// -----------------------------------------------------------------------------

/// Select function type for real GEES
pub type SgeesSelectFn = unsafe extern "C" fn(wr: *const f32, wi: *const f32) -> lapackint;
/// Select function type for real GEES (double)
pub type DgeesSelectFn = unsafe extern "C" fn(wr: *const f64, wi: *const f64) -> lapackint;
/// Select function type for complex GEES (single)
pub type CgeesSelectFn = unsafe extern "C" fn(w: *const Complex32) -> lapackint;
/// Select function type for complex GEES (double)
pub type ZgeesSelectFn = unsafe extern "C" fn(w: *const Complex64) -> lapackint;

/// Fortran sgees function pointer type
pub type SgeesFnPtr = unsafe extern "C" fn(
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
);

/// Fortran dgees function pointer type
pub type DgeesFnPtr = unsafe extern "C" fn(
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
);

/// Fortran cgees function pointer type
pub type CgeesFnPtr = unsafe extern "C" fn(
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
);

/// Fortran zgees function pointer type
pub type ZgeesFnPtr = unsafe extern "C" fn(
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
);

// =============================================================================
// OnceLock storage and registration/getter functions
// =============================================================================

// GESV
static SGESV: OnceLock<SgesvFnPtr> = OnceLock::new();
static DGESV: OnceLock<DgesvFnPtr> = OnceLock::new();
static CGESV: OnceLock<CgesvFnPtr> = OnceLock::new();
static ZGESV: OnceLock<ZgesvFnPtr> = OnceLock::new();

// GETRF
static SGETRF: OnceLock<SgetrfFnPtr> = OnceLock::new();
static DGETRF: OnceLock<DgetrfFnPtr> = OnceLock::new();
static CGETRF: OnceLock<CgetrfFnPtr> = OnceLock::new();
static ZGETRF: OnceLock<ZgetrfFnPtr> = OnceLock::new();

// GETRI
static SGETRI: OnceLock<SgetriFnPtr> = OnceLock::new();
static DGETRI: OnceLock<DgetriFnPtr> = OnceLock::new();
static CGETRI: OnceLock<CgetriFnPtr> = OnceLock::new();
static ZGETRI: OnceLock<ZgetriFnPtr> = OnceLock::new();

// POTRF
static SPOTRF: OnceLock<SpotrfFnPtr> = OnceLock::new();
static DPOTRF: OnceLock<DpotrfFnPtr> = OnceLock::new();
static CPOTRF: OnceLock<CpotrfFnPtr> = OnceLock::new();
static ZPOTRF: OnceLock<ZpotrfFnPtr> = OnceLock::new();

// GELS
static SGELS: OnceLock<SgelsFnPtr> = OnceLock::new();
static DGELS: OnceLock<DgelsFnPtr> = OnceLock::new();
static CGELS: OnceLock<CgelsFnPtr> = OnceLock::new();
static ZGELS: OnceLock<ZgelsFnPtr> = OnceLock::new();

// GEQRF
static SGEQRF: OnceLock<SgeqrfFnPtr> = OnceLock::new();
static DGEQRF: OnceLock<DgeqrfFnPtr> = OnceLock::new();
static CGEQRF: OnceLock<CgeqrfFnPtr> = OnceLock::new();
static ZGEQRF: OnceLock<ZgeqrfFnPtr> = OnceLock::new();

// ORGQR
static SORGQR: OnceLock<SorgqrFnPtr> = OnceLock::new();
static DORGQR: OnceLock<DorgqrFnPtr> = OnceLock::new();

// UNGQR
static CUNGQR: OnceLock<CungqrFnPtr> = OnceLock::new();
static ZUNGQR: OnceLock<ZungqrFnPtr> = OnceLock::new();

// GESVD
static SGESVD: OnceLock<SgesvdFnPtr> = OnceLock::new();
static DGESVD: OnceLock<DgesvdFnPtr> = OnceLock::new();
static CGESVD: OnceLock<CgesvdFnPtr> = OnceLock::new();
static ZGESVD: OnceLock<ZgesvdFnPtr> = OnceLock::new();

// GESDD
static SGESDD: OnceLock<SgesddFnPtr> = OnceLock::new();
static DGESDD: OnceLock<DgesddFnPtr> = OnceLock::new();
static CGESDD: OnceLock<CgesddFnPtr> = OnceLock::new();
static ZGESDD: OnceLock<ZgesddFnPtr> = OnceLock::new();

// GEEV
static SGEEV: OnceLock<SgeevFnPtr> = OnceLock::new();
static DGEEV: OnceLock<DgeevFnPtr> = OnceLock::new();
static CGEEV: OnceLock<CgeevFnPtr> = OnceLock::new();
static ZGEEV: OnceLock<ZgeevFnPtr> = OnceLock::new();

// SYEV
static SSYEV: OnceLock<SsyevFnPtr> = OnceLock::new();
static DSYEV: OnceLock<DsyevFnPtr> = OnceLock::new();

// HEEV
static CHEEV: OnceLock<CheevFnPtr> = OnceLock::new();
static ZHEEV: OnceLock<ZheevFnPtr> = OnceLock::new();

// GEES
static SGEES: OnceLock<SgeesFnPtr> = OnceLock::new();
static DGEES: OnceLock<DgeesFnPtr> = OnceLock::new();
static CGEES: OnceLock<CgeesFnPtr> = OnceLock::new();
static ZGEES: OnceLock<ZgeesFnPtr> = OnceLock::new();

// =============================================================================
// Registration functions
// =============================================================================

// GESV
#[no_mangle]
pub unsafe extern "C" fn register_sgesv(f: SgesvFnPtr) {
    let _ = SGESV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgesv(f: DgesvFnPtr) {
    let _ = DGESV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgesv(f: CgesvFnPtr) {
    let _ = CGESV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgesv(f: ZgesvFnPtr) {
    let _ = ZGESV.set(f);
}

// GETRF
#[no_mangle]
pub unsafe extern "C" fn register_sgetrf(f: SgetrfFnPtr) {
    let _ = SGETRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgetrf(f: DgetrfFnPtr) {
    let _ = DGETRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgetrf(f: CgetrfFnPtr) {
    let _ = CGETRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgetrf(f: ZgetrfFnPtr) {
    let _ = ZGETRF.set(f);
}

// GETRI
#[no_mangle]
pub unsafe extern "C" fn register_sgetri(f: SgetriFnPtr) {
    let _ = SGETRI.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgetri(f: DgetriFnPtr) {
    let _ = DGETRI.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgetri(f: CgetriFnPtr) {
    let _ = CGETRI.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgetri(f: ZgetriFnPtr) {
    let _ = ZGETRI.set(f);
}

// POTRF
#[no_mangle]
pub unsafe extern "C" fn register_spotrf(f: SpotrfFnPtr) {
    let _ = SPOTRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dpotrf(f: DpotrfFnPtr) {
    let _ = DPOTRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cpotrf(f: CpotrfFnPtr) {
    let _ = CPOTRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zpotrf(f: ZpotrfFnPtr) {
    let _ = ZPOTRF.set(f);
}

// GELS
#[no_mangle]
pub unsafe extern "C" fn register_sgels(f: SgelsFnPtr) {
    let _ = SGELS.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgels(f: DgelsFnPtr) {
    let _ = DGELS.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgels(f: CgelsFnPtr) {
    let _ = CGELS.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgels(f: ZgelsFnPtr) {
    let _ = ZGELS.set(f);
}

// GEQRF
#[no_mangle]
pub unsafe extern "C" fn register_sgeqrf(f: SgeqrfFnPtr) {
    let _ = SGEQRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgeqrf(f: DgeqrfFnPtr) {
    let _ = DGEQRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgeqrf(f: CgeqrfFnPtr) {
    let _ = CGEQRF.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgeqrf(f: ZgeqrfFnPtr) {
    let _ = ZGEQRF.set(f);
}

// ORGQR
#[no_mangle]
pub unsafe extern "C" fn register_sorgqr(f: SorgqrFnPtr) {
    let _ = SORGQR.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dorgqr(f: DorgqrFnPtr) {
    let _ = DORGQR.set(f);
}

// UNGQR
#[no_mangle]
pub unsafe extern "C" fn register_cungqr(f: CungqrFnPtr) {
    let _ = CUNGQR.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zungqr(f: ZungqrFnPtr) {
    let _ = ZUNGQR.set(f);
}

// GESVD
#[no_mangle]
pub unsafe extern "C" fn register_sgesvd(f: SgesvdFnPtr) {
    let _ = SGESVD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgesvd(f: DgesvdFnPtr) {
    let _ = DGESVD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgesvd(f: CgesvdFnPtr) {
    let _ = CGESVD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgesvd(f: ZgesvdFnPtr) {
    let _ = ZGESVD.set(f);
}

// GESDD
#[no_mangle]
pub unsafe extern "C" fn register_sgesdd(f: SgesddFnPtr) {
    let _ = SGESDD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgesdd(f: DgesddFnPtr) {
    let _ = DGESDD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgesdd(f: CgesddFnPtr) {
    let _ = CGESDD.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgesdd(f: ZgesddFnPtr) {
    let _ = ZGESDD.set(f);
}

// GEEV
#[no_mangle]
pub unsafe extern "C" fn register_sgeev(f: SgeevFnPtr) {
    let _ = SGEEV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgeev(f: DgeevFnPtr) {
    let _ = DGEEV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgeev(f: CgeevFnPtr) {
    let _ = CGEEV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgeev(f: ZgeevFnPtr) {
    let _ = ZGEEV.set(f);
}

// SYEV
#[no_mangle]
pub unsafe extern "C" fn register_ssyev(f: SsyevFnPtr) {
    let _ = SSYEV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dsyev(f: DsyevFnPtr) {
    let _ = DSYEV.set(f);
}

// HEEV
#[no_mangle]
pub unsafe extern "C" fn register_cheev(f: CheevFnPtr) {
    let _ = CHEEV.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zheev(f: ZheevFnPtr) {
    let _ = ZHEEV.set(f);
}

// GEES
#[no_mangle]
pub unsafe extern "C" fn register_sgees(f: SgeesFnPtr) {
    let _ = SGEES.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_dgees(f: DgeesFnPtr) {
    let _ = DGEES.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_cgees(f: CgeesFnPtr) {
    let _ = CGEES.set(f);
}

#[no_mangle]
pub unsafe extern "C" fn register_zgees(f: ZgeesFnPtr) {
    let _ = ZGEES.set(f);
}

// =============================================================================
// Getter functions (internal use)
// =============================================================================

// GESV
pub(crate) fn get_sgesv() -> SgesvFnPtr {
    *SGESV.get().expect("sgesv not registered")
}

pub(crate) fn get_dgesv() -> DgesvFnPtr {
    *DGESV.get().expect("dgesv not registered")
}

pub(crate) fn get_cgesv() -> CgesvFnPtr {
    *CGESV.get().expect("cgesv not registered")
}

pub(crate) fn get_zgesv() -> ZgesvFnPtr {
    *ZGESV.get().expect("zgesv not registered")
}

// GETRF
pub(crate) fn get_sgetrf() -> SgetrfFnPtr {
    *SGETRF.get().expect("sgetrf not registered")
}

pub(crate) fn get_dgetrf() -> DgetrfFnPtr {
    *DGETRF.get().expect("dgetrf not registered")
}

pub(crate) fn get_cgetrf() -> CgetrfFnPtr {
    *CGETRF.get().expect("cgetrf not registered")
}

pub(crate) fn get_zgetrf() -> ZgetrfFnPtr {
    *ZGETRF.get().expect("zgetrf not registered")
}

// GETRI
pub(crate) fn get_sgetri() -> SgetriFnPtr {
    *SGETRI.get().expect("sgetri not registered")
}

pub(crate) fn get_dgetri() -> DgetriFnPtr {
    *DGETRI.get().expect("dgetri not registered")
}

pub(crate) fn get_cgetri() -> CgetriFnPtr {
    *CGETRI.get().expect("cgetri not registered")
}

pub(crate) fn get_zgetri() -> ZgetriFnPtr {
    *ZGETRI.get().expect("zgetri not registered")
}

// POTRF
pub(crate) fn get_spotrf() -> SpotrfFnPtr {
    *SPOTRF.get().expect("spotrf not registered")
}

pub(crate) fn get_dpotrf() -> DpotrfFnPtr {
    *DPOTRF.get().expect("dpotrf not registered")
}

pub(crate) fn get_cpotrf() -> CpotrfFnPtr {
    *CPOTRF.get().expect("cpotrf not registered")
}

pub(crate) fn get_zpotrf() -> ZpotrfFnPtr {
    *ZPOTRF.get().expect("zpotrf not registered")
}

// GELS
pub(crate) fn get_sgels() -> SgelsFnPtr {
    *SGELS.get().expect("sgels not registered")
}

pub(crate) fn get_dgels() -> DgelsFnPtr {
    *DGELS.get().expect("dgels not registered")
}

pub(crate) fn get_cgels() -> CgelsFnPtr {
    *CGELS.get().expect("cgels not registered")
}

pub(crate) fn get_zgels() -> ZgelsFnPtr {
    *ZGELS.get().expect("zgels not registered")
}

// GEQRF
pub(crate) fn get_sgeqrf() -> SgeqrfFnPtr {
    *SGEQRF.get().expect("sgeqrf not registered")
}

pub(crate) fn get_dgeqrf() -> DgeqrfFnPtr {
    *DGEQRF.get().expect("dgeqrf not registered")
}

pub(crate) fn get_cgeqrf() -> CgeqrfFnPtr {
    *CGEQRF.get().expect("cgeqrf not registered")
}

pub(crate) fn get_zgeqrf() -> ZgeqrfFnPtr {
    *ZGEQRF.get().expect("zgeqrf not registered")
}

// ORGQR
pub(crate) fn get_sorgqr() -> SorgqrFnPtr {
    *SORGQR.get().expect("sorgqr not registered")
}

pub(crate) fn get_dorgqr() -> DorgqrFnPtr {
    *DORGQR.get().expect("dorgqr not registered")
}

// UNGQR
pub(crate) fn get_cungqr() -> CungqrFnPtr {
    *CUNGQR.get().expect("cungqr not registered")
}

pub(crate) fn get_zungqr() -> ZungqrFnPtr {
    *ZUNGQR.get().expect("zungqr not registered")
}

// GESVD
pub(crate) fn get_sgesvd() -> SgesvdFnPtr {
    *SGESVD.get().expect("sgesvd not registered")
}

pub(crate) fn get_dgesvd() -> DgesvdFnPtr {
    *DGESVD.get().expect("dgesvd not registered")
}

pub(crate) fn get_cgesvd() -> CgesvdFnPtr {
    *CGESVD.get().expect("cgesvd not registered")
}

pub(crate) fn get_zgesvd() -> ZgesvdFnPtr {
    *ZGESVD.get().expect("zgesvd not registered")
}

// GESDD
pub(crate) fn get_sgesdd() -> SgesddFnPtr {
    *SGESDD.get().expect("sgesdd not registered")
}

pub(crate) fn get_dgesdd() -> DgesddFnPtr {
    *DGESDD.get().expect("dgesdd not registered")
}

pub(crate) fn get_cgesdd() -> CgesddFnPtr {
    *CGESDD.get().expect("cgesdd not registered")
}

pub(crate) fn get_zgesdd() -> ZgesddFnPtr {
    *ZGESDD.get().expect("zgesdd not registered")
}

// GEEV
pub(crate) fn get_sgeev() -> SgeevFnPtr {
    *SGEEV.get().expect("sgeev not registered")
}

pub(crate) fn get_dgeev() -> DgeevFnPtr {
    *DGEEV.get().expect("dgeev not registered")
}

pub(crate) fn get_cgeev() -> CgeevFnPtr {
    *CGEEV.get().expect("cgeev not registered")
}

pub(crate) fn get_zgeev() -> ZgeevFnPtr {
    *ZGEEV.get().expect("zgeev not registered")
}

// SYEV
pub(crate) fn get_ssyev() -> SsyevFnPtr {
    *SSYEV.get().expect("ssyev not registered")
}

pub(crate) fn get_dsyev() -> DsyevFnPtr {
    *DSYEV.get().expect("dsyev not registered")
}

// HEEV
pub(crate) fn get_cheev() -> CheevFnPtr {
    *CHEEV.get().expect("cheev not registered")
}

pub(crate) fn get_zheev() -> ZheevFnPtr {
    *ZHEEV.get().expect("zheev not registered")
}

// GEES
pub(crate) fn get_sgees() -> SgeesFnPtr {
    *SGEES.get().expect("sgees not registered")
}

pub(crate) fn get_dgees() -> DgeesFnPtr {
    *DGEES.get().expect("dgees not registered")
}

pub(crate) fn get_cgees() -> CgeesFnPtr {
    *CGEES.get().expect("cgees not registered")
}

pub(crate) fn get_zgees() -> ZgeesFnPtr {
    *ZGEES.get().expect("zgees not registered")
}
