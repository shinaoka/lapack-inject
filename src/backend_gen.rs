// Auto-generated: LP64/ILP64 dual function pointer types.

pub type Cgesc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *const num_complex::Complex32,
    lda: *const i32,
    rhs: *mut num_complex::Complex32,
    ipiv: *const i32,
    jpiv: *const i32,
    scale: *mut f32,
);
pub type Cgesc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *const num_complex::Complex32,
    lda: *const i64,
    rhs: *mut num_complex::Complex32,
    ipiv: *const i64,
    jpiv: *const i64,
    scale: *mut f32,
);
define_dual_backend!(cgesc2, Cgesc2Lp64FnPtr, Cgesc2Ilp64FnPtr);

pub type CgesvLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    nrhs: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    ipiv: *mut i32,
    B: *mut num_complex::Complex32,
    ldb: *const i32,
    info: *mut i32,
);
pub type CgesvIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    nrhs: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    ipiv: *mut i64,
    B: *mut num_complex::Complex32,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(cgesv, CgesvLp64FnPtr, CgesvIlp64FnPtr);

pub type CgesvdLp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i32,
    n: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    S: *mut f32,
    U: *mut num_complex::Complex32,
    ldu: *const i32,
    VT: *mut num_complex::Complex32,
    ldvt: *const i32,
    work: *mut num_complex::Complex32,
    lwork: *const i32,
    rwork: *mut f32,
    info: *mut i32,
);
pub type CgesvdIlp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i64,
    n: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    S: *mut f32,
    U: *mut num_complex::Complex32,
    ldu: *const i64,
    VT: *mut num_complex::Complex32,
    ldvt: *const i64,
    work: *mut num_complex::Complex32,
    lwork: *const i64,
    rwork: *mut f32,
    info: *mut i64,
);
define_dual_backend!(cgesvd, CgesvdLp64FnPtr, CgesvdIlp64FnPtr);

pub type Cgetc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    ipiv: *mut i32,
    jpiv: *mut i32,
    info: *mut i32,
);
pub type Cgetc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(cgetc2, Cgetc2Lp64FnPtr, Cgetc2Ilp64FnPtr);

pub type CgetrfLp64FnPtr = unsafe extern "C" fn(
    m: *const i32,
    n: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    ipiv: *mut i32,
    info: *mut i32,
);
pub type CgetrfIlp64FnPtr = unsafe extern "C" fn(
    m: *const i64,
    n: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(cgetrf, CgetrfLp64FnPtr, CgetrfIlp64FnPtr);

pub type CgetriLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    ipiv: *const i32,
    work: *mut num_complex::Complex32,
    lwork: *const i32,
    info: *mut i32,
);
pub type CgetriIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    ipiv: *const i64,
    work: *mut num_complex::Complex32,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(cgetri, CgetriLp64FnPtr, CgetriIlp64FnPtr);

pub type CgetrsLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i32,
    nrhs: *const i32,
    A: *const num_complex::Complex32,
    lda: *const i32,
    ipiv: *const i32,
    B: *mut num_complex::Complex32,
    ldb: *const i32,
    info: *mut i32,
);
pub type CgetrsIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i64,
    nrhs: *const i64,
    A: *const num_complex::Complex32,
    lda: *const i64,
    ipiv: *const i64,
    B: *mut num_complex::Complex32,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(cgetrs, CgetrsLp64FnPtr, CgetrsIlp64FnPtr);

pub type CpotrfLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i32,
    A: *mut num_complex::Complex32,
    lda: *const i32,
    info: *mut i32,
);
pub type CpotrfIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i64,
    A: *mut num_complex::Complex32,
    lda: *const i64,
    info: *mut i64,
);
define_dual_backend!(cpotrf, CpotrfLp64FnPtr, CpotrfIlp64FnPtr);

pub type Dgesc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *const f64,
    lda: *const i32,
    rhs: *mut f64,
    ipiv: *const i32,
    jpiv: *const i32,
    scale: *mut f64,
);
pub type Dgesc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *const f64,
    lda: *const i64,
    rhs: *mut f64,
    ipiv: *const i64,
    jpiv: *const i64,
    scale: *mut f64,
);
define_dual_backend!(dgesc2, Dgesc2Lp64FnPtr, Dgesc2Ilp64FnPtr);

pub type DgesvLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    nrhs: *const i32,
    A: *mut f64,
    lda: *const i32,
    ipiv: *mut i32,
    B: *mut f64,
    ldb: *const i32,
    info: *mut i32,
);
pub type DgesvIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    nrhs: *const i64,
    A: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    B: *mut f64,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(dgesv, DgesvLp64FnPtr, DgesvIlp64FnPtr);

pub type DgesvdLp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i32,
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    S: *mut f64,
    U: *mut f64,
    ldu: *const i32,
    VT: *mut f64,
    ldvt: *const i32,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
);
pub type DgesvdIlp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i64,
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    S: *mut f64,
    U: *mut f64,
    ldu: *const i64,
    VT: *mut f64,
    ldvt: *const i64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(dgesvd, DgesvdLp64FnPtr, DgesvdIlp64FnPtr);

pub type Dgetc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    ipiv: *mut i32,
    jpiv: *mut i32,
    info: *mut i32,
);
pub type Dgetc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(dgetc2, Dgetc2Lp64FnPtr, Dgetc2Ilp64FnPtr);

pub type DgetrfLp64FnPtr = unsafe extern "C" fn(
    m: *const i32,
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    ipiv: *mut i32,
    info: *mut i32,
);
pub type DgetrfIlp64FnPtr = unsafe extern "C" fn(
    m: *const i64,
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(dgetrf, DgetrfLp64FnPtr, DgetrfIlp64FnPtr);

pub type DgetriLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    ipiv: *const i32,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
);
pub type DgetriIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    ipiv: *const i64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(dgetri, DgetriLp64FnPtr, DgetriIlp64FnPtr);

pub type DgetrsLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i32,
    nrhs: *const i32,
    A: *const f64,
    lda: *const i32,
    ipiv: *const i32,
    B: *mut f64,
    ldb: *const i32,
    info: *mut i32,
);
pub type DgetrsIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i64,
    nrhs: *const i64,
    A: *const f64,
    lda: *const i64,
    ipiv: *const i64,
    B: *mut f64,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(dgetrs, DgetrsLp64FnPtr, DgetrsIlp64FnPtr);

pub type DpotrfLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    info: *mut i32,
);
pub type DpotrfIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    info: *mut i64,
);
define_dual_backend!(dpotrf, DpotrfLp64FnPtr, DpotrfIlp64FnPtr);

pub type DsyevLp64FnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i32,
    A: *mut f64,
    lda: *const i32,
    W: *mut f64,
    work: *mut f64,
    lwork: *const i32,
    info: *mut i32,
);
pub type DsyevIlp64FnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i64,
    A: *mut f64,
    lda: *const i64,
    W: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(dsyev, DsyevLp64FnPtr, DsyevIlp64FnPtr);

pub type Sgesc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *const f32,
    lda: *const i32,
    rhs: *mut f32,
    ipiv: *const i32,
    jpiv: *const i32,
    scale: *mut f32,
);
pub type Sgesc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *const f32,
    lda: *const i64,
    rhs: *mut f32,
    ipiv: *const i64,
    jpiv: *const i64,
    scale: *mut f32,
);
define_dual_backend!(sgesc2, Sgesc2Lp64FnPtr, Sgesc2Ilp64FnPtr);

pub type SgesvLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    nrhs: *const i32,
    A: *mut f32,
    lda: *const i32,
    ipiv: *mut i32,
    B: *mut f32,
    ldb: *const i32,
    info: *mut i32,
);
pub type SgesvIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    nrhs: *const i64,
    A: *mut f32,
    lda: *const i64,
    ipiv: *mut i64,
    B: *mut f32,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(sgesv, SgesvLp64FnPtr, SgesvIlp64FnPtr);

pub type SgesvdLp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i32,
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    S: *mut f32,
    U: *mut f32,
    ldu: *const i32,
    VT: *mut f32,
    ldvt: *const i32,
    work: *mut f32,
    lwork: *const i32,
    info: *mut i32,
);
pub type SgesvdIlp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i64,
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    S: *mut f32,
    U: *mut f32,
    ldu: *const i64,
    VT: *mut f32,
    ldvt: *const i64,
    work: *mut f32,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(sgesvd, SgesvdLp64FnPtr, SgesvdIlp64FnPtr);

pub type Sgetc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    ipiv: *mut i32,
    jpiv: *mut i32,
    info: *mut i32,
);
pub type Sgetc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(sgetc2, Sgetc2Lp64FnPtr, Sgetc2Ilp64FnPtr);

pub type SgetrfLp64FnPtr = unsafe extern "C" fn(
    m: *const i32,
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    ipiv: *mut i32,
    info: *mut i32,
);
pub type SgetrfIlp64FnPtr = unsafe extern "C" fn(
    m: *const i64,
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(sgetrf, SgetrfLp64FnPtr, SgetrfIlp64FnPtr);

pub type SgetriLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    ipiv: *const i32,
    work: *mut f32,
    lwork: *const i32,
    info: *mut i32,
);
pub type SgetriIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    ipiv: *const i64,
    work: *mut f32,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(sgetri, SgetriLp64FnPtr, SgetriIlp64FnPtr);

pub type SgetrsLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i32,
    nrhs: *const i32,
    A: *const f32,
    lda: *const i32,
    ipiv: *const i32,
    B: *mut f32,
    ldb: *const i32,
    info: *mut i32,
);
pub type SgetrsIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i64,
    nrhs: *const i64,
    A: *const f32,
    lda: *const i64,
    ipiv: *const i64,
    B: *mut f32,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(sgetrs, SgetrsLp64FnPtr, SgetrsIlp64FnPtr);

pub type SpotrfLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    info: *mut i32,
);
pub type SpotrfIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    info: *mut i64,
);
define_dual_backend!(spotrf, SpotrfLp64FnPtr, SpotrfIlp64FnPtr);

pub type SsyevLp64FnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i32,
    A: *mut f32,
    lda: *const i32,
    W: *mut f32,
    work: *mut f32,
    lwork: *const i32,
    info: *mut i32,
);
pub type SsyevIlp64FnPtr = unsafe extern "C" fn(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const i64,
    A: *mut f32,
    lda: *const i64,
    W: *mut f32,
    work: *mut f32,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(ssyev, SsyevLp64FnPtr, SsyevIlp64FnPtr);

pub type Zgesc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *const num_complex::Complex64,
    lda: *const i32,
    rhs: *mut num_complex::Complex64,
    ipiv: *const i32,
    jpiv: *const i32,
    scale: *mut f64,
);
pub type Zgesc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *const num_complex::Complex64,
    lda: *const i64,
    rhs: *mut num_complex::Complex64,
    ipiv: *const i64,
    jpiv: *const i64,
    scale: *mut f64,
);
define_dual_backend!(zgesc2, Zgesc2Lp64FnPtr, Zgesc2Ilp64FnPtr);

pub type ZgesvLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    nrhs: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    ipiv: *mut i32,
    B: *mut num_complex::Complex64,
    ldb: *const i32,
    info: *mut i32,
);
pub type ZgesvIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    nrhs: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    ipiv: *mut i64,
    B: *mut num_complex::Complex64,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(zgesv, ZgesvLp64FnPtr, ZgesvIlp64FnPtr);

pub type ZgesvdLp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i32,
    n: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    S: *mut f64,
    U: *mut num_complex::Complex64,
    ldu: *const i32,
    VT: *mut num_complex::Complex64,
    ldvt: *const i32,
    work: *mut num_complex::Complex64,
    lwork: *const i32,
    rwork: *mut f64,
    info: *mut i32,
);
pub type ZgesvdIlp64FnPtr = unsafe extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const i64,
    n: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    S: *mut f64,
    U: *mut num_complex::Complex64,
    ldu: *const i64,
    VT: *mut num_complex::Complex64,
    ldvt: *const i64,
    work: *mut num_complex::Complex64,
    lwork: *const i64,
    rwork: *mut f64,
    info: *mut i64,
);
define_dual_backend!(zgesvd, ZgesvdLp64FnPtr, ZgesvdIlp64FnPtr);

pub type Zgetc2Lp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    ipiv: *mut i32,
    jpiv: *mut i32,
    info: *mut i32,
);
pub type Zgetc2Ilp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(zgetc2, Zgetc2Lp64FnPtr, Zgetc2Ilp64FnPtr);

pub type ZgetrfLp64FnPtr = unsafe extern "C" fn(
    m: *const i32,
    n: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    ipiv: *mut i32,
    info: *mut i32,
);
pub type ZgetrfIlp64FnPtr = unsafe extern "C" fn(
    m: *const i64,
    n: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    ipiv: *mut i64,
    info: *mut i64,
);
define_dual_backend!(zgetrf, ZgetrfLp64FnPtr, ZgetrfIlp64FnPtr);

pub type ZgetriLp64FnPtr = unsafe extern "C" fn(
    n: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    ipiv: *const i32,
    work: *mut num_complex::Complex64,
    lwork: *const i32,
    info: *mut i32,
);
pub type ZgetriIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    ipiv: *const i64,
    work: *mut num_complex::Complex64,
    lwork: *const i64,
    info: *mut i64,
);
define_dual_backend!(zgetri, ZgetriLp64FnPtr, ZgetriIlp64FnPtr);

pub type ZgetrsLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i32,
    nrhs: *const i32,
    A: *const num_complex::Complex64,
    lda: *const i32,
    ipiv: *const i32,
    B: *mut num_complex::Complex64,
    ldb: *const i32,
    info: *mut i32,
);
pub type ZgetrsIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    n: *const i64,
    nrhs: *const i64,
    A: *const num_complex::Complex64,
    lda: *const i64,
    ipiv: *const i64,
    B: *mut num_complex::Complex64,
    ldb: *const i64,
    info: *mut i64,
);
define_dual_backend!(zgetrs, ZgetrsLp64FnPtr, ZgetrsIlp64FnPtr);

pub type ZpotrfLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i32,
    A: *mut num_complex::Complex64,
    lda: *const i32,
    info: *mut i32,
);
pub type ZpotrfIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const i64,
    A: *mut num_complex::Complex64,
    lda: *const i64,
    info: *mut i64,
);
define_dual_backend!(zpotrf, ZpotrfLp64FnPtr, ZpotrfIlp64FnPtr);
