//! Auto-registration of Fortran LAPACK functions from OpenBLAS.
//!
//! This module uses the `ctor` crate to automatically register Fortran LAPACK
//! function pointers when the library is loaded.

use crate::backend::*;
use num_complex::{Complex32, Complex64};

// Fortran LAPACK declarations (linked from OpenBLAS)
#[link(name = "openblas")]
extern "C" {
    // GESV - Linear solve
    fn sgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f32,
        ldb: *const i32,
        info: *mut i32,
    );
    fn dgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
    fn cgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut Complex32,
        ldb: *const i32,
        info: *mut i32,
    );
    fn zgesv_(
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        b: *mut Complex64,
        ldb: *const i32,
        info: *mut i32,
    );

    // GETRF - LU factorization
    fn sgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn dgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn cgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );
    fn zgetrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        info: *mut i32,
    );

    // GETRI - Matrix inverse
    fn sgetri_(
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgetri_(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn cgetri_(
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn zgetri_(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

    // POTRF - Cholesky factorization
    fn spotrf_(uplo: *const i8, n: *const i32, a: *mut f32, lda: *const i32, info: *mut i32);
    fn dpotrf_(uplo: *const i8, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn cpotrf_(
        uplo: *const i8,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        info: *mut i32,
    );
    fn zpotrf_(
        uplo: *const i8,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        info: *mut i32,
    );

    // GELS - Least squares
    fn sgels_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f32,
        lda: *const i32,
        b: *mut f32,
        ldb: *const i32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgels_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn cgels_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        b: *mut Complex32,
        ldb: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn zgels_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        b: *mut Complex64,
        ldb: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

    // GEQRF - QR factorization
    fn sgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        tau: *mut f32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn cgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        tau: *mut Complex32,
        work: *mut Complex32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn zgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        tau: *mut Complex64,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

    // ORGQR - Generate Q (real)
    fn sorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f32,
        lda: *const i32,
        tau: *const f32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );

    // UNGQR - Generate Q (complex)
    fn cungqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        tau: *const Complex32,
        work: *mut Complex32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn zungqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        tau: *const Complex64,
        work: *mut Complex64,
        lwork: *const i32,
        info: *mut i32,
    );

    // GESVD - SVD (standard)
    fn sgesvd_(
        jobu: *const i8,
        jobvt: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        s: *mut f32,
        u: *mut f32,
        ldu: *const i32,
        vt: *mut f32,
        ldvt: *const i32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgesvd_(
        jobu: *const i8,
        jobvt: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn cgesvd_(
        jobu: *const i8,
        jobvt: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        s: *mut f32,
        u: *mut Complex32,
        ldu: *const i32,
        vt: *mut Complex32,
        ldvt: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        rwork: *mut f32,
        info: *mut i32,
    );
    fn zgesvd_(
        jobu: *const i8,
        jobvt: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: *const i32,
        vt: *mut Complex64,
        ldvt: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    // GESDD - SVD (divide and conquer)
    fn sgesdd_(
        jobz: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        s: *mut f32,
        u: *mut f32,
        ldu: *const i32,
        vt: *mut f32,
        ldvt: *const i32,
        work: *mut f32,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    fn dgesdd_(
        jobz: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    fn cgesdd_(
        jobz: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        s: *mut f32,
        u: *mut Complex32,
        ldu: *const i32,
        vt: *mut Complex32,
        ldvt: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        rwork: *mut f32,
        iwork: *mut i32,
        info: *mut i32,
    );
    fn zgesdd_(
        jobz: *const i8,
        m: *const i32,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        s: *mut f64,
        u: *mut Complex64,
        ldu: *const i32,
        vt: *mut Complex64,
        ldvt: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        iwork: *mut i32,
        info: *mut i32,
    );

    // GEEV - General eigenvalue
    fn sgeev_(
        jobvl: *const i8,
        jobvr: *const i8,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        wr: *mut f32,
        wi: *mut f32,
        vl: *mut f32,
        ldvl: *const i32,
        vr: *mut f32,
        ldvr: *const i32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgeev_(
        jobvl: *const i8,
        jobvr: *const i8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        wr: *mut f64,
        wi: *mut f64,
        vl: *mut f64,
        ldvl: *const i32,
        vr: *mut f64,
        ldvr: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn cgeev_(
        jobvl: *const i8,
        jobvr: *const i8,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        w: *mut Complex32,
        vl: *mut Complex32,
        ldvl: *const i32,
        vr: *mut Complex32,
        ldvr: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        rwork: *mut f32,
        info: *mut i32,
    );
    fn zgeev_(
        jobvl: *const i8,
        jobvr: *const i8,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        w: *mut Complex64,
        vl: *mut Complex64,
        ldvl: *const i32,
        vr: *mut Complex64,
        ldvr: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    // SYEV - Symmetric eigenvalue
    fn ssyev_(
        jobz: *const i8,
        uplo: *const i8,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        w: *mut f32,
        work: *mut f32,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dsyev_(
        jobz: *const i8,
        uplo: *const i8,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        w: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );

    // HEEV - Hermitian eigenvalue
    fn cheev_(
        jobz: *const i8,
        uplo: *const i8,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        w: *mut f32,
        work: *mut Complex32,
        lwork: *const i32,
        rwork: *mut f32,
        info: *mut i32,
    );
    fn zheev_(
        jobz: *const i8,
        uplo: *const i8,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        w: *mut f64,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        info: *mut i32,
    );

    // GEES - Schur decomposition
    fn sgees_(
        jobvs: *const i8,
        sort: *const i8,
        select: Option<SgeesSelectFn>,
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        sdim: *mut i32,
        wr: *mut f32,
        wi: *mut f32,
        vs: *mut f32,
        ldvs: *const i32,
        work: *mut f32,
        lwork: *const i32,
        bwork: *mut i32,
        info: *mut i32,
    );
    fn dgees_(
        jobvs: *const i8,
        sort: *const i8,
        select: Option<DgeesSelectFn>,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        sdim: *mut i32,
        wr: *mut f64,
        wi: *mut f64,
        vs: *mut f64,
        ldvs: *const i32,
        work: *mut f64,
        lwork: *const i32,
        bwork: *mut i32,
        info: *mut i32,
    );
    fn cgees_(
        jobvs: *const i8,
        sort: *const i8,
        select: Option<CgeesSelectFn>,
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        sdim: *mut i32,
        w: *mut Complex32,
        vs: *mut Complex32,
        ldvs: *const i32,
        work: *mut Complex32,
        lwork: *const i32,
        rwork: *mut f32,
        bwork: *mut i32,
        info: *mut i32,
    );
    fn zgees_(
        jobvs: *const i8,
        sort: *const i8,
        select: Option<ZgeesSelectFn>,
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        sdim: *mut i32,
        w: *mut Complex64,
        vs: *mut Complex64,
        ldvs: *const i32,
        work: *mut Complex64,
        lwork: *const i32,
        rwork: *mut f64,
        bwork: *mut i32,
        info: *mut i32,
    );
}

#[ctor::ctor]
fn register_all_lapack() {
    unsafe {
        // GESV
        register_sgesv(std::mem::transmute(sgesv_ as *const ()));
        register_dgesv(std::mem::transmute(dgesv_ as *const ()));
        register_cgesv(std::mem::transmute(cgesv_ as *const ()));
        register_zgesv(std::mem::transmute(zgesv_ as *const ()));

        // GETRF
        register_sgetrf(std::mem::transmute(sgetrf_ as *const ()));
        register_dgetrf(std::mem::transmute(dgetrf_ as *const ()));
        register_cgetrf(std::mem::transmute(cgetrf_ as *const ()));
        register_zgetrf(std::mem::transmute(zgetrf_ as *const ()));

        // GETRI
        register_sgetri(std::mem::transmute(sgetri_ as *const ()));
        register_dgetri(std::mem::transmute(dgetri_ as *const ()));
        register_cgetri(std::mem::transmute(cgetri_ as *const ()));
        register_zgetri(std::mem::transmute(zgetri_ as *const ()));

        // POTRF
        register_spotrf(std::mem::transmute(spotrf_ as *const ()));
        register_dpotrf(std::mem::transmute(dpotrf_ as *const ()));
        register_cpotrf(std::mem::transmute(cpotrf_ as *const ()));
        register_zpotrf(std::mem::transmute(zpotrf_ as *const ()));

        // GELS
        register_sgels(std::mem::transmute(sgels_ as *const ()));
        register_dgels(std::mem::transmute(dgels_ as *const ()));
        register_cgels(std::mem::transmute(cgels_ as *const ()));
        register_zgels(std::mem::transmute(zgels_ as *const ()));

        // GEQRF
        register_sgeqrf(std::mem::transmute(sgeqrf_ as *const ()));
        register_dgeqrf(std::mem::transmute(dgeqrf_ as *const ()));
        register_cgeqrf(std::mem::transmute(cgeqrf_ as *const ()));
        register_zgeqrf(std::mem::transmute(zgeqrf_ as *const ()));

        // ORGQR
        register_sorgqr(std::mem::transmute(sorgqr_ as *const ()));
        register_dorgqr(std::mem::transmute(dorgqr_ as *const ()));

        // UNGQR
        register_cungqr(std::mem::transmute(cungqr_ as *const ()));
        register_zungqr(std::mem::transmute(zungqr_ as *const ()));

        // GESVD
        register_sgesvd(std::mem::transmute(sgesvd_ as *const ()));
        register_dgesvd(std::mem::transmute(dgesvd_ as *const ()));
        register_cgesvd(std::mem::transmute(cgesvd_ as *const ()));
        register_zgesvd(std::mem::transmute(zgesvd_ as *const ()));

        // GESDD
        register_sgesdd(std::mem::transmute(sgesdd_ as *const ()));
        register_dgesdd(std::mem::transmute(dgesdd_ as *const ()));
        register_cgesdd(std::mem::transmute(cgesdd_ as *const ()));
        register_zgesdd(std::mem::transmute(zgesdd_ as *const ()));

        // GEEV
        register_sgeev(std::mem::transmute(sgeev_ as *const ()));
        register_dgeev(std::mem::transmute(dgeev_ as *const ()));
        register_cgeev(std::mem::transmute(cgeev_ as *const ()));
        register_zgeev(std::mem::transmute(zgeev_ as *const ()));

        // SYEV
        register_ssyev(std::mem::transmute(ssyev_ as *const ()));
        register_dsyev(std::mem::transmute(dsyev_ as *const ()));

        // HEEV
        register_cheev(std::mem::transmute(cheev_ as *const ()));
        register_zheev(std::mem::transmute(zheev_ as *const ()));

        // GEES
        register_sgees(std::mem::transmute(sgees_ as *const ()));
        register_dgees(std::mem::transmute(dgees_ as *const ()));
        register_cgees(std::mem::transmute(cgees_ as *const ()));
        register_zgees(std::mem::transmute(zgees_ as *const ()));
    }
}
