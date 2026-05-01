// Auto-generated: Fortran LAPACK symbol exports with dual LP64/ILP64 dispatch.

#[no_mangle]
pub unsafe extern "C" fn cgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    W: *mut Complex32,
    VL: *mut Complex32,
    ldvl: *const lapackint,
    VR: *mut Complex32,
    ldvr: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgeev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgeev_for_lp64();
    match provider {
        CgeevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldvl_lp64: *const i32 = ldvl as *const i32;
            let ldvr_lp64: *const i32 = ldvr as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobvl, jobvr, n_lp64, A, lda_lp64, W, VL, ldvl_lp64, VR, ldvr_lp64, work, lwork_lp64, rwork, info_lp64)
        }
        CgeevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldvl_ilp64: *const i64 = ldvl as *const i64;
            let ldvr_ilp64: *const i64 = ldvr as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobvl, jobvr, n_ilp64, A, lda_ilp64, W, VL, ldvl_ilp64, VR, ldvr_ilp64, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    tau: *mut Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgeqrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgeqrf_for_lp64();
    match provider {
        CgeqrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        CgeqrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgesc2_(
    n: *const lapackint,
    A: *const Complex32,
    lda: *const lapackint,
    rhs: *mut Complex32,
    ipiv: *const lapackint,
    jpiv: *const lapackint,
    scale: *mut f32,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgesc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgesc2_for_lp64();
    match provider {
        Cgesc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let jpiv_lp64: *const i32 = jpiv as *const i32;
            fun(n_lp64, A, lda_lp64, rhs, ipiv_lp64, jpiv_lp64, scale)
        }
        Cgesc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let jpiv_ilp64: *const i64 = jpiv as *const i64;
            fun(n_ilp64, A, lda_ilp64, rhs, ipiv_ilp64, jpiv_ilp64, scale)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    B: *mut Complex32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgesv_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgesv_for_lp64();
    match provider {
        CgesvProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        CgesvProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    S: *mut f32,
    U: *mut Complex32,
    ldu: *const lapackint,
    VT: *mut Complex32,
    ldvt: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgesvd_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgesvd_for_lp64();
    match provider {
        CgesvdProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldu_lp64: *const i32 = ldu as *const i32;
            let ldvt_lp64: *const i32 = ldvt as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobu, jobvt, m_lp64, n_lp64, A, lda_lp64, S, U, ldu_lp64, VT, ldvt_lp64, work, lwork_lp64, rwork, info_lp64)
        }
        CgesvdProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldu_ilp64: *const i64 = ldu as *const i64;
            let ldvt_ilp64: *const i64 = ldvt as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobu, jobvt, m_ilp64, n_ilp64, A, lda_ilp64, S, U, ldu_ilp64, VT, ldvt_ilp64, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgetc2_(
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    jpiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgetc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgetc2_for_lp64();
    match provider {
        Cgetc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let jpiv_lp64: *mut i32 = jpiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, jpiv_lp64, info_lp64)
        }
        Cgetc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let jpiv_ilp64: *mut i64 = jpiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, jpiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgetrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgetrf_for_lp64();
    match provider {
        CgetrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, ipiv_lp64, info_lp64)
        }
        CgetrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, ipiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgetri_(
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgetri_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgetri_for_lp64();
    match provider {
        CgetriProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, work, lwork_lp64, info_lp64)
        }
        CgetriProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cgetrs_(
    trans: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const Complex32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    B: *mut Complex32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cgetrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cgetrs_for_lp64();
    match provider {
        CgetrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(trans, n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        CgetrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(trans, n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cheev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    W: *mut f32,
    work: *mut Complex32,
    lwork: *const lapackint,
    rwork: *mut f32,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cheev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cheev_for_lp64();
    match provider {
        CheevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobz, uplo, n_lp64, A, lda_lp64, W, work, lwork_lp64, rwork, info_lp64)
        }
        CheevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobz, uplo, n_ilp64, A, lda_ilp64, W, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cpotrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cpotrf_for_lp64();
    match provider {
        CpotrfProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, n_lp64, A, lda_lp64, info_lp64)
        }
        CpotrfProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, n_ilp64, A, lda_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ctrtrs_(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const Complex32,
    lda: *const lapackint,
    B: *mut Complex32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_ctrtrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_ctrtrs_for_lp64();
    match provider {
        CtrtrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, trans, diag, n_lp64, nrhs_lp64, A, lda_lp64, B, ldb_lp64, info_lp64)
        }
        CtrtrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, trans, diag, n_ilp64, nrhs_ilp64, A, lda_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cungqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    A: *mut Complex32,
    lda: *const lapackint,
    tau: *const Complex32,
    work: *mut Complex32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_cungqr_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_cungqr_for_lp64();
    match provider {
        CungqrProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let k_lp64: *const i32 = k as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, k_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        CungqrProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let k_ilp64: *const i64 = k as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, k_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    WR: *mut f64,
    WI: *mut f64,
    VL: *mut f64,
    ldvl: *const lapackint,
    VR: *mut f64,
    ldvr: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgeev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgeev_for_lp64();
    match provider {
        DgeevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldvl_lp64: *const i32 = ldvl as *const i32;
            let ldvr_lp64: *const i32 = ldvr as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobvl, jobvr, n_lp64, A, lda_lp64, WR, WI, VL, ldvl_lp64, VR, ldvr_lp64, work, lwork_lp64, info_lp64)
        }
        DgeevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldvl_ilp64: *const i64 = ldvl as *const i64;
            let ldvr_ilp64: *const i64 = ldvr as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobvl, jobvr, n_ilp64, A, lda_ilp64, WR, WI, VL, ldvl_ilp64, VR, ldvr_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgeqrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgeqrf_for_lp64();
    match provider {
        DgeqrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        DgeqrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgesc2_(
    n: *const lapackint,
    A: *const f64,
    lda: *const lapackint,
    rhs: *mut f64,
    ipiv: *const lapackint,
    jpiv: *const lapackint,
    scale: *mut f64,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgesc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgesc2_for_lp64();
    match provider {
        Dgesc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let jpiv_lp64: *const i32 = jpiv as *const i32;
            fun(n_lp64, A, lda_lp64, rhs, ipiv_lp64, jpiv_lp64, scale)
        }
        Dgesc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let jpiv_ilp64: *const i64 = jpiv as *const i64;
            fun(n_ilp64, A, lda_ilp64, rhs, ipiv_ilp64, jpiv_ilp64, scale)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    B: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgesv_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgesv_for_lp64();
    match provider {
        DgesvProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        DgesvProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    S: *mut f64,
    U: *mut f64,
    ldu: *const lapackint,
    VT: *mut f64,
    ldvt: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgesvd_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgesvd_for_lp64();
    match provider {
        DgesvdProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldu_lp64: *const i32 = ldu as *const i32;
            let ldvt_lp64: *const i32 = ldvt as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobu, jobvt, m_lp64, n_lp64, A, lda_lp64, S, U, ldu_lp64, VT, ldvt_lp64, work, lwork_lp64, info_lp64)
        }
        DgesvdProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldu_ilp64: *const i64 = ldu as *const i64;
            let ldvt_ilp64: *const i64 = ldvt as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobu, jobvt, m_ilp64, n_ilp64, A, lda_ilp64, S, U, ldu_ilp64, VT, ldvt_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgetc2_(
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    jpiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetc2_for_lp64();
    match provider {
        Dgetc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let jpiv_lp64: *mut i32 = jpiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, jpiv_lp64, info_lp64)
        }
        Dgetc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let jpiv_ilp64: *mut i64 = jpiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, jpiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetrf_for_lp64();
    match provider {
        DgetrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, ipiv_lp64, info_lp64)
        }
        DgetrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, ipiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgetri_(
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetri_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetri_for_lp64();
    match provider {
        DgetriProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, work, lwork_lp64, info_lp64)
        }
        DgetriProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dgetrs_(
    trans: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const f64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    B: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetrs_for_lp64();
    match provider {
        DgetrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(trans, n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        DgetrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(trans, n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dorgqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    tau: *const f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dorgqr_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dorgqr_for_lp64();
    match provider {
        DorgqrProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let k_lp64: *const i32 = k as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, k_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        DorgqrProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let k_ilp64: *const i64 = k as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, k_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dpotrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dpotrf_for_lp64();
    match provider {
        DpotrfProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, n_lp64, A, lda_lp64, info_lp64)
        }
        DpotrfProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, n_ilp64, A, lda_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dsyev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut f64,
    lda: *const lapackint,
    W: *mut f64,
    work: *mut f64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dsyev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dsyev_for_lp64();
    match provider {
        DsyevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobz, uplo, n_lp64, A, lda_lp64, W, work, lwork_lp64, info_lp64)
        }
        DsyevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobz, uplo, n_ilp64, A, lda_ilp64, W, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn dtrtrs_(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const f64,
    lda: *const lapackint,
    B: *mut f64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_dtrtrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dtrtrs_for_lp64();
    match provider {
        DtrtrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, trans, diag, n_lp64, nrhs_lp64, A, lda_lp64, B, ldb_lp64, info_lp64)
        }
        DtrtrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, trans, diag, n_ilp64, nrhs_ilp64, A, lda_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    WR: *mut f32,
    WI: *mut f32,
    VL: *mut f32,
    ldvl: *const lapackint,
    VR: *mut f32,
    ldvr: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgeev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgeev_for_lp64();
    match provider {
        SgeevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldvl_lp64: *const i32 = ldvl as *const i32;
            let ldvr_lp64: *const i32 = ldvr as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobvl, jobvr, n_lp64, A, lda_lp64, WR, WI, VL, ldvl_lp64, VR, ldvr_lp64, work, lwork_lp64, info_lp64)
        }
        SgeevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldvl_ilp64: *const i64 = ldvl as *const i64;
            let ldvr_ilp64: *const i64 = ldvr as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobvl, jobvr, n_ilp64, A, lda_ilp64, WR, WI, VL, ldvl_ilp64, VR, ldvr_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    tau: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgeqrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgeqrf_for_lp64();
    match provider {
        SgeqrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        SgeqrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgesc2_(
    n: *const lapackint,
    A: *const f32,
    lda: *const lapackint,
    rhs: *mut f32,
    ipiv: *const lapackint,
    jpiv: *const lapackint,
    scale: *mut f32,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgesc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgesc2_for_lp64();
    match provider {
        Sgesc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let jpiv_lp64: *const i32 = jpiv as *const i32;
            fun(n_lp64, A, lda_lp64, rhs, ipiv_lp64, jpiv_lp64, scale)
        }
        Sgesc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let jpiv_ilp64: *const i64 = jpiv as *const i64;
            fun(n_ilp64, A, lda_ilp64, rhs, ipiv_ilp64, jpiv_ilp64, scale)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    B: *mut f32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgesv_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgesv_for_lp64();
    match provider {
        SgesvProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        SgesvProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    S: *mut f32,
    U: *mut f32,
    ldu: *const lapackint,
    VT: *mut f32,
    ldvt: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgesvd_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgesvd_for_lp64();
    match provider {
        SgesvdProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldu_lp64: *const i32 = ldu as *const i32;
            let ldvt_lp64: *const i32 = ldvt as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobu, jobvt, m_lp64, n_lp64, A, lda_lp64, S, U, ldu_lp64, VT, ldvt_lp64, work, lwork_lp64, info_lp64)
        }
        SgesvdProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldu_ilp64: *const i64 = ldu as *const i64;
            let ldvt_ilp64: *const i64 = ldvt as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobu, jobvt, m_ilp64, n_ilp64, A, lda_ilp64, S, U, ldu_ilp64, VT, ldvt_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgetc2_(
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    jpiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgetc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgetc2_for_lp64();
    match provider {
        Sgetc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let jpiv_lp64: *mut i32 = jpiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, jpiv_lp64, info_lp64)
        }
        Sgetc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let jpiv_ilp64: *mut i64 = jpiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, jpiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgetrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgetrf_for_lp64();
    match provider {
        SgetrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, ipiv_lp64, info_lp64)
        }
        SgetrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, ipiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgetri_(
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgetri_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgetri_for_lp64();
    match provider {
        SgetriProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, work, lwork_lp64, info_lp64)
        }
        SgetriProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sgetrs_(
    trans: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const f32,
    lda: *const lapackint,
    ipiv: *const lapackint,
    B: *mut f32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sgetrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sgetrs_for_lp64();
    match provider {
        SgetrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(trans, n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        SgetrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(trans, n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn sorgqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    tau: *const f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_sorgqr_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_sorgqr_for_lp64();
    match provider {
        SorgqrProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let k_lp64: *const i32 = k as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, k_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        SorgqrProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let k_ilp64: *const i64 = k as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, k_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn spotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_spotrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_spotrf_for_lp64();
    match provider {
        SpotrfProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, n_lp64, A, lda_lp64, info_lp64)
        }
        SpotrfProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, n_ilp64, A, lda_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ssyev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut f32,
    lda: *const lapackint,
    W: *mut f32,
    work: *mut f32,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_ssyev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_ssyev_for_lp64();
    match provider {
        SsyevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobz, uplo, n_lp64, A, lda_lp64, W, work, lwork_lp64, info_lp64)
        }
        SsyevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobz, uplo, n_ilp64, A, lda_ilp64, W, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn strtrs_(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const f32,
    lda: *const lapackint,
    B: *mut f32,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_strtrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_strtrs_for_lp64();
    match provider {
        StrtrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, trans, diag, n_lp64, nrhs_lp64, A, lda_lp64, B, ldb_lp64, info_lp64)
        }
        StrtrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, trans, diag, n_ilp64, nrhs_ilp64, A, lda_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgeev_(
    jobvl: *const c_char,
    jobvr: *const c_char,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    W: *mut Complex64,
    VL: *mut Complex64,
    ldvl: *const lapackint,
    VR: *mut Complex64,
    ldvr: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgeev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgeev_for_lp64();
    match provider {
        ZgeevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldvl_lp64: *const i32 = ldvl as *const i32;
            let ldvr_lp64: *const i32 = ldvr as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobvl, jobvr, n_lp64, A, lda_lp64, W, VL, ldvl_lp64, VR, ldvr_lp64, work, lwork_lp64, rwork, info_lp64)
        }
        ZgeevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldvl_ilp64: *const i64 = ldvl as *const i64;
            let ldvr_ilp64: *const i64 = ldvr as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobvl, jobvr, n_ilp64, A, lda_ilp64, W, VL, ldvl_ilp64, VR, ldvr_ilp64, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgeqrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    tau: *mut Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgeqrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgeqrf_for_lp64();
    match provider {
        ZgeqrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        ZgeqrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgesc2_(
    n: *const lapackint,
    A: *const Complex64,
    lda: *const lapackint,
    rhs: *mut Complex64,
    ipiv: *const lapackint,
    jpiv: *const lapackint,
    scale: *mut f64,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgesc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgesc2_for_lp64();
    match provider {
        Zgesc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let jpiv_lp64: *const i32 = jpiv as *const i32;
            fun(n_lp64, A, lda_lp64, rhs, ipiv_lp64, jpiv_lp64, scale)
        }
        Zgesc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let jpiv_ilp64: *const i64 = jpiv as *const i64;
            fun(n_ilp64, A, lda_ilp64, rhs, ipiv_ilp64, jpiv_ilp64, scale)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgesv_(
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    B: *mut Complex64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgesv_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgesv_for_lp64();
    match provider {
        ZgesvProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        ZgesvProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgesvd_(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    S: *mut f64,
    U: *mut Complex64,
    ldu: *const lapackint,
    VT: *mut Complex64,
    ldvt: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgesvd_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgesvd_for_lp64();
    match provider {
        ZgesvdProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldu_lp64: *const i32 = ldu as *const i32;
            let ldvt_lp64: *const i32 = ldvt as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobu, jobvt, m_lp64, n_lp64, A, lda_lp64, S, U, ldu_lp64, VT, ldvt_lp64, work, lwork_lp64, rwork, info_lp64)
        }
        ZgesvdProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldu_ilp64: *const i64 = ldu as *const i64;
            let ldvt_ilp64: *const i64 = ldvt as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobu, jobvt, m_ilp64, n_ilp64, A, lda_ilp64, S, U, ldu_ilp64, VT, ldvt_ilp64, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgetc2_(
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    jpiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgetc2_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgetc2_for_lp64();
    match provider {
        Zgetc2Provider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let jpiv_lp64: *mut i32 = jpiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, jpiv_lp64, info_lp64)
        }
        Zgetc2Provider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let jpiv_ilp64: *mut i64 = jpiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, jpiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgetrf_(
    m: *const lapackint,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgetrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgetrf_for_lp64();
    match provider {
        ZgetrfProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *mut i32 = ipiv as *mut i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, A, lda_lp64, ipiv_lp64, info_lp64)
        }
        ZgetrfProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *mut i64 = ipiv as *mut i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, A, lda_ilp64, ipiv_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgetri_(
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgetri_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgetri_for_lp64();
    match provider {
        ZgetriProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(n_lp64, A, lda_lp64, ipiv_lp64, work, lwork_lp64, info_lp64)
        }
        ZgetriProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(n_ilp64, A, lda_ilp64, ipiv_ilp64, work, lwork_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zgetrs_(
    trans: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const Complex64,
    lda: *const lapackint,
    ipiv: *const lapackint,
    B: *mut Complex64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zgetrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zgetrs_for_lp64();
    match provider {
        ZgetrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ipiv_lp64: *const i32 = ipiv as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(trans, n_lp64, nrhs_lp64, A, lda_lp64, ipiv_lp64, B, ldb_lp64, info_lp64)
        }
        ZgetrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ipiv_ilp64: *const i64 = ipiv as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(trans, n_ilp64, nrhs_ilp64, A, lda_ilp64, ipiv_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zheev_(
    jobz: *const c_char,
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    W: *mut f64,
    work: *mut Complex64,
    lwork: *const lapackint,
    rwork: *mut f64,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zheev_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zheev_for_lp64();
    match provider {
        ZheevProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(jobz, uplo, n_lp64, A, lda_lp64, W, work, lwork_lp64, rwork, info_lp64)
        }
        ZheevProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(jobz, uplo, n_ilp64, A, lda_ilp64, W, work, lwork_ilp64, rwork, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zpotrf_(
    uplo: *const c_char,
    n: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zpotrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zpotrf_for_lp64();
    match provider {
        ZpotrfProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, n_lp64, A, lda_lp64, info_lp64)
        }
        ZpotrfProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, n_ilp64, A, lda_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ztrtrs_(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const lapackint,
    nrhs: *const lapackint,
    A: *const Complex64,
    lda: *const lapackint,
    B: *mut Complex64,
    ldb: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_ztrtrs_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_ztrtrs_for_lp64();
    match provider {
        ZtrtrsProvider::Lp64(fun) => {
            let n_lp64: *const i32 = n as *const i32;
            let nrhs_lp64: *const i32 = nrhs as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let ldb_lp64: *const i32 = ldb as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(uplo, trans, diag, n_lp64, nrhs_lp64, A, lda_lp64, B, ldb_lp64, info_lp64)
        }
        ZtrtrsProvider::Ilp64(fun) => {
            let n_ilp64: *const i64 = n as *const i64;
            let nrhs_ilp64: *const i64 = nrhs as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let ldb_ilp64: *const i64 = ldb as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(uplo, trans, diag, n_ilp64, nrhs_ilp64, A, lda_ilp64, B, ldb_ilp64, info_ilp64)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn zungqr_(
    m: *const lapackint,
    n: *const lapackint,
    k: *const lapackint,
    A: *mut Complex64,
    lda: *const lapackint,
    tau: *const Complex64,
    work: *mut Complex64,
    lwork: *const lapackint,
    info: *mut lapackint,
) {
    #[cfg(feature = "ilp64")]
    let provider = get_zungqr_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_zungqr_for_lp64();
    match provider {
        ZungqrProvider::Lp64(fun) => {
            let m_lp64: *const i32 = m as *const i32;
            let n_lp64: *const i32 = n as *const i32;
            let k_lp64: *const i32 = k as *const i32;
            let lda_lp64: *const i32 = lda as *const i32;
            let lwork_lp64: *const i32 = lwork as *const i32;
            let info_lp64: *mut i32 = info as *mut i32;
            fun(m_lp64, n_lp64, k_lp64, A, lda_lp64, tau, work, lwork_lp64, info_lp64)
        }
        ZungqrProvider::Ilp64(fun) => {
            let m_ilp64: *const i64 = m as *const i64;
            let n_ilp64: *const i64 = n as *const i64;
            let k_ilp64: *const i64 = k as *const i64;
            let lda_ilp64: *const i64 = lda as *const i64;
            let lwork_ilp64: *const i64 = lwork as *const i64;
            let info_ilp64: *mut i64 = info as *mut i64;
            fun(m_ilp64, n_ilp64, k_ilp64, A, lda_ilp64, tau, work, lwork_ilp64, info_ilp64)
        }
    }
}
