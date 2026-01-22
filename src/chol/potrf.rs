//! POTRF - Cholesky factorization of a symmetric/Hermitian positive definite matrix.
//!
//! LAPACKE interface following OpenBLAS implementation exactly.

use crate::backend::{get_cpotrf, get_dpotrf, get_spotrf, get_zpotrf};
use crate::types::{lapack_int, lapack_max, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR, LAPACK_TRANSPOSE_MEMORY_ERROR};
use crate::utils::{
    alloc_c32, alloc_c64, alloc_f32, alloc_f64, lapacke_cpo_trans, lapacke_dpo_trans,
    lapacke_spo_trans, lapacke_zpo_trans,
};
use crate::xerbla::lapacke_xerbla_internal;
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;

// =============================================================================
// LAPACKE high-level interface
// =============================================================================

/// LAPACKE_spotrf - Computes Cholesky factorization (single precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_spotrf(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_spotrf", -1);
        return -1;
    }
    LAPACKE_spotrf_work(matrix_layout, uplo, n, a, lda)
}

/// LAPACKE_dpotrf - Computes Cholesky factorization (double precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dpotrf(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_dpotrf", -1);
        return -1;
    }
    LAPACKE_dpotrf_work(matrix_layout, uplo, n, a, lda)
}

/// LAPACKE_cpotrf - Computes Cholesky factorization (single precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cpotrf(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_cpotrf", -1);
        return -1;
    }
    LAPACKE_cpotrf_work(matrix_layout, uplo, n, a, lda)
}

/// LAPACKE_zpotrf - Computes Cholesky factorization (double precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zpotrf(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_zpotrf", -1);
        return -1;
    }
    LAPACKE_zpotrf_work(matrix_layout, uplo, n, a, lda)
}

// =============================================================================
// LAPACKE work interface (following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_spotrf_work - Computes Cholesky factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_spotrf_work(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let uplo_char = uplo as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let spotrf = get_spotrf();
        spotrf(&uplo_char, &n, a, &lda, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_spotrf_work", info);
            return info;
        }

        let mut a_t = match alloc_f32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_spotrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_spo_trans(matrix_layout, uplo, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let spotrf = get_spotrf();
        spotrf(&uplo_char, &n, a_t.as_mut_ptr(), &lda_t, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_spo_trans(LAPACK_COL_MAJOR, uplo, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_spotrf_work", info);
    }

    info
}

/// LAPACKE_dpotrf_work - Computes Cholesky factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dpotrf_work(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let uplo_char = uplo as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let dpotrf = get_dpotrf();
        dpotrf(&uplo_char, &n, a, &lda, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_dpotrf_work", info);
            return info;
        }

        let mut a_t = match alloc_f64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dpotrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_dpo_trans(matrix_layout, uplo, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let dpotrf = get_dpotrf();
        dpotrf(&uplo_char, &n, a_t.as_mut_ptr(), &lda_t, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_dpo_trans(LAPACK_COL_MAJOR, uplo, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_dpotrf_work", info);
    }

    info
}

/// LAPACKE_cpotrf_work - Computes Cholesky factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cpotrf_work(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let uplo_char = uplo as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let cpotrf = get_cpotrf();
        cpotrf(&uplo_char, &n, a, &lda, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_cpotrf_work", info);
            return info;
        }

        let mut a_t = match alloc_c32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cpotrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_cpo_trans(matrix_layout, uplo, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let cpotrf = get_cpotrf();
        cpotrf(&uplo_char, &n, a_t.as_mut_ptr(), &lda_t, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_cpo_trans(LAPACK_COL_MAJOR, uplo, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_cpotrf_work", info);
    }

    info
}

/// LAPACKE_zpotrf_work - Computes Cholesky factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zpotrf_work(
    matrix_layout: i32,
    uplo: u8,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;
    let uplo_char = uplo as c_char;

    if matrix_layout == LAPACK_COL_MAJOR {
        let zpotrf = get_zpotrf();
        zpotrf(&uplo_char, &n, a, &lda, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_zpotrf_work", info);
            return info;
        }

        let mut a_t = match alloc_c64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zpotrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_zpo_trans(matrix_layout, uplo, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let zpotrf = get_zpotrf();
        zpotrf(&uplo_char, &n, a_t.as_mut_ptr(), &lda_t, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_zpo_trans(LAPACK_COL_MAJOR, uplo, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_zpotrf_work", info);
    }

    info
}
