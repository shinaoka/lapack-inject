//! GETRF - LU factorization with partial pivoting.
//!
//! LAPACKE interface following OpenBLAS implementation exactly.

use crate::backend::{get_cgetrf, get_dgetrf, get_sgetrf, get_zgetrf};
use crate::types::{
    lapack_int, lapack_max, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR, LAPACK_TRANSPOSE_MEMORY_ERROR,
};
use crate::utils::{
    alloc_c32, alloc_c64, alloc_f32, alloc_f64, lapacke_cge_trans, lapacke_dge_trans,
    lapacke_sge_trans, lapacke_zge_trans,
};
use crate::xerbla::lapacke_xerbla_internal;
use num_complex::{Complex32, Complex64};

// =============================================================================
// LAPACKE high-level interface
// =============================================================================

/// LAPACKE_sgetrf - Computes LU factorization of a general M-by-N matrix (single precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgetrf(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_sgetrf", -1);
        return -1;
    }
    LAPACKE_sgetrf_work(matrix_layout, m, n, a, lda, ipiv)
}

/// LAPACKE_dgetrf - Computes LU factorization of a general M-by-N matrix (double precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetrf(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_dgetrf", -1);
        return -1;
    }
    LAPACKE_dgetrf_work(matrix_layout, m, n, a, lda, ipiv)
}

/// LAPACKE_cgetrf - Computes LU factorization of a general M-by-N matrix (single precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgetrf(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_cgetrf", -1);
        return -1;
    }
    LAPACKE_cgetrf_work(matrix_layout, m, n, a, lda, ipiv)
}

/// LAPACKE_zgetrf - Computes LU factorization of a general M-by-N matrix (double precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgetrf(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_zgetrf", -1);
        return -1;
    }
    LAPACKE_zgetrf_work(matrix_layout, m, n, a, lda, ipiv)
}

// =============================================================================
// LAPACKE work interface (following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_sgetrf_work - Computes LU factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgetrf_work(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let sgetrf = get_sgetrf();
        sgetrf(&m, &n, a, &lda, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_sgetrf_work", info);
            return info;
        }

        let mut a_t = match alloc_f32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgetrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_sge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let sgetrf = get_sgetrf();
        sgetrf(&m, &n, a_t.as_mut_ptr(), &lda_t, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_sge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_sgetrf_work", info);
    }

    info
}

/// LAPACKE_dgetrf_work - Computes LU factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetrf_work(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let dgetrf = get_dgetrf();
        dgetrf(&m, &n, a, &lda, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_dgetrf_work", info);
            return info;
        }

        let mut a_t = match alloc_f64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgetrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_dge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let dgetrf = get_dgetrf();
        dgetrf(&m, &n, a_t.as_mut_ptr(), &lda_t, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_dge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_dgetrf_work", info);
    }

    info
}

/// LAPACKE_cgetrf_work - Computes LU factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgetrf_work(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let cgetrf = get_cgetrf();
        cgetrf(&m, &n, a, &lda, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_cgetrf_work", info);
            return info;
        }

        let mut a_t = match alloc_c32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgetrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_cge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let cgetrf = get_cgetrf();
        cgetrf(&m, &n, a_t.as_mut_ptr(), &lda_t, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_cge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_cgetrf_work", info);
    }

    info
}

/// LAPACKE_zgetrf_work - Computes LU factorization (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgetrf_work(
    matrix_layout: i32,
    m: lapack_int,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let zgetrf = get_zgetrf();
        zgetrf(&m, &n, a, &lda, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, m);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_zgetrf_work", info);
            return info;
        }

        let mut a_t = match alloc_c64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgetrf_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_zge_trans(matrix_layout, m, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let zgetrf = get_zgetrf();
        zgetrf(&m, &n, a_t.as_mut_ptr(), &lda_t, ipiv, &mut info);
        if info < 0 {
            info -= 1;
        }

        lapacke_zge_trans(LAPACK_COL_MAJOR, m, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_zgetrf_work", info);
    }

    info
}
