//! GESV - Solves a general system of linear equations AX = B.
//!
//! LAPACKE interface following OpenBLAS implementation exactly.

use crate::backend::{get_cgesv, get_dgesv, get_sgesv, get_zgesv};
use crate::types::{
    lapack_int, lapack_max, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR, LAPACK_TRANSPOSE_MEMORY_ERROR,
};
use crate::utils::{alloc_c32, alloc_c64, alloc_f32, alloc_f64, lapacke_cge_trans, lapacke_dge_trans, lapacke_sge_trans, lapacke_zge_trans};
use crate::xerbla::lapacke_xerbla_internal;
use num_complex::{Complex32, Complex64};

// =============================================================================
// LAPACKE high-level interface (allocates workspace automatically)
// =============================================================================

/// LAPACKE_sgesv - Solves A * X = B for single precision.
///
/// # Arguments
/// * `matrix_layout` - LAPACK_ROW_MAJOR (101) or LAPACK_COL_MAJOR (102)
/// * `n` - Number of linear equations
/// * `nrhs` - Number of right-hand sides
/// * `a` - Matrix A (n x n), overwritten with L and U
/// * `lda` - Leading dimension of A
/// * `ipiv` - Pivot indices (output)
/// * `b` - Matrix B (n x nrhs), overwritten with solution X
/// * `ldb` - Leading dimension of B
///
/// # Returns
/// * 0 on success
/// * < 0 if argument -i had an illegal value
/// * > 0 if U(i,i) is exactly zero
///
/// # Safety
/// Caller must ensure all pointers are valid and arrays have correct dimensions.
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgesv(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f32,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_sgesv", -1);
        return -1;
    }
    LAPACKE_sgesv_work(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
}

/// LAPACKE_dgesv - Solves A * X = B for double precision.
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f64,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_dgesv", -1);
        return -1;
    }
    LAPACKE_dgesv_work(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
}

/// LAPACKE_cgesv - Solves A * X = B for single precision complex.
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgesv(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut Complex32,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_cgesv", -1);
        return -1;
    }
    LAPACKE_cgesv_work(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
}

/// LAPACKE_zgesv - Solves A * X = B for double precision complex.
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgesv(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut Complex64,
    ldb: lapack_int,
) -> lapack_int {
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_zgesv", -1);
        return -1;
    }
    LAPACKE_zgesv_work(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
}

// =============================================================================
// LAPACKE work interface (middle-level, following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_sgesv_work - Solves A * X = B for single precision (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgesv_work(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f32,
    ldb: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        // Call LAPACK function and adjust info
        let sgesv = get_sgesv();
        sgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);
        let ldb_t = lapack_max(1, n);

        // Check leading dimension(s)
        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_sgesv_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -8;
            lapacke_xerbla_internal("LAPACKE_sgesv_work", info);
            return info;
        }

        // Allocate memory for temporary array(s)
        let mut a_t = match alloc_f32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };
        let mut b_t = match alloc_f32((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        // Transpose input matrices
        lapacke_sge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_sge_trans(matrix_layout, n, nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        // Call LAPACK function and adjust info
        let sgesv = get_sgesv();
        sgesv(
            &n,
            &nrhs,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            b_t.as_mut_ptr(),
            &ldb_t,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        // Transpose output matrices
        lapacke_sge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_sge_trans(LAPACK_COL_MAJOR, n, nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_sgesv_work", info);
    }

    info
}

/// LAPACKE_dgesv_work - Solves A * X = B for double precision (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv_work(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f64,
    ldb: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let dgesv = get_dgesv();
        dgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);
        let ldb_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_dgesv_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -8;
            lapacke_xerbla_internal("LAPACKE_dgesv_work", info);
            return info;
        }

        let mut a_t = match alloc_f64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };
        let mut b_t = match alloc_f64((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_dge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_dge_trans(matrix_layout, n, nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let dgesv = get_dgesv();
        dgesv(
            &n,
            &nrhs,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            b_t.as_mut_ptr(),
            &ldb_t,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_dge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_dge_trans(LAPACK_COL_MAJOR, n, nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_dgesv_work", info);
    }

    info
}

/// LAPACKE_cgesv_work - Solves A * X = B for single precision complex (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgesv_work(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut Complex32,
    ldb: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let cgesv = get_cgesv();
        cgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);
        let ldb_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_cgesv_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -8;
            lapacke_xerbla_internal("LAPACKE_cgesv_work", info);
            return info;
        }

        let mut a_t = match alloc_c32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };
        let mut b_t = match alloc_c32((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_cge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_cge_trans(matrix_layout, n, nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let cgesv = get_cgesv();
        cgesv(
            &n,
            &nrhs,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            b_t.as_mut_ptr(),
            &ldb_t,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_cge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_cge_trans(LAPACK_COL_MAJOR, n, nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_cgesv_work", info);
    }

    info
}

/// LAPACKE_zgesv_work - Solves A * X = B for double precision complex (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgesv_work(
    matrix_layout: i32,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut Complex64,
    ldb: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let zgesv = get_zgesv();
        zgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);
        let ldb_t = lapack_max(1, n);

        if lda < n {
            info = -5;
            lapacke_xerbla_internal("LAPACKE_zgesv_work", info);
            return info;
        }
        if ldb < nrhs {
            info = -8;
            lapacke_xerbla_internal("LAPACKE_zgesv_work", info);
            return info;
        }

        let mut a_t = match alloc_c64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };
        let mut b_t = match alloc_c64((ldb_t * lapack_max(1, nrhs)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgesv_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_zge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);
        lapacke_zge_trans(matrix_layout, n, nrhs, b, ldb, b_t.as_mut_ptr(), ldb_t);

        let zgesv = get_zgesv();
        zgesv(
            &n,
            &nrhs,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            b_t.as_mut_ptr(),
            &ldb_t,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_zge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
        lapacke_zge_trans(LAPACK_COL_MAJOR, n, nrhs, b_t.as_ptr(), ldb_t, b, ldb);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_zgesv_work", info);
    }

    info
}
