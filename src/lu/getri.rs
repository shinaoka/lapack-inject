//! GETRI - Computes the inverse of a matrix using LU factorization.
//!
//! LAPACKE interface following OpenBLAS implementation exactly.

use crate::backend::{get_cgetri, get_dgetri, get_sgetri, get_zgetri};
use crate::types::{
    lapack_int, lapack_max, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR, LAPACK_TRANSPOSE_MEMORY_ERROR,
    LAPACK_WORK_MEMORY_ERROR,
};
use crate::utils::{
    alloc_c32, alloc_c64, alloc_f32, alloc_f64, lapacke_cge_trans, lapacke_dge_trans,
    lapacke_sge_trans, lapacke_zge_trans,
};
use crate::xerbla::lapacke_xerbla_internal;
use num_complex::{Complex32, Complex64};

// =============================================================================
// LAPACKE high-level interface (allocates workspace automatically)
// =============================================================================

/// LAPACKE_sgetri - Computes the inverse of a matrix (single precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgetri(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *const lapack_int,
) -> lapack_int {
    let mut info: lapack_int;
    let mut lwork: lapack_int = -1;
    let mut work_query: f32 = 0.0;

    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_sgetri", -1);
        return -1;
    }

    // Query optimal working array size
    info = LAPACKE_sgetri_work(matrix_layout, n, a, lda, ipiv, &mut work_query, lwork);
    if info != 0 {
        if info == LAPACK_WORK_MEMORY_ERROR {
            lapacke_xerbla_internal("LAPACKE_sgetri", info);
        }
        return info;
    }

    lwork = work_query as lapack_int;
    let mut work = match alloc_f32(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_sgetri", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    info = LAPACKE_sgetri_work(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork);
    info
}

/// LAPACKE_dgetri - Computes the inverse of a matrix (double precision).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *const lapack_int,
) -> lapack_int {
    let mut info: lapack_int;
    let mut lwork: lapack_int = -1;
    let mut work_query: f64 = 0.0;

    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_dgetri", -1);
        return -1;
    }

    info = LAPACKE_dgetri_work(matrix_layout, n, a, lda, ipiv, &mut work_query, lwork);
    if info != 0 {
        if info == LAPACK_WORK_MEMORY_ERROR {
            lapacke_xerbla_internal("LAPACKE_dgetri", info);
        }
        return info;
    }

    lwork = work_query as lapack_int;
    let mut work = match alloc_f64(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_dgetri", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    info = LAPACKE_dgetri_work(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork);
    info
}

/// LAPACKE_cgetri - Computes the inverse of a matrix (single precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgetri(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *const lapack_int,
) -> lapack_int {
    let mut info: lapack_int;
    let mut lwork: lapack_int = -1;
    let mut work_query = Complex32::new(0.0, 0.0);

    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_cgetri", -1);
        return -1;
    }

    info = LAPACKE_cgetri_work(matrix_layout, n, a, lda, ipiv, &mut work_query, lwork);
    if info != 0 {
        if info == LAPACK_WORK_MEMORY_ERROR {
            lapacke_xerbla_internal("LAPACKE_cgetri", info);
        }
        return info;
    }

    lwork = work_query.re as lapack_int;
    let mut work = match alloc_c32(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_cgetri", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    info = LAPACKE_cgetri_work(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork);
    info
}

/// LAPACKE_zgetri - Computes the inverse of a matrix (double precision complex).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgetri(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *const lapack_int,
) -> lapack_int {
    let mut info: lapack_int;
    let mut lwork: lapack_int = -1;
    let mut work_query = Complex64::new(0.0, 0.0);

    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        lapacke_xerbla_internal("LAPACKE_zgetri", -1);
        return -1;
    }

    info = LAPACKE_zgetri_work(matrix_layout, n, a, lda, ipiv, &mut work_query, lwork);
    if info != 0 {
        if info == LAPACK_WORK_MEMORY_ERROR {
            lapacke_xerbla_internal("LAPACKE_zgetri", info);
        }
        return info;
    }

    lwork = work_query.re as lapack_int;
    let mut work = match alloc_c64(lwork as usize) {
        Some(v) => v,
        None => {
            lapacke_xerbla_internal("LAPACKE_zgetri", LAPACK_WORK_MEMORY_ERROR);
            return LAPACK_WORK_MEMORY_ERROR;
        }
    };

    info = LAPACKE_zgetri_work(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork);
    info
}

// =============================================================================
// LAPACKE work interface (following OpenBLAS exactly)
// =============================================================================

/// LAPACKE_sgetri_work - Computes the inverse of a matrix (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_sgetri_work(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut f32,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut f32,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let sgetri = get_sgetri();
        sgetri(&n, a, &lda, ipiv, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -4;
            lapacke_xerbla_internal("LAPACKE_sgetri_work", info);
            return info;
        }

        // Query optimal working array size if requested
        if lwork == -1 {
            let sgetri = get_sgetri();
            sgetri(&n, a, &lda_t, ipiv, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_f32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_sgetri_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_sge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let sgetri = get_sgetri();
        sgetri(
            &n,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            work,
            &lwork,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_sge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_sgetri_work", info);
    }

    info
}

/// LAPACKE_dgetri_work - Computes the inverse of a matrix (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri_work(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut f64,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let dgetri = get_dgetri();
        dgetri(&n, a, &lda, ipiv, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -4;
            lapacke_xerbla_internal("LAPACKE_dgetri_work", info);
            return info;
        }

        if lwork == -1 {
            let dgetri = get_dgetri();
            dgetri(&n, a, &lda_t, ipiv, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_f64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_dgetri_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_dge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let dgetri = get_dgetri();
        dgetri(
            &n,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            work,
            &lwork,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_dge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_dgetri_work", info);
    }

    info
}

/// LAPACKE_cgetri_work - Computes the inverse of a matrix (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_cgetri_work(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut Complex32,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut Complex32,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let cgetri = get_cgetri();
        cgetri(&n, a, &lda, ipiv, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -4;
            lapacke_xerbla_internal("LAPACKE_cgetri_work", info);
            return info;
        }

        if lwork == -1 {
            let cgetri = get_cgetri();
            cgetri(&n, a, &lda_t, ipiv, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_c32((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_cgetri_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_cge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let cgetri = get_cgetri();
        cgetri(
            &n,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            work,
            &lwork,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_cge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_cgetri_work", info);
    }

    info
}

/// LAPACKE_zgetri_work - Computes the inverse of a matrix (work interface).
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_zgetri_work(
    matrix_layout: i32,
    n: lapack_int,
    a: *mut Complex64,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut Complex64,
    lwork: lapack_int,
) -> lapack_int {
    let mut info: lapack_int = 0;

    if matrix_layout == LAPACK_COL_MAJOR {
        let zgetri = get_zgetri();
        zgetri(&n, a, &lda, ipiv, work, &lwork, &mut info);
        if info < 0 {
            info -= 1;
        }
    } else if matrix_layout == LAPACK_ROW_MAJOR {
        let lda_t = lapack_max(1, n);

        if lda < n {
            info = -4;
            lapacke_xerbla_internal("LAPACKE_zgetri_work", info);
            return info;
        }

        if lwork == -1 {
            let zgetri = get_zgetri();
            zgetri(&n, a, &lda_t, ipiv, work, &lwork, &mut info);
            return if info < 0 { info - 1 } else { info };
        }

        let mut a_t = match alloc_c64((lda_t * lapack_max(1, n)) as usize) {
            Some(v) => v,
            None => {
                lapacke_xerbla_internal("LAPACKE_zgetri_work", LAPACK_TRANSPOSE_MEMORY_ERROR);
                return LAPACK_TRANSPOSE_MEMORY_ERROR;
            }
        };

        lapacke_zge_trans(matrix_layout, n, n, a, lda, a_t.as_mut_ptr(), lda_t);

        let zgetri = get_zgetri();
        zgetri(
            &n,
            a_t.as_mut_ptr(),
            &lda_t,
            ipiv,
            work,
            &lwork,
            &mut info,
        );
        if info < 0 {
            info -= 1;
        }

        lapacke_zge_trans(LAPACK_COL_MAJOR, n, n, a_t.as_ptr(), lda_t, a, lda);
    } else {
        info = -1;
        lapacke_xerbla_internal("LAPACKE_zgetri_work", info);
    }

    info
}
