//! Minimal LAPACKE-compatible entry points for the generated Phase 1 surface.

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_char;

use crate::backend::*;
use crate::{lapack_int, LAPACK_COL_MAJOR, LAPACK_ROW_MAJOR};

#[inline]
fn valid_layout(matrix_layout: lapack_int) -> bool {
    matrix_layout == LAPACK_COL_MAJOR || matrix_layout == LAPACK_ROW_MAJOR
}

#[inline]
fn matrix_len(rows: lapack_int, cols: lapack_int) -> Option<usize> {
    if rows < 0 || cols < 0 {
        return None;
    }
    (rows as usize).checked_mul(cols as usize)
}

#[inline]
fn valid_layout_i64(matrix_layout: i64) -> bool {
    matrix_layout == LAPACK_COL_MAJOR as i64 || matrix_layout == LAPACK_ROW_MAJOR as i64
}

#[inline]
fn matrix_len_i64(rows: i64, cols: i64) -> Option<usize> {
    if rows < 0 || cols < 0 {
        return None;
    }
    (rows as usize).checked_mul(cols as usize)
}

unsafe fn row_to_col(
    rows: lapack_int,
    cols: lapack_int,
    src: *const f64,
    ld_src: lapack_int,
    dst: *mut f64,
    ld_dst: lapack_int,
) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add((i + j * ld_dst) as usize) = *src.add((i * ld_src + j) as usize);
        }
    }
}

unsafe fn col_to_row(
    rows: lapack_int,
    cols: lapack_int,
    src: *const f64,
    ld_src: lapack_int,
    dst: *mut f64,
    ld_dst: lapack_int,
) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add((i * ld_dst + j) as usize) = *src.add((i + j * ld_src) as usize);
        }
    }
}

unsafe fn row_to_col_i64(
    rows: i64,
    cols: i64,
    src: *const f64,
    ld_src: i64,
    dst: *mut f64,
    ld_dst: i64,
) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add((i + j * ld_dst) as usize) = *src.add((i * ld_src + j) as usize);
        }
    }
}

unsafe fn col_to_row_i64(
    rows: i64,
    cols: i64,
    src: *const f64,
    ld_src: i64,
    dst: *mut f64,
    ld_dst: i64,
) {
    for i in 0..rows {
        for j in 0..cols {
            *dst.add((i * ld_dst + j) as usize) = *src.add((i + j * ld_src) as usize);
        }
    }
}

unsafe fn dgesv_colmajor(
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f64,
    ldb: lapack_int,
) -> lapack_int {
    #[cfg(feature = "ilp64")]
    let provider = get_dgesv_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgesv_for_lp64();

    let piv_len = n.max(0) as usize;
    match provider {
        DgesvProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let nrhs_i32 = nrhs as i32;
            let lda_i32 = lda as i32;
            let ldb_i32 = ldb as i32;
            let mut ipiv_i32 = vec![0_i32; piv_len];
            let mut info_i32 = 0_i32;
            fun(
                &n_i32,
                &nrhs_i32,
                a,
                &lda_i32,
                ipiv_i32.as_mut_ptr(),
                b,
                &ldb_i32,
                &mut info_i32,
            );
            for (idx, value) in ipiv_i32.into_iter().enumerate() {
                *ipiv.add(idx) = value as lapack_int;
            }
            info_i32 as lapack_int
        }
        DgesvProvider::Ilp64(fun) => {
            let n_i64 = n as i64;
            let nrhs_i64 = nrhs as i64;
            let lda_i64 = lda as i64;
            let ldb_i64 = ldb as i64;
            let mut ipiv_i64 = vec![0_i64; piv_len];
            let mut info_i64 = 0_i64;
            fun(
                &n_i64,
                &nrhs_i64,
                a,
                &lda_i64,
                ipiv_i64.as_mut_ptr(),
                b,
                &ldb_i64,
                &mut info_i64,
            );
            for (idx, value) in ipiv_i64.into_iter().enumerate() {
                *ipiv.add(idx) = value as lapack_int;
            }
            info_i64 as lapack_int
        }
    }
}

unsafe fn dgetrf_colmajor(
    m: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetrf_for_lp64();

    let piv_len = m.min(n).max(0) as usize;
    match provider {
        DgetrfProvider::Lp64(fun) => {
            let m_i32 = m as i32;
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let mut ipiv_i32 = vec![0_i32; piv_len];
            let mut info_i32 = 0_i32;
            fun(
                &m_i32,
                &n_i32,
                a,
                &lda_i32,
                ipiv_i32.as_mut_ptr(),
                &mut info_i32,
            );
            for (idx, value) in ipiv_i32.into_iter().enumerate() {
                *ipiv.add(idx) = value as lapack_int;
            }
            info_i32 as lapack_int
        }
        DgetrfProvider::Ilp64(fun) => {
            let m_i64 = m as i64;
            let n_i64 = n as i64;
            let lda_i64 = lda as i64;
            let mut ipiv_i64 = vec![0_i64; piv_len];
            let mut info_i64 = 0_i64;
            fun(
                &m_i64,
                &n_i64,
                a,
                &lda_i64,
                ipiv_i64.as_mut_ptr(),
                &mut info_i64,
            );
            for (idx, value) in ipiv_i64.into_iter().enumerate() {
                *ipiv.add(idx) = value as lapack_int;
            }
            info_i64 as lapack_int
        }
    }
}

unsafe fn dgetri_colmajor(
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut f64,
    lwork: lapack_int,
) -> lapack_int {
    #[cfg(feature = "ilp64")]
    let provider = get_dgetri_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dgetri_for_lp64();

    let piv_len = n.max(0) as usize;
    match provider {
        DgetriProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let lwork_i32 = lwork as i32;
            let ipiv_i32: Vec<i32> = (0..piv_len).map(|idx| *ipiv.add(idx) as i32).collect();
            let mut info_i32 = 0_i32;
            fun(
                &n_i32,
                a,
                &lda_i32,
                ipiv_i32.as_ptr(),
                work,
                &lwork_i32,
                &mut info_i32,
            );
            info_i32 as lapack_int
        }
        DgetriProvider::Ilp64(fun) => {
            let n_i64 = n as i64;
            let lda_i64 = lda as i64;
            let lwork_i64 = lwork as i64;
            let ipiv_i64: Vec<i64> = (0..piv_len).map(|idx| *ipiv.add(idx) as i64).collect();
            let mut info_i64 = 0_i64;
            fun(
                &n_i64,
                a,
                &lda_i64,
                ipiv_i64.as_ptr(),
                work,
                &lwork_i64,
                &mut info_i64,
            );
            info_i64 as lapack_int
        }
    }
}

unsafe fn dpotrf_colmajor(uplo: c_char, n: lapack_int, a: *mut f64, lda: lapack_int) -> lapack_int {
    #[cfg(feature = "ilp64")]
    let provider = get_dpotrf_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_dpotrf_for_lp64();

    match provider {
        DpotrfProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let mut info_i32 = 0_i32;
            fun(&uplo, &n_i32, a, &lda_i32, &mut info_i32);
            info_i32 as lapack_int
        }
        DpotrfProvider::Ilp64(fun) => {
            let n_i64 = n as i64;
            let lda_i64 = lda as i64;
            let mut info_i64 = 0_i64;
            fun(&uplo, &n_i64, a, &lda_i64, &mut info_i64);
            info_i64 as lapack_int
        }
    }
}

unsafe fn dgesv_colmajor_64(
    n: i64,
    nrhs: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *mut i64,
    b: *mut f64,
    ldb: i64,
) -> i64 {
    let provider = get_dgesv_for_ilp64();
    let piv_len = n.max(0) as usize;
    match provider {
        DgesvProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let nrhs_i32 = nrhs as i32;
            let lda_i32 = lda as i32;
            let ldb_i32 = ldb as i32;
            let mut ipiv_i32 = vec![0_i32; piv_len];
            let mut info_i32 = 0_i32;
            fun(
                &n_i32,
                &nrhs_i32,
                a,
                &lda_i32,
                ipiv_i32.as_mut_ptr(),
                b,
                &ldb_i32,
                &mut info_i32,
            );
            for (idx, value) in ipiv_i32.into_iter().enumerate() {
                *ipiv.add(idx) = value as i64;
            }
            info_i32 as i64
        }
        DgesvProvider::Ilp64(fun) => {
            let mut info = 0_i64;
            fun(&n, &nrhs, a, &lda, ipiv, b, &ldb, &mut info);
            info
        }
    }
}

unsafe fn dgetrf_colmajor_64(m: i64, n: i64, a: *mut f64, lda: i64, ipiv: *mut i64) -> i64 {
    let provider = get_dgetrf_for_ilp64();
    let piv_len = m.min(n).max(0) as usize;
    match provider {
        DgetrfProvider::Lp64(fun) => {
            let m_i32 = m as i32;
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let mut ipiv_i32 = vec![0_i32; piv_len];
            let mut info_i32 = 0_i32;
            fun(
                &m_i32,
                &n_i32,
                a,
                &lda_i32,
                ipiv_i32.as_mut_ptr(),
                &mut info_i32,
            );
            for (idx, value) in ipiv_i32.into_iter().enumerate() {
                *ipiv.add(idx) = value as i64;
            }
            info_i32 as i64
        }
        DgetrfProvider::Ilp64(fun) => {
            let mut info = 0_i64;
            fun(&m, &n, a, &lda, ipiv, &mut info);
            info
        }
    }
}

unsafe fn dgetri_colmajor_64(
    n: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *const i64,
    work: *mut f64,
    lwork: i64,
) -> i64 {
    let provider = get_dgetri_for_ilp64();
    let piv_len = n.max(0) as usize;
    match provider {
        DgetriProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let lwork_i32 = lwork as i32;
            let ipiv_i32: Vec<i32> = (0..piv_len).map(|idx| *ipiv.add(idx) as i32).collect();
            let mut info_i32 = 0_i32;
            fun(
                &n_i32,
                a,
                &lda_i32,
                ipiv_i32.as_ptr(),
                work,
                &lwork_i32,
                &mut info_i32,
            );
            info_i32 as i64
        }
        DgetriProvider::Ilp64(fun) => {
            let mut info = 0_i64;
            fun(&n, a, &lda, ipiv, work, &lwork, &mut info);
            info
        }
    }
}

unsafe fn dpotrf_colmajor_64(uplo: c_char, n: i64, a: *mut f64, lda: i64) -> i64 {
    let provider = get_dpotrf_for_ilp64();
    match provider {
        DpotrfProvider::Lp64(fun) => {
            let n_i32 = n as i32;
            let lda_i32 = lda as i32;
            let mut info_i32 = 0_i32;
            fun(&uplo, &n_i32, a, &lda_i32, &mut info_i32);
            info_i32 as i64
        }
        DpotrfProvider::Ilp64(fun) => {
            let mut info = 0_i64;
            fun(&uplo, &n, a, &lda, &mut info);
            info
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv(
    matrix_layout: lapack_int,
    n: lapack_int,
    nrhs: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
    b: *mut f64,
    ldb: lapack_int,
) -> lapack_int {
    if !valid_layout(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR {
        return dgesv_colmajor(n, nrhs, a, lda, ipiv, b, ldb);
    }

    let Some(a_len) = matrix_len(n, n) else {
        return -2;
    };
    let Some(b_len) = matrix_len(n, nrhs) else {
        return -3;
    };
    let mut a_col = vec![0.0; a_len];
    let mut b_col = vec![0.0; b_len];
    row_to_col(n, n, a, lda, a_col.as_mut_ptr(), n);
    row_to_col(n, nrhs, b, ldb, b_col.as_mut_ptr(), n);

    let info = dgesv_colmajor(n, nrhs, a_col.as_mut_ptr(), n, ipiv, b_col.as_mut_ptr(), n);
    col_to_row(n, n, a_col.as_ptr(), n, a, lda);
    col_to_row(n, nrhs, b_col.as_ptr(), n, b, ldb);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetrf(
    matrix_layout: lapack_int,
    m: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *mut lapack_int,
) -> lapack_int {
    if !valid_layout(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR {
        return dgetrf_colmajor(m, n, a, lda, ipiv);
    }

    let Some(a_len) = matrix_len(m, n) else {
        return -2;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col(m, n, a, lda, a_col.as_mut_ptr(), m);

    let info = dgetrf_colmajor(m, n, a_col.as_mut_ptr(), m, ipiv);
    col_to_row(m, n, a_col.as_ptr(), m, a, lda);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri(
    matrix_layout: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *const lapack_int,
) -> lapack_int {
    let lwork = n.max(1) * 64;
    let mut work = vec![0.0; lwork as usize];
    LAPACKE_dgetri_work(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork)
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri_work(
    matrix_layout: lapack_int,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
    ipiv: *const lapack_int,
    work: *mut f64,
    lwork: lapack_int,
) -> lapack_int {
    if !valid_layout(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR {
        return dgetri_colmajor(n, a, lda, ipiv, work, lwork);
    }

    let Some(a_len) = matrix_len(n, n) else {
        return -2;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col(n, n, a, lda, a_col.as_mut_ptr(), n);

    let info = dgetri_colmajor(n, a_col.as_mut_ptr(), n, ipiv, work, lwork);
    col_to_row(n, n, a_col.as_ptr(), n, a, lda);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dpotrf(
    matrix_layout: lapack_int,
    uplo: c_char,
    n: lapack_int,
    a: *mut f64,
    lda: lapack_int,
) -> lapack_int {
    if !valid_layout(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR {
        return dpotrf_colmajor(uplo, n, a, lda);
    }

    let Some(a_len) = matrix_len(n, n) else {
        return -3;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col(n, n, a, lda, a_col.as_mut_ptr(), n);

    let info = dpotrf_colmajor(uplo, n, a_col.as_mut_ptr(), n);
    col_to_row(n, n, a_col.as_ptr(), n, a, lda);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv_64(
    matrix_layout: i64,
    n: i64,
    nrhs: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *mut i64,
    b: *mut f64,
    ldb: i64,
) -> i64 {
    if !valid_layout_i64(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR as i64 {
        return dgesv_colmajor_64(n, nrhs, a, lda, ipiv, b, ldb);
    }

    let Some(a_len) = matrix_len_i64(n, n) else {
        return -2;
    };
    let Some(b_len) = matrix_len_i64(n, nrhs) else {
        return -3;
    };
    let mut a_col = vec![0.0; a_len];
    let mut b_col = vec![0.0; b_len];
    row_to_col_i64(n, n, a, lda, a_col.as_mut_ptr(), n);
    row_to_col_i64(n, nrhs, b, ldb, b_col.as_mut_ptr(), n);

    let info = dgesv_colmajor_64(n, nrhs, a_col.as_mut_ptr(), n, ipiv, b_col.as_mut_ptr(), n);
    col_to_row_i64(n, n, a_col.as_ptr(), n, a, lda);
    col_to_row_i64(n, nrhs, b_col.as_ptr(), n, b, ldb);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetrf_64(
    matrix_layout: i64,
    m: i64,
    n: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *mut i64,
) -> i64 {
    if !valid_layout_i64(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR as i64 {
        return dgetrf_colmajor_64(m, n, a, lda, ipiv);
    }

    let Some(a_len) = matrix_len_i64(m, n) else {
        return -2;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col_i64(m, n, a, lda, a_col.as_mut_ptr(), m);

    let info = dgetrf_colmajor_64(m, n, a_col.as_mut_ptr(), m, ipiv);
    col_to_row_i64(m, n, a_col.as_ptr(), m, a, lda);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri_64(
    matrix_layout: i64,
    n: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *const i64,
) -> i64 {
    let lwork = n.max(1) * 64;
    let mut work = vec![0.0; lwork as usize];
    LAPACKE_dgetri_work_64(matrix_layout, n, a, lda, ipiv, work.as_mut_ptr(), lwork)
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgetri_work_64(
    matrix_layout: i64,
    n: i64,
    a: *mut f64,
    lda: i64,
    ipiv: *const i64,
    work: *mut f64,
    lwork: i64,
) -> i64 {
    if !valid_layout_i64(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR as i64 {
        return dgetri_colmajor_64(n, a, lda, ipiv, work, lwork);
    }

    let Some(a_len) = matrix_len_i64(n, n) else {
        return -2;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col_i64(n, n, a, lda, a_col.as_mut_ptr(), n);

    let info = dgetri_colmajor_64(n, a_col.as_mut_ptr(), n, ipiv, work, lwork);
    col_to_row_i64(n, n, a_col.as_ptr(), n, a, lda);
    info
}

#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dpotrf_64(
    matrix_layout: i64,
    uplo: c_char,
    n: i64,
    a: *mut f64,
    lda: i64,
) -> i64 {
    if !valid_layout_i64(matrix_layout) {
        return -1;
    }
    if matrix_layout == LAPACK_COL_MAJOR as i64 {
        return dpotrf_colmajor_64(uplo, n, a, lda);
    }

    let Some(a_len) = matrix_len_i64(n, n) else {
        return -3;
    };
    let mut a_col = vec![0.0; a_len];
    row_to_col_i64(n, n, a, lda, a_col.as_mut_ptr(), n);

    let info = dpotrf_colmajor_64(uplo, n, a_col.as_mut_ptr(), n);
    col_to_row_i64(n, n, a_col.as_ptr(), n, a, lda);
    info
}
