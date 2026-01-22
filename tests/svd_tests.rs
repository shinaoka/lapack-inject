//! Tests for SVD (GESVD, GESDD)

extern crate blas_src;
extern crate lapack_sys;
extern crate openblas_src;

use lapack_inject::*;

fn register_dgesvd_from_lapack() {
    extern "C" {
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
    }
    unsafe {
        register_dgesvd(std::mem::transmute(dgesvd_ as *const ()));
    }
}

fn register_dgesdd_from_lapack() {
    extern "C" {
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
    }
    unsafe {
        register_dgesdd(std::mem::transmute(dgesdd_ as *const ()));
    }
}

#[test]
fn test_dgesvd() {
    register_dgesvd_from_lapack();

    // SVD of A = [[1, 2], [3, 4], [5, 6]]
    let m = 3i32;
    let n = 2i32;
    let min_mn = 2usize;
    let mut a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major
    let lda = 3i32;
    let mut s = vec![0.0f64; min_mn];
    let mut u = vec![0.0f64; (m * m) as usize];
    let ldu = m;
    let mut vt = vec![0.0f64; (n * n) as usize];
    let ldvt = n;
    let mut info = 0i32;

    let jobu = b'A' as i8;
    let jobvt = b'A' as i8;

    // Query workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgesvd(
            jobu, jobvt, m, n, a.as_mut_ptr(), lda,
            s.as_mut_ptr(), u.as_mut_ptr(), ldu, vt.as_mut_ptr(), ldvt,
            work_query.as_mut_ptr(), lwork, &mut info
        );
    }
    assert_eq!(info, 0, "DGESVD workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Perform SVD
    unsafe {
        lapack_dgesvd(
            jobu, jobvt, m, n, a.as_mut_ptr(), lda,
            s.as_mut_ptr(), u.as_mut_ptr(), ldu, vt.as_mut_ptr(), ldvt,
            work.as_mut_ptr(), lwork, &mut info
        );
    }
    assert_eq!(info, 0, "DGESVD should succeed");

    // Singular values should be positive and decreasing
    assert!(s[0] > 0.0, "First singular value should be positive");
    assert!(s[0] >= s[1], "Singular values should be in decreasing order");
}

#[test]
fn test_dgesdd() {
    register_dgesdd_from_lapack();

    // SVD of A = [[1, 2], [3, 4], [5, 6]] using divide and conquer
    let m = 3i32;
    let n = 2i32;
    let min_mn = 2usize;
    let mut a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major
    let lda = 3i32;
    let mut s = vec![0.0f64; min_mn];
    let mut u = vec![0.0f64; (m * m) as usize];
    let ldu = m;
    let mut vt = vec![0.0f64; (n * n) as usize];
    let ldvt = n;
    let mut iwork = vec![0i32; 8 * min_mn];
    let mut info = 0i32;

    let jobz = b'A' as i8;

    // Query workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgesdd(
            jobz, m, n, a.as_mut_ptr(), lda,
            s.as_mut_ptr(), u.as_mut_ptr(), ldu, vt.as_mut_ptr(), ldvt,
            work_query.as_mut_ptr(), lwork, iwork.as_mut_ptr(), &mut info
        );
    }
    assert_eq!(info, 0, "DGESDD workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Perform SVD
    unsafe {
        lapack_dgesdd(
            jobz, m, n, a.as_mut_ptr(), lda,
            s.as_mut_ptr(), u.as_mut_ptr(), ldu, vt.as_mut_ptr(), ldvt,
            work.as_mut_ptr(), lwork, iwork.as_mut_ptr(), &mut info
        );
    }
    assert_eq!(info, 0, "DGESDD should succeed");

    // Singular values should be positive and decreasing
    assert!(s[0] > 0.0, "First singular value should be positive");
    assert!(s[0] >= s[1], "Singular values should be in decreasing order");
}
