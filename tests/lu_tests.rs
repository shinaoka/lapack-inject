//! Tests for GETRF and GETRI (LU factorization)

extern crate blas_src;
extern crate lapack_sys;
extern crate openblas_src;

use lapack_inject::*;

fn register_dgetrf_from_lapack() {
    extern "C" {
        fn dgetrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_dgetrf(std::mem::transmute(dgetrf_ as *const ()));
    }
}

fn register_dgetri_from_lapack() {
    extern "C" {
        fn dgetri_(
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );
    }
    unsafe {
        register_dgetri(std::mem::transmute(dgetri_ as *const ()));
    }
}

#[test]
fn test_dgetrf() {
    register_dgetrf_from_lapack();

    // LU factorization of A = [[1, 2], [3, 4]]
    let m = 2i32;
    let n = 2i32;
    let mut a = vec![1.0, 3.0, 2.0, 4.0]; // column-major
    let lda = 2i32;
    let mut ipiv = vec![0i32; 2];
    let mut info = 0i32;

    unsafe {
        lapack_dgetrf(m, n, a.as_mut_ptr(), lda, ipiv.as_mut_ptr(), &mut info);
    }

    assert_eq!(info, 0, "DGETRF should succeed");
    // After LU factorization, the matrix contains L and U combined
}

#[test]
fn test_dgetri() {
    register_dgetrf_from_lapack();
    register_dgetri_from_lapack();

    // Compute inverse of A = [[1, 2], [3, 4]]
    // inv(A) = [[-2, 1], [1.5, -0.5]]
    let n = 2i32;
    let mut a = vec![1.0, 3.0, 2.0, 4.0]; // column-major
    let lda = 2i32;
    let mut ipiv = vec![0i32; 2];
    let mut info = 0i32;

    // First do LU factorization
    unsafe {
        lapack_dgetrf(n, n, a.as_mut_ptr(), lda, ipiv.as_mut_ptr(), &mut info);
    }
    assert_eq!(info, 0, "DGETRF should succeed");

    // Query optimal workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgetri(n, a.as_mut_ptr(), lda, ipiv.as_ptr(), work_query.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DGETRI workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Compute inverse
    unsafe {
        lapack_dgetri(n, a.as_mut_ptr(), lda, ipiv.as_ptr(), work.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DGETRI should succeed");

    // Check inverse values (column-major)
    // inv(A) = [[-2, 1], [1.5, -0.5]]
    assert!((a[0] - (-2.0)).abs() < 1e-10, "inv[0,0] should be -2.0");
    assert!((a[1] - 1.5).abs() < 1e-10, "inv[1,0] should be 1.5");
    assert!((a[2] - 1.0).abs() < 1e-10, "inv[0,1] should be 1.0");
    assert!((a[3] - (-0.5)).abs() < 1e-10, "inv[1,1] should be -0.5");
}
