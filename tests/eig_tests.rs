//! Tests for eigenvalue decomposition (GEEV, SYEV)

extern crate blas_src;
extern crate lapack_sys;
extern crate openblas_src;

use lapack_inject::*;

fn register_dgeev_from_lapack() {
    extern "C" {
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
    }
    unsafe {
        register_dgeev(std::mem::transmute(dgeev_ as *const ()));
    }
}

fn register_dsyev_from_lapack() {
    extern "C" {
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
    }
    unsafe {
        register_dsyev(std::mem::transmute(dsyev_ as *const ()));
    }
}

#[test]
fn test_dgeev() {
    register_dgeev_from_lapack();

    // Eigenvalues of A = [[1, 2], [3, 4]]
    // Eigenvalues: (5 + sqrt(33))/2 ≈ 5.372, (5 - sqrt(33))/2 ≈ -0.372
    let n = 2i32;
    let mut a = vec![1.0, 3.0, 2.0, 4.0]; // column-major
    let lda = 2i32;
    let mut wr = vec![0.0f64; 2];
    let mut wi = vec![0.0f64; 2];
    let mut vl = vec![0.0f64; 4];
    let ldvl = 2i32;
    let mut vr = vec![0.0f64; 4];
    let ldvr = 2i32;
    let mut info = 0i32;

    let jobvl = b'N' as i8;
    let jobvr = b'V' as i8;

    // Query workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dgeev(
            jobvl, jobvr, n, a.as_mut_ptr(), lda,
            wr.as_mut_ptr(), wi.as_mut_ptr(),
            vl.as_mut_ptr(), ldvl, vr.as_mut_ptr(), ldvr,
            work_query.as_mut_ptr(), lwork, &mut info
        );
    }
    assert_eq!(info, 0, "DGEEV workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Compute eigenvalues
    unsafe {
        lapack_dgeev(
            jobvl, jobvr, n, a.as_mut_ptr(), lda,
            wr.as_mut_ptr(), wi.as_mut_ptr(),
            vl.as_mut_ptr(), ldvl, vr.as_mut_ptr(), ldvr,
            work.as_mut_ptr(), lwork, &mut info
        );
    }
    assert_eq!(info, 0, "DGEEV should succeed");

    // Check eigenvalues (imaginary parts should be zero for this matrix)
    assert!(wi[0].abs() < 1e-10, "Imaginary part should be zero");
    assert!(wi[1].abs() < 1e-10, "Imaginary part should be zero");

    // Sort eigenvalues for comparison
    let mut eigs: Vec<f64> = wr.clone();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending

    let sqrt33 = 33.0f64.sqrt();
    let expected1 = (5.0 + sqrt33) / 2.0;
    let expected2 = (5.0 - sqrt33) / 2.0;

    assert!((eigs[0] - expected1).abs() < 1e-10, "First eigenvalue should be ~5.372");
    assert!((eigs[1] - expected2).abs() < 1e-10, "Second eigenvalue should be ~-0.372");
}

#[test]
fn test_dsyev() {
    register_dsyev_from_lapack();

    // Eigenvalues of symmetric A = [[2, 1], [1, 2]]
    // Eigenvalues: 3, 1
    let n = 2i32;
    let mut a = vec![2.0, 1.0, 1.0, 2.0]; // column-major, symmetric
    let lda = 2i32;
    let mut w = vec![0.0f64; 2];
    let mut info = 0i32;

    let jobz = b'V' as i8;
    let uplo = b'U' as i8;

    // Query workspace size
    let lwork = -1i32;
    let mut work_query = vec![0.0f64; 1];
    unsafe {
        lapack_dsyev(jobz, uplo, n, a.as_mut_ptr(), lda, w.as_mut_ptr(), work_query.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DSYEV workspace query should succeed");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Compute eigenvalues
    unsafe {
        lapack_dsyev(jobz, uplo, n, a.as_mut_ptr(), lda, w.as_mut_ptr(), work.as_mut_ptr(), lwork, &mut info);
    }
    assert_eq!(info, 0, "DSYEV should succeed");

    // Eigenvalues are returned in ascending order
    assert!((w[0] - 1.0).abs() < 1e-10, "First eigenvalue should be 1.0");
    assert!((w[1] - 3.0).abs() < 1e-10, "Second eigenvalue should be 3.0");
}
