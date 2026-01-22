//! Functional test using system LAPACK/OpenBLAS as the backend.
//!
//! This test tries to find and load a LAPACK library dynamically,
//! then verifies computation results through lapack-sys.

// Link lapack-inject to provide symbols
extern crate lapack_inject;

use lapack_inject::*;
use std::sync::OnceLock;

// Keep the library loaded for the entire test run to avoid use-after-free
static LAPACK_LIB: OnceLock<Option<libloading::Library>> = OnceLock::new();

fn get_lapack_library() -> Option<&'static libloading::Library> {
    LAPACK_LIB.get_or_init(|| {
        let candidates = [
            "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
            "/System/Library/Frameworks/Accelerate.framework/Accelerate",
            "libopenblas.so",
            "libopenblas.so.0",
            "liblapack.so",
            "liblapack.so.3",
            "libmkl_rt.so",
        ];

        for path in &candidates {
            if let Ok(lib) = unsafe { libloading::Library::new(path) } {
                eprintln!("Found LAPACK library: {}", path);
                return Some(lib);
            }
        }
        None
    }).as_ref()
}

#[test]
#[ignore] // Run with: cargo test --test functional_test -- --ignored
fn test_dgesv() {
    let lib = match get_lapack_library() {
        Some(l) => l,
        None => {
            eprintln!("No LAPACK library found on system. Skipping test.");
            return;
        }
    };

    // Get and register dgesv_
    let dgesv: libloading::Symbol<DgesvFnPtr> = match unsafe { lib.get(b"dgesv_") } {
        Ok(f) => f,
        Err(_) => {
            eprintln!("dgesv_ not found. Skipping.");
            return;
        }
    };
    unsafe { register_dgesv(*dgesv); }

    // Test: solve A * X = B
    // A = [[1, 2], [3, 4]], B = [[5], [11]]
    // Expected X = [[1], [2]]
    let n: i32 = 2;
    let nrhs: i32 = 1;
    let mut a = vec![1.0_f64, 3.0, 2.0, 4.0]; // column-major
    let lda: i32 = 2;
    let mut ipiv = vec![0_i32; 2];
    let mut b = vec![5.0_f64, 11.0];
    let ldb: i32 = 2;
    let mut info: i32 = 0;

    unsafe {
        lapack_sys::dgesv_(
            &n, &nrhs,
            a.as_mut_ptr(), &lda,
            ipiv.as_mut_ptr(),
            b.as_mut_ptr(), &ldb,
            &mut info,
        );
    }

    assert_eq!(info, 0, "DGESV should succeed");
    assert!((b[0] - 1.0).abs() < 1e-10, "x[0] should be 1.0, got {}", b[0]);
    assert!((b[1] - 2.0).abs() < 1e-10, "x[1] should be 2.0, got {}", b[1]);

    println!("DGESV test passed!");
}

#[test]
#[ignore]
fn test_dgetrf_dgetri() {
    let lib = match get_lapack_library() {
        Some(l) => l,
        None => {
            eprintln!("No LAPACK library found. Skipping test.");
            return;
        }
    };

    // Register functions
    let dgetrf: libloading::Symbol<DgetrfFnPtr> = match unsafe { lib.get(b"dgetrf_") } {
        Ok(f) => f,
        Err(_) => return,
    };
    let dgetri: libloading::Symbol<DgetriFnPtr> = match unsafe { lib.get(b"dgetri_") } {
        Ok(f) => f,
        Err(_) => return,
    };
    unsafe {
        register_dgetrf(*dgetrf);
        register_dgetri(*dgetri);
    }

    // Test: compute inverse of A = [[1, 2], [3, 4]]
    let n: i32 = 2;
    let mut a = vec![1.0_f64, 3.0, 2.0, 4.0];
    let lda: i32 = 2;
    let mut ipiv = vec![0_i32; 2];
    let mut info: i32 = 0;

    unsafe {
        lapack_sys::dgetrf_(&n, &n, a.as_mut_ptr(), &lda, ipiv.as_mut_ptr(), &mut info);
    }
    assert_eq!(info, 0);

    let lwork: i32 = 64;
    let mut work = vec![0.0_f64; lwork as usize];
    unsafe {
        lapack_sys::dgetri_(
            &n, a.as_mut_ptr(), &lda,
            ipiv.as_ptr(),
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }
    assert_eq!(info, 0);

    // A^-1 = [[-2, 1], [1.5, -0.5]]
    assert!((a[0] - (-2.0)).abs() < 1e-10);
    assert!((a[1] - 1.5).abs() < 1e-10);
    assert!((a[2] - 1.0).abs() < 1e-10);
    assert!((a[3] - (-0.5)).abs() < 1e-10);

    println!("DGETRF/DGETRI test passed!");
}

#[test]
#[ignore]
fn test_dpotrf() {
    let lib = match get_lapack_library() {
        Some(l) => l,
        None => {
            eprintln!("No LAPACK library found. Skipping test.");
            return;
        }
    };

    let dpotrf: libloading::Symbol<DpotrfFnPtr> = match unsafe { lib.get(b"dpotrf_") } {
        Ok(f) => f,
        Err(_) => {
            eprintln!("dpotrf_ not found. Skipping.");
            return;
        }
    };
    unsafe { register_dpotrf(*dpotrf); }

    // Test: Cholesky factorization of A = [[4, 2], [2, 5]]
    // L = [[2, 0], [1, 2]]
    let uplo: i8 = b'L' as i8;
    let n: i32 = 2;
    let mut a = vec![4.0_f64, 2.0, 2.0, 5.0]; // column-major
    let lda: i32 = 2;
    let mut info: i32 = 0;

    unsafe {
        lapack_sys::dpotrf_(
            &uplo, &n,
            a.as_mut_ptr(), &lda,
            &mut info,
        );
    }

    assert_eq!(info, 0, "DPOTRF should succeed");
    // Lower triangular: L[0,0]=2, L[1,0]=1, L[1,1]=2
    assert!((a[0] - 2.0).abs() < 1e-10, "L[0,0] should be 2.0");
    assert!((a[1] - 1.0).abs() < 1e-10, "L[1,0] should be 1.0");
    assert!((a[3] - 2.0).abs() < 1e-10, "L[1,1] should be 2.0");

    println!("DPOTRF test passed!");
}

#[test]
#[ignore]
fn test_dsyev() {
    let lib = match get_lapack_library() {
        Some(l) => l,
        None => {
            eprintln!("No LAPACK library found. Skipping test.");
            return;
        }
    };

    let dsyev: libloading::Symbol<DsyevFnPtr> = match unsafe { lib.get(b"dsyev_") } {
        Ok(f) => f,
        Err(_) => {
            eprintln!("dsyev_ not found. Skipping.");
            return;
        }
    };
    unsafe { register_dsyev(*dsyev); }

    // Test: eigenvalues of A = [[2, 1], [1, 2]]
    // Eigenvalues: 1, 3
    let jobz: i8 = b'N' as i8; // eigenvalues only
    let uplo: i8 = b'U' as i8;
    let n: i32 = 2;
    let mut a = vec![2.0_f64, 1.0, 1.0, 2.0]; // column-major
    let lda: i32 = 2;
    let mut w = vec![0.0_f64; 2]; // eigenvalues
    let lwork: i32 = 64;
    let mut work = vec![0.0_f64; lwork as usize];
    let mut info: i32 = 0;

    unsafe {
        lapack_sys::dsyev_(
            &jobz, &uplo, &n,
            a.as_mut_ptr(), &lda,
            w.as_mut_ptr(),
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }

    assert_eq!(info, 0, "DSYEV should succeed");
    // Eigenvalues in ascending order: 1.0, 3.0
    assert!((w[0] - 1.0).abs() < 1e-10, "First eigenvalue should be 1.0, got {}", w[0]);
    assert!((w[1] - 3.0).abs() < 1e-10, "Second eigenvalue should be 3.0, got {}", w[1]);

    println!("DSYEV test passed!");
}

#[test]
#[ignore]
fn test_dgesvd() {
    let lib = match get_lapack_library() {
        Some(l) => l,
        None => {
            eprintln!("No LAPACK library found. Skipping test.");
            return;
        }
    };

    let dgesvd: libloading::Symbol<DgesvdFnPtr> = match unsafe { lib.get(b"dgesvd_") } {
        Ok(f) => f,
        Err(_) => {
            eprintln!("dgesvd_ not found. Skipping.");
            return;
        }
    };
    unsafe { register_dgesvd(*dgesvd); }

    // Test: SVD of A = [[3, 0], [0, 4]]
    // Singular values: 4, 3
    let jobu: i8 = b'N' as i8;
    let jobvt: i8 = b'N' as i8;
    let m: i32 = 2;
    let n: i32 = 2;
    let mut a = vec![3.0_f64, 0.0, 0.0, 4.0]; // column-major
    let lda: i32 = 2;
    let mut s = vec![0.0_f64; 2]; // singular values
    let mut u = vec![0.0_f64; 1]; // not computed
    let ldu: i32 = 1;
    let mut vt = vec![0.0_f64; 1]; // not computed
    let ldvt: i32 = 1;
    let lwork: i32 = 64;
    let mut work = vec![0.0_f64; lwork as usize];
    let mut info: i32 = 0;

    unsafe {
        lapack_sys::dgesvd_(
            &jobu, &jobvt,
            &m, &n,
            a.as_mut_ptr(), &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(), &ldu,
            vt.as_mut_ptr(), &ldvt,
            work.as_mut_ptr(), &lwork,
            &mut info,
        );
    }

    assert_eq!(info, 0, "DGESVD should succeed");
    // Singular values in descending order: 4.0, 3.0
    assert!((s[0] - 4.0).abs() < 1e-10, "First singular value should be 4.0, got {}", s[0]);
    assert!((s[1] - 3.0).abs() < 1e-10, "Second singular value should be 3.0, got {}", s[1]);

    println!("DGESVD test passed!");
}
