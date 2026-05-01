//! Tests for supplemental Netlib LAPACK complete-pivoting LU symbols.
//!
//! lapack-sys does not declare xGETC2/xGESC2, so these tests declare the
//! Fortran symbols directly and verify lapack-inject's registration dispatch.

extern crate lapack_inject;

use lapack_inject::*;
use num_complex::{Complex32, Complex64};
use std::sync::atomic::{AtomicI32, Ordering};

#[cfg(not(feature = "ilp64"))]
type Sgetc2CurrentFnPtr = Sgetc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Sgetc2CurrentFnPtr = Sgetc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Dgetc2CurrentFnPtr = Dgetc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Dgetc2CurrentFnPtr = Dgetc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Cgetc2CurrentFnPtr = Cgetc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Cgetc2CurrentFnPtr = Cgetc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Zgetc2CurrentFnPtr = Zgetc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Zgetc2CurrentFnPtr = Zgetc2Ilp64FnPtr;

#[cfg(not(feature = "ilp64"))]
type Sgesc2CurrentFnPtr = Sgesc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Sgesc2CurrentFnPtr = Sgesc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Dgesc2CurrentFnPtr = Dgesc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Dgesc2CurrentFnPtr = Dgesc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Cgesc2CurrentFnPtr = Cgesc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Cgesc2CurrentFnPtr = Cgesc2Ilp64FnPtr;
#[cfg(not(feature = "ilp64"))]
type Zgesc2CurrentFnPtr = Zgesc2Lp64FnPtr;
#[cfg(feature = "ilp64")]
type Zgesc2CurrentFnPtr = Zgesc2Ilp64FnPtr;

unsafe fn register_dgetc2_current(f: Dgetc2CurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dgetc2_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dgetc2_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

unsafe fn register_dgesc2_current(f: Dgesc2CurrentFnPtr) {
    #[cfg(not(feature = "ilp64"))]
    let status = register_dgesc2_lp64(f);
    #[cfg(feature = "ilp64")]
    let status = register_dgesc2_ilp64(f);
    assert!(matches!(status, 0 | 2));
}

extern "C" {
    fn sgetc2_(
        n: *const lapackint,
        a: *mut f32,
        lda: *const lapackint,
        ipiv: *mut lapackint,
        jpiv: *mut lapackint,
        info: *mut lapackint,
    );
    fn dgetc2_(
        n: *const lapackint,
        a: *mut f64,
        lda: *const lapackint,
        ipiv: *mut lapackint,
        jpiv: *mut lapackint,
        info: *mut lapackint,
    );
    fn cgetc2_(
        n: *const lapackint,
        a: *mut Complex32,
        lda: *const lapackint,
        ipiv: *mut lapackint,
        jpiv: *mut lapackint,
        info: *mut lapackint,
    );
    fn zgetc2_(
        n: *const lapackint,
        a: *mut Complex64,
        lda: *const lapackint,
        ipiv: *mut lapackint,
        jpiv: *mut lapackint,
        info: *mut lapackint,
    );
    fn sgesc2_(
        n: *const lapackint,
        a: *const f32,
        lda: *const lapackint,
        rhs: *mut f32,
        ipiv: *const lapackint,
        jpiv: *const lapackint,
        scale: *mut f32,
    );
    fn dgesc2_(
        n: *const lapackint,
        a: *const f64,
        lda: *const lapackint,
        rhs: *mut f64,
        ipiv: *const lapackint,
        jpiv: *const lapackint,
        scale: *mut f64,
    );
    fn cgesc2_(
        n: *const lapackint,
        a: *const Complex32,
        lda: *const lapackint,
        rhs: *mut Complex32,
        ipiv: *const lapackint,
        jpiv: *const lapackint,
        scale: *mut f32,
    );
    fn zgesc2_(
        n: *const lapackint,
        a: *const Complex64,
        lda: *const lapackint,
        rhs: *mut Complex64,
        ipiv: *const lapackint,
        jpiv: *const lapackint,
        scale: *mut f64,
    );
}

static DGETC2_STATUS: AtomicI32 = AtomicI32::new(0);
static DGESC2_STATUS: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" fn fake_dgetc2(
    n: *const lapackint,
    a: *mut f64,
    lda: *const lapackint,
    ipiv: *mut lapackint,
    jpiv: *mut lapackint,
    info: *mut lapackint,
) {
    if *n != 1 {
        DGETC2_STATUS.store(1, Ordering::SeqCst);
    }
    if *lda != 1 {
        DGETC2_STATUS.store(2, Ordering::SeqCst);
    }
    if (*a - 4.0).abs() > f64::EPSILON {
        DGETC2_STATUS.store(3, Ordering::SeqCst);
    }

    *ipiv = 1;
    *jpiv = 1;
    *info = 0;
}

unsafe extern "C" fn fake_dgesc2(
    n: *const lapackint,
    a: *const f64,
    lda: *const lapackint,
    rhs: *mut f64,
    ipiv: *const lapackint,
    jpiv: *const lapackint,
    scale: *mut f64,
) {
    if *n != 1 {
        DGESC2_STATUS.store(1, Ordering::SeqCst);
    }
    if *lda != 1 {
        DGESC2_STATUS.store(2, Ordering::SeqCst);
    }
    if (*ipiv, *jpiv) != (1, 1) {
        DGESC2_STATUS.store(3, Ordering::SeqCst);
    }

    *rhs /= *a;
    *scale = 1.0;
}

#[test]
fn getc2_and_gesc2_symbols_have_expected_function_pointer_types() {
    let _: Sgetc2CurrentFnPtr = sgetc2_;
    let _: Dgetc2CurrentFnPtr = dgetc2_;
    let _: Cgetc2CurrentFnPtr = cgetc2_;
    let _: Zgetc2CurrentFnPtr = zgetc2_;

    let _: Sgesc2CurrentFnPtr = sgesc2_;
    let _: Dgesc2CurrentFnPtr = dgesc2_;
    let _: Cgesc2CurrentFnPtr = cgesc2_;
    let _: Zgesc2CurrentFnPtr = zgesc2_;
}

#[test]
fn dgetc2_and_dgesc2_dispatch_through_registered_function_pointers() {
    unsafe {
        register_dgetc2_current(fake_dgetc2);
        register_dgesc2_current(fake_dgesc2);
    }

    let n: lapackint = 1;
    let lda: lapackint = 1;
    let mut a = [4.0_f64];
    let mut rhs = [8.0_f64];
    let mut ipiv: [lapackint; 1] = [0];
    let mut jpiv: [lapackint; 1] = [0];
    let mut info: lapackint = -1;
    let mut scale = 0.0_f64;

    unsafe {
        dgetc2_(
            &n,
            a.as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            jpiv.as_mut_ptr(),
            &mut info,
        );
        dgesc2_(
            &n,
            a.as_ptr(),
            &lda,
            rhs.as_mut_ptr(),
            ipiv.as_ptr(),
            jpiv.as_ptr(),
            &mut scale,
        );
    }

    assert_eq!(DGETC2_STATUS.load(Ordering::SeqCst), 0);
    assert_eq!(DGESC2_STATUS.load(Ordering::SeqCst), 0);
    assert_eq!(info, 0);
    assert_eq!(ipiv, [1]);
    assert_eq!(jpiv, [1]);
    assert_eq!(rhs, [2.0]);
    assert_eq!(scale, 1.0);
}
