//! LP64/ILP64 dual backend for LAPACK function pointer registration.
//!
//! This module provides dual LP64/ILP64 fn pointer types and registration
//! functions for each LAPACK function. Each function stores both an LP64 and
//! an ILP64 provider via `OnceLock`, allowing runtime dispatch to whichever
//! integer width the caller registered.
//!
//! Auto-generated from lapack-sys plus supplemental Netlib LAPACK routines.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_char;
use std::sync::OnceLock;

use num_complex::{Complex32, Complex64};

// =============================================================================
// Select function types for eigenvalue routines
// =============================================================================

/// Select function type for real GEES (single)
pub type SSelectFn2 = unsafe extern "C" fn(ar: *const f32, ai: *const f32) -> i32;
/// Select function type for real GEES (double)
pub type DSelectFn2 = unsafe extern "C" fn(ar: *const f64, ai: *const f64) -> i32;
/// Select function type for real (single) with 3 args
pub type SSelectFn3 = unsafe extern "C" fn(ar: *const f32, ai: *const f32, b: *const f32) -> i32;
/// Select function type for real (double) with 3 args
pub type DSelectFn3 = unsafe extern "C" fn(ar: *const f64, ai: *const f64, b: *const f64) -> i32;
/// Select function type for complex GEES (single)
pub type CSelectFn1 = unsafe extern "C" fn(w: *const Complex32) -> i32;
/// Select function type for complex GEES (double)
pub type ZSelectFn1 = unsafe extern "C" fn(w: *const Complex64) -> i32;
/// Select function type for complex (single) with 2 args
pub type CSelectFn2 = unsafe extern "C" fn(a: *const Complex32, b: *const Complex32) -> i32;
/// Select function type for complex (double) with 2 args
pub type ZSelectFn2 = unsafe extern "C" fn(a: *const Complex64, b: *const Complex64) -> i32;

// =============================================================================
// Macro for defining dual LP64/ILP64 function types, storage, and registration
// =============================================================================

/// Macro to define dual LP64/ILP64 LAPACK function pointer types,
/// OnceLock storage, registration functions, and getter functions.
macro_rules! define_dual_backend {
    ($name:ident, $lp64_type:ty, $ilp64_type:ty) => {
        paste::paste! {
            static [<$name:upper _LP64>]: OnceLock<$lp64_type> = OnceLock::new();
            static [<$name:upper _ILP64>]: OnceLock<$ilp64_type> = OnceLock::new();

            /// Register the LP64 (i32) Fortran function pointer.
            /// Returns 0 on success, 2 if already registered.
            #[no_mangle]
            pub unsafe extern "C" fn [<register_ $name _lp64>](f: $lp64_type) -> i32 {
                match [<$name:upper _LP64>].set(f) {
                    Ok(()) => 0,
                    Err(_) => 2,
                }
            }

            /// Register the ILP64 (i64) Fortran function pointer.
            /// Returns 0 on success, 2 if already registered.
            #[no_mangle]
            pub unsafe extern "C" fn [<register_ $name _ilp64>](f: $ilp64_type) -> i32 {
                match [<$name:upper _ILP64>].set(f) {
                    Ok(()) => 0,
                    Err(_) => 2,
                }
            }

            /// Enum representing either an LP64 or ILP64 provider for this function.
            #[allow(dead_code)]
            pub enum [<$name:camel Provider>] {
                Lp64($lp64_type),
                Ilp64($ilp64_type),
            }

            /// Get the provider preferring LP64 (falls back to ILP64).
            #[allow(dead_code)]
            pub(crate) fn [<get_ $name _for_lp64>]() -> [<$name:camel Provider>] {
                if let Some(f) = [<$name:upper _LP64>].get() {
                    return [<$name:camel Provider>]::Lp64(*f);
                }
                if let Some(f) = [<$name:upper _ILP64>].get() {
                    return [<$name:camel Provider>]::Ilp64(*f);
                }
                panic!(concat!(
                    "lapack function `", stringify!($name),
                    "` is not registered (call register_", stringify!($name),
                    "_lp64 or register_", stringify!($name), "_ilp64 first)"
                ));
            }

            /// Get the provider preferring ILP64 (falls back to LP64).
            #[allow(dead_code)]
            pub(crate) fn [<get_ $name _for_ilp64>]() -> [<$name:camel Provider>] {
                if let Some(f) = [<$name:upper _ILP64>].get() {
                    return [<$name:camel Provider>]::Ilp64(*f);
                }
                if let Some(f) = [<$name:upper _LP64>].get() {
                    return [<$name:camel Provider>]::Lp64(*f);
                }
                panic!(concat!(
                    "lapack function `", stringify!($name),
                    "` is not registered (call register_", stringify!($name),
                    "_lp64 or register_", stringify!($name), "_ilp64 first)"
                ));
            }
        }
    };
}

// =============================================================================
// Generated function types and registrations
// =============================================================================

include!("backend_gen.rs");
