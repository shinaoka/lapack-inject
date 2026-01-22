//! LAPACK type definitions.

#![allow(non_camel_case_types)]

/// Integer type for LAPACK operations (LP64: 32-bit).
#[cfg(not(feature = "ilp64"))]
pub type lapack_int = i32;

/// Integer type for LAPACK operations (ILP64: 64-bit).
#[cfg(feature = "ilp64")]
pub type lapack_int = i64;

/// Alias for backward compatibility
pub type lapackint = lapack_int;

// Complex type aliases for lapack-sys compatibility
pub type lapack_complex_float = num_complex::Complex32;
pub type lapack_complex_double = num_complex::Complex64;
