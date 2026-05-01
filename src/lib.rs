//! # lapack-inject
//!
//! LAPACK compatible interface backed by Fortran LAPACK function pointers.
//!
//! This crate allows you to use LAPACK functions while the actual computation
//! is performed by Fortran LAPACK functions provided at runtime.
//! This is useful for integrating with Python (scipy) or Julia (libblastrampoline)
//! where Fortran LAPACK pointers are available.
//!
//! ## Usage
//!
//! ```ignore
//! use lapack_inject::register_dgesv_lp64;
//!
//! // Register Fortran dgesv pointer (e.g., from scipy or Julia)
//! unsafe {
//!     let status = register_dgesv_lp64(dgesv_ptr);
//!     assert_eq!(status, 0);
//! }
//!
//! // Now lapack_inject exports dgesv_ symbol that can be used by other crates
//! ```
//!
//! ## lapack-src/lapack-sys Compatibility
//!
//! This crate exports a generated subset of Fortran-style LAPACK symbols such
//! as `dgesv_`, `dgetrf_`, and `dgesc2_`. Register LP64 or ILP64 function
//! pointers at runtime, and this crate provides those symbols to downstream
//! crates that expect a LAPACK provider.
//!
//! LAPACKE-style C entry points are exported for `dgesv`, `dgetrf`, `dgetri`,
//! and `dpotrf`, including `_64` variants and row-major layout handling.
//!
//! ## Supported Functions
//!
//! The generated Fortran surface supports:
//!
//! - `xGESV`, `xGETRF`, `xGETRS`, `xGETRI`, `xPOTRF`
//! - `xGESVD`
//! - `xGEQRF`, real `xORGQR`, complex `xUNGQR`
//! - `xTRTRS`
//! - `s/dSYEV`, `c/zHEEV`
//! - `xGEEV`
//! - supplemental complete-pivoting LU routines `xGETC2` and `xGESC2`
//!
//! The default build keeps the consumer-facing Fortran symbols LP64-compatible.
//! Register ILP64 host providers with the `_ilp64` registration functions; do not
//! enable the `ilp64` feature just because the host provider is ILP64. The feature
//! changes the consumer ABI too.

mod backend;
pub mod fortran;
pub mod lapacke;
mod types;

pub use backend::*;
pub use fortran::*;
pub use lapacke::*;
pub use types::*;
