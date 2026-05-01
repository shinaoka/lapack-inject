//! Fortran LAPACK symbol exports with dual LP64/ILP64 dispatch.
//!
//! This module exports Fortran-style LAPACK symbols (e.g., `dgesv_`) that
//! dynamically dispatch to the registered LP64 or ILP64 provider.
//! This allows lapack-inject to be a drop-in replacement for lapack-src
//! supporting both integer widths simultaneously.
//!
//! Auto-generated from lapack-sys plus supplemental Netlib LAPACK routines.

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_char;

use num_complex::{Complex32, Complex64};

use crate::backend::*;
use crate::lapackint;

// =============================================================================
// Generated Fortran symbol exports with dual LP64/ILP64 dispatch
// =============================================================================

include!("fortran_gen.rs");
