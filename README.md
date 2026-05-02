# lapack-inject

LAPACK compatible interface backed by runtime-registered Fortran LAPACK function pointers.

## Overview

This crate allows you to use LAPACK functions while the actual computation is performed by Fortran LAPACK functions provided at runtime. This is useful for integrating with Python (scipy) or Julia (libblastrampoline) where Fortran LAPACK pointers are available.

## Usage

```rust
use lapack_inject::register_dgesv_lp64;

// Register Fortran dgesv pointer (e.g., from scipy or Julia)
unsafe {
    let status = register_dgesv_lp64(dgesv_ptr);
    assert_eq!(status, 0);
}

// Now lapack_inject exports dgesv_ symbol that can be used by other crates
```

The LAPACKE-style C entry points are also exported for the current Phase 2
surface:

- `LAPACKE_dgesv` / `LAPACKE_dgesv_64`
- `LAPACKE_dgetrf` / `LAPACKE_dgetrf_64`
- `LAPACKE_dgetri` / `LAPACKE_dgetri_64`
- `LAPACKE_dpotrf` / `LAPACKE_dpotrf_64`

Both row-major (`LAPACK_ROW_MAJOR`) and column-major (`LAPACK_COL_MAJOR`) layouts
are supported for those wrappers.

## lapack-src/lapack-sys Compatibility

This crate exports a generated subset of Fortran-style LAPACK symbols such as
`dgesv_`, `dgetrf_`, and `dgesc2_`. Register function pointers at runtime, and
this crate provides those symbols to downstream crates that expect a LAPACK
provider.

Each generated routine has explicit LP64 and ILP64 registration functions:

- `register_dgesv_lp64(f)` for providers using 32-bit LAPACK integers
- `register_dgesv_ilp64(f)` for providers using 64-bit LAPACK integers

Registration returns `0` on success and `2` if that provider was already
registered.

## Supported Functions

The generated Fortran surface supports:

- `xGESV`, `xGETRF`, `xGETRS`, `xGETRI`, `xPOTRF`
- `xGESVD`
- `xGEQRF`, real `xORGQR`, complex `xUNGQR`
- `xTRTRS`
- `s/dSYEV`, `c/zHEEV`
- `xGEEV`
- Supplemental complete-pivoting LU routines `xGETC2` and `xGESC2`

The LAPACKE surface currently covers the double-precision routines listed
above in the Usage section.

## Features

- `ilp64`: Export Fortran symbols with 64-bit `lapackint` parameters. This is
  still needed for Fortran ABI compatibility because `dgesv_` has the same
  symbol name for LP64 and ILP64 callers. The LAPACKE C API exposes `_64`
  functions separately, so those C entry points do not require this feature to
  select 64-bit integer arguments.

The default build keeps the consumer-facing Fortran symbols LP64-compatible.
Register ILP64 host providers with the `_ilp64` registration functions; do not
enable the `ilp64` feature just because the host provider is ILP64. The feature
changes the consumer ABI too.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
