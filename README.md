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

The current generated Phase 1 surface supports:

- `xGESV`, `xGETRF`, `xGETRS`, `xGETRI`, `xPOTRF`, and `xGESVD`
- `sSYEV` and `dSYEV`
- Supplemental complete-pivoting LU routines `xGETC2` and `xGESC2`

Full LAPACKE C wrappers and row-major support are planned as a later phase.

## Features

- `ilp64`: Export Fortran symbols with 64-bit `lapackint` parameters.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
