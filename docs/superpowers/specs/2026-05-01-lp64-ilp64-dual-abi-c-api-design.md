# LP64/ILP64 Dual ABI C API Design for lapack-inject

## Status

Design spec for adding dual LP64/ILP64 LAPACKE C API to lapack-inject,
following the same pattern as cblas-inject.

## Context

lapack-inject provides LAPACK-compatible symbols backed by runtime-registered
Fortran LAPACK function pointers. Currently, the integer width (LP64=32-bit,
ILP64=64-bit) is determined at compile time via the `ilp64` feature flag.

Julia may provide either LP64 or ILP64 Fortran LAPACK providers at runtime.
The library must support both simultaneously: register whichever provider is
available, and dispatch LP64/ILP64 C API calls to whichever provider is
registered.

## Scope

- **Backend**: Dual LP64/ILP64 function pointer registration for all ~1323
  LAPACK functions
- **Fortran symbols**: Updated to dispatch dynamically between LP64/ILP64
  providers
- **LAPACKE C API**: Full LAPACKE-compatible C wrapper layer with row-major
  support, for all LAPACK functions
- **Tests**: Rust unit tests + C integration tests
- **README**: Updated with new API documentation

## Architecture

```
                    C API Users (C, Julia, Python)
                              │
                      ┌───────┴───────┐
                      │  LAPACKE C API │  lapacke.rs
                      │ (LAPACKE_dgesv)│
                      │ (_64 variants) │
                      └───────┬───────┘
                              │
                      ┌───────┴───────┐
                      │     backend   │  backend.rs
                      │ LP64/ILP64    │
                      │ dual provider │
                      └───────┬───────┘
                              │
                      ┌───────┴───────┐
                      │  fortran.rs   │  Fortran symbols
                      │ (dgesv_, etc) │
                      └───────┬───────┘
                              │
                   Fortran LAPACK providers
                   (runtime-registered)
```

### Key Design Decisions

1. **Backward compatibility**: Not required. Old `register_*` (panicking) is
   replaced by `register_*_lp64` / `register_*_ilp64` (returning status code).
2. **No feature gates**: All functions enabled unconditionally (measured first,
   optional features added later if needed).
3. **Python generator used at development time only**: Generated code is
   committed to repo. No build-time Python dependency.
4. **Row-major fully supported**: Reference LAPACKE-compatible transposition
   with temporary buffer allocation.

## Layer 1: Backend (`src/backend.rs`)

### File pointer types

For each LAPACK function, two fn pointer types are generated (by Python
script), differing only in integer width:

```rust
pub type DgesvLp64FnPtr = unsafe extern "C" fn(
    n: *const i32, nrhs: *const i32, a: *mut f64,
    lda: *const i32, ipiv: *mut i32, b: *mut f64,
    ldb: *const i32, info: *mut i32,
);
pub type DgesvIlp64FnPtr = unsafe extern "C" fn(
    n: *const i64, nrhs: *const i64, a: *mut f64,
    lda: *const i64, ipiv: *mut i64, b: *mut f64,
    ldb: *const i64, info: *mut i64,
);
```

### `define_dual_backend!` macro

A single macro invocation generates all storage/registration boilerplate:

```rust
define_dual_backend!(dgesv, DgesvLp64FnPtr, DgesvIlp64FnPtr);
```

Expands to:

```rust
static DGESV_LP64: OnceLock<DgesvLp64FnPtr> = OnceLock::new();
static DGESV_ILP64: OnceLock<DgesvIlp64FnPtr> = OnceLock::new();

pub unsafe extern "C" fn register_dgesv_lp64(f: DgesvLp64FnPtr) -> i32 {
    if f.is_null() { return 1; }
    match DGESV_LP64.set(f) { Ok(_) => 0, Err(_) => 2 }
}
pub unsafe extern "C" fn register_dgesv_ilp64(f: DgesvIlp64FnPtr) -> i32 {
    if f.is_null() { return 1; }
    match DGESV_ILP64.set(f) { Ok(_) => 0, Err(_) => 2 }
}

pub enum DgesvProvider {
    Lp64(DgesvLp64FnPtr),
    Ilp64(DgesvIlp64FnPtr),
}

pub(crate) fn get_dgesv_for_lp64() -> DgesvProvider {
    if let Some(f) = DGESV_LP64.get() {
        return DgesvProvider::Lp64(*f);
    }
    if let Some(f) = DGESV_ILP64.get() {
        return DgesvProvider::Ilp64(*f);
    }
    panic!("dgesv not registered");
}

pub(crate) fn get_dgesv_for_ilp64() -> DgesvProvider {
    if let Some(f) = DGESV_ILP64.get() {
        return DgesvProvider::Ilp64(*f);
    }
    if let Some(f) = DGESV_LP64.get() {
        return DgesvProvider::Lp64(*f);
    }
    panic!("dgesv not registered");
}

pub unsafe extern "C" fn lapack_inject_supports_lp64() -> i32 {
    if DGESV_LP64.get().is_some() { 1 } else { 0 }
}
pub unsafe extern "C" fn lapack_inject_supports_ilp64() -> i32 {
    if DGESV_ILP64.get().is_some() { 1 } else { 0 }
}
```

### Select function types

The select function types (for eigenvalue problem routines like GEES) are
unchanged:

```rust
pub type SSelectFn2 = unsafe extern "C" fn(ar: *const f32, ai: *const f32) -> lapackint;
pub type DSelectFn2 = unsafe extern "C" fn(ar: *const f64, ai: *const f64) -> lapackint;
// etc.
```

## Layer 2: Fortran Symbol Exports (`src/fortran.rs`)

Each Fortran symbol dispatches dynamically to the available provider:

```rust
#[no_mangle]
pub unsafe extern "C" fn dgesv_(
    n: *const lapackint, nrhs: *const lapackint,
    a: *mut f64, lda: *const lapackint, ipiv: *mut lapackint,
    b: *mut f64, ldb: *const lapackint, info: *mut lapackint,
) {
    let provider = get_dgesv_for_lp64();  // prefer LP64
    match provider {
        DgesvProvider::Lp64(f) => f(n, nrhs, a, lda, ipiv, b, ldb, info),
        DgesvProvider::Ilp64(f) => {
            // Cast i32 pointers to i64 (same ABI on little-endian)
            // User compiled with matching integer width
            let n_i64 = n as *const i64;
            let nrhs_i64 = nrhs as *const i64;
            let lda_i64 = lda as *const i64;
            let ldb_i64 = ldb as *const i64;
            let ipiv_i64 = ipiv as *mut i64;
            let info_i64 = info as *mut i64;
            f(n_i64, nrhs_i64, a, lda_i64, ipiv_i64, b, ldb_i64, info_i64);
        }
    }
}
```

The `lapackint` type remains compile-time determined (i32 default, i64 with
`ilp64` feature). The dynamic dispatch casts when the provider width differs
from the compile-time width.

## Layer 3: LAPACKE C API (`src/lapacke.rs`)

### Layout constants (in `types.rs`)

```rust
pub const LAPACK_ROW_MAJOR: lapack_int = 101;
pub const LAPACK_COL_MAJOR: lapack_int = 102;
```

### Function triples

For each LAPACK function, three entry points:

#### Simple variant
```rust
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv(
    matrix_layout: lapack_int, n: lapack_int, nrhs: lapack_int,
    a: *mut f64, lda: lapack_int, ipiv: *mut lapack_int,
    b: *mut f64, ldb: lapack_int,
) -> lapack_int {
    // Validate layout
    if matrix_layout != LAPACK_COL_MAJOR && matrix_layout != LAPACK_ROW_MAJOR {
        return -1;
    }
    LAPACKE_dgesv_work(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb)
}
```

#### Work variant (actual dispatch)
```rust
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv_work(
    matrix_layout: lapack_int, n: lapack_int, nrhs: lapack_int,
    a: *mut f64, lda: lapack_int, ipiv: *mut lapack_int,
    b: *mut f64, ldb: lapack_int,
) -> lapack_int {
    if matrix_layout == LAPACK_COL_MAJOR {
        dispatch_col_major!(dgesv, n, nrhs, a, lda, ipiv, b, ldb)
    } else {
        dispatch_row_major_ge!(dgesv, [n, n], [n, nrhs], n, nrhs, a, lda, ipiv, b, ldb)
    }
}
```

#### 64 variant (explicit int64_t)
```rust
#[no_mangle]
pub unsafe extern "C" fn LAPACKE_dgesv_64(
    matrix_layout: i64, n: i64, nrhs: i64,
    a: *mut f64, lda: i64, ipiv: *mut i64,
    b: *mut f64, ldb: i64,
) -> i64 {
    // Similar dispatch but using ILP64 getter
    ...
}
```

### Dispatch macros

```rust
/// Column-major: dispatch to Fortran with i32→i64 widening if needed
macro_rules! dispatch_col_major {
    ($name:ident, $($param:ident),*) => {
        paste::paste! {
            let provider = [<get_ $name _for_lp64>]();
            match provider {
                [<$name:camel Provider>]::Lp64(f) => {
                    let mut info: lapack_int = 0;
                    f($(&$param),*, &mut info);
                    info
                }
                [<$name:camel Provider>]::Ilp64(f) => {
                    let mut info: i64 = 0;
                    f($((&($param as i64))),*, &mut info);
                    info as lapack_int
                }
            }
        }
    };
}

/// Row-major for general (ge) matrices: transpose input, dispatch, transpose back
macro_rules! dispatch_row_major_ge {
    ($name:ident, [$a_dims:expr], [$b_dims:expr],
     $m:ident, $n:ident, $a:ident, $lda:ident, $ipiv:ident, $b:ident, $ldb:ident) => {
        // Allocate temp buffers, transpose, dispatch, transpose back
        ...
    };
}
```

### Matrix type categories

Each LAPACK function falls into a matrix type category that determines
row-major handling:

| Category | Transposition | Examples |
|----------|---------------|----------|
| `ge` (general) | `LAPACKE_dge_trans` | GESV, GETRF, GELS, GEEV |
| `sy` (symmetric) | `LAPACKE_dsy_trans` | SYEV, SYTRF |
| `he` (Hermitian) | `LAPACKE_zhe_trans` | HEEV, HETRF |
| `tr` (triangular) | `LAPACKE_dtr_trans` | TRTRS, TRTRI |
| `gb` (general band) | special handling | GBSV, GBTRF |
| `sb` (symmetric band) | special handling | SBSV |
| `hb` (Hermitian band) | special handling | HBSV |
| `tb` (triangular band) | special handling | TBTRS |
| `po` (positive definite) | `LAPACKE_dge_trans` | POTRF, POSV |
| `pp` (packed) | special handling | PPTRF, PPSV |
| `sp` (symmetric packed) | special handling | SPTRF |
| `hp` (Hermitian packed) | special handling | HPTRF |
| `op` (orthogonal) | n/a | ORGQR, ORMQR |
| `un` (unitary) | n/a | UNGQR, UNMQR |

### Transposition helpers

```rust
pub(crate) unsafe fn lapacke_dge_trans(
    layout: lapack_int, m: lapack_int, n: lapack_int,
    src: *const f64, ld_src: lapack_int,
    dst: *mut f64, ld_dst: lapack_int,
) {
    for i in 0..m {
        for j in 0..n {
            *dst.offset(j * ld_dst as isize + i as isize) =
                *src.offset(i * ld_src as isize + j as isize);
        }
    }
}
// Similar for f32, Complex32, Complex64
```

### Generator macro for LAPACKE

The Python generator produces `impl_lapacke_sdcz!` invocations that expand to
all 4 precision variants:

```rust
// Macro for generating all 4 precision variants
macro_rules! define_lapacke_quad {
    ($name:ident,
     d: [$($dparam:ident: $dty:ty),*],
     s: [$($sparam:ident: $sty:ty),*],
     z: [$($zparam:ident: $zty:ty),*],
     c: [$($cparam:ident: $cty:ty),*],
     matrix_type: ge,  // category tag
    ) => {
        paste::paste! {
            define_lapacke_func!([<LAPACKE_d $name>], [<LAPACKE_d $name _work>], [<LAPACKE_d $name _64>],
                matrix_type: ge, $($dparam: $dty),*);
            define_lapacke_func!([<LAPACKE_s $name>], [<LAPACKE_s $name _work>], [<LAPACKE_s $name _64>],
                matrix_type: ge, $($sparam: $sty),*);
            // ... same for z, c
        }
    };
}
```

## Generator Script Update

The existing `scripts/generate_lapack_bindings.py` is updated to:

1. Read `lapack.rs` from the cargo registry (configurable path)
2. For each Fortran function, output:
   - LP64 fn pointer type (`*const i32` / `*mut i32`)
   - ILP64 fn pointer type (`*const i64` / `*mut i64`)
   - `define_dual_backend!(name, Lp64FnPtr, Ilp64FnPtr);`
3. Output updated Fortran symbol exports with dual dispatch
4. Output LAPACKE wrapper invocations (`define_lapacke_quad!` calls)

## Implementation Order

1. Create `define_dual_backend!` macro infrastructure
2. Update generator script path and output format
3. Regenerate `backend.rs` with dual types
4. Regenerate `fortran.rs` with dual dispatch
5. Add `LAPACK_ROW_MAJOR` / `LAPACK_COL_MAJOR` to `types.rs`
6. Create `lapacke.rs` with dispatch macros and transposition helpers
7. Add `define_lapacke_quad!` and `define_lapacke_func!` macros
8. Generate LAPACKE wrappers for all functions
9. Build and fix compilation errors
10. Write tests
11. Update `README.md` and `lib.rs`

## File Changes

| File | Action |
|------|--------|
| `src/types.rs` | Add `LAPACK_ROW_MAJOR`, `LAPACK_COL_MAJOR` constants |
| `src/backend.rs` | Rewrite with dual types + `define_dual_backend!` |
| `src/fortran.rs` | Rewrite with dual dispatch |
| `src/lapacke.rs` | **New**: LAPACKE wrapper layer |
| `src/lib.rs` | Add `pub mod lapacke` |
| `scripts/generate_lapack_bindings.py` | Update for new output format |
| `tests/*.rs` | Update/add tests |
| `ctest/*` | Add LAPACKE C tests |
| `README.md` | Document new API |

## Testing

### Rust Unit Tests (`tests/`)

| Test | Description |
|------|-------------|
| `dual_registration` | Register LP64 mock, verify dispatch; register ILP64, verify dispatch |
| `dual_fallback` | Register ILP64 only, verify LP64 C API falls back correctly |
| `fortran_symbols` | Test Fortran symbols dispatch through dual backend |
| `lapacke_simple` | Test LAPACKE_dgesv, LAPACKE_dgetrf, LAPACKE_dpotrf with mock |

### C Integration Tests (`ctest/`)

| Test | Description |
|------|-------------|
| `test_LAPACKE_DGESV` | LAPACKE_dgesv col-major + row-major against OpenBLAS Fortran |
| `test_LAPACKE_DGETRF` | LAPACKE_dgetrf col-major + row-major |
| `test_LAPACKE_DPOTRF` | LAPACKE_dpotrf col-major + row-major |

### Test strategy

1. Load OpenBLAS Fortran LAPACK symbols via `libloading`
2. Register via `register_*_lp64()`
3. Call LAPACKE functions and verify results against reference

## README Updates

Key additions to `README.md`:

- New `lapacke` module documentation
- LP64/ILP64 dual registration API
- C API usage examples (both C and Rust)
- Registration status codes
- Row-major support notes
- Updated feature flags documentation
