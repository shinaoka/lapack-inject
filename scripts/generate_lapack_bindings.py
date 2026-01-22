#!/usr/bin/env python3
"""
Generate lapack-inject bindings from lapack-sys.
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Param:
    name: str
    type_: str

@dataclass
class Function:
    name: str  # e.g., "dgesv_"
    params: List[Param]

def parse_lapack_rs(path: Path) -> List[Function]:
    """Parse lapack.rs and extract function signatures."""
    content = path.read_text()

    # Match extern "C" blocks with function definitions
    # Pattern to match: pub fn name_(params) -> return_type; or pub fn name_(params);
    pattern = r'pub fn (\w+_)\s*\(([\s\S]*?)\)(?:\s*->\s*\w+)?;'

    functions = []
    for match in re.finditer(pattern, content):
        name = match.group(1)
        params_str = match.group(2)

        # Parse parameters
        params = []
        if params_str.strip():
            # Split by comma, but handle multi-line
            param_list = re.split(r',\s*', params_str.strip())
            for param in param_list:
                param = param.strip()
                if not param:
                    continue
                # Parse "name: type"
                if ':' in param:
                    parts = param.split(':', 1)
                    pname = parts[0].strip()
                    ptype = parts[1].strip()
                    params.append(Param(pname, ptype))

        functions.append(Function(name, params))

    return functions

def convert_type(rust_type: str) -> str:
    """Convert lapack-sys types to lapack-inject types."""
    # Map types
    type_map = {
        '*const c_int': '*const lapackint',
        '*mut c_int': '*mut lapackint',
        '*const c_char': '*const c_char',
        '*mut c_char': '*mut c_char',
        '*const f32': '*const f32',
        '*mut f32': '*mut f32',
        '*const f64': '*const f64',
        '*mut f64': '*mut f64',
        '*const __BindgenComplex<f32>': '*const Complex32',
        '*mut __BindgenComplex<f32>': '*mut Complex32',
        '*const __BindgenComplex<f64>': '*const Complex64',
        '*mut __BindgenComplex<f64>': '*mut Complex64',
        'c_int': 'c_int',
        'size_t': 'usize',
        # SELECT function types
        'LAPACK_S_SELECT2': 'Option<SSelectFn2>',
        'LAPACK_S_SELECT3': 'Option<SSelectFn3>',
        'LAPACK_D_SELECT2': 'Option<DSelectFn2>',
        'LAPACK_D_SELECT3': 'Option<DSelectFn3>',
        'LAPACK_C_SELECT1': 'Option<CSelectFn1>',
        'LAPACK_C_SELECT2': 'Option<CSelectFn2>',
        'LAPACK_Z_SELECT1': 'Option<ZSelectFn1>',
        'LAPACK_Z_SELECT2': 'Option<ZSelectFn2>',
    }

    for old, new in type_map.items():
        rust_type = rust_type.replace(old, new)

    return rust_type

def get_base_name(func_name: str) -> str:
    """Get base name without precision prefix and trailing underscore."""
    # Remove trailing underscore
    name = func_name.rstrip('_')
    # First char is precision (s, d, c, z)
    if name and name[0] in 'sdcz':
        return name[1:]
    return name

def generate_type_name(func_name: str) -> str:
    """Generate type alias name for function pointer."""
    # dgesv_ -> DgesvFnPtr
    name = func_name.rstrip('_')
    return name[0].upper() + name[1:] + 'FnPtr'

def generate_backend_macro_call(func: Function) -> str:
    """Generate define_lapack_ffi! macro call for a function."""
    name = func.name.rstrip('_')
    type_name = generate_type_name(func.name)

    params = []
    for p in func.params:
        converted_type = convert_type(p.type_)
        params.append(f"    {p.name}: {converted_type}")

    params_str = ',\n'.join(params)

    return f"""define_lapack_ffi!({name}, {type_name},
{params_str},
);"""

def generate_fortran_macro_call(func: Function) -> str:
    """Generate export_fortran_symbol! macro call for a function."""
    name = func.name.rstrip('_')

    params = []
    for p in func.params:
        converted_type = convert_type(p.type_)
        params.append(f"    {p.name}: {converted_type}")

    params_str = ',\n'.join(params)

    return f"""export_fortran_symbol!({name},
{params_str},
);"""

def generate_autoregister_extern(func: Function) -> str:
    """Generate extern declaration for autoregister."""
    params = []
    for p in func.params:
        # Use original types for extern declaration
        params.append(f"        {p.name}: {p.type_}")

    params_str = ',\n'.join(params)

    return f"""    fn {func.name}(
{params_str},
    );"""

def generate_autoregister_call(func: Function) -> str:
    """Generate register call for autoregister."""
    name = func.name.rstrip('_')
    return f"        register_{name}(std::mem::transmute({func.name} as *const ()));"

def categorize_functions(functions: List[Function]) -> dict:
    """Categorize functions by their base operation."""
    categories = {}

    for func in functions:
        base = get_base_name(func.name)
        if base not in categories:
            categories[base] = []
        categories[base].append(func)

    return categories

def main():
    lapack_sys_path = Path("/Users/hiroshi/git/lapack-sys/src/lapack.rs")
    output_dir = Path("/Users/hiroshi/projects/tensor4all/lapack-inject/src")

    print(f"Parsing {lapack_sys_path}...")
    functions = parse_lapack_rs(lapack_sys_path)
    print(f"Found {len(functions)} functions")

    # Filter out lsame_ and other utility functions we handle separately
    functions = [f for f in functions if f.name not in ['lsame_']]

    # Categorize
    categories = categorize_functions(functions)
    print(f"Found {len(categories)} unique operations")

    # Generate backend.rs
    print("Generating backend.rs...")
    backend_lines = [
        '//! Fortran LAPACK function pointer registration.',
        '//!',
        '//! This module provides the infrastructure for registering Fortran LAPACK',
        '//! function pointers at runtime. Each function has its own `OnceLock` to allow',
        '//! partial registration (only register the functions you need).',
        '//!',
        '//! Auto-generated from lapack-sys.',
        '',
        '#![allow(non_camel_case_types)]',
        '#![allow(non_snake_case)]',
        '#![allow(clippy::too_many_arguments)]',
        '',
        'use std::ffi::c_char;',
        'use std::sync::OnceLock;',
        '',
        'use num_complex::{Complex32, Complex64};',
        '',
        'use crate::lapackint;',
        '',
        '// =============================================================================',
        '// Select function types for eigenvalue routines',
        '// =============================================================================',
        '',
        '/// Select function type for real GEES (single)',
        'pub type SSelectFn2 = unsafe extern "C" fn(ar: *const f32, ai: *const f32) -> lapackint;',
        '/// Select function type for real GEES (double)',
        'pub type DSelectFn2 = unsafe extern "C" fn(ar: *const f64, ai: *const f64) -> lapackint;',
        '/// Select function type for real (single) with 3 args',
        'pub type SSelectFn3 = unsafe extern "C" fn(ar: *const f32, ai: *const f32, b: *const f32) -> lapackint;',
        '/// Select function type for real (double) with 3 args',
        'pub type DSelectFn3 = unsafe extern "C" fn(ar: *const f64, ai: *const f64, b: *const f64) -> lapackint;',
        '/// Select function type for complex GEES (single)',
        'pub type CSelectFn1 = unsafe extern "C" fn(w: *const Complex32) -> lapackint;',
        '/// Select function type for complex GEES (double)',
        'pub type ZSelectFn1 = unsafe extern "C" fn(w: *const Complex64) -> lapackint;',
        '/// Select function type for complex (single) with 2 args',
        'pub type CSelectFn2 = unsafe extern "C" fn(a: *const Complex32, b: *const Complex32) -> lapackint;',
        '/// Select function type for complex (double) with 2 args',
        'pub type ZSelectFn2 = unsafe extern "C" fn(a: *const Complex64, b: *const Complex64) -> lapackint;',
        '',
        '// =============================================================================',
        '// Macro for defining LAPACK function types, storage, and registration',
        '// =============================================================================',
        '',
        '/// Macro to define a LAPACK function pointer type, OnceLock storage,',
        '/// registration function, and getter function.',
        'macro_rules! define_lapack_ffi {',
        '    ($func_name:ident, $type_name:ident, $( $param:ident : $param_ty:ty ),* $(,)?) => {',
        '        paste::paste! {',
        '            /// Fortran function pointer type',
        '            pub type $type_name = unsafe extern "C" fn( $( $param : $param_ty ),* );',
        '',
        '            static [<$func_name:upper>]: OnceLock<$type_name> = OnceLock::new();',
        '',
        '            #[doc = concat!("Register the Fortran ", stringify!($func_name), " function pointer.")]',
        '            #[no_mangle]',
        '            pub unsafe extern "C" fn [<register_ $func_name>](f: $type_name) {',
        '                let _ = [<$func_name:upper>].set(f);',
        '            }',
        '',
        '            #[doc = concat!("Get the registered ", stringify!($func_name), " function pointer.")]',
        '            #[allow(dead_code)]',
        '            pub(crate) fn [<get_ $func_name>]() -> $type_name {',
        '                *[<$func_name:upper>].get().expect(concat!(stringify!($func_name), " not registered"))',
        '            }',
        '        }',
        '    };',
        '}',
        '',
    ]

    # Add function definitions by category
    for base_name, funcs in sorted(categories.items()):
        backend_lines.append(f'// {base_name.upper()}')
        for func in sorted(funcs, key=lambda f: f.name):
            backend_lines.append(generate_backend_macro_call(func))
            backend_lines.append('')

    (output_dir / "backend.rs").write_text('\n'.join(backend_lines))

    # Generate fortran.rs
    print("Generating fortran.rs...")
    fortran_lines = [
        '//! Fortran LAPACK symbol exports.',
        '//!',
        '//! This module exports Fortran-style LAPACK symbols (e.g., `dgesv_`) that call',
        '//! the registered function pointers. This allows lapack-inject to be a drop-in',
        '//! replacement for lapack-src.',
        '//!',
        '//! Auto-generated from lapack-sys.',
        '',
        '#![allow(non_snake_case)]',
        '#![allow(clippy::too_many_arguments)]',
        '',
        'use crate::backend::*;',
        'use crate::lapackint;',
        'use num_complex::{Complex32, Complex64};',
        'use std::ffi::c_char;',
        '',
        '/// Macro to export a Fortran LAPACK symbol that delegates to the registered function pointer.',
        'macro_rules! export_fortran_symbol {',
        '    ($func_name:ident, $( $param:ident : $param_ty:ty ),* $(,)?) => {',
        '        paste::paste! {',
        '            #[no_mangle]',
        '            pub unsafe extern "C" fn [<$func_name _>](',
        '                $( $param : $param_ty ),*',
        '            ) {',
        '                let f = [<get_ $func_name>]();',
        '                f($( $param ),*);',
        '            }',
        '        }',
        '    };',
        '}',
        '',
    ]

    for base_name, funcs in sorted(categories.items()):
        fortran_lines.append(f'// {base_name.upper()}')
        for func in sorted(funcs, key=lambda f: f.name):
            fortran_lines.append(generate_fortran_macro_call(func))
            fortran_lines.append('')

    (output_dir / "fortran.rs").write_text('\n'.join(fortran_lines))

    print("Done!")
    print(f"Generated {len(functions)} function bindings")

if __name__ == "__main__":
    main()
