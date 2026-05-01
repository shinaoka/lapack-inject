#!/usr/bin/env python3
"""
Generate lapack-inject bindings from lapack-sys.
"""

import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Param:
    name: str
    type_: str

@dataclass
class Function:
    name: str
    params: List[Param]

# Only generate bindings for these core LAPACK functions.
CORE_FUNCTIONS = {
    "sgesv_", "dgesv_", "cgesv_", "zgesv_",
    "sgetrf_", "dgetrf_", "cgetrf_", "zgetrf_",
    "sgetrs_", "dgetrs_", "cgetrs_", "zgetrs_",
    "sgetri_", "dgetri_", "cgetri_", "zgetri_",
    "spotrf_", "dpotrf_", "cpotrf_", "zpotrf_",
    "ssyev_", "dsyev_",
    "sgesvd_", "dgesvd_", "cgesvd_", "zgesvd_",
    "sgeqrf_", "dgeqrf_", "cgeqrf_", "zgeqrf_",
    "sorgqr_", "dorgqr_",
    "cungqr_", "zungqr_",
    "strtrs_", "dtrtrs_", "ctrtrs_", "ztrtrs_",
    "sgeev_", "dgeev_", "cgeev_", "zgeev_",
    "cheev_", "zheev_",
    "sgetc2_", "dgetc2_", "cgetc2_", "zgetc2_",
    "sgesc2_", "dgesc2_", "cgesc2_", "zgesc2_",
}

SUPPLEMENTAL_FUNCTIONS = [
    Function("cgesc2_", [
        Param("n", "*const c_int"),
        Param("A", "*const __BindgenComplex<f32>"),
        Param("lda", "*const c_int"),
        Param("rhs", "*mut __BindgenComplex<f32>"),
        Param("ipiv", "*const c_int"),
        Param("jpiv", "*const c_int"),
        Param("scale", "*mut f32"),
    ]),
    Function("dgesc2_", [
        Param("n", "*const c_int"),
        Param("A", "*const f64"),
        Param("lda", "*const c_int"),
        Param("rhs", "*mut f64"),
        Param("ipiv", "*const c_int"),
        Param("jpiv", "*const c_int"),
        Param("scale", "*mut f64"),
    ]),
    Function("sgesc2_", [
        Param("n", "*const c_int"),
        Param("A", "*const f32"),
        Param("lda", "*const c_int"),
        Param("rhs", "*mut f32"),
        Param("ipiv", "*const c_int"),
        Param("jpiv", "*const c_int"),
        Param("scale", "*mut f32"),
    ]),
    Function("zgesc2_", [
        Param("n", "*const c_int"),
        Param("A", "*const __BindgenComplex<f64>"),
        Param("lda", "*const c_int"),
        Param("rhs", "*mut __BindgenComplex<f64>"),
        Param("ipiv", "*const c_int"),
        Param("jpiv", "*const c_int"),
        Param("scale", "*mut f64"),
    ]),
    Function("cgetc2_", [
        Param("n", "*const c_int"),
        Param("A", "*mut __BindgenComplex<f32>"),
        Param("lda", "*const c_int"),
        Param("ipiv", "*mut c_int"),
        Param("jpiv", "*mut c_int"),
        Param("info", "*mut c_int"),
    ]),
    Function("dgetc2_", [
        Param("n", "*const c_int"),
        Param("A", "*mut f64"),
        Param("lda", "*const c_int"),
        Param("ipiv", "*mut c_int"),
        Param("jpiv", "*mut c_int"),
        Param("info", "*mut c_int"),
    ]),
    Function("sgetc2_", [
        Param("n", "*const c_int"),
        Param("A", "*mut f32"),
        Param("lda", "*const c_int"),
        Param("ipiv", "*mut c_int"),
        Param("jpiv", "*mut c_int"),
        Param("info", "*mut c_int"),
    ]),
    Function("zgetc2_", [
        Param("n", "*const c_int"),
        Param("A", "*mut __BindgenComplex<f64>"),
        Param("lda", "*const c_int"),
        Param("ipiv", "*mut c_int"),
        Param("jpiv", "*mut c_int"),
        Param("info", "*mut c_int"),
    ]),
]

def parse_lapack_rs(path: Path) -> List[Function]:
    """Parse lapack.rs and extract function signatures."""
    content = path.read_text()

    pattern = r'pub fn (\w+_)\s*\(([\s\S]*?)\)(?:\s*->\s*\w+)?;'

    functions = []
    for match in re.finditer(pattern, content):
        name = match.group(1)
        params_str = match.group(2)

        params = []
        if params_str.strip():
            param_list = re.split(r',\s*', params_str.strip())
            for param in param_list:
                param = param.strip()
                if not param:
                    continue
                if ':' in param:
                    parts = param.split(':', 1)
                    pname = parts[0].strip()
                    ptype = parts[1].strip()
                    params.append(Param(pname, ptype))

        functions.append(Function(name, params))

    return functions

def convert_type_concrete(rust_type: str, int_ty: str) -> str:
    """Convert lapack-sys types to concrete i32/i64 types."""
    type_map = {
        '*const c_int': f'*const {int_ty}',
        '*mut c_int': f'*mut {int_ty}',
        '*const c_char': '*const c_char',
        '*mut c_char': '*mut c_char',
        '*const f32': '*const f32',
        '*mut f32': '*mut f32',
        '*const f64': '*const f64',
        '*mut f64': '*mut f64',
        '*const __BindgenComplex<f32>': '*const num_complex::Complex32',
        '*mut __BindgenComplex<f32>': '*mut num_complex::Complex32',
        '*const __BindgenComplex<f64>': '*const num_complex::Complex64',
        '*mut __BindgenComplex<f64>': '*mut num_complex::Complex64',
        'c_int': 'c_int',
        'size_t': 'usize',
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

def convert_type_lapackint(rust_type: str) -> str:
    """Convert lapack-sys types using lapackint (for Fortran exports)."""
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
        'c_int': 'lapackint',
        'size_t': 'usize',
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

def to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase (matching paste::paste! :camel)."""
    return ''.join(p.capitalize() for p in name.split('_'))

def get_type_name_prefix(func_name: str) -> str:
    """Get CamelCase prefix from function name (e.g., dgesv_ -> Dgesv, zungtsqr_row_ -> ZungtsqrRow)."""
    name = func_name.rstrip('_')
    return to_camel_case(name)

def generate_dual_types(func: Function) -> str:
    """Generate LP64 and ILP64 fn pointer types + define_dual_backend! call."""
    name = func.name.rstrip('_')
    prefix = get_type_name_prefix(func.name)

    lp64_params = []
    ilp64_params = []
    for p in func.params:
        lp64_type = convert_type_concrete(p.type_, 'i32')
        ilp64_type = convert_type_concrete(p.type_, 'i64')
        lp64_params.append(f"    {p.name}: {lp64_type}")
        ilp64_params.append(f"    {p.name}: {ilp64_type}")

    lp64_str = ',\n'.join(lp64_params)
    ilp64_str = ',\n'.join(ilp64_params)

    return f"""pub type {prefix}Lp64FnPtr = unsafe extern "C" fn(
{lp64_str},
);
pub type {prefix}Ilp64FnPtr = unsafe extern "C" fn(
{ilp64_str},
);
define_dual_backend!({name}, {prefix}Lp64FnPtr, {prefix}Ilp64FnPtr);"""

def generate_fortran_export(func: Function) -> str:
    """Generate Fortran symbol export with dual dispatch."""
    name = func.name.rstrip('_')
    prefix = get_type_name_prefix(func.name)
    provider_name = f"{prefix}Provider"

    params = []
    for p in func.params:
        converted = convert_type_lapackint(p.type_)
        params.append(f"    {p.name}: {converted}")

    lp64_lines = []
    lp64_params = []
    ilp64_lines = []
    ilp64_params = []
    lp64_writeback = []
    ilp64_writeback = []
    for p in func.params:
        if p.type_.startswith('*const c_int'):
            lp64_name = f"{p.name}_lp64"
            ilp64_name = f"{p.name}_ilp64"
            lp64_lines.append(f"            let {lp64_name}: i32 = *{p.name} as i32;")
            ilp64_lines.append(f"            let {ilp64_name}: i64 = *{p.name} as i64;")
            lp64_params.append(f"&{lp64_name}")
            ilp64_params.append(f"&{ilp64_name}")
        elif p.type_.startswith('*mut c_int'):
            lp64_name = f"{p.name}_lp64"
            ilp64_name = f"{p.name}_ilp64"
            lp64_lines.append(f"            let mut {lp64_name}: i32 = *{p.name} as i32;")
            ilp64_lines.append(f"            let mut {ilp64_name}: i64 = *{p.name} as i64;")
            lp64_params.append(f"&mut {lp64_name}")
            ilp64_params.append(f"&mut {ilp64_name}")
            lp64_writeback.append(f"            *{p.name} = {lp64_name} as lapackint;")
            ilp64_writeback.append(f"            *{p.name} = {ilp64_name} as lapackint;")
        else:
            lp64_params.append(p.name)
            ilp64_params.append(p.name)

    lp64_wb_str = '\n'.join(lp64_writeback)
    ilp64_wb_str = '\n'.join(ilp64_writeback)
    lp64_lines_str = '\n'.join(lp64_lines)
    ilp64_lines_str = '\n'.join(ilp64_lines)
    lp64_calls_str = ', '.join(lp64_params)
    ilp64_calls_str = ', '.join(ilp64_params)
    params_str = ',\n'.join(params)

    has_lp64_wb = bool(lp64_writeback)
    has_ilp64_wb = bool(ilp64_writeback)
    lp64_call_block = f"""            fun({lp64_calls_str})""" + (f""";
{lp64_wb_str}""" if has_lp64_wb else "")
    ilp64_call_block = f"""            fun({ilp64_calls_str})""" + (f""";
{ilp64_wb_str}""" if has_ilp64_wb else "")

    return f"""#[no_mangle]
pub unsafe extern "C" fn {func.name}(
{params_str},
) {{
    #[cfg(feature = "ilp64")]
    let provider = get_{name}_for_ilp64();
    #[cfg(not(feature = "ilp64"))]
    let provider = get_{name}_for_lp64();
    match provider {{
        {provider_name}::Lp64(fun) => {{
{lp64_lines_str}
{lp64_call_block}
        }}
        {provider_name}::Ilp64(fun) => {{
{ilp64_lines_str}
{ilp64_call_block}
        }}
    }}
}}"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lapack-sys-path', type=str, required=True,
                        help='Path to lapack-sys lapack.rs')
    parser.add_argument('--output-dir', type=str, default='src',
                        help='Output directory for generated files')
    args = parser.parse_args()
    lapack_sys_path = Path(args.lapack_sys_path)
    output_dir = Path(args.output_dir)

    print(f"Parsing {lapack_sys_path}...")
    functions = parse_lapack_rs(lapack_sys_path)
    print(f"Found {len(functions)} functions")

    functions = [f for f in functions if f.name in CORE_FUNCTIONS]
    existing_names = {f.name for f in functions}
    functions.extend(f for f in SUPPLEMENTAL_FUNCTIONS if f.name in CORE_FUNCTIONS and f.name not in existing_names)

    print("Generating backend_gen.rs...")
    backend_lines = [
        '// Auto-generated: LP64/ILP64 dual function pointer types.',
        '',
    ]

    functions.sort(key=lambda f: f.name)
    for func in functions:
        backend_lines.append(generate_dual_types(func))
        backend_lines.append('')

    (output_dir / "backend_gen.rs").write_text('\n'.join(backend_lines))

    print("Generating fortran_gen.rs...")
    fortran_lines = [
        '// Auto-generated: Fortran LAPACK symbol exports with dual LP64/ILP64 dispatch.',
        '',
    ]

    for func in functions:
        fortran_lines.append(generate_fortran_export(func))
        fortran_lines.append('')

    (output_dir / "fortran_gen.rs").write_text('\n'.join(fortran_lines))

    print("Done!")
    print(f"Generated {len(functions)} function bindings")

if __name__ == "__main__":
    main()
