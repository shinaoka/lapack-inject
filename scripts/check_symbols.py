#!/usr/bin/env python3
"""
Check that all expected LAPACK symbols are exported from the built library.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_expected_symbols(lapack_sys_path: Path) -> set:
    """Extract expected symbol names from lapack-sys."""
    import re
    content = lapack_sys_path.read_text()
    pattern = r'pub fn (\w+_)\s*\('
    symbols = set()
    for match in re.finditer(pattern, content):
        symbols.add(match.group(1))
    return symbols


def get_exported_symbols(dylib_path: Path) -> set:
    """Get exported symbols from the built library."""
    result = subprocess.run(
        ['nm', '-g', str(dylib_path)],
        capture_output=True,
        text=True
    )

    symbols = set()
    for line in result.stdout.split('\n'):
        parts = line.split()
        if len(parts) >= 3 and parts[1] == 'T':
            name = parts[2].lstrip('_')
            if name.endswith('_'):
                symbols.add(name)
    return symbols


def get_generated_set() -> set:
    """Return the set of LAPACK symbols the generator currently produces."""
    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lapack-sys-path', type=str, required=True,
                        help='Path to lapack-sys lapack.rs')
    parser.add_argument('--library', type=str, required=True,
                        help='Path to the built liblapack_inject library')
    parser.add_argument('--all', action='store_true',
                        help='Check against all lapack-sys symbols instead of generated set')
    args = parser.parse_args()

    lapack_sys = Path(args.lapack_sys_path)
    dylib = Path(args.library)

    if not dylib.exists():
        print(f"Error: Library not found: {dylib}")
        sys.exit(1)

    print(f"Checking {dylib}...")

    if args.all:
        expected = get_expected_symbols(lapack_sys)
    else:
        expected = get_generated_set()

    exported = get_exported_symbols(dylib)
    lapack_exported = {s for s in exported if s[0] in 'sdcz' and s.endswith('_')}

    print(f"Expected symbols: {len(expected)}")
    print(f"Exported symbols: {len(lapack_exported)}")

    missing = expected - lapack_exported
    extra = lapack_exported - expected

    if missing:
        print(f"\nMissing symbols ({len(missing)}):")
        for s in sorted(missing)[:20]:
            print(f"  {s}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if extra:
        print(f"\nExtra symbols ({len(extra)}):")
        for s in sorted(extra)[:10]:
            print(f"  {s}")

    if not missing:
        print("\nAll expected symbols are exported!")
        return 0
    else:
        print(f"\n{len(missing)} symbols missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
