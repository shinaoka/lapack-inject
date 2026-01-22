#!/usr/bin/env python3
"""
Check that all expected LAPACK symbols are exported from the built library.
"""

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
            # Remove leading underscore (macOS convention)
            name = parts[2].lstrip('_')
            if name.endswith('_'):
                symbols.add(name)
    return symbols

def main():
    lapack_sys = Path("/Users/hiroshi/git/lapack-sys/src/lapack.rs")

    # Try both debug and release
    for profile in ['debug', 'release']:
        dylib = Path(f"/Users/hiroshi/projects/tensor4all/lapack-inject/target/{profile}/liblapack_inject.dylib")
        if dylib.exists():
            break
    else:
        print("Error: Library not found. Run 'cargo build' first.")
        sys.exit(1)

    print(f"Checking {dylib}...")

    expected = get_expected_symbols(lapack_sys)
    exported = get_exported_symbols(dylib)

    # Filter to only LAPACK symbols (s/d/c/z prefix)
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
        print("\n✓ All expected symbols are exported!")
        return 0
    else:
        print(f"\n✗ {len(missing)} symbols missing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
