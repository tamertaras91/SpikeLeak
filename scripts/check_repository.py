#!/usr/bin/env python
"""Static repository sanity check that does not require datasets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    py_files = sorted(
        p for p in ROOT.rglob("*.py")
        if "archive/source_snapshots" not in p.as_posix()
    )

    failures = []
    for path in py_files:
        try:
            compile(path.read_text(), str(path), "exec")
        except Exception as exc:
            failures.append((path, exc))

    print(f"Checked {len(py_files)} Python files.")
    if failures:
        for path, exc in failures:
            print(f"FAIL: {path.relative_to(ROOT)} -> {exc}")
        raise SystemExit(1)

    print("Syntax check: PASS")
    print("No dataset download or experiment was executed.")

if __name__ == "__main__":
    main()
