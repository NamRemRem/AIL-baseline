"""
setup_data.py
Creates a unified 'data/' directory for PatchCore training on carpet + vial.
Uses directory junctions (Windows) so no actual copying is needed.
"""

import os
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))


def make_junction(src, dst):
    """Create a Windows directory junction from src to dst."""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.exists(dst):
        print(f"  [SKIP] Already exists: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", dst, src], capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  [OK]   {src} -> {dst}")
    else:
        print(f"  [FAIL] {result.stderr.strip()}")


def setup():
    print("=" * 60)
    print("Setting up data/ directory for carpet + vial")
    print("=" * 60)

    # ── carpet ──────────────────────────────────────────────────
    # carpet/carpet/ already has correct MVTec structure (train, test, ground_truth)
    # data/carpet/ -> carpet/carpet/
    print("\n[1] carpet:")
    carpet_src = os.path.join(BASE, "carpet", "carpet")
    carpet_dst = os.path.join(BASE, "data", "carpet")
    make_junction(carpet_src, carpet_dst)

    # ── vial ─────────────────────────────────────────────────────
    # vial/ has non-standard layout:
    #   train/good/                  (OK - maps directly)
    #   test_public/bad/             (needs to be test/bad/)
    #   test_public/good/            (needs to be test/good/)
    #   test_public/ground_truth/bad/ (needs to be ground_truth/bad/)
    print("\n[2] vial:")
    vial_base = os.path.join(BASE, "vial")

    # train → data/vial/train/
    make_junction(
        os.path.join(vial_base, "train"),
        os.path.join(BASE, "data", "vial", "train"),
    )

    # test_public/bad → data/vial/test/bad/
    make_junction(
        os.path.join(vial_base, "test_public", "bad"),
        os.path.join(BASE, "data", "vial", "test", "bad"),
    )

    # test_public/good → data/vial/test/good/
    make_junction(
        os.path.join(vial_base, "test_public", "good"),
        os.path.join(BASE, "data", "vial", "test", "good"),
    )

    # test_public/ground_truth/bad → data/vial/ground_truth/bad/
    make_junction(
        os.path.join(vial_base, "test_public", "ground_truth", "bad"),
        os.path.join(BASE, "data", "vial", "ground_truth", "bad"),
    )

    print("\n" + "=" * 60)
    print("Done! Verify with: tree data /f /a")
    print("=" * 60)

    # Quick sanity checks
    print("\n[Sanity check]")
    checks = [
        ("data/carpet/train/good", True),
        ("data/carpet/test", True),
        ("data/carpet/ground_truth", True),
        ("data/vial/train/good", True),
        ("data/vial/test/bad", True),
        ("data/vial/test/good", True),
        ("data/vial/ground_truth/bad", True),
    ]
    all_ok = True
    for relpath, expected in checks:
        full = os.path.join(BASE, relpath)
        exists = os.path.exists(full)
        status = "OK  " if exists == expected else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  [{status}] {relpath}")
    print(
        "\nAll checks passed!"
        if all_ok
        else "\nSome checks FAILED. Please investigate."
    )


if __name__ == "__main__":
    setup()
