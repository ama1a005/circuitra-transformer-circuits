#!/usr/bin/env python
"""scripts/build_final_artifacts.py
=====================================
Reads all result JSON files from a results directory, generates summary CSV,
statistical tests, all tables, all plots, and a README.

Usage
-----
    python scripts/build_final_artifacts.py --results-root results/final
    python scripts/build_final_artifacts.py --results-root runs/dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final analysis artifacts")
    parser.add_argument(
        "--results-root", type=str, default="results/final",
        help="Directory containing run result JSON files",
    )
    args = parser.parse_args()

    print(f"\n[build_final_artifacts] Results root: {args.results_root}")

    try:
        from hrl_circuits.analysis.final_report import FinalReport
        report = FinalReport(args.results_root)
        report.generate()
        print("[OK] Final artifacts built successfully.")
        return 0
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Install with: pip install pandas numpy matplotlib")
        return 1
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
