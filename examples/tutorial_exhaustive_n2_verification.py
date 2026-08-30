"""Tutorial: exhaustively verify every case14 line N-2 pair."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, complete_diagnostic_security_analysis
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def main() -> None:
    started = time.perf_counter()
    report = complete_diagnostic_security_analysis(
        pn.case14(),
        k=2,
        elements=("line",),
        exhaustive_nk=True,
        ac_mode="all",
        ac_backend="cpu",
        ac_batch_size=32,
        ac_warm_start=True,
        build_options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
        cpu_options=CpuNewtonOptions(max_iterations=30, tolerance=1e-9, jacobian_reuse=1),
    )
    print(report.to_markdown())
    print(f"Elapsed: {time.perf_counter() - started:.3f}s")
    print("Interpretation: SAFE requires zero DC failures, no AC violations, exhaustive_dc=True, exhaustive_ac=True, and all requested AC cases resolved.")
    print(report.certificate.to_json())


if __name__ == "__main__":
    main()
