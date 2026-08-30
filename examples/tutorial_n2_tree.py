"""Tutorial: fast N-2 triage using an N-1 parent warm-start tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, complete_diagnostic_security_analysis
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def main() -> None:
    report = complete_diagnostic_security_analysis(
        pn.case14(),
        k=2,
        elements=("line",),
        n1_pool=6,
        top_n=6,
        max_cases=15,
        ac_mode="top",
        ac_backend="cpu",
        ac_batch_size=8,
        ac_warm_start=True,
        build_options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
        cpu_options=CpuNewtonOptions(max_iterations=30, tolerance=1e-10, jacobian_reuse=2),
    )
    print(report.to_markdown())
    print("N-2 note: parent N-1 solutions are used only as initial guesses; safety is still judged by each solved contingency.")


if __name__ == "__main__":
    main()
