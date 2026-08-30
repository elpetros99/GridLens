"""Tutorial: complete DC + exhaustive AC N-1 validation with batched CPU AC."""

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
        k=1,
        elements=("line", "trafo"),
        ac_mode="all",
        ac_backend="cpu",
        ac_batch_size=8,
        ac_warm_start=True,
        build_options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
        cpu_options=CpuNewtonOptions(max_iterations=30, tolerance=1e-10, jacobian_reuse=1),
    )
    print(report.to_markdown())


if __name__ == "__main__":
    main()
