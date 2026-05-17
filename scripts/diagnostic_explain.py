from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, explain_powerflow, from_pandapower, solve_batch_cpu_sparse
from batched_powerflow.contingency import Contingency, Outage, apply_contingency
from batched_powerflow.dc_screening import score_contingencies_dc
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def _parse_outage(value: str) -> Outage:
    try:
        element, index = value.split(":", 1)
        return Outage(element, int(index))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("outage must use ELEMENT:INDEX, e.g. line:4") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic explainability tool for singular, infeasible, or non-convergent power-flow cases.")
    parser.add_argument("--case", required=True, help="pandapower.networks case name, e.g. case118")
    parser.add_argument("--outage", action="append", type=_parse_outage, default=[], help="Outage as ELEMENT:INDEX. Repeat for N-k.")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--preconditioner", default="none", choices=["none", "auto", "fast_decoupled_dense", "fast_decoupled_diagonal"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = getattr(pn, args.case)()
    contingency = Contingency(tuple(args.outage)) if args.outage else None
    work = apply_contingency(net, contingency) if contingency is not None else net

    model = None
    result = None
    dc_result = None
    exception = None
    try:
        model = from_pandapower(
            work,
            options=PandapowerBuildOptions(check_connectivity=False, preconditioner=args.preconditioner),
        )
        result = solve_batch_cpu_sparse(
            model,
            options=CpuNewtonOptions(max_iterations=args.max_iterations, tolerance=args.tolerance),
        )
    except Exception as exc:
        exception = exc

    if contingency is not None:
        try:
            dc_result = next(
                score_contingencies_dc(
                    net,
                    [contingency],
                    options=PandapowerBuildOptions(check_connectivity=False, preconditioner=args.preconditioner),
                    chunk_size=1,
                )
            )
        except Exception:
            dc_result = None

    report = explain_powerflow(
        work,
        model=model,
        result=result,
        dc_result=dc_result,
        contingency=contingency,
        exception=exception,
    )
    print(report.to_markdown())


if __name__ == "__main__":
    main()
