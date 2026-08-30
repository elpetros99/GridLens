from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, NewtonOptions, complete_diagnostic_security_analysis
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete diagnostic N-k security analysis with stability indicators.")
    parser.add_argument("--case", required=True, help="pandapower.networks case name, e.g. case118")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--elements", nargs="+", default=["line", "trafo"])
    parser.add_argument("--max-cases", type=int, default=None, help="Cap N-k combinations. If set, the report is not a complete guarantee.")
    parser.add_argument("--n1-pool", type=int, default=None, help="For k>1, screen N-k over this many highest-risk N-1 candidates.")
    parser.add_argument("--exhaustive-nk", action="store_true", help="Enumerate every active branch N-k combination; disables risk-pool pruning.")
    parser.add_argument("--pruning", choices=["none", "risk_pool"], default=None, help="Use risk-pool pruning by default, or none automatically with --exhaustive-nk.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--ac-mode", choices=["none", "top", "all"], default="top")
    parser.add_argument("--ac-backend", choices=["cpu", "jax"], default="cpu", help="AC validation backend; JAX uses the GPU-compatible topology batch path.")
    parser.add_argument("--ac-batch-size", type=int, default=64)
    parser.add_argument("--no-ac-warm-start", action="store_true")
    parser.add_argument("--voltage-min", type=float, default=0.90)
    parser.add_argument("--voltage-max", type=float, default=1.10)
    parser.add_argument("--loading-limit-percent", type=float, default=100.0)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--jacobian-reuse", type=int, default=1, help="Reuse each CPU numerical Jacobian/factorization for this many Newton steps.")
    parser.add_argument("--jax-max-iterations", type=int, default=12)
    parser.add_argument("--preconditioner", default="none", choices=["none", "auto", "fast_decoupled_dense", "fast_decoupled_diagonal"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = getattr(pn, args.case)()
    pruning = args.pruning if args.pruning is not None else ("none" if args.exhaustive_nk else "risk_pool")
    report = complete_diagnostic_security_analysis(
        net,
        k=args.k,
        elements=tuple(args.elements),
        max_cases=args.max_cases,
        n1_pool=args.n1_pool,
        top_n=args.top_n,
        ac_mode=args.ac_mode,
        voltage_min=args.voltage_min,
        voltage_max=args.voltage_max,
        loading_limit_percent=args.loading_limit_percent,
        build_options=PandapowerBuildOptions(check_connectivity=False, preconditioner=args.preconditioner),
        cpu_options=CpuNewtonOptions(max_iterations=args.max_iterations, tolerance=args.tolerance, jacobian_reuse=args.jacobian_reuse),
        ac_backend=args.ac_backend,
        ac_batch_size=args.ac_batch_size,
        ac_warm_start=not args.no_ac_warm_start,
        jax_options=NewtonOptions(max_iterations=args.jax_max_iterations, tolerance=args.tolerance),
        exhaustive_nk=args.exhaustive_nk,
        pruning=pruning,
    )
    print(report.to_markdown())


if __name__ == "__main__":
    main()
