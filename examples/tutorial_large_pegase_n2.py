"""Scalable N-2 screening on a large PEGASE transmission case.

This tutorial deliberately uses a complete N-1 DC screen, then an exact
multi-outage DC calculation over a bounded high-risk N-1 pool. It keeps the
work bounded for large networks while retaining an auditable certificate for
the selected universe. It is not a full-universe N-2 AC proof.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, complete_diagnostic_security_analysis
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalable large-case PEGASE N-2 tutorial.")
    parser.add_argument("--case", default="case1354pegase", help="pandapower network case, e.g. case1354pegase or case9241pegase")
    parser.add_argument("--n1-pool", type=int, default=80, help="High-risk N-1 pool used to form the exact N-2 DC candidate universe.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of N-2 cases to validate with AC.")
    parser.add_argument("--ac-mode", choices=["top", "all"], default="top")
    parser.add_argument("--ac-backend", choices=["cpu", "jax"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default=None, help="Optional directory for report.md and certificate.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = getattr(pn, args.case)()
    started = time.perf_counter()
    report = complete_diagnostic_security_analysis(
        net,
        k=2,
        elements=("line", "trafo"),
        n1_pool=args.n1_pool,
        top_n=args.top_n,
        ac_mode=args.ac_mode,
        ac_backend=args.ac_backend,
        ac_batch_size=args.batch_size,
        ac_warm_start=True,
        pruning="risk_pool",
        build_options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
        cpu_options=CpuNewtonOptions(max_iterations=30, tolerance=1e-9, jacobian_reuse=2),
    )
    elapsed = time.perf_counter() - started
    certificate = report.certificate
    summary = {
        "case": args.case,
        "buses": len(net.bus),
        "lines": len(net.line),
        "transformers": len(net.trafo),
        "elapsed_s": elapsed,
        "verdict": "SAFE" if report.safe else "UNSAFE",
        "guarantee_scope": report.guarantee_scope,
        "n2_dc_cases_in_pool": report.total_screened,
        "ac_cases_requested": report.ac_validated,
        "ac_cases_converged": report.ac_converged,
        "ac_unresolved": report.ac_unresolved,
        "parent_tree_warm_start_children": 0 if certificate is None else certificate.parent_tree_warm_start_children,
        "candidate_hash": None if certificate is None else certificate.candidate_hash,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(report.to_markdown())
    if certificate is not None:
        print(certificate.to_json())

    if args.out_dir:
        output_dir = ROOT / args.out_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{args.case}.md").write_text(report.to_markdown(), encoding="utf-8")
        if certificate is not None:
            (output_dir / f"{args.case}.certificate.json").write_text(certificate.to_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
