from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, from_pandapower, solve_batch_cpu_sparse
from batched_powerflow.contingency import apply_contingency
from batched_powerflow.dc_screening import screen_n1_lodf_top, screen_nk_lodf_top
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["case9241pegase", "case6470rte"])
    parser.add_argument("--elements", nargs="+", default=["line", "trafo"])
    parser.add_argument("--n1-pool", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--max-combinations", type=int, default=20000)
    parser.add_argument("--ac-validate", action="store_true")
    parser.add_argument("--ac-top-n", type=int, default=10)
    parser.add_argument("--run-powerflow-init", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--linear-solver", choices=["direct", "gmres_ilu"], default="direct")
    parser.add_argument("--out-dir", default="benchmark_results/smart_nk_screening")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    options = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")
    ac_options = PandapowerBuildOptions(
        check_connectivity=False,
        preconditioner="none",
        run_powerflow=args.run_powerflow_init,
    )
    cpu_options = CpuNewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        linear_solver=args.linear_solver,
    )
    rows = []
    details = []
    for case in args.cases:
        net = getattr(pn, case)()
        start = time.perf_counter()
        n1 = screen_n1_lodf_top(net, top_n=args.n1_pool, elements=tuple(args.elements), options=options)
        n1_screen_s = time.perf_counter() - start
        for k in args.k_values:
            max_combinations = None if k == 2 else args.max_combinations
            start = time.perf_counter()
            nk = screen_nk_lodf_top(
                net,
                k=k,
                candidate_results=n1.screened,
                top_n=args.top_n,
                max_combinations=max_combinations,
                options=options,
            )
            nk_screen_s = time.perf_counter() - start
            row = {
                "case": case,
                "k": k,
                "n1_total": n1.total,
                "n1_failed": n1.failed,
                "n1_screen_s": n1_screen_s,
                "candidate_pool": nk.candidate_pool_size,
                "nk_combinations": nk.total,
                "nk_failed": nk.failed,
                "nk_screen_s": nk_screen_s,
                "exhaustive_over_pool": nk.exhaustive_over_pool,
                "top_score": nk.screened[0].score if nk.screened else 0.0,
                "top_contingency": str(nk.screened[0].contingency) if nk.screened else "",
            }
            ac_validate_s = 0.0
            ac_converged = 0
            ac_failed = 0
            ac_max_mismatch = float("nan")
            finite_mismatches = []
            rows.append(row)
            print(row, flush=True)
            for rank, result in enumerate(nk.screened, start=1):
                detail = {
                    "case": case,
                    "k": k,
                    "rank": rank,
                    "contingency": str(result.contingency),
                    "score": result.score,
                    "failed": result.failed,
                    "reason": result.reason,
                }
                if args.ac_validate and rank <= args.ac_top_n:
                    start = time.perf_counter()
                    try:
                        model = from_pandapower(apply_contingency(net, result.contingency), options=ac_options)
                        ac = solve_batch_cpu_sparse(model, options=cpu_options)
                        elapsed = time.perf_counter() - start
                        mismatch = float(ac.mismatch_norm[0])
                        ok = bool(ac.converged[0])
                        detail.update({"ac_s": elapsed, "ac_converged": ok, "ac_mismatch": mismatch})
                        ac_validate_s += elapsed
                        ac_converged += int(ok)
                        ac_failed += int(not ok)
                        if math.isfinite(mismatch):
                            finite_mismatches.append(mismatch)
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        detail.update({"ac_s": elapsed, "ac_converged": False, "ac_mismatch": float("nan"), "ac_error": repr(exc)})
                        ac_validate_s += elapsed
                        ac_failed += 1
                details.append(detail)
            if args.ac_validate:
                ac_max_mismatch = max(finite_mismatches) if finite_mismatches else float("nan")
                row.update(
                    {
                        "ac_top_n": min(args.ac_top_n, len(nk.screened)),
                        "run_powerflow_init": args.run_powerflow_init,
                        "ac_validate_s": ac_validate_s,
                        "ac_converged": ac_converged,
                        "ac_failed": ac_failed,
                        "ac_max_mismatch": ac_max_mismatch,
                    }
                )

    _write_csv(out_dir / "summary.csv", rows)
    _write_csv(out_dir / "details.csv", details)
    lines = [
        "# Smart N-k Screening",
        "",
        "N-1 is complete over all active selected elements. N-k is exact DC over the selected N-1 risk pool.",
        "",
        "|case|k|N-1 total|N-1 failed|N-1 screen s|candidate pool|N-k combos|N-k failed|N-k screen s|exhaustive over pool|top score|AC converged|AC max mismatch|top contingency|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row['case']}|{row['k']}|{row['n1_total']}|{row['n1_failed']}|{row['n1_screen_s']:.3f}|"
            f"{row['candidate_pool']}|{row['nk_combinations']}|{row['nk_failed']}|{row['nk_screen_s']:.3f}|"
            f"{row['exhaustive_over_pool']}|{row['top_score']:.3f}|{row.get('ac_converged', '')}|"
            f"{row.get('ac_max_mismatch', '')}|{row['top_contingency']}|"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
