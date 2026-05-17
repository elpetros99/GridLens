from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, from_pandapower, solve_batch_cpu_sparse
from batched_powerflow.contingency import apply_contingency
from batched_powerflow.dc_screening import screen_n1_lodf_top
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _screen_case(case: str, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    net = getattr(pn, case)()
    screen_options = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")
    ac_build_options = PandapowerBuildOptions(
        check_connectivity=False,
        preconditioner="none",
        run_powerflow=args.run_powerflow_init,
    )

    started = time.perf_counter()
    screening = screen_n1_lodf_top(
        net,
        top_n=args.top_n,
        elements=tuple(args.elements),
        options=screen_options,
        chunk_size=args.chunk_size,
    )
    dc_screen_s = time.perf_counter() - started

    cpu_options = CpuNewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        linear_solver=args.linear_solver,
        start=args.cpu_start,
    )
    details = []
    ac_validate_s = 0.0
    ac_converged = 0
    ac_failed = 0
    for rank, result in enumerate(screening.screened, start=1):
        row = {
            "case": case,
            "rank": rank,
            "contingency": str(result.contingency),
            "dc_score": result.score,
            "dc_failed": result.failed,
            "dc_reason": result.reason,
        }
        started = time.perf_counter()
        try:
            contingency_net = apply_contingency(net, result.contingency)
            model = from_pandapower(contingency_net, options=ac_build_options)
            ac = solve_batch_cpu_sparse(model, options=cpu_options)
            elapsed = time.perf_counter() - started
            row.update(
                {
                    "ac_s": elapsed,
                    "ac_converged": bool(ac.converged[0]),
                    "ac_mismatch": float(ac.mismatch_norm[0]),
                }
            )
            ac_converged += int(bool(ac.converged[0]))
            ac_failed += int(not bool(ac.converged[0]))
            ac_validate_s += elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - started
            row.update({"ac_s": elapsed, "ac_converged": False, "ac_mismatch": float("nan"), "ac_error": repr(exc)})
            ac_failed += 1
            ac_validate_s += elapsed
        details.append(row)

    finite_mismatches = [row["ac_mismatch"] for row in details if row.get("ac_mismatch") == row.get("ac_mismatch")]
    summary = {
        "case": case,
        "total_n1": screening.total,
        "dc_failed": screening.failed,
        "dc_screen_s": dc_screen_s,
        "max_base_loading_pct": screening.max_base_loading_percent,
        "top_n": len(screening.screened),
        "top_dc_score": screening.screened[0].score if screening.screened else 0.0,
        "run_powerflow_init": args.run_powerflow_init,
        "cpu_start": args.cpu_start,
        "ac_validate_s": ac_validate_s,
        "ac_converged": ac_converged,
        "ac_failed": ac_failed,
        "max_ac_mismatch": max(finite_mismatches) if finite_mismatches else float("nan"),
    }
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["case6470rte", "case6495rte", "case6515rte", "case9241pegase"])
    parser.add_argument("--elements", nargs="+", default=["line", "trafo"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--linear-solver", choices=["direct", "gmres_ilu"], default="direct")
    parser.add_argument("--cpu-start", choices=["model", "pandapower_result"], default="model")
    parser.add_argument("--run-powerflow-init", action="store_true")
    parser.add_argument("--out-dir", default="benchmark_results/realistic_large_screening")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    summaries = []
    details = []
    for case in args.cases:
        summary, case_details = _screen_case(case, args)
        summaries.append(summary)
        details.extend(case_details)
        print(summary, flush=True)

    _write_csv(out_dir / "summary.csv", summaries)
    _write_csv(out_dir / "top_details.csv", details)

    lines = [
        "# Realistic Large N-1 Screening",
        "",
        "Complete all active line+trafo N-1 DC screening with LODF/PTDF; top contingencies AC-validated with CPU sparse direct.",
        "",
        "|case|N-1 candidates|DC failed/islanded|DC screen s|max base loading %|top DC score|runpp init|AC validation s|AC converged topN|AC failed topN|max AC mismatch|",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"|{row['case']}|{row['total_n1']}|{row['dc_failed']}|{row['dc_screen_s']:.3f}|"
            f"{row['max_base_loading_pct']:.3f}|{row['top_dc_score']:.3f}|{row['run_powerflow_init']}|{row['ac_validate_s']:.3f}|"
            f"{row['ac_converged']}|{row['ac_failed']}|{row['max_ac_mismatch']:.3g}|"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
