from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandapower as pp
import pandapower.contingency as ppcont
import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, from_pandapower, solve_batch_cpu_sparse
from batched_powerflow.contingency import apply_contingency
from batched_powerflow.dc_screening import screen_n1_lodf_top
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


PF_KWARGS = {
    "algorithm": "nr",
    "calculate_voltage_angles": True,
    "init": "auto",
    "max_iteration": 30,
    "tolerance_mva": 1e-10,
    "trafo_model": "t",
    "trafo_loading": "current",
    "enforce_q_lims": False,
    "check_connectivity": False,
    "voltage_depend_loads": False,
    "numba": False,
}


def _now() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _case(name: str):
    return getattr(pn, name)()


def _counts(net) -> dict[str, int]:
    return {
        "n_bus": len(net.bus),
        "n_line": len(net.line) if hasattr(net, "line") else 0,
        "n_trafo": len(net.trafo) if hasattr(net, "trafo") else 0,
    }


def _active_indices(net, element: str) -> list[int]:
    if not hasattr(net, element):
        return []
    table = getattr(net, element)
    if table is None or len(table) == 0:
        return []
    if "in_service" in table:
        return [int(i) for i in table.index[table["in_service"].astype(bool)]]
    return [int(i) for i in table.index]


def _nminus1_cases(net, elements: tuple[str, ...], max_cases: int | None = None) -> dict[str, dict[str, list[int]]]:
    remaining = max_cases
    cases: dict[str, dict[str, list[int]]] = {}
    for element in elements:
        indices = _active_indices(net, element)
        if remaining is not None:
            indices = indices[:remaining]
            remaining -= len(indices)
        if indices:
            cases[element] = {"index": indices}
        if remaining == 0:
            break
    return cases


def _n_contingencies(cases: dict[str, dict[str, list[int]]]) -> int:
    return sum(len(values["index"]) for values in cases.values())


def _time_repeated(fn: Callable[[], Any], *, warmups: int, repeats: int) -> tuple[list[float], Any]:
    last = None
    for _ in range(warmups):
        last = fn()
    times = []
    for _ in range(repeats):
        start = _now()
        last = fn()
        times.append(_elapsed(start))
    return times, last


def _summarize_times(times: list[float]) -> dict[str, float]:
    return {
        "min_s": min(times),
        "median_s": float(np.median(times)),
        "mean_s": float(np.mean(times)),
        "max_s": max(times),
    }


def benchmark_base(cases: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        net = _case(case)
        counts = _counts(net)

        start = _now()
        model = from_pandapower(
            net,
            options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
        )
        build_s = _elapsed(start)
        cpu_options = CpuNewtonOptions(max_iterations=args.max_iterations, tolerance=args.tolerance)
        times, result = _time_repeated(
            lambda: solve_batch_cpu_sparse(model, options=cpu_options),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        rows.append(
            {
                "case": case,
                "backend": "batched_powerflow_cpu_sparse",
                "scope": "base_ac_solve_only",
                **counts,
                "n_state": model.n_state,
                "build_s_excluded": build_s,
                "converged": bool(result.converged[0]),
                "mismatch": float(result.mismatch_norm[0]),
                **_summarize_times(times),
            }
        )

        for backend, extra in (
            ("pandapower_runpp", {"lightsim2grid": False}),
            ("pandapower_lightsim2grid_runpp", {"lightsim2grid": True}),
        ):
            work = copy.deepcopy(net)

            def runpp_once():
                pp.runpp(work, **PF_KWARGS, **extra)
                return work

            try:
                times, solved = _time_repeated(runpp_once, warmups=args.warmups, repeats=args.repeats)
                rows.append(
                    {
                        "case": case,
                        "backend": backend,
                        "scope": "base_ac_runpp_wall",
                        **counts,
                        "converged": bool(solved.converged),
                        "mismatch": math.nan,
                        **_summarize_times(times),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case,
                        "backend": backend,
                        "scope": "base_ac_runpp_wall",
                        **counts,
                        "error": repr(exc),
                    }
                )
    return rows


def _ls2g_direct_profile(net, cases: dict[str, dict[str, list[int]]], args: argparse.Namespace) -> dict[str, Any]:
    from lightsim2grid import SolverType

    work = copy.deepcopy(net)
    pp.runpp(work, **PF_KWARGS, lightsim2grid=False)
    grid_model = ppcont.init_ls2g(work)
    analysis = ppcont.ContingencyAnalysisCPP(grid_model)
    solver_type = SolverType.KLUSingleSlack if ppcont.KLU_solver_available else SolverType.SparseLUSingleSlack
    analysis.change_solver(solver_type)

    n_lines = len(work.line)
    ids = []
    if "line" in cases:
        line_index = work.line.index.to_numpy()
        ids.extend(np.searchsorted(line_index, np.asarray(cases["line"]["index"], dtype=line_index.dtype)).tolist())
    if "trafo" in cases:
        trafo_index = work.trafo.index.to_numpy()
        trafo_ids = np.searchsorted(trafo_index, np.asarray(cases["trafo"]["index"], dtype=trafo_index.dtype)) + n_lines
        ids.extend(trafo_ids.tolist())
    analysis.add_multiple_n1(np.asarray(ids, dtype=np.int64))
    v_init = work._ppc["internal"]["V"]

    analysis.compute(v_init, args.max_iterations, args.tolerance)
    analysis.compute_flows()
    analysis.clear_results_only()

    start = _now()
    analysis.compute(v_init, args.max_iterations, args.tolerance)
    compute_wall_s = _elapsed(start)
    start = _now()
    analysis.compute_flows()
    flows_wall_s = _elapsed(start)
    total_wall_s = compute_wall_s + flows_wall_s

    return {
        "solver_type": str(solver_type),
        "nb_solved": int(analysis.nb_solved()),
        "ls2g_compute_wall_s": compute_wall_s,
        "ls2g_flows_wall_s": flows_wall_s,
        "ls2g_total_wall_s": total_wall_s,
        "ls2g_solver_time_s": float(analysis.solver_time()),
        "ls2g_preprocessing_time_s": float(analysis.preprocessing_time()),
        "ls2g_modif_ybus_time_s": float(analysis.modif_Ybus_time()),
        "ls2g_amps_computation_time_s": float(analysis.amps_computation_time()),
        "ls2g_internal_total_time_s": float(analysis.total_time()),
    }


def benchmark_security(cases: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elements = tuple(args.elements)
    for case in cases:
        net = _case(case)
        counts = _counts(net)
        nminus1 = _nminus1_cases(net, elements, args.max_contingencies)
        n_cont = _n_contingencies(nminus1)
        if n_cont == 0:
            continue

        start = _now()
        screening = screen_n1_lodf_top(
            net,
            top_n=args.top_n,
            elements=elements,
            options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
            chunk_size=args.chunk_size,
        )
        dc_s = _elapsed(start)
        rows.append(
            {
                "case": case,
                "backend": "batched_powerflow_dc_lodf",
                "scope": "complete_n1_dc_screening",
                **counts,
                "n_contingencies": screening.total,
                "failed_or_islanded": screening.failed,
                "total_wall_s": dc_s,
                "s_per_contingency": dc_s / max(screening.total, 1),
                "top_n": len(screening.screened),
                "top_score": screening.screened[0].score if screening.screened else math.nan,
            }
        )

        cpu_options = CpuNewtonOptions(max_iterations=args.max_iterations, tolerance=args.tolerance)
        ac_models = [
            from_pandapower(
                apply_contingency(net, result.contingency),
                options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
            )
            for result in screening.screened
        ]
        start = _now()
        ac_results = [solve_batch_cpu_sparse(model, options=cpu_options) for model in ac_models]
        ac_s = _elapsed(start)
        finite_mismatch = [
            float(result.mismatch_norm[0])
            for result in ac_results
            if bool(result.converged[0]) and np.isfinite(float(result.mismatch_norm[0]))
        ]
        rows.append(
            {
                "case": case,
                "backend": "batched_powerflow_cpu_sparse",
                "scope": "selected_top_n_ac_solve_only",
                **counts,
                "n_contingencies": len(ac_models),
                "total_wall_s": ac_s,
                "s_per_contingency": ac_s / max(len(ac_models), 1),
                "converged_count": int(sum(bool(result.converged[0]) for result in ac_results)),
                "max_finite_mismatch": max(finite_mismatch) if finite_mismatch else math.nan,
            }
        )

        if not args.skip_lightsim:
            work = copy.deepcopy(net)
            start = _now()
            try:
                ppcont.run_contingency_ls2g(work, nminus1, **PF_KWARGS, lightsim2grid=False)
                runtime = _elapsed(start)
                rows.append(
                    {
                        "case": case,
                        "backend": "lightsim2grid_run_contingency_ls2g",
                        "scope": "complete_n1_ac_wall",
                        **counts,
                        "n_contingencies": n_cont,
                        "total_wall_s": runtime,
                        "s_per_contingency": runtime / n_cont,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case,
                        "backend": "lightsim2grid_run_contingency_ls2g",
                        "scope": "complete_n1_ac_wall",
                        **counts,
                        "n_contingencies": n_cont,
                        "error": repr(exc),
                    }
                )

            try:
                profile = _ls2g_direct_profile(net, nminus1, args)
                rows.append(
                    {
                        "case": case,
                        "backend": "lightsim2grid_cpp_direct",
                        "scope": "complete_n1_ac_compute_only",
                        **counts,
                        "n_contingencies": n_cont,
                        "s_per_contingency": profile["ls2g_total_wall_s"] / n_cont,
                        **profile,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case,
                        "backend": "lightsim2grid_cpp_direct",
                        "scope": "complete_n1_ac_compute_only",
                        **counts,
                        "n_contingencies": n_cont,
                        "error": repr(exc),
                    }
                )

        if not args.skip_pandapower_contingency:
            work = copy.deepcopy(net)
            start = _now()
            try:
                ppcont.run_contingency(
                    work,
                    nminus1,
                    pf_options={**PF_KWARGS, "lightsim2grid": False},
                    pf_options_nminus1={**PF_KWARGS, "lightsim2grid": False},
                    write_to_net=True,
                )
                runtime = _elapsed(start)
                rows.append(
                    {
                        "case": case,
                        "backend": "pandapower_run_contingency",
                        "scope": "complete_n1_ac_wall",
                        **counts,
                        "n_contingencies": n_cont,
                        "total_wall_s": runtime,
                        "s_per_contingency": runtime / n_cont,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "case": case,
                        "backend": "pandapower_run_contingency",
                        "scope": "complete_n1_ac_wall",
                        **counts,
                        "n_contingencies": n_cont,
                        "error": repr(exc),
                    }
                )
    return rows


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(key, "")) for key in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "security", "all"], default="all")
    parser.add_argument("--base-cases", nargs="+", default=["case118", "case300", "case1354pegase", "case6470rte", "case9241pegase"])
    parser.add_argument("--security-cases", nargs="+", default=["case118", "case300", "case6470rte", "case9241pegase"])
    parser.add_argument("--elements", nargs="+", default=["line", "trafo"])
    parser.add_argument("--max-contingencies", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--skip-lightsim", action="store_true")
    parser.add_argument("--skip-pandapower-contingency", action="store_true")
    parser.add_argument("--out-dir", default="benchmark_results/backend_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    if args.mode in {"base", "all"}:
        rows = benchmark_base(args.base_cases, args)
        _write_csv(out_dir / "base.csv", rows)
        write_markdown(out_dir / "base.md", "Base AC Backend Comparison", rows)
        for row in rows:
            print(row, flush=True)
    if args.mode in {"security", "all"}:
        rows = benchmark_security(args.security_cases, args)
        _write_csv(out_dir / "security.csv", rows)
        write_markdown(out_dir / "security.md", "N-1 Security Backend Comparison", rows)
        for row in rows:
            print(row, flush=True)


if __name__ == "__main__":
    main()
