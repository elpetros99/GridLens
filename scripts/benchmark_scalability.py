from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import jax.numpy as jnp

from batched_powerflow import BatchedPowerFlowSolver, NewtonOptions, from_pandapower
from batched_powerflow.cpu_solver import CpuNewtonOptions, solve_batch_cpu_sparse
from batched_powerflow.contingency import (
    ContingencyAnalysis,
    apply_contingency,
    build_contingency_batch_data,
    enumerate_nk,
)
from batched_powerflow.dc_screening import iter_nk_contingencies, screen_top_contingencies
from batched_powerflow.model import TopologyBatchResult
from batched_powerflow.pandapower_adapter import (
    PandapowerBuildOptions,
    make_random_load_sbus_batch,
)
from batched_powerflow.solver import _compile_topology_solver, compute_branch_flows


DEFAULT_CASES = [
    "case118",
    "case300",
    "case1354pegase",
    "case1888rte",
    "case2848rte",
    "case2869pegase",
    "case3120sp",
    "case6470rte",
    "case6495rte",
    "case6515rte",
    "case9241pegase",
]


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def sync_result(result):
    # Conversion to numpy already blocks JAX execution in the public API.
    return result


def load_case(name: str):
    return getattr(pn, name)()


def network_counts(net) -> dict[str, int]:
    return {
        "n_bus": int(len(net.bus)),
        "n_line": int(len(net.line)) if hasattr(net, "line") else 0,
        "n_trafo": int(len(net.trafo)) if hasattr(net, "trafo") else 0,
        "n_gen": int(len(net.gen)) if hasattr(net, "gen") else 0,
        "n_load": int(len(net.load)) if hasattr(net, "load") else 0,
        "n_ext_grid": int(len(net.ext_grid)) if hasattr(net, "ext_grid") else 0,
    }


def pandapower_reference(net) -> tuple[bool, float]:
    work = copy.deepcopy(net)
    start = now()
    try:
        pp.runpp(
            work,
            algorithm="nr",
            init="flat",
            calculate_voltage_angles=True,
            tolerance_mva=1e-8,
            max_iteration=30,
            enforce_q_lims=False,
            voltage_depend_loads=False,
            numba=False,
        )
        return bool(work.converged), elapsed(start)
    except Exception:
        return False, elapsed(start)


def benchmark_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    net = load_case(name)
    row: dict[str, Any] = {"case": name, **network_counts(net)}

    if args.reference_pandapower:
        converged, runtime = pandapower_reference(net)
        row["pandapower_converged"] = converged
        row["pandapower_s"] = runtime

    build_options = PandapowerBuildOptions(
        check_connectivity=args.check_connectivity,
        preconditioner=args.preconditioner,
        max_dense_preconditioner_state=args.max_dense_preconditioner_state,
        run_powerflow=args.run_powerflow_init,
    )
    solver_options = NewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        gmres_tol=args.gmres_tol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )

    start = now()
    try:
        model = from_pandapower(net, name=name, options=build_options)
        row["build_ok"] = True
        row["build_s"] = elapsed(start)
    except Exception as exc:
        row.update({"build_ok": False, "build_s": elapsed(start), "error": repr(exc)})
        return row

    row.update(
        {
            "n_state": model.n_state,
            "nnz_ybus": int(len(model.ybus_data)),
            "preconditioner": "fast_decoupled_diagonal" if model.bp_inv.ndim == 1 else ("fast_decoupled_dense" if model.bp_inv.size else "none"),
        }
    )

    if args.backend == "jax":
        solver = BatchedPowerFlowSolver(model, solver_options)

        start = now()
        try:
            first = sync_result(solver.solve())
            row["first_solve_ok"] = True
            row["first_solve_s"] = elapsed(start)
            row["first_converged"] = bool(first.converged[0])
            row["first_mismatch"] = float(first.mismatch_norm[0])
        except Exception as exc:
            row.update({"first_solve_ok": False, "first_solve_s": elapsed(start), "error": repr(exc)})
            return row

        start = now()
        try:
            warm = sync_result(solver.solve())
            row["warm_solve_ok"] = True
            row["warm_solve_s"] = elapsed(start)
            row["warm_converged"] = bool(warm.converged[0])
            row["warm_mismatch"] = float(warm.mismatch_norm[0])
        except Exception as exc:
            row.update({"warm_solve_ok": False, "warm_solve_s": elapsed(start), "error": repr(exc)})
    else:
        cpu_options = CpuNewtonOptions(
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            linear_solver=args.cpu_linear_solver,
            gmres_tol=args.gmres_tol,
            gmres_restart=args.gmres_restart,
            gmres_maxiter=args.gmres_maxiter,
            start=args.cpu_start,
        )
        start = now()
        try:
            first = solve_batch_cpu_sparse(model, options=cpu_options)
            row["first_solve_ok"] = True
            row["first_solve_s"] = elapsed(start)
            row["first_converged"] = bool(first.converged[0])
            row["first_mismatch"] = float(first.mismatch_norm[0])
            row["warm_solve_ok"] = None
            row["warm_solve_s"] = None
            row["warm_converged"] = None
            row["warm_mismatch"] = None
        except Exception as exc:
            row.update({"first_solve_ok": False, "first_solve_s": elapsed(start), "error": repr(exc)})

    return row


def benchmark_batch_scaling(case: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    net = load_case(case)
    build_options = PandapowerBuildOptions(
        check_connectivity=args.check_connectivity,
        preconditioner=args.preconditioner,
        max_dense_preconditioner_state=args.max_dense_preconditioner_state,
        run_powerflow=args.run_powerflow_init,
    )
    solver_options = NewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        gmres_tol=args.gmres_tol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )
    model = from_pandapower(net, name=case, options=build_options)
    rows = []
    for batch_size in args.batch_sizes:
        start = now()
        sbus_batch, _ = make_random_load_sbus_batch(
            net,
            n_scenarios=batch_size,
            low=args.load_low,
            high=args.load_high,
            seed=args.seed,
            options=build_options,
        )
        scenario_build_s = elapsed(start)
        if args.backend == "jax":
            solver = BatchedPowerFlowSolver(model, solver_options)
            start = now()
            first = sync_result(solver.solve(sbus_batch))
            first_s = elapsed(start)
            start = now()
            warm = sync_result(solver.solve(sbus_batch))
            warm_s = elapsed(start)
        else:
            cpu_options = CpuNewtonOptions(
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                linear_solver=args.cpu_linear_solver,
                gmres_tol=args.gmres_tol,
                gmres_restart=args.gmres_restart,
                gmres_maxiter=args.gmres_maxiter,
                start=args.cpu_start,
            )
            start = now()
            first = solve_batch_cpu_sparse(model, sbus_batch, options=cpu_options)
            first_s = elapsed(start)
            warm = first
            warm_s = first_s

        rows.append(
            {
                "case": case,
                "batch_size": batch_size,
                "n_bus": model.n_bus,
                "n_state": model.n_state,
                "nnz_ybus": int(len(model.ybus_data)),
                "scenario_build_s": scenario_build_s,
                "first_solve_s": first_s,
                "warm_solve_s": warm_s,
                "warm_s_per_pf": warm_s / batch_size,
                "first_converged_rate": float(np.mean(first.converged)),
                "warm_converged_rate": float(np.mean(warm.converged)),
                "warm_max_mismatch": float(np.max(warm.mismatch_norm)),
                "preconditioner": "fast_decoupled_diagonal" if model.bp_inv.ndim == 1 else ("fast_decoupled_dense" if model.bp_inv.size else "none"),
                "backend": args.backend,
            }
        )
    return rows


def _timed_ac_contingency_workflow(
    net,
    contingencies,
    *,
    build_options: PandapowerBuildOptions,
    solver_options: NewtonOptions,
) -> dict[str, Any]:
    """Time AC contingency phases without changing the solve or diagnostics math."""

    contingencies = tuple(contingencies)
    start = now()
    batch = build_contingency_batch_data(
        net,
        contingencies,
        options=build_options,
    )
    ac_build_s = elapsed(start)

    sbus_batch = np.broadcast_to(
        batch.template.sbus0[None, None, :],
        (len(batch.ybus_data_batch), 1, batch.template.n_bus),
    )
    solve_topologies = _compile_topology_solver(batch.template, solver_options)

    def solve_once() -> TopologyBatchResult:
        voltages, converged, mismatch_norm = solve_topologies(
            jnp.asarray(sbus_batch),
            jnp.asarray(batch.ybus_data_batch),
            jnp.asarray(batch.bp_inv_batch),
            jnp.asarray(batch.bpp_inv_batch),
            jnp.asarray(batch.g_qtheta_batch),
            jnp.asarray(batch.vm0_batch),
            jnp.asarray(batch.va0_batch),
        )
        return TopologyBatchResult(
            contingencies=tuple(batch.contingencies),
            voltages=np.asarray(voltages),
            converged=np.asarray(converged),
            mismatch_norm=np.asarray(mismatch_norm),
            iterations=solver_options.max_iterations,
        )

    start = now()
    first_result = solve_once()
    first_ac_solve_s = elapsed(start)

    start = now()
    warm_result = solve_once()
    warm_ac_solve_s = elapsed(start)

    start = now()
    models = [from_pandapower(apply_contingency(net, c), name=str(c), options=build_options) for c in contingencies]

    from pandapower.pypower.idx_brch import RATE_A
    from pandapower.pypower.idx_bus import VMAX, VMIN

    s_from = []
    s_to = []
    loading = []
    for i, model in enumerate(models):
        sf, st = compute_branch_flows(model, first_result.voltages[i])
        s_from.append(sf)
        s_to.append(st)
        rate_a = model.ppc["branch"][:, RATE_A].astype(float)
        branch_mva = np.maximum(np.abs(sf), np.abs(st))
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(branch_mva * 100.0, rate_a[None, :], out=np.zeros_like(branch_mva.real), where=rate_a[None, :] > 0)
        loading.append(pct)

    template_model = models[0]
    vmin = template_model.ppc["bus"][:, VMIN].astype(float)
    vmax = template_model.ppc["bus"][:, VMAX].astype(float)
    loading_percent = np.stack(loading)
    analysis = ContingencyAnalysis(
        result=first_result,
        s_from_mva=np.stack(s_from),
        s_to_mva=np.stack(s_to),
        loading_percent=loading_percent,
        voltage_low=first_result.vm < vmin[None, None, :],
        voltage_high=first_result.vm > vmax[None, None, :],
        branch_overloaded=loading_percent > 100.0,
    )
    diagnostics_flows_s = elapsed(start)

    return {
        "analysis": analysis,
        "warm_result": warm_result,
        "ac_build_s": ac_build_s,
        "first_ac_solve_s": first_ac_solve_s,
        "jax_compile_first_ac_solve_s": first_ac_solve_s,
        "warm_ac_solve_s": warm_ac_solve_s,
        "diagnostics_flows_s": diagnostics_flows_s,
    }


def benchmark_contingency(case: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    net = load_case(case)
    build_options = PandapowerBuildOptions(
        check_connectivity=False,
        preconditioner=args.preconditioner,
        max_dense_preconditioner_state=args.max_dense_preconditioner_state,
        run_powerflow=args.run_powerflow_init,
    )
    solver_options = NewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        gmres_tol=args.gmres_tol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )
    rows = []
    for k in args.contingency_k:
        contingencies = enumerate_nk(net, k=k, elements=tuple(args.contingency_elements), max_cases=args.max_contingencies)
        if not contingencies:
            continue
        start = now()
        try:
            timed = _timed_ac_contingency_workflow(
                net,
                contingencies,
                build_options=build_options,
                solver_options=solver_options,
            )
            runtime = elapsed(start)
            analysis = timed["analysis"]
            rows.append(
                {
                    "case": case,
                    "k": k,
                    "n_contingencies": len(contingencies),
                    "selected_for_ac": len(contingencies),
                    "dc_screening_s": 0.0,
                    "selected_ac_build_s": timed["ac_build_s"],
                    "first_ac_solve_s": timed["first_ac_solve_s"],
                    "jax_compile_first_ac_solve_s": timed["jax_compile_first_ac_solve_s"],
                    "warm_ac_solve_s": timed["warm_ac_solve_s"],
                    "diagnostics_flows_s": timed["diagnostics_flows_s"],
                    "total_s": runtime,
                    "runtime_s": runtime,
                    "s_per_pf": runtime / len(contingencies),
                    "converged_rate": float(np.mean(analysis.result.converged)),
                    "max_mismatch": float(np.max(analysis.result.mismatch_norm)),
                    "warm_converged_rate": float(np.mean(timed["warm_result"].converged)),
                    "warm_max_mismatch": float(np.max(timed["warm_result"].mismatch_norm)),
                    "any_voltage_low": bool(np.any(analysis.voltage_low)),
                    "any_voltage_high": bool(np.any(analysis.voltage_high)),
                    "any_branch_overloaded": bool(np.any(analysis.branch_overloaded)),
                }
            )
        except Exception as exc:
            runtime = elapsed(start)
            rows.append(
                {
                    "case": case,
                    "k": k,
                    "n_contingencies": len(contingencies),
                    "selected_for_ac": len(contingencies),
                    "dc_screening_s": 0.0,
                    "runtime_s": runtime,
                    "total_s": runtime,
                    "error": repr(exc),
                }
            )
    return rows


def benchmark_hierarchical_contingency(case: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    net = load_case(case)
    screen_options = PandapowerBuildOptions(
        check_connectivity=False,
        preconditioner=args.preconditioner,
        max_dense_preconditioner_state=args.max_dense_preconditioner_state,
    )
    ac_build_options = PandapowerBuildOptions(
        check_connectivity=False,
        preconditioner=args.preconditioner,
        max_dense_preconditioner_state=args.max_dense_preconditioner_state,
        run_powerflow=args.run_powerflow_init,
    )
    solver_options = NewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        gmres_tol=args.gmres_tol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )
    rows = []
    for k in args.contingency_k:
        total = sum(1 for _ in iter_nk_contingencies(net, k=k, elements=tuple(args.contingency_elements), max_cases=args.max_contingencies))
        if total == 0:
            continue

        start = now()
        screened = screen_top_contingencies(
            net,
            k=k,
            top_n=min(args.top_contingencies, total),
            elements=tuple(args.contingency_elements),
            max_cases=args.max_contingencies,
            chunk_size=args.screening_chunk_size,
            options=screen_options,
            include_failed=False,
        )
        screening_s = elapsed(start)

        selected = tuple(result.contingency for result in screened if not result.failed)
        if not selected:
            rows.append(
                {
                    "case": case,
                    "k": k,
                    "total_screened": total,
                    "selected_for_ac": 0,
                    "screening_s": screening_s,
                    "dc_screening_s": screening_s,
                    "selected_ac_build_s": 0.0,
                    "first_ac_solve_s": 0.0,
                    "jax_compile_first_ac_solve_s": 0.0,
                    "warm_ac_solve_s": 0.0,
                    "diagnostics_flows_s": 0.0,
                    "ac_s": 0.0,
                    "total_s": screening_s,
                    "screening_s_per_case": screening_s / total,
                    "dc_failures_selected": int(sum(result.failed for result in screened)),
                    "error": "all selected contingencies failed DC screening",
                }
            )
            continue
        start = now()
        try:
            timed = _timed_ac_contingency_workflow(
                net,
                selected,
                build_options=ac_build_options,
                solver_options=solver_options,
            )
            ac_s = elapsed(start)
            total_s = screening_s + ac_s
            analysis = timed["analysis"]
            rows.append(
                {
                    "case": case,
                    "k": k,
                    "total_screened": total,
                    "selected_for_ac": len(selected),
                    "screening_s": screening_s,
                    "dc_screening_s": screening_s,
                    "selected_ac_build_s": timed["ac_build_s"],
                    "first_ac_solve_s": timed["first_ac_solve_s"],
                    "jax_compile_first_ac_solve_s": timed["jax_compile_first_ac_solve_s"],
                    "warm_ac_solve_s": timed["warm_ac_solve_s"],
                    "diagnostics_flows_s": timed["diagnostics_flows_s"],
                    "ac_s": ac_s,
                    "total_s": total_s,
                    "screening_s_per_case": screening_s / total,
                    "total_s_per_screened_case": total_s / total,
                    "ac_converged_rate": float(np.mean(analysis.result.converged)),
                    "ac_max_mismatch": float(np.max(analysis.result.mismatch_norm)),
                    "warm_ac_converged_rate": float(np.mean(timed["warm_result"].converged)),
                    "warm_ac_max_mismatch": float(np.max(timed["warm_result"].mismatch_norm)),
                    "max_dc_score": float(max(result.score for result in screened)) if screened else 0.0,
                    "dc_failures_selected": int(sum(result.failed for result in screened)),
                    "any_voltage_low": bool(np.any(analysis.voltage_low)),
                    "any_voltage_high": bool(np.any(analysis.voltage_high)),
                    "any_branch_overloaded": bool(np.any(analysis.branch_overloaded)),
                }
            )
        except Exception as exc:
            ac_s = elapsed(start)
            rows.append(
                {
                    "case": case,
                    "k": k,
                    "total_screened": total,
                    "selected_for_ac": len(selected),
                    "screening_s": screening_s,
                    "dc_screening_s": screening_s,
                    "ac_s": ac_s,
                    "total_s": screening_s + ac_s,
                    "error": repr(exc),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(key, "")) for key in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cases", "batch", "contingency", "hierarchical", "all"], default="all")
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--batch-case", default="case118")
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--contingency-case", default="case118")
    parser.add_argument("--contingency-k", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--contingency-elements", nargs="*", default=["line"])
    parser.add_argument("--max-contingencies", type=int, default=32)
    parser.add_argument("--top-contingencies", type=int, default=8)
    parser.add_argument("--screening-chunk-size", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--gmres-tol", type=float, default=1e-8)
    parser.add_argument("--gmres-restart", type=int, default=30)
    parser.add_argument("--gmres-maxiter", type=int, default=30)
    parser.add_argument("--preconditioner", choices=["auto", "fast_decoupled_dense", "fast_decoupled_diagonal", "none"], default="auto")
    parser.add_argument("--backend", choices=["jax", "cpu_sparse"], default="jax")
    parser.add_argument("--cpu-linear-solver", choices=["direct", "gmres_ilu"], default="direct")
    parser.add_argument("--cpu-start", choices=["model", "pandapower_result"], default="model")
    parser.add_argument("--run-powerflow-init", action="store_true")
    parser.add_argument("--max-dense-preconditioner-state", type=int, default=1500)
    parser.add_argument("--check-connectivity", action="store_true")
    parser.add_argument("--reference-pandapower", action="store_true")
    parser.add_argument("--load-low", type=float, default=0.8)
    parser.add_argument("--load-high", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="benchmark_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    metadata = {
        "args": vars(args),
        "python": sys.version,
    }
    (out_dir / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.mode in {"cases", "all"}:
        rows = [benchmark_case(case, args) for case in args.cases]
        write_rows(out_dir / "case_scalability.csv", rows)
        write_markdown(out_dir / "case_scalability.md", "Case Scalability", rows)

    if args.mode in {"batch", "all"}:
        rows = benchmark_batch_scaling(args.batch_case, args)
        write_rows(out_dir / "batch_scaling.csv", rows)
        write_markdown(out_dir / "batch_scaling.md", "Batch Scaling", rows)

    if args.mode in {"contingency", "all"}:
        rows = benchmark_contingency(args.contingency_case, args)
        write_rows(out_dir / "contingency_scaling.csv", rows)
        write_markdown(out_dir / "contingency_scaling.md", "Contingency Scaling", rows)

    if args.mode in {"hierarchical", "all"}:
        rows = benchmark_hierarchical_contingency(args.contingency_case, args)
        write_rows(out_dir / "hierarchical_contingency.csv", rows)
        write_markdown(out_dir / "hierarchical_contingency.md", "Hierarchical Contingency", rows)


if __name__ == "__main__":
    main()
