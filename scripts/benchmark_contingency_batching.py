"""Profile CPU N-1 batching and the GPU-compatible JAX topology path.

Examples
--------
python scripts/benchmark_contingency_batching.py --cases case14 case30 case57 case118
python scripts/benchmark_contingency_batching.py --cases case300 --max-contingencies 64 --include-jax
"""

from __future__ import annotations

import argparse
import csv
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandapower.networks as pn

from batched_powerflow.contingency import (
    build_contingency_batch_data,
    enumerate_nk,
)
from batched_powerflow.cpu_solver import CpuNewtonOptions, solve_batch_cpu_sparse
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions, from_pandapower
from batched_powerflow.solver import NewtonOptions, solve_topology_batch


DEFAULT_CASES = ["case14", "case30", "case57", "case118", "case300"]
BUILD = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def load_case(name: str):
    try:
        return getattr(pn, name)()
    except AttributeError as exc:
        raise SystemExit(f"Unknown pandapower network {name!r}.") from exc


def base_start(net) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    model = from_pandapower(net, options=BUILD)
    build_s = elapsed(started)
    started = time.perf_counter()
    result = solve_batch_cpu_sparse(
        model,
        options=CpuNewtonOptions(max_iterations=30, tolerance=1e-10, jacobian_reuse=1),
    )
    solve_s = elapsed(started)
    if not bool(result.converged[0]):
        raise RuntimeError(f"Base case did not converge for {len(net.bus)} buses.")
    return result.voltages[0], {
        "base_build_s": build_s,
        "base_solve_s": solve_s,
        "base_mismatch": float(result.mismatch_norm[0]),
    }


def network_counts(net) -> dict[str, int]:
    return {
        "buses": int(len(net.bus)),
        "lines": int(len(net.line)) if hasattr(net, "line") else 0,
        "trafos": int(len(net.trafo)) if hasattr(net, "trafo") else 0,
    }


def result_row(
    case: str,
    net,
    contingencies,
    mode: str,
    batch_size: int,
    build_s: float,
    solve_s: float,
    converged: int,
    mismatch: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "case": case,
        **network_counts(net),
        "contingencies": len(contingencies),
        "mode": mode,
        "batch_size": batch_size,
        "build_s": build_s,
        "solve_s": solve_s,
        "total_s": build_s + solve_s,
        "converged": converged,
        "mismatch": mismatch,
    }
    if extra:
        row.update(extra)
    return row


def run_cpu_sequential(net, contingencies, base_voltage, *, warm: bool, jacobian_reuse: int) -> dict[str, Any]:
    started = time.perf_counter()
    build_s = 0.0
    solve_s = 0.0
    converged = 0
    mismatch = 0.0
    batch_fallbacks = 0
    options = CpuNewtonOptions(max_iterations=30, tolerance=1e-10, jacobian_reuse=jacobian_reuse)
    for contingency in contingencies:
        t0 = time.perf_counter()
        try:
            batch = build_contingency_batch_data(net, [contingency], options=BUILD)
        except Exception:
            build_s += elapsed(t0)
            mismatch = float("inf")
            continue
        build_s += elapsed(t0)
        t0 = time.perf_counter()
        starts = None if not warm else base_voltage[None, :]
        try:
            result = solve_batch_cpu_sparse(batch.models[0], options=options, initial_voltages=starts)
        except Exception:
            result = None
        solve_s += elapsed(t0)
        if result is None:
            mismatch = float("inf")
        else:
            converged += int(result.converged[0])
            mismatch = max(mismatch, float(result.mismatch_norm[0]))
    return {
        "build_s": build_s,
        "solve_s": solve_s,
        "converged": converged,
        "mismatch": mismatch,
        "batch_fallbacks": batch_fallbacks,
        "wall_s": elapsed(started),
    }


def run_cpu_batched(net, contingencies, base_voltage, *, batch_size: int, jacobian_reuse: int) -> dict[str, Any]:
    started = time.perf_counter()
    build_s = 0.0
    solve_s = 0.0
    converged = 0
    mismatch = 0.0
    batch_fallbacks = 0
    options = CpuNewtonOptions(max_iterations=30, tolerance=1e-10, jacobian_reuse=jacobian_reuse)
    for offset in range(0, len(contingencies), batch_size):
        chunk = tuple(contingencies[offset : offset + batch_size])
        t0 = time.perf_counter()
        try:
            batch = build_contingency_batch_data(net, chunk, options=BUILD)
        except Exception:
            build_s += elapsed(t0)
            batch_fallbacks += len(chunk)
            for contingency in chunk:
                single_start = time.perf_counter()
                try:
                    single = build_contingency_batch_data(net, [contingency], options=BUILD)
                    build_s += elapsed(single_start)
                    solve_start = time.perf_counter()
                    result = solve_batch_cpu_sparse(
                        single.models[0],
                        options=options,
                        initial_voltages=base_voltage[None, :],
                    )
                    solve_s += elapsed(solve_start)
                    converged += int(result.converged[0])
                    mismatch = max(mismatch, float(result.mismatch_norm[0]))
                except Exception:
                    build_s += elapsed(single_start)
                    mismatch = float("inf")
            continue
        build_s += elapsed(t0)
        starts = np.broadcast_to(base_voltage, (len(chunk), len(base_voltage)))
        t0 = time.perf_counter()
        for index, model in enumerate(batch.models):
            try:
                result = solve_batch_cpu_sparse(model, options=options, initial_voltages=starts[index : index + 1])
            except Exception:
                result = None
            if result is None:
                mismatch = float("inf")
            else:
                converged += int(result.converged[0])
                mismatch = max(mismatch, float(result.mismatch_norm[0]))
        solve_s += elapsed(t0)
    return {
        "build_s": build_s,
        "solve_s": solve_s,
        "converged": converged,
        "mismatch": mismatch,
        "batch_fallbacks": batch_fallbacks,
        "wall_s": elapsed(started),
    }


def run_jax_topology(net, contingencies, base_voltage, *, batch_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    build_s = 0.0
    solve_s = 0.0
    converged = 0
    mismatch = 0.0
    batch_fallbacks = 0
    options = NewtonOptions(max_iterations=12, tolerance=1e-10)
    for offset in range(0, len(contingencies), batch_size):
        chunk = tuple(contingencies[offset : offset + batch_size])
        t0 = time.perf_counter()
        try:
            batch = build_contingency_batch_data(net, chunk, options=BUILD)
        except Exception:
            build_s += elapsed(t0)
            batch_fallbacks += len(chunk)
            mismatch = float("inf")
            continue
        build_s += elapsed(t0)
        starts = np.broadcast_to(base_voltage, (len(chunk), len(base_voltage)))
        t0 = time.perf_counter()
        try:
            result = solve_topology_batch(
                batch.template,
                ybus_data_batch=batch.ybus_data_batch,
                bp_inv_batch=batch.bp_inv_batch,
                bpp_inv_batch=batch.bpp_inv_batch,
                g_qtheta_batch=batch.g_qtheta_batch,
                vm0_batch=np.abs(starts),
                va0_batch=np.angle(starts),
                sbus_batch=batch.template.sbus0[None, :],
                contingencies=chunk,
                options=options,
            )
        except Exception:
            batch_fallbacks += len(chunk)
            mismatch = float("inf")
            continue
        solve_s += elapsed(t0)
        converged += int(np.count_nonzero(result.converged[:, 0]))
        mismatch = max(mismatch, float(np.nanmax(result.mismatch_norm)))
    return {
        "build_s": build_s,
        "solve_s": solve_s,
        "converged": converged,
        "mismatch": mismatch,
        "batch_fallbacks": batch_fallbacks,
        "wall_s": elapsed(started),
    }


def benchmark_case(case: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    net = load_case(case)
    contingencies = list(enumerate_nk(net, k=1, elements=("line", "trafo"), max_cases=args.max_contingencies))
    if not contingencies:
        return []
    base_voltage, base_metadata = base_start(net)
    rows = []
    for warm, mode in ((False, "cpu_sequential_cold"), (True, "cpu_sequential_warm")):
        values = run_cpu_sequential(net, contingencies, base_voltage, warm=warm, jacobian_reuse=1)
        rows.append(result_row(case, net, contingencies, mode, 1, values["build_s"], values["solve_s"], values["converged"], values["mismatch"], {**base_metadata, "batch_fallbacks": values["batch_fallbacks"]}))
    for batch_size in args.batch_sizes:
        values = run_cpu_batched(net, contingencies, base_voltage, batch_size=batch_size, jacobian_reuse=1)
        rows.append(result_row(case, net, contingencies, "cpu_batch_warm_superlu", batch_size, values["build_s"], values["solve_s"], values["converged"], values["mismatch"], {**base_metadata, "batch_fallbacks": values["batch_fallbacks"]}))
        values = run_cpu_batched(net, contingencies, base_voltage, batch_size=batch_size, jacobian_reuse=2)
        rows.append(result_row(case, net, contingencies, "cpu_batch_warm_superlu_jacobian2", batch_size, values["build_s"], values["solve_s"], values["converged"], values["mismatch"], {**base_metadata, "batch_fallbacks": values["batch_fallbacks"]}))
        if args.include_jax:
            values = run_jax_topology(net, contingencies, base_voltage, batch_size=batch_size)
            rows.append(result_row(case, net, contingencies, "jax_topology_cpu", batch_size, values["build_s"], values["solve_s"], values["converged"], values["mismatch"], {**base_metadata, "batch_fallbacks": values["batch_fallbacks"]}))
    return rows


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path, markdown_path: Path | None):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Contingency batching benchmark",
            "",
            "CPU rows use SciPy SuperLU (`splu`) and explicit sparse Newton. `jacobian2` reuses a numerical Jacobian/factorization for two Newton steps. JAX rows run on the detected JAX device and are included only with `--include-jax`.",
            "",
        ]
        if rows:
            columns = ["case", "buses", "contingencies", "mode", "batch_size", "build_s", "solve_s", "total_s", "converged", "batch_fallbacks", "mismatch"]
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("|" + "|".join("---" for _ in columns) + "|")
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(column, "")) if not isinstance(row.get(column), float) else f"{row[column]:.6g}" for column in columns) + " |")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile_call(call: Callable[[], Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    call()
    profiler.disable()
    profiler.dump_stats(str(output))
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--max-contingencies", type=int, default=32)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--include-jax", action="store_true", help="Also benchmark the JAX topology path on the available device.")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results" / "contingency_batching.csv")
    parser.add_argument("--json-output", type=Path, default=ROOT / "benchmark_results" / "contingency_batching.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "benchmark_results" / "contingency_batching.md")
    parser.add_argument("--profile-case", type=str, default=None, help="Write a cProfile dump for this case's benchmark.")
    parser.add_argument("--profile-output", type=Path, default=ROOT / "benchmark_results" / "contingency_batching.prof")
    args = parser.parse_args()
    if any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.max_contingencies < 1:
        parser.error("--max-contingencies must be positive")

    import jax

    print(f"JAX devices: {jax.devices()}")
    rows: list[dict[str, Any]] = []
    for case in args.cases:
        print(f"Benchmarking {case} ...", flush=True)
        if args.profile_case == case:
            case_rows: list[dict[str, Any]] = []
            profile_call(lambda: case_rows.extend(benchmark_case(case, args)), args.profile_output)
        else:
            case_rows = benchmark_case(case, args)
        rows.extend(case_rows)
    write_outputs(rows, args.output, args.json_output, args.markdown_output)
    for row in rows:
        print(f"{row['case']:>16} {row['mode']:<34} b={row['batch_size']:<3} total={row['total_s']:.4f}s converged={row['converged']}/{row['contingencies']} fallback={row['batch_fallbacks']} mismatch={row['mismatch']:.3e}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
