"""Benchmark batched_powerflow on named pandapower cases.

Examples
--------
Run base-case benchmarks:

    python benchmarks/benchmark_powerflow.py --cases case14 case33bw

Run batch-scaling benchmarks with random load scenarios:

    python benchmarks/benchmark_powerflow.py --mode batch-scaling --cases case14 --batch-sizes 1 8 64

The script assumes it is executed from an environment with JAX, pandapower, and
batched_powerflow installed, such as the project's JAX conda environment. When
run from a source checkout, it also adds ``src/`` to ``sys.path``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandapower.networks as pn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp

from batched_powerflow import NewtonOptions, from_pandapower
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions, make_random_load_sbus_batch
from batched_powerflow.solver import _compile_solver


DEFAULT_CASES = ("case14", "case33bw")
DEFAULT_BATCH_SIZES = (1, 8, 32, 128)


@dataclass(frozen=True)
class BenchmarkRow:
    timestamp_utc: str
    mode: str
    case: str
    batch_size: int
    n_bus: int
    n_state: int
    nnz_ybus: int
    build_time_s: float
    first_solve_compile_s: float
    warm_solve_s: float
    convergence_rate: float
    max_mismatch: float
    max_iterations: int
    tolerance: float
    seed: int
    load_low: float
    load_high: float
    per_load: bool
    jax_platform: str
    python: str


def _available_case_names() -> list[str]:
    names = []
    for name in dir(pn):
        if name.startswith("_"):
            continue
        value = getattr(pn, name)
        if callable(value) and (name.startswith("case") or name.startswith("simple")):
            names.append(name)
    return sorted(names)


def _case_factory(name: str) -> Callable:
    factory = getattr(pn, name, None)
    if not callable(factory):
        available = ", ".join(_available_case_names())
        raise ValueError(f"Unknown pandapower case '{name}'. Available cases include: {available}")
    return factory


def _time_call(fn):
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    return value, elapsed


def _solve_with_callable(solve_many, model, sbus_batch):
    if sbus_batch is None:
        sbus = model.sbus0[None, :]
    else:
        sbus = np.asarray(sbus_batch, dtype=np.complex128)
        if sbus.ndim == 1:
            sbus = sbus[None, :]
    voltages, converged, mismatch_norm, _ = solve_many(
        jnp.asarray(sbus),
        jnp.asarray(model.ybus_data),
        jnp.asarray(model.bp_inv),
        jnp.asarray(model.bpp_inv),
        jnp.asarray(model.g_qtheta),
    )
    return np.asarray(voltages), np.asarray(converged), np.asarray(mismatch_norm)


def _jax_platform() -> str:
    try:
        import jax

        return jax.default_backend()
    except Exception as exc:  # pragma: no cover - diagnostic fallback only.
        return f"unavailable: {exc}"


def _scenario_batch(
    net,
    *,
    model,
    mode: str,
    batch_size: int,
    seed: int,
    low: float,
    high: float,
    per_load: bool,
    build_options: PandapowerBuildOptions,
) -> np.ndarray | None:
    if mode == "base":
        if batch_size == 1:
            return None
        return np.broadcast_to(model.sbus0, (batch_size, model.n_bus)).copy()
    if mode == "batch-scaling":
        sbus_batch, _ = make_random_load_sbus_batch(
            net,
            n_scenarios=batch_size,
            low=low,
            high=high,
            seed=seed,
            per_load=per_load,
            options=build_options,
        )
        return sbus_batch
    raise ValueError(f"Unsupported mode: {mode}")


def run_one(
    *,
    case_name: str,
    mode: str,
    batch_size: int,
    timestamp_utc: str,
    solver_options: NewtonOptions,
    build_options: PandapowerBuildOptions,
    seed: int,
    low: float,
    high: float,
    per_load: bool,
) -> BenchmarkRow:
    factory = _case_factory(case_name)
    net = factory()

    model, build_time_s = _time_call(lambda: from_pandapower(net, name=case_name, options=build_options))
    sbus_batch = _scenario_batch(
        net,
        model=model,
        mode=mode,
        batch_size=batch_size,
        seed=seed,
        low=low,
        high=high,
        per_load=per_load,
        build_options=build_options,
    )

    solve_many = _compile_solver(model, solver_options)
    _, first_solve_compile_s = _time_call(lambda: _solve_with_callable(solve_many, model, sbus_batch))
    warm_result, warm_solve_s = _time_call(lambda: _solve_with_callable(solve_many, model, sbus_batch))

    _, warm_converged, warm_mismatch_norm = warm_result
    convergence_rate = float(np.mean(warm_converged)) if warm_converged.size else 0.0
    max_mismatch = float(np.max(warm_mismatch_norm)) if warm_mismatch_norm.size else float("nan")

    return BenchmarkRow(
        timestamp_utc=timestamp_utc,
        mode=mode,
        case=case_name,
        batch_size=batch_size,
        n_bus=model.n_bus,
        n_state=model.n_state,
        nnz_ybus=int(model.ybus_data.size),
        build_time_s=build_time_s,
        first_solve_compile_s=first_solve_compile_s,
        warm_solve_s=warm_solve_s,
        convergence_rate=convergence_rate,
        max_mismatch=max_mismatch,
        max_iterations=solver_options.max_iterations,
        tolerance=solver_options.tolerance,
        seed=seed,
        load_low=low,
        load_high=high,
        per_load=per_load,
        jax_platform=_jax_platform(),
        python=platform.python_version(),
    )


def _write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _format_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.{digits}g}"


def _write_markdown(path: Path, rows: list[BenchmarkRow], args: argparse.Namespace, csv_path: Path) -> None:
    headers = [
        "case",
        "mode",
        "batch",
        "n_bus",
        "n_state",
        "nnz_ybus",
        "build_s",
        "first_s",
        "warm_s",
        "conv_rate",
        "max_mismatch",
    ]
    lines = [
        "# batched_powerflow benchmark summary",
        "",
        f"- Generated UTC: `{rows[0].timestamp_utc}`",
        f"- CSV: `{csv_path.name}`",
        f"- Python: `{rows[0].python}`",
        f"- JAX backend: `{rows[0].jax_platform}`",
        f"- Solver options: max_iterations=`{args.max_iterations}`, tolerance=`{args.tolerance}`, gmres_tol=`{args.gmres_tol}`",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    row.case,
                    row.mode,
                    str(row.batch_size),
                    str(row.n_bus),
                    str(row.n_state),
                    str(row.nnz_ybus),
                    _format_float(row.build_time_s),
                    _format_float(row.first_solve_compile_s),
                    _format_float(row.warm_solve_s),
                    _format_float(row.convergence_rate),
                    _format_float(row.max_mismatch, digits=4),
                ]
            )
            + "|"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES), help="pandapower.networks case function names.")
    parser.add_argument(
        "--mode",
        choices=("base", "batch-scaling"),
        default="base",
        help="base repeats the original Sbus; batch-scaling creates random load scenarios.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load-low", type=float, default=0.8)
    parser.add_argument("--load-high", type=float, default=1.2)
    parser.add_argument("--per-load", action="store_true", help="Scale each load independently in batch-scaling mode.")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--gmres-tol", type=float, default=1e-11)
    parser.add_argument("--gmres-restart", type=int, default=40)
    parser.add_argument("--gmres-maxiter", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "benchmark_results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(size <= 0 for size in args.batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers.")

    timestamp_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    solver_options = NewtonOptions(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        gmres_tol=args.gmres_tol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )
    build_options = PandapowerBuildOptions(check_connectivity=False, regularization=1e-8)

    rows: list[BenchmarkRow] = []
    for case_name in args.cases:
        for batch_size in args.batch_sizes:
            row = run_one(
                case_name=case_name,
                mode=args.mode,
                batch_size=batch_size,
                timestamp_utc=timestamp_utc,
                solver_options=solver_options,
                build_options=build_options,
                seed=args.seed,
                low=args.load_low,
                high=args.load_high,
                per_load=args.per_load,
            )
            rows.append(row)
            print(
                f"{row.case} batch={row.batch_size}: "
                f"build={row.build_time_s:.4f}s first={row.first_solve_compile_s:.4f}s "
                f"warm={row.warm_solve_s:.4f}s conv={row.convergence_rate:.3f} "
                f"max_mismatch={row.max_mismatch:.3e}"
            )

    suffix = f"{args.mode}_{timestamp_utc}"
    csv_path = args.output_dir / f"benchmark_{suffix}.csv"
    md_path = args.output_dir / f"benchmark_{suffix}.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, args, csv_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
