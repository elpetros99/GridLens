"""Tutorial: solve many operating scenarios with CPU sparse Newton and JAX."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, NewtonOptions, from_pandapower, solve_batch, solve_batch_cpu_sparse
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions, make_random_load_sbus_batch


def main() -> None:
    net = pn.case14()
    build = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")
    model = from_pandapower(net, options=build)
    sbus_batch, _ = make_random_load_sbus_batch(net, n_scenarios=128, low=0.8, high=1.2, seed=7, options=build)
    print(f"JAX devices: {jax.devices()}")
    started = time.perf_counter()
    cpu = solve_batch_cpu_sparse(model, sbus_batch, options=CpuNewtonOptions(max_iterations=20, tolerance=1e-10))
    print(f"CPU SuperLU: {cpu.converged.sum()}/{len(cpu.converged)} converged in {time.perf_counter() - started:.4f}s")
    started = time.perf_counter()
    accelerated = solve_batch(model, sbus_batch, options=NewtonOptions(max_iterations=12, tolerance=1e-10))
    print(f"JAX batch:   {accelerated.converged.sum()}/{len(accelerated.converged)} converged in {time.perf_counter() - started:.4f}s")
    print(f"Voltage tensor shape: {accelerated.vm.shape} = (scenarios, buses)")


if __name__ == "__main__":
    main()
