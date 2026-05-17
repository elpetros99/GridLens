from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
from jax.scipy.sparse.linalg import gmres

from .model import PowerFlowModel, PowerFlowResult, TopologyBatchResult


@dataclass(frozen=True)
class NewtonOptions:
    max_iterations: int = 12
    tolerance: float = 1e-10
    gmres_tol: float = 1e-10
    gmres_atol: float = 0.0
    gmres_restart: int = 50
    gmres_maxiter: int | None = 50
    jit: bool = True


class BatchedPowerFlowSolver:
    """Reusable compiled solver for one model structure and option set."""

    def __init__(self, model: PowerFlowModel, options: NewtonOptions | None = None):
        self.model = model
        self.options = options or NewtonOptions()
        self._solve_many = _compile_solver(model, self.options)

    def solve(self, sbus_batch: np.ndarray | None = None) -> PowerFlowResult:
        sbus = _as_batch_sbus(self.model, sbus_batch)
        voltages, converged, mismatch_norm, _ = self._solve_many(
            jnp.asarray(sbus),
            jnp.asarray(self.model.ybus_data),
            jnp.asarray(self.model.bp_inv),
            jnp.asarray(self.model.bpp_inv),
            jnp.asarray(self.model.g_qtheta),
        )
        return PowerFlowResult(
            voltages=np.asarray(voltages),
            converged=np.asarray(converged),
            mismatch_norm=np.asarray(mismatch_norm),
            iterations=self.options.max_iterations,
        )


def _as_batch_sbus(model: PowerFlowModel, sbus_batch: np.ndarray | None) -> np.ndarray:
    if sbus_batch is None:
        sbus = model.sbus0[None, :]
    else:
        sbus = np.asarray(sbus_batch, dtype=np.complex128)
        if sbus.ndim == 1:
            sbus = sbus[None, :]
    if sbus.shape[1] != model.n_bus:
        raise ValueError(f"Sbus has {sbus.shape[1]} buses, expected {model.n_bus}.")
    return sbus


def _compile_solver(model: PowerFlowModel, options: NewtonOptions):
    n_bus = model.n_bus
    theta = jnp.asarray(model.theta, dtype=jnp.int32)
    pq = jnp.asarray(model.pq, dtype=jnp.int32)
    ybus_indices = jnp.asarray(model.ybus_indices, dtype=jnp.int32)
    vm0 = jnp.asarray(model.vm0, dtype=jnp.float64)
    va0 = jnp.asarray(model.va0, dtype=jnp.float64)
    n_theta = int(len(model.theta))
    n_pq = int(len(model.pq))
    shape = (n_bus, n_bus)
    use_preconditioner = model.bp_inv.size > 0 and model.bpp_inv.size > 0
    use_diagonal_preconditioner = use_preconditioner and model.bp_inv.ndim == 1

    def solve_one(sbus, ybus_data, bp_inv, bpp_inv, g_qtheta, vm0_start, va0_start):
        ybus = jsparse.BCOO((ybus_data, ybus_indices), shape=shape)
        if use_preconditioner:
            bp_inv = jnp.asarray(bp_inv, dtype=jnp.float64)
            bpp_inv = jnp.asarray(bpp_inv, dtype=jnp.float64)
            if not use_diagonal_preconditioner:
                g_qtheta = jnp.asarray(g_qtheta, dtype=jnp.float64)

        x0 = jnp.concatenate([va0_start[theta], vm0_start[pq]])

        def unpack(x):
            va = va0_start.at[theta].set(x[:n_theta])
            vm = vm0_start.at[pq].set(x[n_theta:])
            return vm * jnp.exp(1j * va)

        def mismatch(x):
            voltage = unpack(x)
            current = ybus @ voltage
            scalc = voltage * jnp.conj(current)
            p_mis = scalc.real[theta] - sbus.real[theta]
            q_mis = scalc.imag[pq] - sbus.imag[pq]
            return jnp.concatenate([p_mis, q_mis])

        def newton_body(x, _):
            residual, jvp = jax.linearize(mismatch, x)
            voltage = unpack(x)
            vm = jnp.abs(voltage)

            preconditioner = None
            if use_preconditioner:
                if use_diagonal_preconditioner:
                    def apply_preconditioner(r):
                        r_theta = r[:n_theta]
                        r_q = r[n_theta:]
                        z_theta = bp_inv * (r_theta / vm[theta])
                        z_q = bpp_inv * (r_q / vm[pq])
                        return jnp.concatenate([z_theta, z_q])
                else:
                    def apply_preconditioner(r):
                        r_theta = r[:n_theta]
                        r_q = r[n_theta:]
                        z_theta = bp_inv @ (r_theta / vm[theta])
                        q_rhs = r_q - g_qtheta @ z_theta
                        z_q = bpp_inv @ (q_rhs / vm[pq])
                        return jnp.concatenate([z_theta, z_q])

                preconditioner = apply_preconditioner

            dx, _ = gmres(
                jvp,
                -residual,
                tol=options.gmres_tol,
                atol=options.gmres_atol,
                restart=options.gmres_restart,
                maxiter=options.gmres_maxiter,
                M=preconditioner,
            )
            return x + dx, jnp.max(jnp.abs(residual))

        x_final, residual_history = lax.scan(
            newton_body,
            x0,
            xs=None,
            length=options.max_iterations,
        )
        final_residual = mismatch(x_final)
        final_norm = jnp.max(jnp.abs(final_residual))
        converged = final_norm <= options.tolerance
        return unpack(x_final), converged, final_norm, residual_history

    def solve_many(sbus_batch, ybus_data, bp_inv, bpp_inv, g_qtheta, vm0_start=vm0, va0_start=va0):
        return jax.vmap(solve_one, in_axes=(0, None, None, None, None, None, None))(
            sbus_batch,
            ybus_data,
            bp_inv,
            bpp_inv,
            g_qtheta,
            vm0_start,
            va0_start,
        )

    return jax.jit(solve_many) if options.jit else solve_many


def solve_batch(
    model: PowerFlowModel,
    sbus_batch: np.ndarray | None = None,
    options: NewtonOptions | None = None,
) -> PowerFlowResult:
    """Solve a batch of operating scenarios for one pandapower topology."""

    options = options or NewtonOptions()
    return BatchedPowerFlowSolver(model, options).solve(sbus_batch)


def _compile_topology_solver(model: PowerFlowModel, options: NewtonOptions):
    solve_many = _compile_solver(model, options)

    def solve_topologies(
        sbus_by_topology,
        ybus_data_batch,
        bp_inv_batch,
        bpp_inv_batch,
        g_qtheta_batch,
        vm0_batch,
        va0_batch,
    ):
        def one_topology(sbus_batch, ybus_data, bp_inv, bpp_inv, g_qtheta, vm0_start, va0_start):
            voltages, converged, mismatch_norm, _ = solve_many(
                sbus_batch,
                ybus_data,
                bp_inv,
                bpp_inv,
                g_qtheta,
                vm0_start,
                va0_start,
            )
            return voltages, converged, mismatch_norm

        return jax.vmap(one_topology, in_axes=(0, 0, 0, 0, 0, 0, 0))(
            sbus_by_topology,
            ybus_data_batch,
            bp_inv_batch,
            bpp_inv_batch,
            g_qtheta_batch,
            vm0_batch,
            va0_batch,
        )

    return jax.jit(solve_topologies) if options.jit else solve_topologies


def solve_topology_batch(
    template_model: PowerFlowModel,
    *,
    ybus_data_batch: np.ndarray,
    bp_inv_batch: np.ndarray,
    bpp_inv_batch: np.ndarray,
    g_qtheta_batch: np.ndarray,
    vm0_batch: np.ndarray | None = None,
    va0_batch: np.ndarray | None = None,
    sbus_batch: np.ndarray,
    contingencies: tuple = (),
    options: NewtonOptions | None = None,
) -> TopologyBatchResult:
    """Solve compatible topology and operating-scenario batches.

    ``template_model.ybus_indices`` defines the shared sparse COO pattern.
    ``ybus_data_batch`` supplies the per-topology sparse values over that
    pattern.
    """

    options = options or NewtonOptions()
    sbus = np.asarray(sbus_batch, dtype=np.complex128)
    if sbus.ndim == 2:
        sbus = np.broadcast_to(sbus[None, :, :], (len(ybus_data_batch), *sbus.shape))
    if sbus.ndim != 3:
        raise ValueError("sbus_batch must have shape (scenario, bus) or (topology, scenario, bus).")
    if sbus.shape[0] != len(ybus_data_batch):
        raise ValueError("Topology dimension mismatch between sbus_batch and ybus_data_batch.")
    if vm0_batch is None:
        vm0_batch = np.broadcast_to(template_model.vm0[None, :], (len(ybus_data_batch), template_model.n_bus))
    if va0_batch is None:
        va0_batch = np.broadcast_to(template_model.va0[None, :], (len(ybus_data_batch), template_model.n_bus))
    vm0_batch = np.asarray(vm0_batch, dtype=np.float64)
    va0_batch = np.asarray(va0_batch, dtype=np.float64)
    expected_start_shape = (len(ybus_data_batch), template_model.n_bus)
    if vm0_batch.shape != expected_start_shape or va0_batch.shape != expected_start_shape:
        raise ValueError(f"Topology start arrays must have shape {expected_start_shape}.")

    solve_topologies = _compile_topology_solver(template_model, options)
    voltages, converged, mismatch_norm = solve_topologies(
        jnp.asarray(sbus),
        jnp.asarray(ybus_data_batch),
        jnp.asarray(bp_inv_batch),
        jnp.asarray(bpp_inv_batch),
        jnp.asarray(g_qtheta_batch),
        jnp.asarray(vm0_batch),
        jnp.asarray(va0_batch),
    )
    return TopologyBatchResult(
        contingencies=tuple(contingencies),
        voltages=np.asarray(voltages),
        converged=np.asarray(converged),
        mismatch_norm=np.asarray(mismatch_norm),
        iterations=options.max_iterations,
    )


def compute_branch_flows(model: PowerFlowModel, voltages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute complex branch powers at from/to ends in MVA."""

    voltage = jnp.asarray(voltages)
    if voltage.ndim == 1:
        voltage = voltage[None, :]

    yf = jsparse.BCOO(
        (jnp.asarray(model.yf_data), jnp.asarray(model.yf_indices, dtype=jnp.int32)),
        shape=(len(model.branch_from), model.n_bus),
    )
    yt = jsparse.BCOO(
        (jnp.asarray(model.yt_data), jnp.asarray(model.yt_indices, dtype=jnp.int32)),
        shape=(len(model.branch_to), model.n_bus),
    )
    branch_from = jnp.asarray(model.branch_from, dtype=jnp.int32)
    branch_to = jnp.asarray(model.branch_to, dtype=jnp.int32)

    def one(v):
        current_from = yf @ v
        current_to = yt @ v
        s_from = v[branch_from] * jnp.conj(current_from) * model.base_mva
        s_to = v[branch_to] * jnp.conj(current_to) * model.base_mva
        return s_from, s_to

    s_from, s_to = jax.vmap(one)(voltage)
    return np.asarray(s_from), np.asarray(s_to)
