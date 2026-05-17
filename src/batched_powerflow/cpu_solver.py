from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .model import PowerFlowModel, PowerFlowResult


@dataclass(frozen=True)
class CpuNewtonOptions:
    """Options for CPU sparse Newton power flow."""

    max_iterations: int = 20
    tolerance: float = 1e-10
    linear_solver: str = "direct"
    gmres_tol: float = 1e-8
    gmres_restart: int = 50
    gmres_maxiter: int = 200
    ilu_drop_tol: float = 1e-4
    ilu_fill_factor: float = 10.0
    start: str = "model"


def _as_batch(model: PowerFlowModel, sbus_batch: np.ndarray | None) -> np.ndarray:
    if sbus_batch is None:
        return model.sbus0[None, :]
    sbus = np.asarray(sbus_batch, dtype=np.complex128)
    if sbus.ndim == 1:
        sbus = sbus[None, :]
    if sbus.shape[1] != model.n_bus:
        raise ValueError(f"Sbus has {sbus.shape[1]} buses, expected {model.n_bus}.")
    return sbus


def _build_jacobian(model: PowerFlowModel, voltage: np.ndarray) -> sparse.csr_matrix:
    from pandapower.pypower.dSbus_dV import dSbus_dV

    dS_dVm, dS_dVa = dSbus_dV(model.ybus_scipy, voltage)
    theta = model.theta
    pq = model.pq
    j11 = dS_dVa[theta, :][:, theta].real
    j12 = dS_dVm[theta, :][:, pq].real
    j21 = dS_dVa[pq, :][:, theta].imag
    j22 = dS_dVm[pq, :][:, pq].imag
    return sparse.vstack(
        [
            sparse.hstack([j11, j12], format="csr"),
            sparse.hstack([j21, j22], format="csr"),
        ],
        format="csr",
    )


def _mismatch(model: PowerFlowModel, voltage: np.ndarray, sbus: np.ndarray) -> np.ndarray:
    scalc = voltage * np.conj(model.ybus_scipy @ voltage)
    return np.r_[scalc.real[model.theta] - sbus.real[model.theta], scalc.imag[model.pq] - sbus.imag[model.pq]]


def _solve_linear(jacobian: sparse.csr_matrix, rhs: np.ndarray, options: CpuNewtonOptions) -> np.ndarray:
    if options.linear_solver == "direct":
        return spla.spsolve(jacobian.tocsc(), rhs)
    if options.linear_solver == "gmres_ilu":
        csc = jacobian.tocsc()
        ilu = spla.spilu(csc, drop_tol=options.ilu_drop_tol, fill_factor=options.ilu_fill_factor)
        preconditioner = spla.LinearOperator(csc.shape, matvec=ilu.solve)
        result = spla.gmres(
            csc,
            rhs,
            M=preconditioner,
            rtol=options.gmres_tol,
            atol=0.0,
            restart=options.gmres_restart,
            maxiter=options.gmres_maxiter,
        )
        dx, info = result
        if info != 0:
            return spla.spsolve(csc, rhs)
        return dx
    raise ValueError("linear_solver must be 'direct' or 'gmres_ilu'.")


def _solve_one(model: PowerFlowModel, sbus: np.ndarray, options: CpuNewtonOptions) -> tuple[np.ndarray, bool, float]:
    if options.start not in {"model", "pandapower_result"}:
        raise ValueError("start must be 'model' or 'pandapower_result'.")
    if options.start == "pandapower_result" and model.ppc is not None and "internal" in model.ppc and "V" in model.ppc["internal"]:
        voltage0 = np.asarray(model.ppc["internal"]["V"], dtype=np.complex128)
        if np.iscomplexobj(model.ppc["internal"]["V"]):
            va = np.angle(voltage0)
            vm = np.abs(voltage0)
        else:
            va = np.array(model.va0, dtype=np.float64, copy=True)
            vm = np.array(model.vm0, dtype=np.float64, copy=True)
    else:
        va = np.array(model.va0, dtype=np.float64, copy=True)
        vm = np.array(model.vm0, dtype=np.float64, copy=True)
    voltage = vm * np.exp(1j * va)

    final_norm = np.inf
    for _ in range(options.max_iterations):
        residual = _mismatch(model, voltage, sbus)
        final_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        if final_norm <= options.tolerance:
            return voltage, True, final_norm

        jacobian = _build_jacobian(model, voltage)
        dx = _solve_linear(jacobian, -residual, options)

        n_theta = len(model.theta)
        va[model.theta] += dx[:n_theta]
        vm[model.pq] += dx[n_theta:]
        voltage = vm * np.exp(1j * va)
        vm = np.abs(voltage)
        va = np.angle(voltage)

    residual = _mismatch(model, voltage, sbus)
    final_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
    return voltage, final_norm <= options.tolerance, final_norm


def solve_batch_cpu_sparse(
    model: PowerFlowModel,
    sbus_batch: np.ndarray | None = None,
    options: CpuNewtonOptions | None = None,
) -> PowerFlowResult:
    """Solve a scenario batch with explicit sparse Newton on CPU."""

    options = options or CpuNewtonOptions()
    sbus = _as_batch(model, sbus_batch)
    voltages = []
    converged = []
    mismatch_norm = []
    for scenario_sbus in sbus:
        voltage, ok, norm = _solve_one(model, scenario_sbus, options)
        voltages.append(voltage)
        converged.append(ok)
        mismatch_norm.append(norm)
    return PowerFlowResult(
        voltages=np.asarray(voltages, dtype=np.complex128),
        converged=np.asarray(converged, dtype=bool),
        mismatch_norm=np.asarray(mismatch_norm, dtype=np.float64),
        iterations=options.max_iterations,
    )
