from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .model import PowerFlowModel, PowerFlowResult


def compute_branch_flows_cpu(model: PowerFlowModel, voltages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute branch powers with the model's SciPy sparse incidence matrices."""

    voltage = np.asarray(voltages, dtype=np.complex128)
    if voltage.ndim == 1:
        voltage = voltage[None, :]
    if voltage.ndim != 2 or voltage.shape[1] != model.n_bus:
        raise ValueError(f"voltages must have shape (scenario, {model.n_bus}) or ({model.n_bus},).")
    if model.yf_scipy is None or model.yt_scipy is None:
        yf = sparse.coo_matrix(
            (model.yf_data, (model.yf_indices[:, 0], model.yf_indices[:, 1])),
            shape=(len(model.branch_from), model.n_bus),
        ).tocsr()
        yt = sparse.coo_matrix(
            (model.yt_data, (model.yt_indices[:, 0], model.yt_indices[:, 1])),
            shape=(len(model.branch_to), model.n_bus),
        ).tocsr()
        current_from = np.asarray((yf @ voltage.T).T)
        current_to = np.asarray((yt @ voltage.T).T)
    else:
        current_from = np.asarray((model.yf_scipy @ voltage.T).T)
        current_to = np.asarray((model.yt_scipy @ voltage.T).T)
    s_from = voltage[:, model.branch_from] * np.conj(current_from) * model.base_mva
    s_to = voltage[:, model.branch_to] * np.conj(current_to) * model.base_mva
    return s_from, s_to


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
    jacobian_reuse: int = 1


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


def _make_linear_solver(jacobian: sparse.csr_matrix, options: CpuNewtonOptions):
    """Factor or precondition a Newton matrix once for optional reuse."""

    if options.linear_solver == "direct":
        factor = spla.splu(jacobian.tocsc())
        return factor.solve
    if options.linear_solver == "gmres_ilu":
        csc = jacobian.tocsc()
        ilu = spla.spilu(csc, drop_tol=options.ilu_drop_tol, fill_factor=options.ilu_fill_factor)
        preconditioner = spla.LinearOperator(csc.shape, matvec=ilu.solve)
        direct_factor = None

        def solve(rhs: np.ndarray) -> np.ndarray:
            nonlocal direct_factor
            dx, info = spla.gmres(
                csc,
                rhs,
                M=preconditioner,
                rtol=options.gmres_tol,
                atol=0.0,
                restart=options.gmres_restart,
                maxiter=options.gmres_maxiter,
            )
            if info != 0:
                if direct_factor is None:
                    direct_factor = spla.splu(csc)
                return direct_factor.solve(rhs)
            return dx

        return solve
    raise ValueError("linear_solver must be 'direct' or 'gmres_ilu'.")


def _solve_one(
    model: PowerFlowModel,
    sbus: np.ndarray,
    options: CpuNewtonOptions,
    initial_voltage: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, float]:
    if options.start not in {"model", "pandapower_result"}:
        raise ValueError("start must be 'model' or 'pandapower_result'.")
    if options.jacobian_reuse < 1:
        raise ValueError("jacobian_reuse must be >= 1.")
    if initial_voltage is not None:
        voltage0 = np.asarray(initial_voltage, dtype=np.complex128)
        if voltage0.shape != (model.n_bus,):
            raise ValueError(f"initial_voltage has shape {voltage0.shape}, expected {(model.n_bus,)}.")
        if not np.all(np.isfinite(voltage0)):
            raise ValueError("initial_voltage must contain only finite values.")
        va = np.angle(voltage0)
        vm = np.abs(voltage0)
    elif options.start == "pandapower_result" and model.ppc is not None and "internal" in model.ppc and "V" in model.ppc["internal"]:
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
    linear_solver = None
    for iteration in range(options.max_iterations):
        residual = _mismatch(model, voltage, sbus)
        final_norm = float(np.max(np.abs(residual))) if residual.size else 0.0
        if final_norm <= options.tolerance:
            return voltage, True, final_norm

        if linear_solver is None or iteration % options.jacobian_reuse == 0:
            jacobian = _build_jacobian(model, voltage)
            linear_solver = _make_linear_solver(jacobian, options)
        dx = linear_solver(-residual)

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
    initial_voltages: np.ndarray | None = None,
) -> PowerFlowResult:
    """Solve a scenario batch with explicit sparse Newton on CPU.

    ``initial_voltages`` may be a single ``(bus,)`` vector or one complex
    starting-voltage vector per scenario with shape ``(scenario, bus)``.
    ``CpuNewtonOptions.jacobian_reuse`` enables an optional inexact-Newton
    mode; the default ``1`` rebuilds the numerical Jacobian every iteration.
    """

    options = options or CpuNewtonOptions()
    sbus = _as_batch(model, sbus_batch)
    starts = None
    if initial_voltages is not None:
        starts = np.asarray(initial_voltages, dtype=np.complex128)
        if starts.ndim == 1:
            starts = starts[None, :]
        if starts.shape != sbus.shape:
            raise ValueError(f"initial_voltages has shape {starts.shape}, expected {sbus.shape}.")
    voltages = []
    converged = []
    mismatch_norm = []
    for index, scenario_sbus in enumerate(sbus):
        voltage, ok, norm = _solve_one(
            model,
            scenario_sbus,
            options,
            None if starts is None else starts[index],
        )
        voltages.append(voltage)
        converged.append(ok)
        mismatch_norm.append(norm)
    return PowerFlowResult(
        voltages=np.asarray(voltages, dtype=np.complex128),
        converged=np.asarray(converged, dtype=bool),
        mismatch_norm=np.asarray(mismatch_norm, dtype=np.float64),
        iterations=options.max_iterations,
    )
