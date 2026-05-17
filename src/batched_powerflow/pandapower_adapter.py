from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import sparse


class UnsupportedFeature(ValueError):
    """Raised when a pandapower feature cannot be represented by static JAX equations."""

    def __init__(self, feature: str, guidance: str, *, details: str = ""):
        self.feature = feature
        self.guidance = guidance
        self.details = details
        message = f"Unsupported pandapower feature {feature!r}: {guidance}"
        if details:
            message = f"{message} ({details})"
        super().__init__(message)


@dataclass(frozen=True)
class PandapowerFeatureOptions:
    """Policy for pandapower features encountered before static extraction."""

    controllers: str = "freeze"
    voltage_depend_loads: str = "raise"
    distributed_slack: str = "raise"
    tdpf: str = "raise"


@dataclass(frozen=True)
class PandapowerBuildOptions:
    calculate_voltage_angles: bool = True
    init: str = "auto"
    max_iteration: int = 20
    tolerance_mva: float = 1e-10
    trafo_model: str = "t"
    trafo_loading: str = "current"
    enforce_q_lims: bool = False
    check_connectivity: bool = True
    voltage_depend_loads: bool = False
    consider_line_temperature: bool = False
    distributed_slack: bool = False
    tdpf: bool = False
    run_control: bool = False
    run_powerflow: bool = False
    numba: bool = False
    regularization: float = 1e-8
    preconditioner: str = "auto"
    max_dense_preconditioner_state: int = 1500
    feature_options: PandapowerFeatureOptions = field(default_factory=PandapowerFeatureOptions)


def _validate_feature_policy(name: str, action: str, allowed: set[str]) -> None:
    if action not in allowed:
        raise ValueError(f"{name} feature policy must be one of {sorted(allowed)}, got {action!r}.")


def _active_controller_mask(net) -> np.ndarray:
    if not hasattr(net, "controller") or net.controller is None or len(net.controller) == 0:
        return np.zeros(0, dtype=bool)
    if "in_service" in net.controller:
        return net.controller["in_service"].to_numpy(dtype=bool)
    return np.ones(len(net.controller), dtype=bool)


def _has_active_controllers(net) -> bool:
    return bool(np.any(_active_controller_mask(net)))


def _active_controller_names(net) -> str:
    mask = _active_controller_mask(net)
    if not len(mask):
        return ""
    objects = net.controller.loc[mask, "object"] if "object" in net.controller else []
    names = [type(obj).__name__ for obj in objects]
    return ", ".join(names[:5])


def _raise_unsupported(feature: str, guidance: str, *, details: str = "") -> None:
    raise UnsupportedFeature(feature, guidance, details=details)


def _runpp_kwargs(options: PandapowerBuildOptions) -> dict[str, object]:
    return {
        "algorithm": "nr",
        "calculate_voltage_angles": options.calculate_voltage_angles,
        "init": options.init,
        "max_iteration": options.max_iteration,
        "tolerance_mva": options.tolerance_mva,
        "trafo_model": options.trafo_model,
        "trafo_loading": options.trafo_loading,
        "enforce_q_lims": options.enforce_q_lims,
        "check_connectivity": options.check_connectivity,
        "voltage_depend_loads": options.voltage_depend_loads,
        "consider_line_temperature": options.consider_line_temperature,
        "distributed_slack": options.distributed_slack,
        "tdpf": options.tdpf,
        "numba": options.numba,
    }


def _freeze_voltage_dependent_loads(work) -> None:
    if not hasattr(work, "load") or len(work.load) == 0 or not hasattr(work, "res_load") or len(work.res_load) == 0:
        return
    work.load.loc[:, "p_mw"] = work.res_load["p_mw"].to_numpy(dtype=float)
    work.load.loc[:, "q_mvar"] = work.res_load["q_mvar"].to_numpy(dtype=float)
    for column in (
        "const_z_p_percent",
        "const_i_p_percent",
        "const_z_q_percent",
        "const_i_q_percent",
    ):
        if column in work.load:
            work.load.loc[:, column] = 0.0


def _preprocess_features(net, options: PandapowerBuildOptions):
    import pandapower as pp

    feature_options = options.feature_options
    requested_voltage_dependent_loads = options.voltage_depend_loads
    _validate_feature_policy("controllers", feature_options.controllers, {"freeze", "raise", "ignore"})
    _validate_feature_policy("voltage_depend_loads", feature_options.voltage_depend_loads, {"raise", "freeze", "ignore"})
    _validate_feature_policy("distributed_slack", feature_options.distributed_slack, {"raise", "ignore"})
    _validate_feature_policy("tdpf", feature_options.tdpf, {"raise", "ignore"})

    if options.voltage_depend_loads:
        if feature_options.voltage_depend_loads == "raise":
            _raise_unsupported(
                "voltage_depend_loads",
                "the JAX solver uses a fixed Sbus; use voltage_depend_loads='freeze' to freeze ZIP loads at the pandapower preprocessing point, or set voltage_depend_loads=False.",
            )
        if feature_options.voltage_depend_loads == "freeze":
            options = copy.copy(options)
            object.__setattr__(options, "voltage_depend_loads", False)
    if options.distributed_slack and feature_options.distributed_slack == "raise":
        _raise_unsupported(
            "distributed_slack",
            "distributed slack changes active-power balancing outside the fixed-PQ/PV equations; disable it for JAX extraction.",
        )
    if options.tdpf:
        if feature_options.tdpf == "raise":
            _raise_unsupported(
                "tdpf",
                "temperature-dependent power flow updates branch parameters during the solve; use consider_line_temperature=True for static line-temperature parameters or run pandapower TDPF separately.",
            )
        if feature_options.tdpf == "ignore":
            options = copy.copy(options)
            object.__setattr__(options, "tdpf", False)

    work = copy.deepcopy(net)
    if requested_voltage_dependent_loads and feature_options.voltage_depend_loads == "freeze":
        run_options = copy.copy(options)
        object.__setattr__(run_options, "voltage_depend_loads", True)
        pp.runpp(work, run_control=False, **_runpp_kwargs(run_options))
        _freeze_voltage_dependent_loads(work)

    has_controllers = options.run_control or _has_active_controllers(work)
    if has_controllers:
        if feature_options.controllers == "raise":
            _raise_unsupported(
                "controllers",
                "controller actions are not part of the JAX nonlinear equations; use controllers='freeze' to run pandapower control before extraction.",
                details=_active_controller_names(work),
            )
        if feature_options.controllers == "freeze":
            pp.runpp(work, run_control=True, **_runpp_kwargs(options))
    elif options.run_powerflow:
        run_options = copy.copy(options)
        if run_options.init == "flat":
            object.__setattr__(run_options, "init", "auto")
        if isinstance(run_options.max_iteration, int) and run_options.max_iteration < 50:
            object.__setattr__(run_options, "max_iteration", 50)
        pp.runpp(work, run_control=False, **_runpp_kwargs(run_options))
        options = copy.copy(options)
        object.__setattr__(options, "init", "results")
    return work, options


def _prepare_ppci(net, options: PandapowerBuildOptions):
    from pandapower.auxiliary import _init_runpp_options
    from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
    from pandapower.pypower.idx_bus import PD
    from pandapower.pypower.idx_gen import PG
    import pandapower.pd2ppc as pd2ppc

    work, options = _preprocess_features(net, options)
    _init_runpp_options(
        work,
        algorithm="nr",
        calculate_voltage_angles=options.calculate_voltage_angles,
        init=options.init,
        max_iteration=options.max_iteration,
        tolerance_mva=options.tolerance_mva,
        trafo_model=options.trafo_model,
        trafo_loading=options.trafo_loading,
        enforce_q_lims=options.enforce_q_lims,
        check_connectivity=options.check_connectivity,
        voltage_depend_loads=options.voltage_depend_loads,
        consider_line_temperature=options.consider_line_temperature,
        distributed_slack=options.distributed_slack,
        tdpf=options.tdpf,
        numba=options.numba,
    )
    ppc, ppci = pd2ppc._pd2ppc(work)
    if work._options.get("init_va_degree") == "dc":
        if work._options.get("distributed_slack"):
            pg_copy = ppci["gen"][:, PG].copy()
            pd_copy = ppci["bus"][:, PD].copy()
            ppci = _run_dc_pf(ppci, work._options.get("recycle"))
            ppci["gen"][:, PG] = pg_copy
            ppci["bus"][:, PD] = pd_copy
        else:
            ppci = _run_dc_pf(ppci, work._options.get("recycle"))
    return work, ppc, ppci


def _coo_arrays(matrix: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    coo = matrix.tocoo()
    indices = np.column_stack([coo.row, coo.col]).astype(np.int32, copy=False)
    data = coo.data.astype(np.complex128, copy=False)
    return data, indices


def _invert_or_empty(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(matrix.shape, dtype=np.float64)
    return np.linalg.inv(matrix)


def _empty_preconditioner() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros((0, 0), dtype=np.float64)
    return empty, empty, empty


def _safe_inverse_diagonal(values: np.ndarray, regularization: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    adjusted = values.copy()
    small = np.abs(adjusted) < regularization
    adjusted[small] = np.where(adjusted[small] < 0.0, -regularization, regularization)
    return 1.0 / adjusted


def _build_preconditioner(
    ybus: sparse.spmatrix,
    theta: np.ndarray,
    pq: np.ndarray,
    options: PandapowerBuildOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if options.preconditioner not in {"auto", "fast_decoupled_dense", "fast_decoupled_diagonal", "none"}:
        raise ValueError("preconditioner must be 'auto', 'fast_decoupled_dense', 'fast_decoupled_diagonal', or 'none'.")
    if options.preconditioner == "none":
        return _empty_preconditioner()

    b_sparse = (-ybus.imag).tocsr()
    eps = float(options.regularization)
    dense_requested = options.preconditioner == "fast_decoupled_dense" or (
        options.preconditioner == "auto" and len(theta) + len(pq) <= options.max_dense_preconditioner_state
    )
    diagonal_requested = options.preconditioner == "fast_decoupled_diagonal" or (
        options.preconditioner == "auto" and not dense_requested
    )

    if diagonal_requested:
        bp_diag = b_sparse.diagonal()[theta]
        bpp_diag = b_sparse.diagonal()[pq]
        return (
            _safe_inverse_diagonal(bp_diag, eps),
            _safe_inverse_diagonal(bpp_diag, eps),
            np.zeros((0, 0), dtype=np.float64),
        )

    g_sparse = (-ybus.real).tocsr()
    bp = b_sparse[theta, :][:, theta].toarray()
    bpp = b_sparse[pq, :][:, pq].toarray()
    if len(theta):
        bp = bp + eps * np.eye(len(theta))
    if len(pq):
        bpp = bpp + eps * np.eye(len(pq))

    g_qtheta = g_sparse[pq, :][:, theta].toarray().astype(np.float64, copy=False)
    return _invert_or_empty(bp), _invert_or_empty(bpp), g_qtheta


def from_pandapower(
    net,
    *,
    name: str | None = None,
    options: PandapowerBuildOptions | None = None,
):
    """Build a JAX-ready balanced AC model from a pandapower network.

    The conversion uses pandapower's own internal ppc/ppci builder and PYPOWER
    admittance routines. This keeps the numerical data source pandapower-native
    while leaving the nonlinear solve to JAX.
    """

    from pandapower.pf.run_newton_raphson_pf import _get_Sbus, _get_pf_variables_from_ppci
    from pandapower.pypower.idx_brch import F_BUS, T_BUS
    from pandapower.pypower.makeYbus import makeYbus

    from .model import PowerFlowModel

    options = options or PandapowerBuildOptions()
    work, ppc, ppci = _prepare_ppci(net, options)
    (
        base_mva,
        bus,
        gen,
        branch,
        _svc,
        _tcsc,
        _ssc,
        _vsc,
        ref,
        pv,
        pq,
        _on,
        _gbus,
        v0,
        _ref_gens,
    ) = _get_pf_variables_from_ppci(ppci, True)
    ybus, yf, yt = makeYbus(base_mva, bus, branch)
    sbus = _get_Sbus(ppci, work._options.get("recycle"))

    ref = np.asarray(ref, dtype=np.int32)
    pv = np.asarray(pv, dtype=np.int32)
    pq = np.asarray(pq, dtype=np.int32)
    theta = np.concatenate([pv, pq]).astype(np.int32, copy=False)

    ybus_data, ybus_indices = _coo_arrays(ybus)
    yf_data, yf_indices = _coo_arrays(yf)
    yt_data, yt_indices = _coo_arrays(yt)

    bp_inv, bpp_inv, g_qtheta = _build_preconditioner(ybus, theta, pq, options)

    branch_from = branch[:, F_BUS].astype(np.int32, copy=False)
    branch_to = branch[:, T_BUS].astype(np.int32, copy=False)

    return PowerFlowModel(
        name=name or getattr(net, "name", None) or "pandapower_net",
        base_mva=float(base_mva),
        n_bus=int(bus.shape[0]),
        ybus_data=ybus_data,
        ybus_indices=ybus_indices,
        yf_data=yf_data,
        yf_indices=yf_indices,
        yt_data=yt_data,
        yt_indices=yt_indices,
        branch_from=branch_from,
        branch_to=branch_to,
        ref=ref,
        pv=pv,
        pq=pq,
        theta=theta,
        vm0=np.abs(v0).astype(np.float64, copy=False),
        va0=np.angle(v0).astype(np.float64, copy=False),
        sbus0=sbus.astype(np.complex128, copy=False),
        bp_inv=bp_inv,
        bpp_inv=bpp_inv,
        g_qtheta=g_qtheta,
        ybus_scipy=ybus,
        yf_scipy=yf,
        yt_scipy=yt,
        ppc=ppci,
        net=work,
    )


def make_sbus_batch_from_nets(
    nets: Iterable,
    *,
    options: PandapowerBuildOptions | None = None,
) -> np.ndarray:
    """Convert a sequence of pandapower nets into a matching Sbus batch."""

    return np.asarray([from_pandapower(net, options=options).sbus0 for net in nets], dtype=np.complex128)


def make_random_load_sbus_batch(
    net,
    *,
    n_scenarios: int,
    low: float = 0.8,
    high: float = 1.2,
    seed: int = 0,
    per_load: bool = False,
    options: PandapowerBuildOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random load-scaling scenarios using pandapower conversion.

    If ``per_load`` is false, each scenario uses one scalar multiplier for all
    loads. If true, every load receives its own multiplier per scenario.
    """

    model = from_pandapower(net, options=options)
    rng = np.random.default_rng(seed)
    n_load = len(net.load) if hasattr(net, "load") else 0
    if per_load and n_load:
        multipliers = rng.uniform(low, high, size=(n_scenarios, n_load))
    else:
        multipliers = rng.uniform(low, high, size=(n_scenarios,))

    sbus_batch = np.broadcast_to(model.sbus0[None, :], (n_scenarios, model.n_bus)).copy()
    if not n_load:
        return sbus_batch, multipliers

    active = net.load["in_service"].to_numpy(dtype=bool) if "in_service" in net.load else np.ones(n_load, dtype=bool)
    scaling = net.load["scaling"].to_numpy(dtype=float) if "scaling" in net.load else np.ones(n_load)
    bus_lookup = model.net._pd2ppc_lookups["bus"]
    load_bus = bus_lookup[net.load["bus"].to_numpy(dtype=int)]
    load_s = (net.load["p_mw"].to_numpy(dtype=float) + 1j * net.load["q_mvar"].to_numpy(dtype=float))
    load_s = load_s * scaling * active / model.base_mva

    if per_load:
        delta = (multipliers - 1.0) * load_s[None, :]
        np.add.at(sbus_batch, (np.arange(n_scenarios)[:, None], load_bus[None, :]), -delta)
    else:
        base_by_bus = np.zeros(model.n_bus, dtype=np.complex128)
        np.add.at(base_by_bus, load_bus, load_s)
        sbus_batch -= (multipliers[:, None] - 1.0) * base_by_bus[None, :]

    return sbus_batch, multipliers


def make_scaled_load_net(net, multiplier: float | np.ndarray):
    """Return a copy of ``net`` with pandapower load powers scaled."""

    work = copy.deepcopy(net)
    if len(work.load):
        work.load.loc[:, "p_mw"] = work.load["p_mw"].to_numpy(dtype=float) * multiplier
        work.load.loc[:, "q_mvar"] = work.load["q_mvar"].to_numpy(dtype=float) * multiplier
    return work
