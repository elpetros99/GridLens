from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .model import PowerFlowModel
from .pandapower_adapter import PandapowerBuildOptions, from_pandapower
from .solver import NewtonOptions, compute_branch_flows, solve_topology_batch


@dataclass(frozen=True)
class Outage:
    element: str
    index: int

    def __str__(self) -> str:
        return f"{self.element}:{self.index}"


@dataclass(frozen=True)
class Contingency:
    outages: tuple[Outage, ...]
    name: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", "+".join(str(outage) for outage in self.outages))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ContingencyAnalysis:
    """Solved contingency batch with voltage and branch-loading diagnostics."""

    result: object
    s_from_mva: np.ndarray
    s_to_mva: np.ndarray
    loading_percent: np.ndarray
    voltage_low: np.ndarray
    voltage_high: np.ndarray
    branch_overloaded: np.ndarray

    @property
    def failed(self) -> np.ndarray:
        return ~self.result.converged

    @property
    def vm_min(self) -> np.ndarray:
        return self.result.vm.min(axis=-1)

    @property
    def vm_max(self) -> np.ndarray:
        return self.result.vm.max(axis=-1)


@dataclass(frozen=True)
class ContingencyBatchData:
    """Aligned topology-batch data, including per-topology Newton starts."""

    template: PowerFlowModel
    ybus_data_batch: np.ndarray
    bp_inv_batch: np.ndarray
    bpp_inv_batch: np.ndarray
    g_qtheta_batch: np.ndarray
    vm0_batch: np.ndarray
    va0_batch: np.ndarray
    contingencies: tuple[Contingency, ...]


def _active_indices(net, element: str) -> list[int]:
    if not hasattr(net, element):
        return []
    table = getattr(net, element)
    if table is None or len(table) == 0:
        return []
    if "in_service" in table:
        return [int(i) for i in table.index[table["in_service"].astype(bool)]]
    return [int(i) for i in table.index]


def enumerate_nk(
    net,
    *,
    k: int,
    elements: Sequence[str] = ("line", "trafo"),
    max_cases: int | None = None,
) -> tuple[Contingency, ...]:
    """Enumerate active-element N-k contingencies.

    By default this enumerates AC branch outages (`line`, `trafo`), which keeps
    the PV/PQ/ref state dimensions stable for topology-batched solving.
    """

    if k < 1:
        raise ValueError("k must be >= 1.")

    candidates: list[Outage] = []
    for element in elements:
        candidates.extend(Outage(element, index) for index in _active_indices(net, element))

    contingencies = []
    for combo in itertools.combinations(candidates, k):
        contingencies.append(Contingency(tuple(combo)))
        if max_cases is not None and len(contingencies) >= max_cases:
            break
    return tuple(contingencies)


def apply_contingency(net, contingency: Contingency):
    """Return a pandapower net copy with the contingency's elements outaged."""

    work = copy.deepcopy(net)
    for outage in contingency.outages:
        if not hasattr(work, outage.element):
            raise ValueError(f"pandapower net has no element table {outage.element!r}.")
        table = getattr(work, outage.element)
        if outage.index not in table.index:
            raise ValueError(f"{outage.element} index {outage.index} does not exist.")
        if "in_service" not in table:
            raise ValueError(f"{outage.element} has no in_service column and cannot be outaged.")
        table.at[outage.index, "in_service"] = False
    return work


def _same_state_space(a: PowerFlowModel, b: PowerFlowModel) -> bool:
    return (
        a.n_bus == b.n_bus
        and np.array_equal(a.ref, b.ref)
        and np.array_equal(a.pv, b.pv)
        and np.array_equal(a.pq, b.pq)
    )


def _union_ybus_pattern(models: Sequence[PowerFlowModel]) -> np.ndarray:
    coords: set[tuple[int, int]] = set()
    for model in models:
        coords.update(map(tuple, model.ybus_indices.tolist()))
    return np.asarray(sorted(coords), dtype=np.int32)


def _align_ybus_data(model: PowerFlowModel, pattern: np.ndarray) -> np.ndarray:
    dense_map = {tuple(idx): val for idx, val in zip(model.ybus_indices.tolist(), model.ybus_data)}
    return np.asarray([dense_map.get(tuple(idx), 0.0 + 0.0j) for idx in pattern], dtype=np.complex128)


def build_contingency_batch_data(
    net,
    contingencies: Iterable[Contingency],
    *,
    options: PandapowerBuildOptions | None = None,
) -> ContingencyBatchData:
    """Build aligned sparse data for a compatible contingency topology batch."""

    contingencies = tuple(contingencies)
    if not contingencies:
        raise ValueError("At least one contingency is required.")

    build_options = options or PandapowerBuildOptions(check_connectivity=False)
    if options is None:
        build_options = PandapowerBuildOptions(check_connectivity=False)

    models = [from_pandapower(apply_contingency(net, c), name=str(c), options=build_options) for c in contingencies]
    template = models[0]
    incompatible = [str(c) for c, m in zip(contingencies, models) if not _same_state_space(template, m)]
    if incompatible:
        raise ValueError(
            "Contingencies do not share the same bus/PV/PQ/ref state space and cannot be topology-batched: "
            + ", ".join(incompatible[:5])
        )

    pattern = _union_ybus_pattern(models)
    template = template.with_ybus_pattern(
        ybus_indices=pattern,
        ybus_data=_align_ybus_data(template, pattern),
    )
    ybus_data_batch = np.stack([_align_ybus_data(model, pattern) for model in models])
    bp_inv_batch = np.stack([model.bp_inv for model in models])
    bpp_inv_batch = np.stack([model.bpp_inv for model in models])
    g_qtheta_batch = np.stack([model.g_qtheta for model in models])
    vm0_batch = np.stack([model.vm0 for model in models])
    va0_batch = np.stack([model.va0 for model in models])
    return ContingencyBatchData(
        template=template,
        ybus_data_batch=ybus_data_batch,
        bp_inv_batch=bp_inv_batch,
        bpp_inv_batch=bpp_inv_batch,
        g_qtheta_batch=g_qtheta_batch,
        vm0_batch=vm0_batch,
        va0_batch=va0_batch,
        contingencies=contingencies,
    )


def build_contingency_batch(
    net,
    contingencies: Iterable[Contingency],
    *,
    options: PandapowerBuildOptions | None = None,
) -> tuple[PowerFlowModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[Contingency, ...]]:
    """Build aligned sparse data for a compatible contingency topology batch.

    This compatibility wrapper preserves the original tuple return shape. Use
    build_contingency_batch_data when per-topology Newton starts are needed.
    """

    batch = build_contingency_batch_data(net, contingencies, options=options)
    return (
        batch.template,
        batch.ybus_data_batch,
        batch.bp_inv_batch,
        batch.bpp_inv_batch,
        batch.g_qtheta_batch,
        batch.contingencies,
    )


def solve_contingencies_batched(
    net,
    contingencies: Iterable[Contingency],
    sbus_batch: np.ndarray | None = None,
    *,
    build_options: PandapowerBuildOptions | None = None,
    solver_options: NewtonOptions | None = None,
):
    """Run compatible branch N-k contingencies as a JAX topology batch."""

    batch = build_contingency_batch_data(
        net,
        contingencies,
        options=build_options,
    )
    if sbus_batch is None:
        sbus_batch = batch.template.sbus0[None, :]
    return solve_topology_batch(
        batch.template,
        ybus_data_batch=batch.ybus_data_batch,
        bp_inv_batch=batch.bp_inv_batch,
        bpp_inv_batch=batch.bpp_inv_batch,
        g_qtheta_batch=batch.g_qtheta_batch,
        vm0_batch=batch.vm0_batch,
        va0_batch=batch.va0_batch,
        sbus_batch=sbus_batch,
        contingencies=batch.contingencies,
        options=solver_options,
    )


def analyze_contingencies_batched(
    net,
    contingencies: Iterable[Contingency],
    sbus_batch: np.ndarray | None = None,
    *,
    build_options: PandapowerBuildOptions | None = None,
    solver_options: NewtonOptions | None = None,
) -> ContingencyAnalysis:
    """Solve N-k contingencies and compute voltage/thermal violation masks."""

    contingencies = tuple(contingencies)
    result = solve_contingencies_batched(
        net,
        contingencies,
        sbus_batch,
        build_options=build_options,
        solver_options=solver_options,
    )

    options = build_options or PandapowerBuildOptions(check_connectivity=False)
    models = [from_pandapower(apply_contingency(net, c), name=str(c), options=options) for c in contingencies]

    from pandapower.pypower.idx_brch import RATE_A
    from pandapower.pypower.idx_bus import VMAX, VMIN

    s_from = []
    s_to = []
    loading = []
    for i, model in enumerate(models):
        sf, st = compute_branch_flows(model, result.voltages[i])
        s_from.append(sf)
        s_to.append(st)
        rate_a = model.ppc["branch"][:, RATE_A].astype(float)
        branch_mva = np.maximum(np.abs(sf), np.abs(st))
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(branch_mva * 100.0, rate_a[None, :], out=np.zeros_like(branch_mva.real), where=rate_a[None, :] > 0)
        loading.append(pct)

    template = models[0]
    vmin = template.ppc["bus"][:, VMIN].astype(float)
    vmax = template.ppc["bus"][:, VMAX].astype(float)
    vm = result.vm

    loading_percent = np.stack(loading)
    return ContingencyAnalysis(
        result=result,
        s_from_mva=np.stack(s_from),
        s_to_mva=np.stack(s_to),
        loading_percent=loading_percent,
        voltage_low=vm < vmin[None, None, :],
        voltage_high=vm > vmax[None, None, :],
        branch_overloaded=loading_percent > 100.0,
    )
