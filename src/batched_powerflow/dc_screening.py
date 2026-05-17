from __future__ import annotations

import heapq
import itertools
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning

from .contingency import Contingency, ContingencyAnalysis, Outage, _active_indices, analyze_contingencies_batched
from .pandapower_adapter import PandapowerBuildOptions, _prepare_ppci
from .solver import NewtonOptions


@dataclass(frozen=True)
class DCScreeningResult:
    """Cheap DC proxy for one branch contingency."""

    contingency: Contingency
    score: float
    max_loading_percent: float
    failed: bool
    reason: str
    angles_rad: np.ndarray
    branch_flow_mw: np.ndarray
    loading_percent: np.ndarray


@dataclass(frozen=True)
class HierarchicalSecurityAnalysis:
    """DC screening followed by full AC validation of the selected contingencies."""

    screened: tuple[DCScreeningResult, ...]
    ac: ContingencyAnalysis | None
    total_screened: int

    @property
    def selected_contingencies(self) -> tuple[Contingency, ...]:
        return tuple(result.contingency for result in self.screened)


@dataclass(frozen=True)
class N1LODFScreening:
    """Complete LODF/PTDF N-1 DC screening result."""

    screened: tuple[DCScreeningResult, ...]
    total: int
    failed: int
    max_base_loading_percent: float


@dataclass(frozen=True)
class NKLODFScreening:
    """Candidate-reduced LODF/PTDF N-k DC screening result."""

    screened: tuple[DCScreeningResult, ...]
    total: int
    failed: int
    k: int
    candidate_pool_size: int
    exhaustive_over_pool: bool


def iter_nk_contingencies(
    net,
    *,
    k: int,
    elements: Sequence[str] = ("line", "trafo"),
    max_cases: int | None = None,
) -> Iterator[Contingency]:
    """Yield active branch N-k contingencies lazily."""

    if k < 1:
        raise ValueError("k must be >= 1.")

    candidates: list[Outage] = []
    for element in elements:
        candidates.extend(Outage(element, index) for index in _active_indices(net, element))

    count = 0
    for combo in itertools.combinations(candidates, k):
        yield Contingency(tuple(combo))
        count += 1
        if max_cases is not None and count >= max_cases:
            return


def chunk_contingencies(
    contingencies: Iterable[Contingency],
    *,
    chunk_size: int,
) -> Iterator[tuple[Contingency, ...]]:
    """Yield fixed-size contingency chunks without materializing all cases."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    iterator = iter(contingencies)
    while True:
        chunk = tuple(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


@dataclass(frozen=True)
class _DCBaseData:
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    va0: np.ndarray
    branch_rows_by_outage: dict[Outage, int]


def _build_dc_base_data(net, options: PandapowerBuildOptions | None) -> _DCBaseData:
    from pandapower.pypower.bustypes import bustypes
    from pandapower.pypower.idx_bus import VA

    build_options = options or PandapowerBuildOptions(check_connectivity=False)
    work, _, ppci = _prepare_ppci(net, build_options)
    branch_lookup = work._pd2ppc_lookups.get("branch", {})

    rows_by_outage: dict[Outage, int] = {}
    for element, bounds in branch_lookup.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            continue
        start, stop = bounds
        active = _active_indices(work, element)
        for index, row in zip(active, range(int(start), int(stop))):
            rows_by_outage[Outage(element, index)] = row

    ref, pv, pq = bustypes(ppci["bus"], ppci["gen"])
    return _DCBaseData(
        base_mva=float(ppci["baseMVA"]),
        bus=ppci["bus"].copy(),
        gen=ppci["gen"].copy(),
        branch=ppci["branch"].copy(),
        ref=np.asarray(ref, dtype=np.int32),
        pv=np.asarray(pv, dtype=np.int32),
        pq=np.asarray(pq, dtype=np.int32),
        va0=np.deg2rad(ppci["bus"][:, VA].astype(float, copy=False)),
        branch_rows_by_outage=rows_by_outage,
    )


def _score_failed(contingency: Contingency, branch_count: int, reason: str) -> DCScreeningResult:
    return DCScreeningResult(
        contingency=contingency,
        score=float("inf"),
        max_loading_percent=float("inf"),
        failed=True,
        reason=reason,
        angles_rad=np.full(0, np.nan, dtype=np.float64),
        branch_flow_mw=np.full(branch_count, np.nan, dtype=np.float64),
        loading_percent=np.full(branch_count, np.nan, dtype=np.float64),
    )


def _all_buses_reachable_from_ref(branch: np.ndarray, ref: np.ndarray, n_bus: int) -> bool:
    from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS

    if n_bus == 0:
        return True
    if len(ref) == 0:
        return False

    adjacency: list[list[int]] = [[] for _ in range(n_bus)]
    active = branch[:, BR_STATUS] > 0.0
    for from_bus, to_bus in branch[active][:, [F_BUS, T_BUS]].astype(int, copy=False):
        adjacency[from_bus].append(to_bus)
        adjacency[to_bus].append(from_bus)

    seen = set(int(bus) for bus in ref)
    stack = list(seen)
    while stack:
        bus = stack.pop()
        for neighbor in adjacency[bus]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return len(seen) == n_bus


def score_contingency_dc(
    base: _DCBaseData,
    contingency: Contingency,
) -> DCScreeningResult:
    """Score a contingency with a PYPOWER DC power-flow proxy."""

    from pandapower.pypower.dcpf import dcpf
    from pandapower.pypower.idx_brch import BR_STATUS, RATE_A
    from pandapower.pypower.makeBdc import makeBdc
    from pandapower.pypower.makeSbus import makeSbus

    branch = base.branch.copy()
    for outage in contingency.outages:
        row = base.branch_rows_by_outage.get(outage)
        if row is None:
            return _score_failed(contingency, branch.shape[0], f"no DC branch row for outage {outage}")
        branch[row, BR_STATUS] = 0.0

    if not _all_buses_reachable_from_ref(branch, base.ref, base.bus.shape[0]):
        return _score_failed(contingency, branch.shape[0], "islanded DC topology")

    try:
        bbus, bf, pbusinj, pfinj, _ = makeBdc(base.bus, branch)
        sbus = makeSbus(base.base_mva, base.bus, base.gen)
        pbus = sbus.real - pbusinj
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            angles = dcpf(bbus, pbus, base.va0, base.ref, base.pv, base.pq)
        branch_flow_mw = np.asarray((bf @ angles + pfinj) * base.base_mva, dtype=np.float64)
    except Exception as exc:  # singular islanded cases are deliberately ranked first
        return _score_failed(contingency, branch.shape[0], type(exc).__name__)

    rate_a = branch[:, RATE_A].astype(float, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.divide(
            np.abs(branch_flow_mw) * 100.0,
            rate_a,
            out=np.zeros_like(branch_flow_mw, dtype=np.float64),
            where=rate_a > 0.0,
        )

    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(branch_flow_mw)):
        return _score_failed(contingency, branch.shape[0], "non-finite DC solution")

    max_loading = float(np.nanmax(loading)) if loading.size else 0.0
    return DCScreeningResult(
        contingency=contingency,
        score=max_loading,
        max_loading_percent=max_loading,
        failed=False,
        reason="",
        angles_rad=angles,
        branch_flow_mw=branch_flow_mw,
        loading_percent=loading,
    )


def score_contingencies_dc(
    net,
    contingencies: Iterable[Contingency],
    *,
    options: PandapowerBuildOptions | None = None,
    chunk_size: int = 256,
) -> Iterator[DCScreeningResult]:
    """Yield DC screening scores for contingencies in bounded chunks."""

    base = _build_dc_base_data(net, options)
    for chunk in chunk_contingencies(contingencies, chunk_size=chunk_size):
        for contingency in chunk:
            yield score_contingency_dc(base, contingency)


def screen_top_contingencies(
    net,
    *,
    k: int = 1,
    top_n: int = 10,
    elements: Sequence[str] = ("line", "trafo"),
    max_cases: int | None = None,
    chunk_size: int = 256,
    options: PandapowerBuildOptions | None = None,
    include_failed: bool = True,
) -> tuple[DCScreeningResult, ...]:
    """Return the highest-risk contingencies for full AC validation.

    Failures in the DC proxy are ranked ahead of successful solves. Successful
    cases are ranked by their maximum branch-loading percentage.
    """

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    contingencies = iter_nk_contingencies(net, k=k, elements=elements, max_cases=max_cases)
    heap: list[tuple[float, int, DCScreeningResult]] = []
    for order, result in enumerate(
        score_contingencies_dc(net, contingencies, options=options, chunk_size=chunk_size)
    ):
        if result.failed and not include_failed:
            continue
        rank = float("inf") if result.failed else result.score
        item = (rank, order, result)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    ranked = [
        item[2]
        for item in sorted(
            heap,
            key=lambda item: (not item[2].failed, -item[0], item[1]),
        )
    ]
    return tuple(ranked)


def screen_n1_lodf_top(
    net,
    *,
    top_n: int = 32,
    elements: Sequence[str] = ("line", "trafo"),
    options: PandapowerBuildOptions | None = None,
    chunk_size: int = 256,
) -> N1LODFScreening:
    """Complete DC N-1 screening using PTDF/LODF factors.

    This is the realistic large-grid N-1 screening path: build the DC factors
    once, evaluate every selected active branch outage, and keep only the
    highest-risk post-contingency loading profiles for AC validation.
    """

    from pandapower.pypower.dcpf import dcpf
    from pandapower.pypower.idx_brch import RATE_A
    from pandapower.pypower.makeBdc import makeBdc
    from pandapower.pypower.makeLODF import makeLODF
    from pandapower.pypower.makePTDF import makePTDF
    from pandapower.pypower.makeSbus import makeSbus

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    base = _build_dc_base_data(net, options)
    candidates: list[tuple[Outage, int]] = []
    allowed = set(elements)
    for outage, row in base.branch_rows_by_outage.items():
        if outage.element in allowed:
            candidates.append((outage, row))
    candidates.sort(key=lambda item: (item[0].element, item[0].index))
    if not candidates:
        return N1LODFScreening((), total=0, failed=0, max_base_loading_percent=0.0)

    bbus, bf, pbusinj, pfinj, _ = makeBdc(base.bus, base.branch)
    sbus = makeSbus(base.base_mva, base.bus, base.gen)
    angles = dcpf(bbus, sbus.real - pbusinj, base.va0, base.ref, base.pv, base.pq)
    base_flow = np.asarray((bf @ angles + pfinj) * base.base_mva, dtype=np.float64)

    rate_a = base.branch[:, RATE_A].astype(float, copy=False)
    monitor = rate_a > 0.0
    base_loading = np.divide(
        np.abs(base_flow) * 100.0,
        rate_a,
        out=np.zeros_like(base_flow, dtype=np.float64),
        where=monitor,
    )

    ptdf = makePTDF(base.base_mva, base.bus, base.branch, using_sparse_solver=True)
    lodf = makeLODF(base.branch, ptdf)

    heap: list[tuple[float, int, DCScreeningResult]] = []
    failed = 0
    rows = np.asarray([row for _, row in candidates], dtype=np.int32)
    outages = [outage for outage, _ in candidates]
    for offset in range(0, len(rows), chunk_size):
        chunk_rows = rows[offset : offset + chunk_size]
        chunk_lodf = np.asarray(lodf[:, chunk_rows], dtype=np.float64)
        finite = np.all(np.isfinite(chunk_lodf), axis=0)
        failed += int(np.sum(~finite))

        if not np.any(finite):
            continue

        valid_rows = chunk_rows[finite]
        valid_lodf = chunk_lodf[:, finite]
        post_flow = base_flow[:, None] + valid_lodf * base_flow[valid_rows][None, :]
        post_flow[valid_rows, np.arange(len(valid_rows))] = 0.0
        loading = np.divide(
            np.abs(post_flow) * 100.0,
            rate_a[:, None],
            out=np.zeros_like(post_flow, dtype=np.float64),
            where=monitor[:, None],
        )
        scores = np.max(loading, axis=0)

        valid_indices = np.nonzero(finite)[0] + offset
        for local_col, candidate_index in enumerate(valid_indices):
            score = float(scores[local_col])
            outage = outages[candidate_index]
            contingency = Contingency((outage,))
            result = DCScreeningResult(
                contingency=contingency,
                score=score,
                max_loading_percent=score,
                failed=False,
                reason="",
                angles_rad=angles,
                branch_flow_mw=np.asarray(post_flow[:, local_col], dtype=np.float64),
                loading_percent=np.asarray(loading[:, local_col], dtype=np.float64),
            )
            item = (score, candidate_index, result)
            if len(heap) < top_n:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    ranked = tuple(item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1])))
    return N1LODFScreening(
        screened=ranked,
        total=len(candidates),
        failed=failed,
        max_base_loading_percent=float(np.max(base_loading)) if base_loading.size else 0.0,
    )


def _dc_lodf_factors(net, options: PandapowerBuildOptions | None):
    from pandapower.pypower.dcpf import dcpf
    from pandapower.pypower.idx_brch import RATE_A
    from pandapower.pypower.makeBdc import makeBdc
    from pandapower.pypower.makeLODF import makeLODF
    from pandapower.pypower.makePTDF import makePTDF
    from pandapower.pypower.makeSbus import makeSbus

    base = _build_dc_base_data(net, options)
    bbus, bf, pbusinj, pfinj, _ = makeBdc(base.bus, base.branch)
    sbus = makeSbus(base.base_mva, base.bus, base.gen)
    angles = dcpf(bbus, sbus.real - pbusinj, base.va0, base.ref, base.pv, base.pq)
    base_flow = np.asarray((bf @ angles + pfinj) * base.base_mva, dtype=np.float64)
    rate_a = base.branch[:, RATE_A].astype(float, copy=False)
    ptdf = makePTDF(base.base_mva, base.bus, base.branch, using_sparse_solver=True)
    lodf = makeLODF(base.branch, ptdf)
    return base, angles, base_flow, rate_a, lodf


def screen_nk_lodf_top(
    net,
    *,
    k: int,
    candidate_results: Sequence[DCScreeningResult],
    top_n: int = 32,
    max_combinations: int | None = None,
    options: PandapowerBuildOptions | None = None,
) -> NKLODFScreening:
    """Screen N-k combinations over a candidate pool using multi-outage LODF.

    This is exact for the DC linear model over the candidate pool. It is not an
    exhaustive network-wide N-k guarantee unless the candidate pool contains
    every active branch being considered.
    """

    if k < 2:
        raise ValueError("Use screen_n1_lodf_top for k=1; this function requires k >= 2.")
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    base, angles, base_flow, rate_a, lodf = _dc_lodf_factors(net, options)
    rows_by_outage = base.branch_rows_by_outage
    candidates: list[tuple[Outage, int]] = []
    seen: set[Outage] = set()
    for result in candidate_results:
        for outage in result.contingency.outages:
            if outage not in seen and outage in rows_by_outage:
                seen.add(outage)
                candidates.append((outage, rows_by_outage[outage]))

    if len(candidates) < k:
        return NKLODFScreening((), total=0, failed=0, k=k, candidate_pool_size=len(candidates), exhaustive_over_pool=True)

    monitor = rate_a > 0.0
    heap: list[tuple[float, int, DCScreeningResult]] = []
    failed = 0
    total = 0
    combos = itertools.combinations(range(len(candidates)), k)
    for order, combo_indices in enumerate(combos):
        if max_combinations is not None and total >= max_combinations:
            break
        total += 1
        outages = tuple(candidates[i][0] for i in combo_indices)
        rows = np.asarray([candidates[i][1] for i in combo_indices], dtype=np.int32)
        contingency = Contingency(outages)
        try:
            lcc = np.asarray(lodf[np.ix_(rows, rows)], dtype=np.float64).copy()
            np.fill_diagonal(lcc, 0.0)
            rhs = base_flow[rows]
            redistribution = np.linalg.solve(np.eye(k) - lcc, rhs)
            post_flow = base_flow + np.asarray(lodf[:, rows], dtype=np.float64) @ redistribution
            post_flow[rows] = 0.0
            if not np.all(np.isfinite(post_flow)):
                raise FloatingPointError("non-finite multi-outage flow")
            loading = np.divide(
                np.abs(post_flow) * 100.0,
                rate_a,
                out=np.zeros_like(post_flow, dtype=np.float64),
                where=monitor,
            )
            score = float(np.max(loading)) if loading.size else 0.0
            result = DCScreeningResult(
                contingency=contingency,
                score=score,
                max_loading_percent=score,
                failed=False,
                reason="",
                angles_rad=angles,
                branch_flow_mw=np.asarray(post_flow, dtype=np.float64),
                loading_percent=np.asarray(loading, dtype=np.float64),
            )
        except Exception as exc:
            failed += 1
            result = _score_failed(contingency, base.branch.shape[0], type(exc).__name__)
            score = float("inf")

        item = (score, order, result)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    ranked = tuple(
        item[2]
        for item in sorted(
            heap,
            key=lambda item: (not item[2].failed, -item[0], item[1]),
        )
    )
    return NKLODFScreening(
        screened=ranked,
        total=total,
        failed=failed,
        k=k,
        candidate_pool_size=len(candidates),
        exhaustive_over_pool=max_combinations is None or total < max_combinations,
    )


def hierarchical_ac_security_analysis(
    net,
    *,
    k: int = 1,
    top_n: int = 32,
    elements: Sequence[str] = ("line", "trafo"),
    max_cases: int | None = None,
    screening_chunk_size: int = 256,
    sbus_batch: np.ndarray | None = None,
    build_options: PandapowerBuildOptions | None = None,
    solver_options: NewtonOptions | None = None,
) -> HierarchicalSecurityAnalysis:
    """Screen many N-k contingencies cheaply, then validate the riskiest by AC PF.

    This mirrors the structure used by fast security-analysis tools: register or
    enumerate many outages, rank them with a cheaper approximation, and reserve
    the full nonlinear AC solve for the subset that matters.
    """

    all_contingencies = tuple(iter_nk_contingencies(net, k=k, elements=elements, max_cases=max_cases))
    screened = screen_top_contingencies(
        net,
        k=k,
        top_n=min(top_n, len(all_contingencies)),
        elements=elements,
        max_cases=max_cases,
        chunk_size=screening_chunk_size,
        options=build_options,
        include_failed=False,
    )
    selected = tuple(result.contingency for result in screened if not result.failed)
    ac = None
    if selected:
        ac = analyze_contingencies_batched(
            net,
            selected,
            sbus_batch=sbus_batch,
            build_options=build_options,
            solver_options=solver_options,
        )
    return HierarchicalSecurityAnalysis(
        screened=screened,
        ac=ac,
        total_screened=len(all_contingencies),
    )
