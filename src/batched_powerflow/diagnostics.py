from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .contingency import Contingency, apply_contingency
from .cpu_solver import CpuNewtonOptions, solve_batch_cpu_sparse
from .dc_screening import DCScreeningResult, iter_nk_contingencies, screen_n1_lodf_top, screen_nk_lodf_top
from .model import PowerFlowModel, PowerFlowResult, TopologyBatchResult
from .pandapower_adapter import PandapowerBuildOptions, from_pandapower


Severity = str


@dataclass(frozen=True)
class DiagnosticEvidence:
    """Machine-checkable evidence used by a diagnostic."""

    key: str
    message: str
    value: Any = None
    severity: Severity = "info"


@dataclass(frozen=True)
class DiagnosticAction:
    """Explainable remedial action proposed by the diagnostic."""

    title: str
    rationale: str
    steps: tuple[str, ...]
    risk: str = ""


@dataclass(frozen=True)
class DiagnosticDiagnosis:
    """One likely cause of a failed or risky power-flow case."""

    code: str
    title: str
    severity: Severity
    confidence: float
    cause: str
    evidence: tuple[DiagnosticEvidence, ...] = ()
    actions: tuple[DiagnosticAction, ...] = ()


@dataclass(frozen=True)
class DiagnosticIndicator:
    """Quasi-static stability indicator derived from available power-flow data."""

    name: str
    category: str
    status: Severity
    value: Any
    unit: str = ""
    explanation: str = ""
    evidence: tuple[DiagnosticEvidence, ...] = ()


@dataclass(frozen=True)
class DiagnosticReport:
    """Human-readable and programmatic explanation of a power-flow failure."""

    status: str
    summary: str
    diagnoses: tuple[DiagnosticDiagnosis, ...]
    indicators: tuple[DiagnosticIndicator, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f"# Diagnostic Power-Flow Explanation", "", f"Status: `{self.status}`", "", self.summary]
        if self.context:
            lines.extend(["", "## Context"])
            for key, value in sorted(self.context.items()):
                lines.append(f"- `{key}`: {value}")
        if self.indicators:
            lines.extend(["", "## Stability Indicators"])
            for indicator in self.indicators:
                unit = f" {indicator.unit}" if indicator.unit else ""
                lines.append(
                    f"- `{indicator.category}.{indicator.name}`: `{indicator.status}` value={indicator.value}{unit}. {indicator.explanation}"
                )
                for item in indicator.evidence:
                    suffix = f" value={item.value}" if item.value is not None else ""
                    lines.append(f"  - evidence [{item.severity}] `{item.key}`: {item.message}{suffix}")
        if self.diagnoses:
            lines.extend(["", "## Diagnoses"])
        for diagnosis in self.diagnoses:
            lines.extend(
                [
                    "",
                    f"### {diagnosis.title}",
                    f"- code: `{diagnosis.code}`",
                    f"- severity: `{diagnosis.severity}`",
                    f"- confidence: `{diagnosis.confidence:.2f}`",
                    f"- cause: {diagnosis.cause}",
                ]
            )
            if diagnosis.evidence:
                lines.append("- evidence:")
                for item in diagnosis.evidence:
                    suffix = f" value={item.value}" if item.value is not None else ""
                    lines.append(f"  - [{item.severity}] `{item.key}`: {item.message}{suffix}")
            if diagnosis.actions:
                lines.append("- possible actions:")
                for action in diagnosis.actions:
                    lines.append(f"  - {action.title}: {action.rationale}")
                    for step in action.steps:
                        lines.append(f"    - {step}")
                    if action.risk:
                        lines.append(f"    - Risk: {action.risk}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DiagnosticSecurityViolation:
    """One contingency-level security or solvability violation."""

    contingency: Contingency
    kind: str
    severity: Severity
    metric: float | None = None
    limit: float | None = None
    message: str = ""
    report: DiagnosticReport | None = None


@dataclass(frozen=True)
class DiagnosticSecurityReport:
    """Complete diagnostic security-analysis summary for a chosen contingency universe."""

    safe: bool
    guarantee_scope: str
    k: int
    total_screened: int
    dc_failed: int
    ac_validated: int
    ac_converged: int
    exhaustive_dc: bool
    exhaustive_ac: bool
    max_dc_loading_percent: float
    max_ac_loading_percent: float | None
    min_ac_vm_pu: float | None
    max_ac_vm_pu: float | None
    violations: tuple[DiagnosticSecurityViolation, ...]
    indicators: tuple[DiagnosticIndicator, ...]

    def to_markdown(self) -> str:
        verdict = "SAFE" if self.safe else "UNSAFE"
        lines = [
            "# Complete Diagnostic Security Analysis",
            "",
            f"Verdict: `{verdict}`",
            "",
            f"Guarantee scope: {self.guarantee_scope}",
            "",
            "## Summary",
            f"- `k`: {self.k}",
            f"- `total_screened`: {self.total_screened}",
            f"- `dc_failed_or_islanded`: {self.dc_failed}",
            f"- `ac_validated`: {self.ac_validated}",
            f"- `ac_converged`: {self.ac_converged}",
            f"- `exhaustive_dc`: {self.exhaustive_dc}",
            f"- `exhaustive_ac`: {self.exhaustive_ac}",
            f"- `max_dc_loading_percent`: {self.max_dc_loading_percent}",
            f"- `max_ac_loading_percent`: {self.max_ac_loading_percent}",
            f"- `min_ac_vm_pu`: {self.min_ac_vm_pu}",
            f"- `max_ac_vm_pu`: {self.max_ac_vm_pu}",
        ]
        if self.indicators:
            lines.extend(["", "## Stability Indicators"])
            for indicator in self.indicators:
                unit = f" {indicator.unit}" if indicator.unit else ""
                lines.append(f"- `{indicator.category}.{indicator.name}`: `{indicator.status}` value={indicator.value}{unit}. {indicator.explanation}")
        if self.violations:
            lines.extend(["", "## Violations"])
            for violation in self.violations:
                metric = "" if violation.metric is None else f" metric={violation.metric}"
                limit = "" if violation.limit is None else f" limit={violation.limit}"
                lines.append(f"- `{violation.kind}` `{violation.severity}` contingency=`{violation.contingency}`{metric}{limit}: {violation.message}")
        return "\n".join(lines) + "\n"


def explain_powerflow(
    net,
    *,
    model: PowerFlowModel | None = None,
    result: PowerFlowResult | TopologyBatchResult | None = None,
    dc_result: DCScreeningResult | None = None,
    contingency: Contingency | None = None,
    exception: BaseException | None = None,
) -> DiagnosticReport:
    """Build a diagnostic report for a failed or suspicious power-flow case.

    The diagnostic tool is intentionally explanatory and conservative. It does not
    prove infeasibility. It reports detected evidence, likely causes, and
    concrete actions to test.
    """

    diagnoses: list[DiagnosticDiagnosis] = []
    context = _context(net, model, contingency)
    indicators = _stability_indicators(net, model=model, result=result)

    diagnoses.extend(_diagnose_exception(exception))
    diagnoses.extend(_diagnose_slack_and_islands(net))
    diagnoses.extend(_diagnose_zero_impedance(net))
    if model is not None:
        diagnoses.extend(_diagnose_model_singularity(model))
    if result is not None:
        diagnoses.extend(_diagnose_solver_result(model, result))
    if dc_result is not None:
        diagnoses.extend(_diagnose_dc_screen(dc_result))

    diagnoses = _dedupe_rank(diagnoses)
    status = _status(diagnoses, indicators)
    summary = _summary(status, diagnoses)
    return DiagnosticReport(status=status, summary=summary, diagnoses=tuple(diagnoses), indicators=tuple(indicators), context=context)


def explain_dc_screening_result(net, result: DCScreeningResult) -> DiagnosticReport:
    """Explain one DC screening result, typically a failed or overloaded contingency."""

    try:
        work = apply_contingency(net, result.contingency)
    except Exception:
        work = net
    return explain_powerflow(work, dc_result=result, contingency=result.contingency)


def complete_diagnostic_security_analysis(
    net,
    *,
    k: int = 1,
    elements: tuple[str, ...] = ("line", "trafo"),
    max_cases: int | None = None,
    n1_pool: int | None = None,
    top_n: int = 20,
    ac_mode: str = "top",
    voltage_min: float = 0.90,
    voltage_max: float = 1.10,
    loading_limit_percent: float = 100.0,
    build_options: PandapowerBuildOptions | None = None,
    cpu_options: CpuNewtonOptions | None = None,
) -> DiagnosticSecurityReport:
    """Run complete diagnostic security analysis for a selected N-k universe.

    `SAFE` is only reported when all evaluated checks pass within the declared
    scope. For k > 1, the default is exhaustive over a selected N-1 risk pool;
    set `n1_pool=None` and avoid `max_cases` only if that combinatorial universe
    is tractable.
    """

    if k < 1:
        raise ValueError("k must be >= 1.")
    if ac_mode not in {"none", "top", "all"}:
        raise ValueError("ac_mode must be 'none', 'top', or 'all'.")

    build_options = build_options or PandapowerBuildOptions(check_connectivity=False, preconditioner="none")
    cpu_options = cpu_options or CpuNewtonOptions(max_iterations=30, tolerance=1e-10)
    violations: list[DiagnosticSecurityViolation] = []
    ac_results: list[tuple[Contingency, PowerFlowResult, DiagnosticReport, np.ndarray | None]] = []

    if k == 1:
        dc = screen_n1_lodf_top(net, top_n=top_n if ac_mode != "all" else max(top_n, 1), elements=elements, options=build_options)
        screened = list(dc.screened)
        total = dc.total
        dc_failed = dc.failed
        max_dc_loading = max((result.max_loading_percent for result in screened if np.isfinite(result.max_loading_percent)), default=0.0)
        exhaustive_dc = max_cases is None
        if max_cases is not None:
            total = min(total, max_cases)
            screened = screened[:max_cases]
    else:
        pool_size = n1_pool or top_n
        n1 = screen_n1_lodf_top(net, top_n=pool_size, elements=elements, options=build_options)
        dc = screen_nk_lodf_top(
            net,
            k=k,
            candidate_results=n1.screened,
            top_n=top_n,
            max_combinations=max_cases,
            options=build_options,
        )
        screened = list(dc.screened)
        total = dc.total
        dc_failed = dc.failed
        max_dc_loading = max((result.max_loading_percent for result in screened if np.isfinite(result.max_loading_percent)), default=0.0)
        exhaustive_dc = dc.exhaustive_over_pool

    for result in screened:
        if result.failed:
            violations.append(
                DiagnosticSecurityViolation(
                    contingency=result.contingency,
                    kind="dc_failed_or_islanded",
                    severity="critical",
                    metric=result.score,
                    message=result.reason or "DC screen returned a failed or non-finite contingency.",
                    report=explain_dc_screening_result(net, result),
                )
            )
        elif result.max_loading_percent > loading_limit_percent:
            violations.append(
                DiagnosticSecurityViolation(
                    contingency=result.contingency,
                    kind="dc_overload",
                    severity="warning",
                    metric=float(result.max_loading_percent),
                    limit=loading_limit_percent,
                    message="DC post-contingency loading exceeds the configured loading limit.",
                    report=explain_dc_screening_result(net, result),
                )
            )

    if ac_mode == "all":
        selected_for_ac = [result.contingency for result in screened]
    elif ac_mode == "top":
        selected_for_ac = [result.contingency for result in screened[:top_n]]
    else:
        selected_for_ac = []

    for contingency in selected_for_ac:
        work = apply_contingency(net, contingency)
        model = None
        ac_result = None
        exception = None
        try:
            model = from_pandapower(work, options=build_options)
            ac_result = solve_batch_cpu_sparse(model, options=cpu_options)
        except Exception as exc:
            exception = exc
        report = explain_powerflow(work, model=model, result=ac_result, contingency=contingency, exception=exception)
        loading = _ac_loading_percent(model, ac_result.voltages[0]) if model is not None and ac_result is not None and ac_result.voltages.size else None
        if ac_result is None or not bool(ac_result.converged[0]):
            violations.append(
                DiagnosticSecurityViolation(
                    contingency=contingency,
                    kind="ac_non_converged",
                    severity="critical",
                    metric=None if ac_result is None else float(ac_result.mismatch_norm[0]),
                    message="AC validation did not converge for this contingency.",
                    report=report,
                )
            )
        else:
            vm = np.abs(ac_result.voltages[0])
            if float(np.min(vm)) < voltage_min:
                violations.append(
                    DiagnosticSecurityViolation(contingency=contingency, kind="ac_voltage_low", severity="warning", metric=float(np.min(vm)), limit=voltage_min, message="Minimum AC voltage is below the configured limit.", report=report)
                )
            if float(np.max(vm)) > voltage_max:
                violations.append(
                    DiagnosticSecurityViolation(contingency=contingency, kind="ac_voltage_high", severity="warning", metric=float(np.max(vm)), limit=voltage_max, message="Maximum AC voltage is above the configured limit.", report=report)
                )
            if loading is not None and float(np.nanmax(loading)) > loading_limit_percent:
                violations.append(
                    DiagnosticSecurityViolation(contingency=contingency, kind="ac_branch_overload", severity="warning", metric=float(np.nanmax(loading)), limit=loading_limit_percent, message="AC branch loading exceeds the configured loading limit.", report=report)
                )
        if ac_result is not None:
            ac_results.append((contingency, ac_result, report, loading))

    ac_converged = int(sum(bool(result.converged[0]) for _, result, _, _ in ac_results))
    max_ac_loading = max((float(np.nanmax(loading)) for _, _, _, loading in ac_results if loading is not None), default=None)
    min_ac_vm = min((float(np.min(np.abs(result.voltages[0]))) for _, result, _, _ in ac_results if result.voltages.size), default=None)
    max_ac_vm = max((float(np.max(np.abs(result.voltages[0]))) for _, result, _, _ in ac_results if result.voltages.size), default=None)
    indicators = list(_frequency_stability_indicators(net))
    if ac_results:
        indicators.extend(_aggregate_ac_voltage_indicators(ac_results, voltage_min, voltage_max))

    exhaustive_ac = ac_mode == "all" and len(selected_for_ac) == total
    if ac_mode == "none":
        guarantee_scope = f"Complete DC N-{k} over selected elements {elements}; no AC validation."
    elif exhaustive_ac:
        guarantee_scope = f"Complete DC and exhaustive selected AC N-{k} over selected elements {elements}."
    else:
        guarantee_scope = f"Complete DC N-{k} over selected elements {elements}; AC validation only for {len(selected_for_ac)} selected contingencies."

    safe = dc_failed == 0 and not violations and exhaustive_dc and (ac_mode == "none" or exhaustive_ac)
    return DiagnosticSecurityReport(
        safe=safe,
        guarantee_scope=guarantee_scope,
        k=k,
        total_screened=total,
        dc_failed=dc_failed,
        ac_validated=len(selected_for_ac),
        ac_converged=ac_converged,
        exhaustive_dc=exhaustive_dc,
        exhaustive_ac=exhaustive_ac,
        max_dc_loading_percent=float(max_dc_loading),
        max_ac_loading_percent=max_ac_loading,
        min_ac_vm_pu=min_ac_vm,
        max_ac_vm_pu=max_ac_vm,
        violations=tuple(violations),
        indicators=tuple(indicators),
    )


def _aggregate_ac_voltage_indicators(
    ac_results: list[tuple[Contingency, PowerFlowResult, DiagnosticReport, np.ndarray | None]],
    voltage_min: float,
    voltage_max: float,
) -> list[DiagnosticIndicator]:
    vm_rows = []
    contingencies = []
    for contingency, result, _report, _loading in ac_results:
        if result.voltages.size:
            vm_rows.append(np.abs(result.voltages[0]))
            contingencies.append(str(contingency))
    if not vm_rows:
        return []
    vm = np.asarray(vm_rows, dtype=np.float64)
    finite = np.isfinite(vm)
    if not np.all(finite):
        return [
            DiagnosticIndicator(
                name="ac_voltage_finite",
                category="voltage",
                status="critical",
                value=False,
                explanation="At least one AC-validated contingency has non-finite voltage magnitude.",
            )
        ]
    low_mask = vm < voltage_min
    high_mask = vm > voltage_max
    min_vm = float(np.min(vm))
    max_vm = float(np.max(vm))
    min_margin = float(min(np.min(vm - voltage_min), np.min(voltage_max - vm)))
    worst_low = _worst_security_voltage(vm, contingencies, voltage_min, low=True)
    worst_high = _worst_security_voltage(vm, contingencies, voltage_max, low=False)
    return [
        DiagnosticIndicator(
            name="ac_min_vm_pu",
            category="voltage",
            status=_indicator_status(critical=min_vm < voltage_min - 0.05, warning=np.any(low_mask)),
            value=min_vm,
            unit="pu",
            explanation="Minimum AC voltage magnitude across all validated contingencies.",
            evidence=(DiagnosticEvidence("worst_low_voltage_contingencies", "Worst low-voltage AC validation points.", value=worst_low, severity="warning"),) if worst_low else (),
        ),
        DiagnosticIndicator(
            name="ac_max_vm_pu",
            category="voltage",
            status=_indicator_status(critical=max_vm > voltage_max + 0.05, warning=np.any(high_mask)),
            value=max_vm,
            unit="pu",
            explanation="Maximum AC voltage magnitude across all validated contingencies.",
            evidence=(DiagnosticEvidence("worst_high_voltage_contingencies", "Worst high-voltage AC validation points.", value=worst_high, severity="warning"),) if worst_high else (),
        ),
        DiagnosticIndicator(
            name="ac_voltage_limit_violations",
            category="voltage",
            status=_indicator_status(critical=min_vm < voltage_min - 0.05 or max_vm > voltage_max + 0.05, warning=np.any(low_mask) or np.any(high_mask)),
            value={"low": int(np.sum(low_mask)), "high": int(np.sum(high_mask)), "validated_contingencies": len(vm_rows)},
            explanation="Counts AC voltage magnitudes outside the configured voltage limits across validated contingencies.",
        ),
        DiagnosticIndicator(
            name="ac_minimum_voltage_margin",
            category="voltage",
            status=_indicator_status(critical=min_margin < -0.05, warning=min_margin < 0.0),
            value=min_margin,
            unit="pu",
            explanation="Smallest distance to configured AC voltage limits across validated contingencies. Negative values indicate a violation.",
        ),
        DiagnosticIndicator(
            name="ac_voltage_spread",
            category="voltage",
            status=_indicator_status(warning=(max_vm - min_vm) > 0.20),
            value=float(max_vm - min_vm),
            unit="pu",
            explanation="Range between maximum and minimum AC voltage over all validated contingencies.",
        ),
    ]


def _worst_security_voltage(vm: np.ndarray, contingencies: list[str], limit: float, *, low: bool, n: int = 5) -> list[dict[str, Any]]:
    margin = vm - limit if low else limit - vm
    worst = np.argsort(margin.ravel())[:n]
    rows = []
    for flat in worst:
        case_idx, bus = np.unravel_index(int(flat), margin.shape)
        if margin[case_idx, bus] >= 0:
            continue
        rows.append(
            {
                "contingency": contingencies[case_idx],
                "internal_bus": int(bus),
                "vm_pu": float(vm[case_idx, bus]),
                "limit_pu": float(limit),
                "margin_pu": float(margin[case_idx, bus]),
            }
        )
    return rows


def _context(net, model: PowerFlowModel | None, contingency: Contingency | None) -> dict[str, Any]:
    context = {
        "buses": len(net.bus) if hasattr(net, "bus") else 0,
        "lines": len(net.line) if hasattr(net, "line") else 0,
        "trafos": len(net.trafo) if hasattr(net, "trafo") else 0,
    }
    if model is not None:
        context.update({"internal_buses": model.n_bus, "state_variables": model.n_state})
    if contingency is not None:
        context["contingency"] = str(contingency)
    return context


def _status(diagnoses: Iterable[DiagnosticDiagnosis], indicators: Iterable[DiagnosticIndicator] = ()) -> str:
    severities = {diagnosis.severity for diagnosis in diagnoses}
    severities.update(indicator.status for indicator in indicators)
    if "critical" in severities:
        return "insecure_or_infeasible"
    if "warning" in severities:
        return "risky"
    return "ok"


def _summary(status: str, diagnoses: list[DiagnosticDiagnosis]) -> str:
    if not diagnoses:
        return "No deterministic failure cause was detected by the diagnostic checks."
    top = diagnoses[0]
    return f"Most likely cause: {top.title}. The case status is `{status}` based on {len(diagnoses)} diagnostic finding(s)."


def _dedupe_rank(diagnoses: list[DiagnosticDiagnosis]) -> list[DiagnosticDiagnosis]:
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    by_code: dict[str, DiagnosticDiagnosis] = {}
    for diagnosis in diagnoses:
        current = by_code.get(diagnosis.code)
        if current is None or diagnosis.confidence > current.confidence:
            by_code[diagnosis.code] = diagnosis
    return sorted(
        by_code.values(),
        key=lambda item: (severity_rank.get(item.severity, 3), -item.confidence, item.code),
    )


def _indicator_status(*, critical: bool = False, warning: bool = False, unknown: bool = False) -> Severity:
    if critical:
        return "critical"
    if warning:
        return "warning"
    if unknown:
        return "unknown"
    return "info"


def _stability_indicators(
    net,
    *,
    model: PowerFlowModel | None = None,
    result: PowerFlowResult | TopologyBatchResult | None = None,
) -> list[DiagnosticIndicator]:
    indicators: list[DiagnosticIndicator] = []
    indicators.extend(_voltage_stability_indicators(model, result))
    indicators.extend(_frequency_stability_indicators(net))
    return indicators


def _voltage_limits(model: PowerFlowModel, n_solutions: int) -> tuple[np.ndarray, np.ndarray]:
    vmin = np.full((model.n_bus,), 0.95, dtype=np.float64)
    vmax = np.full((model.n_bus,), 1.05, dtype=np.float64)
    if model.ppc is not None and "bus" in model.ppc:
        try:
            from pandapower.pypower.idx_bus import VMAX, VMIN

            vmin = np.asarray(model.ppc["bus"][:, VMIN], dtype=np.float64)
            vmax = np.asarray(model.ppc["bus"][:, VMAX], dtype=np.float64)
        except Exception:
            pass
    return (
        np.broadcast_to(vmin[None, :], (n_solutions, model.n_bus)),
        np.broadcast_to(vmax[None, :], (n_solutions, model.n_bus)),
    )


def _voltage_stability_indicators(
    model: PowerFlowModel | None,
    result: PowerFlowResult | TopologyBatchResult | None,
) -> list[DiagnosticIndicator]:
    if model is None or result is None or np.asarray(result.voltages).size == 0:
        return [
            DiagnosticIndicator(
                name="voltage_solution_available",
                category="voltage",
                status="unknown",
                value=False,
                explanation="No solved voltage vector was supplied, so voltage stability indicators cannot be evaluated.",
            )
        ]

    voltages = np.asarray(result.voltages)
    try:
        vm = np.abs(voltages).reshape(-1, model.n_bus)
    except ValueError:
        return [
            DiagnosticIndicator(
                name="voltage_solution_shape",
                category="voltage",
                status="warning",
                value=str(voltages.shape),
                explanation="Voltage array shape does not match the model bus count.",
            )
        ]
    finite = np.isfinite(vm)
    if not np.all(finite):
        return [
            DiagnosticIndicator(
                name="voltage_finite",
                category="voltage",
                status="critical",
                value=False,
                explanation="At least one voltage magnitude is NaN or infinite; voltage stability margins are invalid.",
            )
        ]

    vmin, vmax = _voltage_limits(model, vm.shape[0])
    low_mask = vm < vmin
    high_mask = vm > vmax
    lower_margin = vm - vmin
    upper_margin = vmax - vm
    min_vm = float(np.min(vm))
    max_vm = float(np.max(vm))
    low_count = int(np.sum(low_mask))
    high_count = int(np.sum(high_mask))
    min_margin = float(min(np.min(lower_margin), np.min(upper_margin)))
    hard_critical = min_vm < 0.90 or max_vm > 1.10
    limit_warning = low_count > 0 or high_count > 0

    worst_low = _worst_voltage_buses(vm, vmin, low=True)
    worst_high = _worst_voltage_buses(vm, vmax, low=False)
    return [
        DiagnosticIndicator(
            name="min_vm_pu",
            category="voltage",
            status=_indicator_status(critical=min_vm < 0.90, warning=low_count > 0),
            value=min_vm,
            unit="pu",
            explanation="Minimum solved voltage magnitude across all supplied scenarios/topologies.",
            evidence=(DiagnosticEvidence("worst_low_voltage_buses", "Lowest voltage buses.", value=worst_low, severity="warning"),) if worst_low else (),
        ),
        DiagnosticIndicator(
            name="max_vm_pu",
            category="voltage",
            status=_indicator_status(critical=max_vm > 1.10, warning=high_count > 0),
            value=max_vm,
            unit="pu",
            explanation="Maximum solved voltage magnitude across all supplied scenarios/topologies.",
            evidence=(DiagnosticEvidence("worst_high_voltage_buses", "Highest voltage buses.", value=worst_high, severity="warning"),) if worst_high else (),
        ),
        DiagnosticIndicator(
            name="voltage_limit_violations",
            category="voltage",
            status=_indicator_status(critical=hard_critical, warning=limit_warning),
            value={"low": low_count, "high": high_count},
            explanation="Counts voltage magnitudes outside pandapower VMIN/VMAX limits, defaulting to 0.95/1.05 pu if limits are unavailable.",
        ),
        DiagnosticIndicator(
            name="minimum_voltage_margin",
            category="voltage",
            status=_indicator_status(critical=min_margin < -0.05, warning=min_margin < 0.0),
            value=min_margin,
            unit="pu",
            explanation="Smallest distance to either lower or upper voltage limit. Negative values indicate a limit violation.",
        ),
        DiagnosticIndicator(
            name="voltage_spread",
            category="voltage",
            status=_indicator_status(warning=(max_vm - min_vm) > 0.20),
            value=float(max_vm - min_vm),
            unit="pu",
            explanation="Range between maximum and minimum solved voltage magnitude. Large spread is a static stress indicator.",
        ),
    ]


def _worst_voltage_buses(vm: np.ndarray, limit: np.ndarray, *, low: bool, n: int = 5) -> list[dict[str, Any]]:
    margin = vm - limit if low else limit - vm
    worst = np.argsort(margin.ravel())[:n]
    rows = []
    for flat in worst:
        scenario, bus = np.unravel_index(int(flat), margin.shape)
        if margin[scenario, bus] >= 0:
            continue
        rows.append(
            {
                "solution": int(scenario),
                "internal_bus": int(bus),
                "vm_pu": float(vm[scenario, bus]),
                "limit_pu": float(limit[scenario, bus]),
                "margin_pu": float(margin[scenario, bus]),
            }
        )
    return rows


def _frequency_stability_indicators(net) -> list[DiagnosticIndicator]:
    graph = _bus_graph(net)
    if graph is None or not hasattr(net, "bus") or len(net.bus) == 0:
        return [
            DiagnosticIndicator(
                name="frequency_proxy_available",
                category="frequency",
                status="unknown",
                value=False,
                explanation="No bus graph is available, so island frequency-risk proxies cannot be evaluated.",
            )
        ]

    n_components, labels = csgraph.connected_components(graph, directed=False)
    bus_index = list(net.bus.index)
    bus_to_component = {int(bus): int(labels[pos]) for pos, bus in enumerate(bus_index)}
    slack_buses = _slack_buses(net)
    slack_components = {bus_to_component[bus] for bus in slack_buses if bus in bus_to_component}
    load_by_component = np.zeros(n_components, dtype=np.float64)
    generation_by_component = np.zeros(n_components, dtype=np.float64)
    inertia_by_component = np.zeros(n_components, dtype=np.float64)
    droop_by_component = np.zeros(n_components, dtype=np.float64)

    _accumulate_component_power(net, "load", "bus", "p_mw", bus_to_component, load_by_component, sign=1.0)
    _accumulate_component_power(net, "gen", "bus", "p_mw", bus_to_component, generation_by_component, sign=1.0)
    _accumulate_component_power(net, "sgen", "bus", "p_mw", bus_to_component, generation_by_component, sign=1.0)
    _accumulate_component_power(net, "storage", "bus", "p_mw", bus_to_component, generation_by_component, sign=-1.0)
    _accumulate_component_metadata(net, "gen", "bus", ("h_s", "inertia_s", "H"), bus_to_component, inertia_by_component)
    _accumulate_component_metadata(net, "gen", "bus", ("droop_percent", "droop_pu", "governor_droop"), bus_to_component, droop_by_component)

    component_rows = []
    unsupported = []
    for component in range(n_components):
        buses = [int(bus) for bus, comp in bus_to_component.items() if comp == component]
        load = float(load_by_component[component])
        generation = float(generation_by_component[component])
        imbalance = generation - load
        has_slack = component in slack_components
        component_rows.append(
            {
                "component": component,
                "n_buses": len(buses),
                "has_slack": has_slack,
                "load_mw": load,
                "scheduled_generation_mw": generation,
                "scheduled_imbalance_mw": imbalance,
                "imbalance_to_load": float(abs(imbalance) / max(load, 1e-9)),
                "inertia_metadata": float(inertia_by_component[component]),
                "droop_metadata": float(droop_by_component[component]),
                "sample_buses": buses[:10],
            }
        )
        if load > 1e-9 and not has_slack:
            unsupported.append(component_rows[-1])

    largest_ratio = max((row["imbalance_to_load"] for row in component_rows if row["load_mw"] > 1e-9), default=0.0)
    missing_dynamic_data = bool(np.sum(inertia_by_component) == 0.0 and np.sum(droop_by_component) == 0.0)
    indicators = [
        DiagnosticIndicator(
            name="island_slack_support",
            category="frequency",
            status=_indicator_status(critical=bool(unsupported)),
            value={"unsupported_load_islands": len(unsupported), "components": n_components},
            explanation="Frequency control requires a reference/frequency-forming source in every energized island. This static check flags load islands without slack.",
            evidence=(DiagnosticEvidence("unsupported_load_islands", "Load islands without slack/frequency reference.", value=unsupported, severity="critical"),) if unsupported else (),
        ),
        DiagnosticIndicator(
            name="scheduled_active_power_imbalance",
            category="frequency",
            status=_indicator_status(warning=largest_ratio > 0.10),
            value={"max_abs_imbalance_to_load": largest_ratio, "components": component_rows},
            explanation="Static proxy for frequency stress: scheduled non-slack generation minus load per island. Slack covers this in PF, but a large ratio implies high frequency-control burden after disturbances.",
        ),
        DiagnosticIndicator(
            name="dynamic_frequency_data",
            category="frequency",
            status=_indicator_status(unknown=missing_dynamic_data),
            value={"inertia_metadata_sum": float(np.sum(inertia_by_component)), "droop_metadata_sum": float(np.sum(droop_by_component))},
            explanation="Power flow alone cannot compute RoCoF or frequency nadir. Provide inertia/governor metadata to upgrade this from a proxy to a quantitative frequency-stability indicator.",
        ),
    ]
    return indicators


def _ac_loading_percent(model: PowerFlowModel, voltage: np.ndarray) -> np.ndarray | None:
    if model.ppc is None:
        return None
    try:
        from pandapower.pypower.idx_brch import RATE_A
        from .solver import compute_branch_flows

        sf, st = compute_branch_flows(model, voltage)
        rate_a = np.asarray(model.ppc["branch"][:, RATE_A], dtype=np.float64)
        branch_mva = np.maximum(np.abs(sf[0]), np.abs(st[0]))
        return np.divide(
            branch_mva * 100.0,
            rate_a,
            out=np.zeros_like(branch_mva, dtype=np.float64),
            where=rate_a > 0,
        )
    except Exception:
        return None


def _accumulate_component_power(
    net,
    table_name: str,
    bus_column: str,
    power_column: str,
    bus_to_component: dict[int, int],
    target: np.ndarray,
    *,
    sign: float,
) -> None:
    if not hasattr(net, table_name):
        return
    table = getattr(net, table_name)
    if table is None or len(table) == 0 or bus_column not in table or power_column not in table:
        return
    mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
    for _, row in table.loc[mask].iterrows():
        bus = int(row[bus_column])
        component = bus_to_component.get(bus)
        if component is not None:
            target[component] += sign * float(row[power_column])


def _accumulate_component_metadata(
    net,
    table_name: str,
    bus_column: str,
    columns: tuple[str, ...],
    bus_to_component: dict[int, int],
    target: np.ndarray,
) -> None:
    if not hasattr(net, table_name):
        return
    table = getattr(net, table_name)
    if table is None or len(table) == 0 or bus_column not in table:
        return
    mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
    for column in columns:
        if column not in table:
            continue
        for _, row in table.loc[mask].iterrows():
            bus = int(row[bus_column])
            component = bus_to_component.get(bus)
            value = row[column]
            if component is not None and value == value:
                target[component] += float(value)


def _action_restore_connectivity() -> DiagnosticAction:
    return DiagnosticAction(
        title="Restore electrical connectivity or define island operation",
        rationale="A singular DC/AC system usually means at least one island has no reference/slack or no valid path to supply.",
        steps=(
            "Identify the island created by the outage and restore at least one branch path.",
            "If islanded operation is intended, add one slack/reference source in every island.",
            "If the island has load but no generation, shed or transfer that load before solving.",
        ),
        risk="Adding an artificial slack can hide an actual security violation if the island is not physically supportable.",
    )


def _action_reduce_stress() -> DiagnosticAction:
    return DiagnosticAction(
        title="Reduce electrical stress before AC validation",
        rationale="High DC loading or high Newton residual indicates the operating point is far from a feasible AC solution.",
        steps=(
            "Redispatch generation to reduce transfer across the overloaded corridor.",
            "Switch topology or add parallel paths to reduce post-contingency loading.",
            "Apply remedial load shedding only after verifying operational policy permits it.",
        ),
    )


def _diagnose_exception(exception: BaseException | None) -> list[DiagnosticDiagnosis]:
    if exception is None:
        return []
    message = repr(exception)
    lowered = message.lower()
    if "singular" in lowered or "matrixrankwarning" in lowered:
        return [
            DiagnosticDiagnosis(
                code="linear_system_singular",
                title="Singular Newton or DC linear system",
                severity="critical",
                confidence=0.95,
                cause="The linearized equations could not be solved, commonly because an island lacks a slack/reference or because a zero-impedance/numerically degenerate branch makes the admittance/Jacobian singular.",
                evidence=(DiagnosticEvidence("exception", message, severity="critical"),),
                actions=(_action_restore_connectivity(), _action_fix_zero_impedance()),
            )
        ]
    if "did not converge" in lowered or "powerflownotconverged" in lowered:
        return [
            DiagnosticDiagnosis(
                code="solver_non_convergence",
                title="Newton solver did not converge",
                severity="warning",
                confidence=0.75,
                cause="The nonlinear iterations reached the iteration budget before the mismatch tolerance was met.",
                evidence=(DiagnosticEvidence("exception", message, severity="warning"),),
                actions=(
                    DiagnosticAction(
                        title="Improve initialization and check feasibility",
                        rationale="Poor starts and stressed or islanded cases are common causes of non-convergence.",
                        steps=(
                            "Use pandapower-compatible `init='auto'` or a previous solved state.",
                            "Run DC screening to identify overloaded corridors before AC validation.",
                            "Check reactive limits, voltage support, and island slack assignment.",
                        ),
                    ),
                ),
            )
        ]
    return [
        DiagnosticDiagnosis(
            code="powerflow_exception",
            title="Power-flow exception",
            severity="warning",
            confidence=0.50,
            cause="The backend raised an exception that the diagnostic classifier does not yet map to a specific physical cause.",
            evidence=(DiagnosticEvidence("exception", message, severity="warning"),),
            actions=(
                DiagnosticAction(
                    title="Inspect backend exception and network data",
                    rationale="Unclassified exceptions often come from unsupported elements or inconsistent input data.",
                    steps=("Validate pandapower tables.", "Run the base case before contingencies.", "Report the exception with the network and contingency id."),
                ),
            ),
        )
    ]


def _diagnose_slack_and_islands(net) -> list[DiagnosticDiagnosis]:
    if not hasattr(net, "bus") or len(net.bus) == 0:
        return []
    active_buses = set(int(i) for i in net.bus.index)
    slack_buses = _slack_buses(net)
    evidence: list[DiagnosticEvidence] = []
    diagnoses: list[DiagnosticDiagnosis] = []
    if not slack_buses:
        diagnoses.append(
            DiagnosticDiagnosis(
                code="missing_slack",
                title="No in-service slack/reference source",
                severity="critical",
                confidence=0.98,
                cause="AC and DC power flow need at least one reference bus. Without it, angles and active-power balance are undefined.",
                evidence=(DiagnosticEvidence("slack_buses", "No in-service ext_grid or slack generator was found.", value=[], severity="critical"),),
                actions=(
                    DiagnosticAction(
                        title="Add or restore a reference source",
                        rationale="A reference source anchors voltage angle and absorbs active-power imbalance.",
                        steps=("Restore an in-service ext_grid.", "Mark one generator as slack.", "For islanded operation, add one slack per energized island."),
                    ),
                ),
            )
        )

    graph = _bus_graph(net)
    if graph is None:
        return diagnoses
    n_components, labels = csgraph.connected_components(graph, directed=False)
    component_buses: dict[int, list[int]] = {i: [] for i in range(n_components)}
    bus_positions = list(net.bus.index)
    for pos, label in enumerate(labels):
        bus = int(bus_positions[pos])
        if bus in active_buses:
            component_buses[int(label)].append(bus)
    slack_components = {int(labels[bus_positions.index(bus)]) for bus in slack_buses if bus in bus_positions}
    load_buses = _load_buses(net)
    unsupplied = []
    for component, buses in component_buses.items():
        if component not in slack_components and any(bus in load_buses for bus in buses):
            unsupplied.append({"component": component, "n_buses": len(buses), "sample_buses": buses[:10]})
    if unsupplied:
        evidence.append(DiagnosticEvidence("unsupplied_islands", "One or more energized load islands have no slack/reference source.", value=unsupplied, severity="critical"))
        diagnoses.append(
            DiagnosticDiagnosis(
                code="island_without_slack",
                title="Island without slack/reference",
                severity="critical",
                confidence=0.95,
                cause="A contingency or network state split the grid into at least one island containing load but no reference source.",
                evidence=tuple(evidence),
                actions=(_action_restore_connectivity(),),
            )
        )
    return diagnoses


def _slack_buses(net) -> set[int]:
    slack = set()
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        table = net.ext_grid
        mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
        slack.update(int(bus) for bus in table.loc[mask, "bus"].to_numpy())
    if hasattr(net, "gen") and len(net.gen) and "slack" in net.gen:
        table = net.gen
        mask = table["slack"].astype(bool)
        if "in_service" in table:
            mask &= table["in_service"].astype(bool)
        slack.update(int(bus) for bus in table.loc[mask, "bus"].to_numpy())
    return slack


def _load_buses(net) -> set[int]:
    if not hasattr(net, "load") or len(net.load) == 0:
        return set()
    table = net.load
    mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
    if "p_mw" in table:
        mask &= table["p_mw"].to_numpy(dtype=float) != 0.0
    return set(int(bus) for bus in table.loc[mask, "bus"].to_numpy())


def _bus_graph(net):
    if not hasattr(net, "bus") or len(net.bus) == 0:
        return None
    bus_index = list(net.bus.index)
    bus_pos = {int(bus): pos for pos, bus in enumerate(bus_index)}
    rows: list[int] = []
    cols: list[int] = []

    def add_edge(a: int, b: int) -> None:
        if a in bus_pos and b in bus_pos:
            rows.extend([bus_pos[a], bus_pos[b]])
            cols.extend([bus_pos[b], bus_pos[a]])

    if hasattr(net, "line") and len(net.line):
        table = net.line
        mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
        for _, row in table.loc[mask].iterrows():
            add_edge(int(row.from_bus), int(row.to_bus))
    if hasattr(net, "trafo") and len(net.trafo):
        table = net.trafo
        mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
        for _, row in table.loc[mask].iterrows():
            add_edge(int(row.hv_bus), int(row.lv_bus))
    if not rows:
        return sparse.csr_matrix((len(bus_index), len(bus_index)), dtype=bool)
    data = np.ones(len(rows), dtype=bool)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(bus_index), len(bus_index)))


def _action_fix_zero_impedance() -> DiagnosticAction:
    return DiagnosticAction(
        title="Remove numerical zero-impedance branches",
        rationale="Near-zero branch impedance can create extremely large admittance entries and singular or ill-conditioned Jacobians.",
        steps=(
            "Replace bus-coupler style branches with switches when appropriate.",
            "Merge buses connected by ideal zero-impedance links if they represent the same electrical node.",
            "If a physical low-impedance line is intended, set realistic nonzero R/X and validate the short-circuit assumptions.",
        ),
    )


def _diagnose_zero_impedance(net) -> list[DiagnosticDiagnosis]:
    evidence: list[DiagnosticEvidence] = []
    if hasattr(net, "line") and len(net.line):
        table = net.line
        mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
        zero = []
        for idx, row in table.loc[mask].iterrows():
            length = float(row.get("length_km", 1.0))
            r = abs(float(row.get("r_ohm_per_km", 0.0))) * length
            x = abs(float(row.get("x_ohm_per_km", 0.0))) * length
            if r + x < 1e-9:
                zero.append(int(idx))
        if zero:
            evidence.append(DiagnosticEvidence("zero_impedance_lines", "In-service lines with near-zero series R+X were found.", value=zero[:20], severity="critical"))
    if hasattr(net, "trafo") and len(net.trafo):
        table = net.trafo
        mask = table["in_service"].astype(bool) if "in_service" in table else np.ones(len(table), dtype=bool)
        zero = [int(idx) for idx, row in table.loc[mask].iterrows() if abs(float(row.get("vk_percent", 0.0))) < 1e-9]
        if zero:
            evidence.append(DiagnosticEvidence("zero_impedance_trafos", "In-service transformers with near-zero vk_percent were found.", value=zero[:20], severity="critical"))
    if not evidence:
        return []
    return [
        DiagnosticDiagnosis(
            code="zero_impedance_branch",
            title="Zero or near-zero branch impedance",
            severity="critical",
            confidence=0.90,
            cause="At least one in-service branch has effectively zero impedance, which can make the admittance matrix or Newton Jacobian singular.",
            evidence=tuple(evidence),
            actions=(_action_fix_zero_impedance(),),
        )
    ]


def _diagnose_model_singularity(model: PowerFlowModel) -> list[DiagnosticDiagnosis]:
    if model.ybus_scipy is None:
        return []
    graph = abs(model.ybus_scipy) > 0
    n_components, _ = csgraph.connected_components(graph, directed=False)
    evidence = []
    if n_components > 1:
        evidence.append(DiagnosticEvidence("ybus_components", "The internal Ybus graph has multiple connected components.", value=int(n_components), severity="critical"))
    try:
        structural_rank = int(csgraph.structural_rank(model.ybus_scipy))
        if structural_rank < model.n_bus:
            evidence.append(
                DiagnosticEvidence(
                    "ybus_structural_rank",
                    "The internal Ybus pattern is structurally rank deficient.",
                    value={"rank": structural_rank, "n_bus": model.n_bus},
                    severity="critical",
                )
            )
    except Exception:
        structural_rank = None
    if not evidence:
        return []
    return [
        DiagnosticDiagnosis(
            code="structurally_singular_ybus",
            title="Structurally singular or disconnected admittance model",
            severity="critical",
            confidence=0.85,
            cause="The extracted admittance model is disconnected or structurally rank deficient before Newton iterations.",
            evidence=tuple(evidence),
            actions=(_action_restore_connectivity(), _action_fix_zero_impedance()),
        )
    ]


def _diagnose_solver_result(model: PowerFlowModel | None, result: PowerFlowResult | TopologyBatchResult) -> list[DiagnosticDiagnosis]:
    voltages = np.asarray(result.voltages)
    mismatch = np.asarray(result.mismatch_norm)
    converged = np.asarray(result.converged)
    diagnoses: list[DiagnosticDiagnosis] = []
    if not np.all(np.isfinite(voltages)) or not np.all(np.isfinite(mismatch)):
        diagnoses.append(
            DiagnosticDiagnosis(
                code="non_finite_solution",
                title="Non-finite voltage or residual values",
                severity="critical",
                confidence=0.95,
                cause="The numerical iteration produced NaN or infinite values, usually after singular linear solves, voltage collapse, or invalid branch data.",
                evidence=(
                    DiagnosticEvidence("finite_voltages", "All voltage values finite.", value=bool(np.all(np.isfinite(voltages))), severity="critical"),
                    DiagnosticEvidence("finite_mismatch", "All mismatch values finite.", value=bool(np.all(np.isfinite(mismatch))), severity="critical"),
                ),
                actions=(_action_restore_connectivity(), _action_fix_zero_impedance()),
            )
        )
    if not bool(np.all(converged)):
        evidence = [
            DiagnosticEvidence("converged_count", "Number of converged solves.", value=f"{int(np.sum(converged))}/{converged.size}", severity="warning"),
            DiagnosticEvidence("max_mismatch", "Maximum reported infinity-norm mismatch.", value=float(np.nanmax(mismatch)), severity="warning"),
        ]
        if model is not None and model.ybus_scipy is not None and voltages.size and np.all(np.isfinite(voltages.reshape(-1, model.n_bus)[0])):
            evidence.extend(_top_residual_evidence(model, voltages.reshape(-1, model.n_bus)[0]))
        diagnoses.append(
            DiagnosticDiagnosis(
                code="high_power_mismatch",
                title="High power mismatch after solve",
                severity="warning",
                confidence=0.80,
                cause="The solver stopped before the active/reactive mismatch met tolerance. The listed buses are the strongest local residual contributors if model data was available.",
                evidence=tuple(evidence),
                actions=(
                    DiagnosticAction(
                        title="Target residual hot spots",
                        rationale="Largest residual buses often identify the island, overloaded corridor, or bus type causing the failure.",
                        steps=(
                            "Inspect the reported top mismatch buses and adjacent branches.",
                            "Check whether those buses are in an island, have extreme load/generation, or lack reactive support.",
                            "Try a robust initialization and compare with pandapower for the same static model.",
                        ),
                    ),
                    _action_reduce_stress(),
                ),
            )
        )
    if voltages.size:
        vm = np.abs(voltages)
        if np.nanmin(vm) < 0.5 or np.nanmax(vm) > 1.5:
            diagnoses.append(
                DiagnosticDiagnosis(
                    code="extreme_voltage_magnitude",
                    title="Extreme voltage magnitude during or after solve",
                    severity="warning",
                    confidence=0.70,
                    cause="Very low or high voltage magnitudes indicate voltage instability, bad initialization, or infeasible operating stress.",
                    evidence=(
                        DiagnosticEvidence("min_vm_pu", "Minimum voltage magnitude.", value=float(np.nanmin(vm)), severity="warning"),
                        DiagnosticEvidence("max_vm_pu", "Maximum voltage magnitude.", value=float(np.nanmax(vm)), severity="warning"),
                    ),
                    actions=(
                        DiagnosticAction(
                            title="Add voltage support or reduce stress",
                            rationale="Voltage collapse and reactive infeasibility commonly appear as extreme voltage magnitudes.",
                            steps=("Check generator voltage setpoints and reactive reserves.", "Reduce load or transfer in the affected island.", "Verify transformer taps and shunt devices."),
                        ),
                    ),
                )
            )
    return diagnoses


def _top_residual_evidence(model: PowerFlowModel, voltage: np.ndarray, n: int = 5) -> tuple[DiagnosticEvidence, ...]:
    scalc = voltage * np.conj(model.ybus_scipy @ voltage)
    residual = scalc - model.sbus0
    p = np.abs(residual.real)
    q = np.abs(residual.imag)
    score = p + q
    if score.size == 0:
        return ()
    top = np.argsort(score)[-n:][::-1]
    values = [
        {
            "internal_bus": int(bus),
            "p_mismatch_pu": float(residual.real[bus]),
            "q_mismatch_pu": float(residual.imag[bus]),
        }
        for bus in top
    ]
    return (DiagnosticEvidence("top_mismatch_buses", "Buses with largest local complex-power residual.", value=values, severity="warning"),)


def _diagnose_dc_screen(result: DCScreeningResult) -> list[DiagnosticDiagnosis]:
    if result.failed or not np.isfinite(result.score):
        return [
            DiagnosticDiagnosis(
                code="dc_contingency_singular_or_islanded",
                title="DC contingency is singular or islanded",
                severity="critical",
                confidence=0.95,
                cause="The screened contingency makes the DC susceptance system unsolvable or creates non-finite LODF/PTDF factors. This usually means islanding, a cutset outage, or a degenerate branch model.",
                evidence=(
                    DiagnosticEvidence("contingency", "Contingency under analysis.", value=str(result.contingency), severity="critical"),
                    DiagnosticEvidence("dc_reason", "DC screening failure reason.", value=result.reason, severity="critical"),
                    DiagnosticEvidence("dc_score", "DC screening score.", value=result.score, severity="critical"),
                ),
                actions=(_action_restore_connectivity(),),
            )
        ]
    if result.max_loading_percent > 100.0:
        return [
            DiagnosticDiagnosis(
                code="post_contingency_overload",
                title="Post-contingency overload in DC screen",
                severity="warning",
                confidence=0.80,
                cause="The linear DC screen predicts a thermal overload after the contingency. AC validation is required, but the case is not secure under the DC approximation.",
                evidence=(
                    DiagnosticEvidence("contingency", "Contingency under analysis.", value=str(result.contingency), severity="warning"),
                    DiagnosticEvidence("max_loading_percent", "Maximum DC post-contingency loading.", value=float(result.max_loading_percent), severity="warning"),
                ),
                actions=(_action_reduce_stress(),),
            )
        ]
    return []
