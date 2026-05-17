import numpy as np
import pandapower.networks as pn

from batched_powerflow import (
    CpuNewtonOptions,
    PowerFlowResult,
    complete_diagnostic_security_analysis,
    explain_dc_screening_result,
    explain_powerflow,
    from_pandapower,
    solve_batch_cpu_sparse,
)
from batched_powerflow.dc_screening import iter_nk_contingencies, score_contingencies_dc
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


BUILD = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")


def _codes(report):
    return {diagnosis.code for diagnosis in report.diagnoses}


def _indicator(report, category, name):
    return next(item for item in report.indicators if item.category == category and item.name == name)


def test_diagnostic_explains_missing_slack_reference():
    net = pn.case14()
    net.ext_grid["in_service"] = False
    if hasattr(net, "gen") and "slack" in net.gen:
        net.gen["slack"] = False

    report = explain_powerflow(net)

    assert report.status == "insecure_or_infeasible"
    assert "missing_slack" in _codes(report)
    assert "reference" in report.to_markdown().lower()


def test_diagnostic_explains_radial_dc_contingency_singularity():
    net = pn.case33bw()
    contingency = next(iter_nk_contingencies(net, k=1, elements=("line",), max_cases=1))
    result = next(score_contingencies_dc(net, [contingency], options=BUILD, chunk_size=1))

    report = explain_dc_screening_result(net, result)

    assert result.failed
    assert "dc_contingency_singular_or_islanded" in _codes(report)
    assert any(diagnosis.actions for diagnosis in report.diagnoses)
    assert _indicator(report, "frequency", "island_slack_support").status == "critical"


def test_diagnostic_detects_zero_impedance_line():
    net = pn.case14()
    line = int(net.line.index[0])
    net.line.at[line, "length_km"] = 0.0
    net.line.at[line, "r_ohm_per_km"] = 0.0
    net.line.at[line, "x_ohm_per_km"] = 0.0

    report = explain_powerflow(net)

    assert "zero_impedance_branch" in _codes(report)


def test_diagnostic_reports_non_converged_residual_hotspots():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch_cpu_sparse(model, options=CpuNewtonOptions(max_iterations=0, tolerance=1e-12))

    report = explain_powerflow(net, model=model, result=result)

    assert not result.converged[0]
    assert "high_power_mismatch" in _codes(report)
    mismatch_diagnosis = next(d for d in report.diagnoses if d.code == "high_power_mismatch")
    assert any(evidence.key == "top_mismatch_buses" for evidence in mismatch_diagnosis.evidence)


def test_diagnostic_reports_voltage_and_frequency_indicators_for_solved_case():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch_cpu_sparse(model, options=CpuNewtonOptions(max_iterations=15, tolerance=1e-10))

    report = explain_powerflow(net, model=model, result=result)

    assert _indicator(report, "voltage", "min_vm_pu").status in {"info", "warning"}
    assert _indicator(report, "voltage", "voltage_limit_violations").value["low"] >= 0
    assert _indicator(report, "frequency", "island_slack_support").value["unsupported_load_islands"] == 0
    assert _indicator(report, "frequency", "dynamic_frequency_data").status == "unknown"


def test_diagnostic_flags_extreme_voltage_indicator():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch_cpu_sparse(model, options=CpuNewtonOptions(max_iterations=15, tolerance=1e-10))
    voltages = result.voltages.copy()
    voltages[0, 0] = 0.50 + 0.0j
    stressed = PowerFlowResult(
        voltages=voltages,
        converged=result.converged,
        mismatch_norm=result.mismatch_norm,
        iterations=result.iterations,
    )

    report = explain_powerflow(net, model=model, result=stressed)

    assert _indicator(report, "voltage", "min_vm_pu").status == "critical"
    assert _indicator(report, "voltage", "voltage_limit_violations").status == "critical"


def test_complete_diagnostic_security_analysis_defaults_to_n1_and_reports_unsafe():
    net = pn.case14()
    report = complete_diagnostic_security_analysis(
        net,
        elements=("line",),
        top_n=3,
        ac_mode="top",
        build_options=BUILD,
        cpu_options=CpuNewtonOptions(max_iterations=15, tolerance=1e-9),
    )

    assert report.k == 1
    assert not report.safe
    assert report.total_screened == len(net.line)
    assert report.ac_validated == 3
    assert report.max_dc_loading_percent >= 0.0
    assert report.min_ac_vm_pu is not None
    assert report.max_ac_vm_pu is not None
    assert any(indicator.category == "frequency" for indicator in report.indicators)
    assert any(indicator.category == "voltage" for indicator in report.indicators)
    assert "UNSAFE" in report.to_markdown()


def test_complete_diagnostic_security_analysis_can_report_safe_for_limited_dc_scope():
    net = pn.case14()
    report = complete_diagnostic_security_analysis(
        net,
        elements=("line",),
        top_n=3,
        ac_mode="none",
        loading_limit_percent=1e9,
        build_options=BUILD,
    )

    assert report.k == 1
    assert report.exhaustive_dc
    assert report.exhaustive_ac is False
    assert report.safe is True
    assert "Complete DC" in report.guarantee_scope
