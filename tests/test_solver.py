import copy

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

from batched_powerflow import NewtonOptions, from_pandapower, solve_batch
from batched_powerflow.contingency import (
    analyze_contingencies_batched,
    apply_contingency,
    enumerate_nk,
    solve_contingencies_batched,
)
from batched_powerflow.pandapower_adapter import (
    PandapowerBuildOptions,
    PandapowerFeatureOptions,
    UnsupportedFeature,
    make_random_load_sbus_batch,
    make_scaled_load_net,
)


OPTIONS = NewtonOptions(max_iterations=12, tolerance=1e-9, gmres_tol=1e-11, gmres_restart=40, gmres_maxiter=80)
BUILD = PandapowerBuildOptions(check_connectivity=False, regularization=1e-8)


def _runpp_voltage(net):
    work = copy.deepcopy(net)
    pp.runpp(
        work,
        algorithm="nr",
        init="flat",
        calculate_voltage_angles=True,
        tolerance_mva=1e-10,
        max_iteration=30,
        enforce_q_lims=False,
        voltage_depend_loads=False,
        numba=False,
    )
    return work._ppc["internal"]["V"]


def _assert_close_voltage(actual, expected, atol=1e-6):
    assert np.max(np.abs(np.abs(actual) - np.abs(expected))) < atol
    angle_error = np.angle(actual * np.conj(expected))
    assert np.max(np.abs(angle_error)) < 5e-6


def test_case14_base_matches_pandapower():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch(model, options=OPTIONS)
    assert result.converged[0]
    _assert_close_voltage(result.voltages[0], _runpp_voltage(net))


def test_case14_random_load_batch_matches_pandapower():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    sbus_batch, multipliers = make_random_load_sbus_batch(
        net,
        n_scenarios=3,
        low=0.9,
        high=1.1,
        seed=10,
        options=BUILD,
    )
    result = solve_batch(model, sbus_batch, options=OPTIONS)
    assert np.all(result.converged)
    for i, multiplier in enumerate(multipliers):
        _assert_close_voltage(result.voltages[i], _runpp_voltage(make_scaled_load_net(net, multiplier)))


def test_case33_radial_base_converges():
    net = pn.case33bw()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch(model, options=OPTIONS)
    assert result.converged[0]
    _assert_close_voltage(result.voltages[0], _runpp_voltage(net), atol=2e-6)


@pytest.mark.parametrize(
    ("option_name", "feature_name"),
    [
        ("voltage_depend_loads", "voltage_depend_loads"),
        ("distributed_slack", "distributed_slack"),
        ("tdpf", "tdpf"),
    ],
)
def test_unsupported_dynamic_pandapower_features_raise(option_name, feature_name):
    options = PandapowerBuildOptions(check_connectivity=False, **{option_name: True})
    with pytest.raises(UnsupportedFeature) as exc_info:
        from_pandapower(pn.case14(), options=options)
    assert exc_info.value.feature == feature_name
    assert exc_info.value.guidance


def test_voltage_dependent_loads_can_be_frozen_to_static_loads():
    net = pn.case14()
    net.load.loc[:, "const_z_p_percent"] = 30.0
    net.load.loc[:, "const_i_q_percent"] = 20.0
    options = PandapowerBuildOptions(
        check_connectivity=False,
        voltage_depend_loads=True,
        feature_options=PandapowerFeatureOptions(voltage_depend_loads="freeze"),
    )
    model = from_pandapower(net, options=options)

    assert np.allclose(model.net.load["const_z_p_percent"].to_numpy(dtype=float), 0.0)
    assert np.allclose(model.net.load["const_i_q_percent"].to_numpy(dtype=float), 0.0)
    assert np.any(~np.isclose(model.net.load["p_mw"].to_numpy(dtype=float), net.load["p_mw"].to_numpy(dtype=float)))


def test_controller_policy_raise_reports_controller_guidance():
    net = pn.case14()
    net.controller.loc[0, "object"] = object()
    net.controller.loc[0, "in_service"] = True
    options = PandapowerBuildOptions(
        check_connectivity=False,
        feature_options=PandapowerFeatureOptions(controllers="raise"),
    )
    with pytest.raises(UnsupportedFeature) as exc_info:
        from_pandapower(net, options=options)
    assert exc_info.value.feature == "controllers"
    assert "freeze" in exc_info.value.guidance


def test_controller_freeze_runs_control_on_copy_before_static_extraction(monkeypatch):
    net = pn.case14()
    original_p_mw = net.load["p_mw"].to_numpy(dtype=float).copy()
    net.controller.loc[0, "object"] = object()
    net.controller.loc[0, "in_service"] = True
    calls = []

    def fake_runpp(work, **kwargs):
        calls.append(kwargs)
        assert kwargs["run_control"] is True
        work.load.loc[:, "p_mw"] = work.load["p_mw"].to_numpy(dtype=float) * 0.5

    monkeypatch.setattr(pp, "runpp", fake_runpp)
    options = PandapowerBuildOptions(
        check_connectivity=False,
        feature_options=PandapowerFeatureOptions(controllers="freeze"),
    )
    model = from_pandapower(net, options=options)

    assert calls
    assert np.allclose(model.net.load["p_mw"].to_numpy(dtype=float), original_p_mw * 0.5)
    assert np.allclose(net.load["p_mw"].to_numpy(dtype=float), original_p_mw)


def test_static_line_temperature_option_builds_model():
    net = pn.case14()
    net.line["temperature_degree_celsius"] = 35.0
    model = from_pandapower(net, options=PandapowerBuildOptions(consider_line_temperature=True))
    assert model.n_bus == len(net.bus)


def test_voltage_dependent_load_freeze_policy_builds_static_model():
    net = pn.case14()
    if "const_z_percent" not in net.load:
        net.load["const_z_percent"] = 0.0
    net.load.loc[net.load.index[0], "const_z_percent"] = 10.0
    options = PandapowerBuildOptions(
        voltage_depend_loads=True,
        feature_options=PandapowerFeatureOptions(voltage_depend_loads="freeze"),
    )
    model = from_pandapower(net, options=options)
    assert model.sbus0.shape == (len(net.bus),)


def test_n1_line_contingencies_batched_match_pandapower():
    net = pn.case14()
    contingencies = enumerate_nk(net, k=1, elements=("line",), max_cases=3)
    result = solve_contingencies_batched(net, contingencies, solver_options=OPTIONS, build_options=BUILD)
    assert result.voltages.shape[:2] == (3, 1)
    assert np.all(result.converged)
    for i, contingency in enumerate(contingencies):
        ref_v = _runpp_voltage(apply_contingency(net, contingency))
        _assert_close_voltage(result.voltages[i, 0], ref_v)


def test_contingency_batch_uses_per_topology_runpp_starts():
    net = pn.case14()
    contingencies = enumerate_nk(net, k=1, elements=("line",), max_cases=3)
    result = solve_contingencies_batched(
        net,
        contingencies,
        solver_options=NewtonOptions(max_iterations=0, tolerance=1e-8, jit=False),
        build_options=PandapowerBuildOptions(check_connectivity=False, run_powerflow=True),
    )
    assert np.max(result.mismatch_norm) < 1e-8


def test_n2_line_contingencies_batched_run():
    net = pn.case14()
    contingencies = enumerate_nk(net, k=2, elements=("line",), max_cases=2)
    result = solve_contingencies_batched(net, contingencies, solver_options=OPTIONS, build_options=BUILD)
    assert result.voltages.shape[:2] == (2, 1)
    assert result.converged.dtype == np.bool_


def test_n1_contingency_analysis_reports_flows_and_violations():
    net = pn.case14()
    contingencies = enumerate_nk(net, k=1, elements=("line",), max_cases=2)
    analysis = analyze_contingencies_batched(net, contingencies, solver_options=OPTIONS, build_options=BUILD)
    assert analysis.s_from_mva.shape[:2] == (2, 1)
    assert analysis.loading_percent.shape == analysis.branch_overloaded.shape
    assert analysis.voltage_low.shape == analysis.result.vm.shape
