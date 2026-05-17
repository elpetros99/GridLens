import copy

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from batched_powerflow import CpuNewtonOptions, from_pandapower, solve_batch_cpu_sparse
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions, make_random_load_sbus_batch, make_scaled_load_net


BUILD = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")


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


def test_cpu_sparse_direct_case14_matches_pandapower():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch_cpu_sparse(model, options=CpuNewtonOptions(max_iterations=15, tolerance=1e-10))
    assert result.converged[0]
    assert np.max(np.abs(result.voltages[0] - _runpp_voltage(net))) < 1e-8


def test_cpu_sparse_direct_random_batch_matches_pandapower():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    sbus_batch, multipliers = make_random_load_sbus_batch(net, n_scenarios=2, seed=3, options=BUILD)
    result = solve_batch_cpu_sparse(model, sbus_batch, options=CpuNewtonOptions(max_iterations=15, tolerance=1e-10))
    assert np.all(result.converged)
    for i, multiplier in enumerate(multipliers):
        assert np.max(np.abs(result.voltages[i] - _runpp_voltage(make_scaled_load_net(net, multiplier)))) < 1e-8


def test_cpu_sparse_direct_converges_case118_from_model_start():
    net = pn.case118()
    model = from_pandapower(
        net,
        options=PandapowerBuildOptions(check_connectivity=False, preconditioner="none"),
    )
    result = solve_batch_cpu_sparse(
        model,
        options=CpuNewtonOptions(max_iterations=20, tolerance=1e-8),
    )
    assert result.converged[0]


def test_cpu_sparse_direct_uses_run_powerflow_results_initialization_for_hard_case():
    net = pn.case1888rte()
    model = from_pandapower(
        net,
        options=PandapowerBuildOptions(check_connectivity=True, preconditioner="none", run_powerflow=True),
    )
    result = solve_batch_cpu_sparse(
        model,
        options=CpuNewtonOptions(max_iterations=5, tolerance=1e-8),
    )
    assert result.converged[0]
    assert result.mismatch_norm[0] < 1e-8
