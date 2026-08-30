import numpy as np
import pandapower.networks as pn

from batched_powerflow import (
    CpuNewtonOptions,
    analyze_contingencies_batched,
    analyze_contingencies_cpu_sparse,
    compute_branch_flows_cpu,
    complete_diagnostic_security_analysis,
    from_pandapower,
    solve_batch_cpu_sparse,
)
from batched_powerflow.contingency import Contingency, Outage, apply_contingency
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions
from batched_powerflow.solver import compute_branch_flows


BUILD = PandapowerBuildOptions(check_connectivity=False, preconditioner="none")
CPU = CpuNewtonOptions(max_iterations=20, tolerance=1e-10, jacobian_reuse=1)


def _case14_n1(count=3):
    net = pn.case14()
    return net, tuple(Contingency((Outage("line", int(index)),)) for index in net.line.index[:count])


def test_cpu_contingency_batch_matches_individual_sparse_solves():
    net, contingencies = _case14_n1()
    batched = analyze_contingencies_cpu_sparse(net, contingencies, build_options=BUILD, solver_options=CPU)

    for index, contingency in enumerate(contingencies):
        model = from_pandapower(
            apply_contingency(net, contingency),
            options=BUILD,
        )
        individual = solve_batch_cpu_sparse(model, options=CPU)
        assert individual.converged[0] == batched.result.converged[index, 0]
        assert np.max(np.abs(individual.voltages[0] - batched.result.voltages[index, 0])) < 1e-9


def test_cpu_warm_start_and_jacobian_reuse_converge():
    net, contingencies = _case14_n1(count=2)
    model = from_pandapower(net, options=BUILD)
    base = solve_batch_cpu_sparse(model, options=CPU)
    starts = np.broadcast_to(base.voltages[0], (len(contingencies), model.n_bus))
    warm = analyze_contingencies_cpu_sparse(
        net,
        contingencies,
        build_options=BUILD,
        solver_options=CpuNewtonOptions(max_iterations=20, tolerance=1e-10, jacobian_reuse=2),
        vm0_batch=np.abs(starts),
        va0_batch=np.angle(starts),
    )
    assert np.all(warm.result.converged)
    assert np.all(warm.result.mismatch_norm < 1e-8)


def test_cpu_branch_flow_postprocessing_matches_jax_implementation():
    net = pn.case14()
    model = from_pandapower(net, options=BUILD)
    result = solve_batch_cpu_sparse(model, options=CPU)
    sf_cpu, st_cpu = compute_branch_flows_cpu(model, result.voltages[0])
    sf_jax, st_jax = compute_branch_flows(model, result.voltages[0])
    assert np.allclose(sf_cpu, sf_jax, atol=1e-10, rtol=1e-10)
    assert np.allclose(st_cpu, st_jax, atol=1e-10, rtol=1e-10)


def test_jax_topology_batch_matches_cpu_on_cpu_device():
    net, contingencies = _case14_n1(count=2)
    cpu = analyze_contingencies_cpu_sparse(net, contingencies, build_options=BUILD, solver_options=CPU)
    jax = analyze_contingencies_batched(net, contingencies, build_options=BUILD)
    assert np.all(jax.result.converged)
    assert np.allclose(jax.result.voltages, cpu.result.voltages, atol=1e-6, rtol=1e-6)


def test_diagnostic_security_analysis_exposes_batched_ac_controls():
    net = pn.case14()
    report = complete_diagnostic_security_analysis(
        net,
        elements=("line",),
        top_n=3,
        ac_mode="top",
        ac_backend="cpu",
        ac_batch_size=2,
        ac_warm_start=True,
        build_options=BUILD,
        cpu_options=CPU,
    )
    assert report.ac_backend == "cpu"
    assert report.ac_batch_size == 2
    assert report.ac_warm_start
    assert report.ac_validated == 3
