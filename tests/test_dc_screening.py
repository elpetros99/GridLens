import itertools

import numpy as np
import pandapower.networks as pn

from batched_powerflow import (
    chunk_contingencies,
    hierarchical_ac_security_analysis,
    iter_nk_contingencies,
    screen_n1_lodf_top,
    screen_nk_lodf_top,
    screen_top_contingencies,
    score_contingencies_dc,
)
from batched_powerflow import NewtonOptions
from batched_powerflow.pandapower_adapter import PandapowerBuildOptions


BUILD = PandapowerBuildOptions(check_connectivity=False, regularization=1e-8)


def test_iter_nk_contingencies_is_lazy_and_chunkable():
    net = pn.case14()
    iterator = iter_nk_contingencies(net, k=2, elements=("line",))

    first_five = tuple(itertools.islice(iterator, 5))
    assert len(first_five) == 5
    assert all(len(contingency.outages) == 2 for contingency in first_five)

    chunks = tuple(chunk_contingencies(first_five, chunk_size=2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_screen_top_contingencies_returns_ranked_dc_proxy_results():
    net = pn.case14()
    results = screen_top_contingencies(
        net,
        k=1,
        top_n=3,
        elements=("line",),
        max_cases=6,
        chunk_size=2,
        options=BUILD,
    )

    assert len(results) == 3
    assert all(not result.failed for result in results)
    assert all(np.isfinite(result.max_loading_percent) for result in results)
    assert [result.score for result in results] == sorted((result.score for result in results), reverse=True)
    assert results[0].angles_rad.shape == (len(net.bus),)


def test_dc_screening_ranks_singular_radial_outages_as_failures():
    net = pn.case33bw()
    contingencies = iter_nk_contingencies(net, k=1, elements=("line",), max_cases=1)
    result = next(score_contingencies_dc(net, contingencies, options=BUILD, chunk_size=1))

    assert result.failed
    assert result.score == float("inf")
    assert np.isnan(result.branch_flow_mw).all()


def test_hierarchical_security_analysis_validates_screened_subset():
    net = pn.case14()
    analysis = hierarchical_ac_security_analysis(
        net,
        k=1,
        top_n=2,
        elements=("line",),
        max_cases=6,
        screening_chunk_size=2,
        build_options=BUILD,
        solver_options=NewtonOptions(max_iterations=12, tolerance=1e-9, gmres_tol=1e-11),
    )

    assert analysis.total_screened == 6
    assert len(analysis.screened) == 2
    assert analysis.ac is not None
    assert analysis.ac.result.voltages.shape[:2] == (2, 1)
    assert analysis.selected_contingencies == tuple(result.contingency for result in analysis.screened if not result.failed)


def test_lodf_n1_screening_returns_complete_summary():
    net = pn.case14()
    summary = screen_n1_lodf_top(
        net,
        top_n=3,
        elements=("line",),
        options=BUILD,
        chunk_size=4,
    )

    assert summary.total == len(net.line)
    assert len(summary.screened) == 3
    assert all(not result.failed for result in summary.screened)
    assert [result.score for result in summary.screened] == sorted((result.score for result in summary.screened), reverse=True)


def test_lodf_n2_screening_matches_exact_dc_for_top_result():
    net = pn.case14()
    n1 = screen_n1_lodf_top(net, top_n=6, elements=("line",), options=BUILD, chunk_size=4)
    n2 = screen_nk_lodf_top(net, k=2, candidate_results=n1.screened, top_n=3, options=BUILD)

    assert n2.k == 2
    assert n2.candidate_pool_size == 6
    assert n2.total == 15
    assert len(n2.screened) == 3

    exact = next(score_contingencies_dc(net, [n2.screened[0].contingency], options=BUILD, chunk_size=1))
    if not exact.failed and not n2.screened[0].failed:
        assert np.max(np.abs(exact.branch_flow_mw - n2.screened[0].branch_flow_mw)) < 1e-6
