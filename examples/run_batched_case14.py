from pathlib import Path
import sys

import pandapower.networks as pn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from batched_powerflow import NewtonOptions, from_pandapower, solve_batch
from batched_powerflow.contingency import enumerate_nk, solve_contingencies_batched
from batched_powerflow.pandapower_adapter import make_random_load_sbus_batch


def main():
    net = pn.case14()
    model = from_pandapower(net)
    sbus_batch, multipliers = make_random_load_sbus_batch(net, n_scenarios=16, seed=1)
    result = solve_batch(model, sbus_batch, NewtonOptions(max_iterations=12))
    print("scenario multipliers:", multipliers)
    print("converged:", result.converged)
    print("max mismatch:", result.mismatch_norm.max())

    contingencies = enumerate_nk(net, k=1, elements=("line",), max_cases=5)
    n1 = solve_contingencies_batched(net, contingencies, sbus_batch[:2])
    print("N-1 voltage shape:", n1.voltages.shape)
    print("N-1 converged:", n1.converged)


if __name__ == "__main__":
    main()
