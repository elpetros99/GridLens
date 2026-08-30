"""Tutorial: explain data defects and power-flow risks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandapower.networks as pn

from batched_powerflow import explain_powerflow


def main() -> None:
    missing_slack = pn.case14()
    missing_slack.ext_grid.loc[:, "in_service"] = False
    if "slack" in missing_slack.gen:
        missing_slack.gen.loc[:, "slack"] = False
    print("=== Missing reference ===")
    print(explain_powerflow(missing_slack).to_markdown())

    zero_impedance = pn.case14()
    line = int(zero_impedance.line.index[0])
    zero_impedance.line.loc[line, ["length_km", "r_ohm_per_km", "x_ohm_per_km"]] = 0.0
    print("=== Zero-impedance branch ===")
    print(explain_powerflow(zero_impedance).to_markdown())


if __name__ == "__main__":
    main()
