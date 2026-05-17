from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerFlowModel:
    """Static pandapower-derived AC power-flow data in internal bus order."""

    name: str
    base_mva: float
    n_bus: int
    ybus_data: np.ndarray
    ybus_indices: np.ndarray
    yf_data: np.ndarray
    yf_indices: np.ndarray
    yt_data: np.ndarray
    yt_indices: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    theta: np.ndarray
    vm0: np.ndarray
    va0: np.ndarray
    sbus0: np.ndarray
    bp_inv: np.ndarray
    bpp_inv: np.ndarray
    g_qtheta: np.ndarray
    ybus_scipy: Any | None = None
    yf_scipy: Any | None = None
    yt_scipy: Any | None = None
    ppc: dict[str, Any] | None = None
    net: Any | None = None

    @property
    def n_state(self) -> int:
        return int(len(self.theta) + len(self.pq))

    @property
    def v0(self) -> np.ndarray:
        return self.vm0 * np.exp(1j * self.va0)

    def with_ybus_pattern(
        self,
        *,
        ybus_indices: np.ndarray,
        ybus_data: np.ndarray,
    ) -> "PowerFlowModel":
        return replace(
            self,
            ybus_indices=np.asarray(ybus_indices, dtype=np.int32),
            ybus_data=np.asarray(ybus_data, dtype=np.complex128),
        )


@dataclass(frozen=True)
class PowerFlowResult:
    """Solved voltages and convergence diagnostics for a scenario batch."""

    voltages: np.ndarray
    converged: np.ndarray
    mismatch_norm: np.ndarray
    iterations: int

    @property
    def vm(self) -> np.ndarray:
        return np.abs(self.voltages)

    @property
    def va_degree(self) -> np.ndarray:
        return np.rad2deg(np.angle(self.voltages))


@dataclass(frozen=True)
class TopologyBatchResult:
    """Solved voltages for a contingency/topology batch and scenario batch."""

    contingencies: tuple[Any, ...]
    voltages: np.ndarray
    converged: np.ndarray
    mismatch_norm: np.ndarray
    iterations: int

    @property
    def vm(self) -> np.ndarray:
        return np.abs(self.voltages)

    @property
    def va_degree(self) -> np.ndarray:
        return np.rad2deg(np.angle(self.voltages))
