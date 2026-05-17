from __future__ import annotations

"""Backward-compatible aliases for the renamed diagnostics module."""

from .diagnostics import (
    DiagnosticAction,
    DiagnosticDiagnosis,
    DiagnosticEvidence,
    DiagnosticIndicator,
    DiagnosticReport,
    DiagnosticSecurityReport,
    DiagnosticSecurityViolation,
    complete_diagnostic_security_analysis,
    explain_dc_screening_result,
    explain_powerflow,
)

BetaAction = DiagnosticAction
BetaDiagnosis = DiagnosticDiagnosis
BetaEvidence = DiagnosticEvidence
BetaIndicator = DiagnosticIndicator
BetaReport = DiagnosticReport

__all__ = [
    "BetaAction",
    "BetaDiagnosis",
    "BetaEvidence",
    "BetaIndicator",
    "BetaReport",
    "DiagnosticAction",
    "DiagnosticDiagnosis",
    "DiagnosticEvidence",
    "DiagnosticIndicator",
    "DiagnosticReport",
    "DiagnosticSecurityReport",
    "DiagnosticSecurityViolation",
    "complete_diagnostic_security_analysis",
    "explain_dc_screening_result",
    "explain_powerflow",
]
