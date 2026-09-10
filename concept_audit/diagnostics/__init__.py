from .base import AuditState, Diagnostic, DiagnosticRegistry
from .metrics import Alignment, CrossConceptCorrelation, BlockPurity, TaskProbe, HeadSensitivity


def default_registry():
    return DiagnosticRegistry([Alignment(), CrossConceptCorrelation(), BlockPurity(), TaskProbe(), HeadSensitivity()])
