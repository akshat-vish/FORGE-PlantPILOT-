"""PlantPilot — Backend Framework for Human-Supervised Industrial Operations AI Copilot."""

__version__ = "0.1.0"

# Relative imports work whether package is installed as "plantpilot" or
# imported from app/ as module "plantpilot" (pythonpath=["app"]).
from .config import get_config, ConfigError
from .schemas import (
    Incident,
    IncidentCreate,
    EvidenceItem,
    Hypothesis,
    ToolExecution,
    Recommendation,
    Investigation,
    InvestigationCreate,
)
from .safety import SafetyPolicy, check_safety
from .tools import ToolRegistry, registered_tools
from .storage import Database, get_database
from .agent import AgentController, InvestigationResult

__all__ = [
    "get_config",
    "ConfigError",
    "Incident",
    "IncidentCreate",
    "EvidenceItem",
    "Hypothesis",
    "ToolExecution",
    "Recommendation",
    "Investigation",
    "InvestigationCreate",
    "SafetyPolicy",
    "check_safety",
    "ToolRegistry",
    "registered_tools",
    "Database",
    "get_database",
    "AgentController",
    "InvestigationResult",
]