"""
Enhanced Agent Orchestration Module

This module provides advanced agent orchestration capabilities for AI-Scientist-v2,
including supervisor agents, workflow coordination, and multi-agent collaboration.
"""

from .supervisor_agent import SupervisorAgent
from .agent_profiles import AgentProfileManager
from .workflow_coordinator import WorkflowCoordinator
from .decision_engine import DecisionEngine

__all__ = [
    "SupervisorAgent",
    "AgentProfileManager", 
    "WorkflowCoordinator",
    "DecisionEngine"
]