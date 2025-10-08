"""
Enhanced Agent Orchestration Module

This module provides advanced agent orchestration capabilities for AI-Scientist-v2,
including supervisor agents, workflow coordination, and multi-agent collaboration.
"""

from .supervisor_agent import SupervisorAgent
from .agent_profiles import AgentProfileManager
from .research_orchestrator_agent import (
    ResearchOrchestratorAgent,
    ResearchRequest,
    ResearchResults,
    ServiceRegistry,
    AgentCapability,
    AgentStatus,
    CoordinationMode,
    get_orchestrator,
    create_research_request,
    coordinate_research_simple
)
from .api_gateway import APIGateway, ConnectionManager

__all__ = [
    "SupervisorAgent",
    "AgentProfileManager",
    "ResearchOrchestratorAgent",
    "ResearchRequest",
    "ResearchResults",
    "ServiceRegistry",
    "AgentCapability",
    "AgentStatus",
    "CoordinationMode",
    "get_orchestrator",
    "create_research_request",
    "coordinate_research_simple",
    "APIGateway",
    "ConnectionManager"
]