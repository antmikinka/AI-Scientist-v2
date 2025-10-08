"""
Research Orchestrator Agent - Central Coordination System

This module implements the unified Research Orchestrator Agent that serves as the central
coordination system for the entire multi-agent research ecosystem. It provides a unified
interface for coordinating research workflows, managing agent interactions, and optimizing
resource allocation across the autonomous research platform.

Key Capabilities:
- Multi-agent workflow coordination and management
- Dynamic agent discovery and service registry
- Strategic decision making and adaptive planning
- Resource optimization and load balancing
- Real-time monitoring and performance analytics
- Consensus building and conflict resolution
- Ethical oversight and compliance enforcement
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta

# Import existing orchestration components
from .supervisor_agent import SupervisorAgent, WorkflowPlan, ResearchTask, ProgressEvaluation
from .agent_profiles import AgentProfileManager, AgentProfile
from ..treesearch.agent_manager import AgentManager
from ..theory.theory_evolution_agent import TheoryEvolutionAgent
from ..security.security_manager import SecurityManager
from ..monitoring.performance_monitor import PerformanceMonitor

# Ethical framework integration will be initialized dynamically
# from ..ethical.integration import EthicalIntegrationManager, EthicalOrchestratorWrapper
# from ..ethical.config import get_config


class AgentStatus(Enum):
    """Agent operational status"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class CoordinationMode(Enum):
    """Coordination strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    COMPETITIVE = "competitive"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: AgentStatus
    performance_metrics: Dict[str, float]
    current_load: int
    max_capacity: int
    last_active: datetime
    availability_score: float = 1.0


@dataclass
class ResearchRequest:
    """Research request definition"""
    request_id: str
    objective: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    deadline: Optional[datetime] = None
    coordination_mode: CoordinationMode = CoordinationMode.SEQUENTIAL
    required_agents: List[str] = field(default_factory=list)
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    ethical_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResults:
    """Research results container"""
    request_id: str
    success: bool
    results: Dict[str, Any]
    execution_time: float
    agent_contributions: Dict[str, Any]
    confidence_score: float
    ethical_compliance: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


class ServiceRegistry:
    """Dynamic service discovery and agent registry"""

    def __init__(self):
        self.registered_agents: Dict[str, AgentCapability] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.health_check_interval = 60  # seconds
        self._health_monitor_task = None

    async def register_agent(self, agent: AgentCapability) -> bool:
        """Register a new agent in the service registry"""
        try:
            self.registered_agents[agent.agent_id] = agent

            # Update capabilities index
            for capability in agent.capabilities:
                if capability not in self.agent_capabilities:
                    self.agent_capabilities[capability] = []
                self.agent_capabilities[capability].append(agent.agent_id)

            logging.info(f"Agent {agent.agent_id} registered successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False

    async def discover_agents(self, required_capabilities: List[str],
                            max_agents: int = 5) -> List[AgentCapability]:
        """Discover agents matching required capabilities"""
        matching_agents = []

        for agent_id, agent in self.registered_agents.items():
            if agent.status == AgentStatus.ACTIVE and agent.current_load < agent.max_capacity:
                # Check if agent has all required capabilities
                if all(cap in agent.capabilities for cap in required_capabilities):
                    matching_agents.append(agent)

        # Sort by availability score and current load
        matching_agents.sort(key=lambda a: (a.availability_score, -a.current_load), reverse=True)

        return matching_agents[:max_agents]

    async def start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logging.error(f"Health monitor error: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all registered agents"""
        current_time = datetime.now()

        for agent_id, agent in self.registered_agents.items():
            # Check if agent is stale (no recent activity)
            time_since_active = current_time - agent.last_active

            if time_since_active > timedelta(minutes=5):
                if agent.status == AgentStatus.ACTIVE:
                    logging.warning(f"Agent {agent_id} appears stale, marking as idle")
                    agent.status = AgentStatus.IDLE
                    agent.availability_score *= 0.8

            # Check if agent is overloaded
            if agent.current_load >= agent.max_capacity:
                if agent.status == AgentStatus.ACTIVE:
                    logging.warning(f"Agent {agent_id} is overloaded")
                    agent.availability_score *= 0.7


class ResearchOrchestratorAgent:
    """
    Research Orchestrator Agent - Central Coordination System

    This agent serves as the central coordination system for the entire multi-agent
    research ecosystem, providing unified orchestration, resource management, and
    strategic decision making capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = f"research_orchestrator_{uuid.uuid4().hex[:8]}"

        # Setup logging first
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")

        # Initialize core components
        self.supervisor = SupervisorAgent(config)

        # Initialize AgentManager with required parameters (using config as cfg and current dir as workspace)
        try:
            self.agent_manager = AgentManager(config, workspace_dir=".")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AgentManager: {e}")
            self.agent_manager = None

        self.profile_manager = AgentProfileManager()

        try:
            self.theory_agent = TheoryEvolutionAgent(config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize TheoryEvolutionAgent: {e}")
            self.theory_agent = None

        self.security_manager = SecurityManager(config)
        self.performance_monitor = PerformanceMonitor()
        self.service_registry = ServiceRegistry()

        # Ethical integration will be initialized dynamically
        self.ethical_integration_enabled = config.get('ethical_integration_enabled', True)

        # Orchestrator state
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        self.agent_assignments: Dict[str, List[str]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, Any] = {}

        # Configuration
        self.max_concurrent_workflows = config.get('max_concurrent_workflows', 10)
        self.default_timeout = config.get('default_timeout', 3600)  # 1 hour
        self.consensus_threshold = config.get('consensus_threshold', 0.7)

        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None

    async def initialize(self):
        """Initialize the orchestrator and start background services"""
        try:
            # Start service registry health monitoring
            await self.service_registry.start_health_monitoring()

            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Register orchestrator itself
            orchestrator_capability = AgentCapability(
                agent_id=self.agent_id,
                agent_type="research_orchestrator",
                capabilities=["workflow_coordination", "agent_management", "strategic_planning"],
                status=AgentStatus.ACTIVE,
                performance_metrics={"success_rate": 1.0, "avg_response_time": 0.1},
                current_load=0,
                max_capacity=50,
                last_active=datetime.now()
            )
            await self.service_registry.register_agent(orchestrator_capability)

            self.logger.info(f"Research Orchestrator Agent {self.agent_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def coordinate_research(self, request: ResearchRequest) -> ResearchResults:
        """
        Main research coordination interface

        Args:
            request: Research request with objective and constraints

        Returns:
            ResearchResults with outcomes and recommendations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting research coordination for request {request.request_id}")

            # Ethical pre-check
            ethical_clearance = await self._ethical_compliance_check(request)
            if not ethical_clearance['approved']:
                return ResearchResults(
                    request_id=request.request_id,
                    success=False,
                    results={"error": "Ethical compliance failed", "details": ethical_clearance},
                    execution_time=time.time() - start_time,
                    agent_contributions={},
                    confidence_score=0.0,
                    ethical_compliance=ethical_clearance
                )

            # Discover and assign agents
            assigned_agents = await self._discover_and_assign_agents(request)

            if not assigned_agents:
                return ResearchResults(
                    request_id=request.request_id,
                    success=False,
                    results={"error": "No suitable agents available"},
                    execution_time=time.time() - start_time,
                    agent_contributions={},
                    confidence_score=0.0,
                    ethical_compliance=ethical_clearance
                )

            # Create workflow plan
            workflow_plan = await self.supervisor.coordinate_workflow(
                request.objective,
                {**request.context, "assigned_agents": assigned_agents}
            )

            # Store active workflow
            self.active_workflows[request.request_id] = workflow_plan

            # Execute workflow based on coordination mode
            results = await self._execute_workflow(workflow_plan, request)

            # Integrate with theory evolution agent
            if self.theory_agent:
                theoretical_insights = await self.theory_agent.correlate_findings(
                    results["results"],
                    context={**request.context, "workflow_id": request.request_id}
                )
            else:
                theoretical_insights = {"theory_correlation": "Theory agent not available"}

            # Prepare final results
            execution_time = time.time() - start_time

            final_results = ResearchResults(
                request_id=request.request_id,
                success=results.get("success", False),
                results={**results.get("results", {}), "theoretical_insights": theoretical_insights},
                execution_time=execution_time,
                agent_contributions=results.get("agent_contributions", {}),
                confidence_score=results.get("confidence", 0.5),
                ethical_compliance=ethical_clearance,
                recommendations=results.get("recommendations", []),
                next_steps=results.get("next_steps", [])
            )

            # Store in history
            self.workflow_history.append({
                "request_id": request.request_id,
                "timestamp": datetime.now(),
                "execution_time": execution_time,
                "success": final_results.success,
                "agents_used": assigned_agents
            })

            # Clean up workflow
            await self._cleanup_workflow(request.request_id)

            return final_results

        except Exception as e:
            self.logger.error(f"Research coordination failed for {request.request_id}: {e}")
            return ResearchResults(
                request_id=request.request_id,
                success=False,
                results={"error": str(e), "error_type": type(e).__name__},
                execution_time=time.time() - start_time,
                agent_contributions={},
                confidence_score=0.0,
                ethical_compliance={"approved": False, "error": str(e)}
            )

    async def _ethical_compliance_check(self, request: ResearchRequest) -> Dict[str, Any]:
        """Perform ethical compliance check on research request"""
        try:
            # Use security manager for ethical compliance
            compliance_result = await self.security_manager.check_research_compliance(
                request.objective,
                {**request.context, **request.ethical_requirements}
            )

            return compliance_result

        except Exception as e:
            self.logger.error(f"Ethical compliance check failed: {e}")
            return {"approved": False, "error": str(e)}

    async def _discover_and_assign_agents(self, request: ResearchRequest) -> List[str]:
        """Discover and assign suitable agents for the research request"""
        try:
            # Determine required capabilities based on objective
            required_capabilities = await self._analyze_required_capabilities(request.objective)

            if request.required_agents:
                required_capabilities.extend(request.required_agents)

            # Discover matching agents
            available_agents = await self.service_registry.discover_agents(
                required_capabilities,
                max_agents=self.config.get('max_agents_per_workflow', 5)
            )

            if not available_agents:
                self.logger.warning(f"No agents found for capabilities: {required_capabilities}")
                return []

            # Select optimal agent combination
            selected_agents = await self._select_optimal_agents(available_agents, request)

            # Update agent assignments
            self.agent_assignments[request.request_id] = [agent.agent_id for agent in selected_agents]

            # Update agent loads
            for agent in selected_agents:
                agent.current_load += 1
                agent.last_active = datetime.now()

            self.logger.info(f"Assigned {len(selected_agents)} agents to request {request.request_id}")

            return [agent.agent_id for agent in selected_agents]

        except Exception as e:
            self.logger.error(f"Agent discovery failed: {e}")
            return []

    async def _analyze_required_capabilities(self, objective: str) -> List[str]:
        """Analyze research objective to determine required capabilities"""
        # Simple keyword-based capability analysis
        capability_keywords = {
            "experiment": ["experimental_design", "data_analysis", "methodical_experimentation"],
            "theory": ["theoretical_analysis", "mathematical_modeling", "theory_synthesis"],
            "analysis": ["data_analysis", "statistical_analysis", "pattern_recognition"],
            "creative": ["creative_thinking", "innovation", "hypothesis_generation"],
            "review": ["critical_review", "quality_assessment", "validation"],
            "implementation": ["coding", "algorithm_design", "system_implementation"]
        }

        objective_lower = objective.lower()
        required_capabilities = ["general_research"]  # Default capability

        for keyword, capabilities in capability_keywords.items():
            if keyword in objective_lower:
                required_capabilities.extend(capabilities)

        return list(set(required_capabilities))  # Remove duplicates

    async def _select_optimal_agents(self, available_agents: List[AgentCapability],
                                   request: ResearchRequest) -> List[AgentCapability]:
        """Select optimal combination of agents for the request"""
        if len(available_agents) == 1:
            return available_agents

        # Sort by availability score and current load
        sorted_agents = sorted(
            available_agents,
            key=lambda a: (a.availability_score, -a.current_load),
            reverse=True
        )

        # Select top agents considering coordination mode
        if request.coordination_mode == CoordinationMode.PARALLEL:
            # Select agents with diverse capabilities
            selected = []
            used_capabilities = set()

            for agent in sorted_agents:
                unique_caps = [cap for cap in agent.capabilities if cap not in used_capabilities]
                if unique_caps or len(selected) == 0:
                    selected.append(agent)
                    used_capabilities.update(agent.capabilities)

                    if len(selected) >= 3:  # Limit parallel agents
                        break

            return selected

        elif request.coordination_mode == CoordinationMode.CONSENSUS:
            # Select multiple agents for consensus
            return sorted_agents[:min(3, len(sorted_agents))]

        else:  # SEQUENTIAL or COMPETITIVE
            # Select best single agent
            return [sorted_agents[0]]

    async def _execute_workflow(self, workflow_plan: WorkflowPlan, request: ResearchRequest) -> Dict[str, Any]:
        """Execute the research workflow based on coordination mode"""
        try:
            if request.coordination_mode == CoordinationMode.PARALLEL:
                return await self._execute_parallel_workflow(workflow_plan, request)
            elif request.coordination_mode == CoordinationMode.CONSENSUS:
                return await self._execute_consensus_workflow(workflow_plan, request)
            elif request.coordination_mode == CoordinationMode.COMPETITIVE:
                return await self._execute_competitive_workflow(workflow_plan, request)
            else:  # SEQUENTIAL
                return await self._execute_sequential_workflow(workflow_plan, request)

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_sequential_workflow(self, workflow_plan: WorkflowPlan,
                                         request: ResearchRequest) -> Dict[str, Any]:
        """Execute workflow in sequential mode"""
        results = {}
        agent_contributions = {}

        for task in workflow_plan.tasks:
            try:
                # Delegate task to appropriate agent
                task_result = await self.supervisor.delegate_to_specialist(
                    task,
                    self.agent_assignments[request.request_id][0]  # Use primary agent
                )

                results[task.task_id] = task_result
                agent_contributions[task.task_id] = self.agent_assignments[request.request_id][0]

                # Check if task succeeded before proceeding
                if not task_result.get("success", False):
                    self.logger.warning(f"Task {task.task_id} failed, stopping sequential execution")
                    break

            except Exception as e:
                self.logger.error(f"Task {task.task_id} execution error: {e}")
                results[task.task_id] = {"success": False, "error": str(e)}

        return {
            "success": all(r.get("success", False) for r in results.values()),
            "results": results,
            "agent_contributions": agent_contributions,
            "confidence": sum(r.get("confidence", 0.5) for r in results.values()) / len(results) if results else 0.0
        }

    async def _execute_parallel_workflow(self, workflow_plan: WorkflowPlan,
                                       request: ResearchRequest) -> Dict[str, Any]:
        """Execute workflow in parallel mode"""
        assigned_agents = self.agent_assignments[request.request_id]
        tasks_by_agent = self._distribute_tasks_by_agent(workflow_plan.tasks, assigned_agents)

        # Execute tasks in parallel
        parallel_tasks = []
        for agent_id, tasks in tasks_by_agent.items():
            for task in tasks:
                parallel_tasks.append(
                    self.supervisor.delegate_to_specialist(task, agent_id)
                )

        parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

        # Process results
        results = {}
        agent_contributions = {}

        for i, result in enumerate(parallel_results):
            task_id = workflow_plan.tasks[i].task_id
            agent_id = assigned_agents[i % len(assigned_agents)]

            if isinstance(result, Exception):
                results[task_id] = {"success": False, "error": str(result)}
            else:
                results[task_id] = result

            agent_contributions[task_id] = agent_id

        return {
            "success": all(r.get("success", False) for r in results.values()),
            "results": results,
            "agent_contributions": agent_contributions,
            "confidence": sum(r.get("confidence", 0.5) for r in results.values()) / len(results) if results else 0.0
        }

    async def _execute_consensus_workflow(self, workflow_plan: WorkflowPlan,
                                        request: ResearchRequest) -> Dict[str, Any]:
        """Execute workflow with consensus building"""
        assigned_agents = self.agent_assignments[request.request_id]

        # Each agent executes the workflow independently
        agent_results = {}

        for agent_id in assigned_agents:
            try:
                agent_workflow_result = await self._execute_sequential_workflow(workflow_plan, request)
                agent_results[agent_id] = agent_workflow_result
            except Exception as e:
                self.logger.error(f"Agent {agent_id} consensus execution failed: {e}")
                agent_results[agent_id] = {"success": False, "error": str(e)}

        # Build consensus
        consensus_result = await self._build_consensus(agent_results, request)

        return consensus_result

    async def _execute_competitive_workflow(self, workflow_plan: WorkflowPlan,
                                          request: ResearchRequest) -> Dict[str, Any]:
        """Execute workflow in competitive mode"""
        assigned_agents = self.agent_assignments[request.request_id]

        # Each agent attempts to complete the entire workflow
        agent_results = {}

        for agent_id in assigned_agents:
            try:
                agent_workflow_result = await self._execute_sequential_workflow(workflow_plan, request)
                agent_results[agent_id] = agent_workflow_result
            except Exception as e:
                self.logger.error(f"Agent {agent_id} competitive execution failed: {e}")
                agent_results[agent_id] = {"success": False, "error": str(e)}

        # Select best result
        best_result = None
        best_agent = None
        best_score = 0.0

        for agent_id, result in agent_results.items():
            score = result.get("confidence", 0.0) * (1.0 if result.get("success", False) else 0.0)
            if score > best_score:
                best_score = score
                best_result = result
                best_agent = agent_id

        if best_result:
            best_result["winner_agent"] = best_agent
            best_result["competitive_results"] = {aid: r.get("success", False) for aid, r in agent_results.items()}

        return best_result or {"success": False, "error": "All agents failed"}

    def _distribute_tasks_by_agent(self, tasks: List[ResearchTask], agents: List[str]) -> Dict[str, List[ResearchTask]]:
        """Distribute tasks among available agents"""
        tasks_by_agent = {agent_id: [] for agent_id in agents}

        for i, task in enumerate(tasks):
            agent_id = agents[i % len(agents)]
            tasks_by_agent[agent_id].append(task)

        return tasks_by_agent

    async def _build_consensus(self, agent_results: Dict[str, Dict[str, Any]],
                             request: ResearchRequest) -> Dict[str, Any]:
        """Build consensus from multiple agent results"""
        try:
            successful_results = [r for r in agent_results.values() if r.get("success", False)]

            if not successful_results:
                return {
                    "success": False,
                    "error": "No agent achieved successful result",
                    "agent_results": agent_results
                }

            # Calculate consensus metrics
            avg_confidence = sum(r.get("confidence", 0.0) for r in successful_results) / len(successful_results)
            consensus_threshold = self.consensus_threshold

            # Check if confidence meets threshold
            if avg_confidence >= consensus_threshold:
                # Combine results from all successful agents
                combined_results = {}
                for i, result in enumerate(successful_results):
                    for key, value in result.get("results", {}).items():
                        if key not in combined_results:
                            combined_results[key] = []
                        combined_results[key].append(value)

                # Average or aggregate results
                final_results = {}
                for key, values in combined_results.items():
                    if isinstance(values[0], (int, float)):
                        final_results[key] = sum(values) / len(values)
                    elif isinstance(values[0], str):
                        # For strings, use most common value
                        from collections import Counter
                        final_results[key] = Counter(values).most_common(1)[0][0]
                    else:
                        # For complex objects, use first result
                        final_results[key] = values[0]

                return {
                    "success": True,
                    "results": final_results,
                    "confidence": avg_confidence,
                    "consensus_met": True,
                    "participating_agents": len(successful_results),
                    "agent_results": agent_results
                }
            else:
                return {
                    "success": False,
                    "error": f"Consensus threshold not met ({avg_confidence:.2f} < {consensus_threshold})",
                    "confidence": avg_confidence,
                    "consensus_met": False,
                    "agent_results": agent_results
                }

        except Exception as e:
            self.logger.error(f"Consensus building failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        try:
            return {
                "orchestrator_id": self.agent_id,
                "status": "active",
                "active_workflows": len(self.active_workflows),
                "total_workflows_completed": len(self.workflow_history),
                "registered_agents": len(self.service_registry.registered_agents),
                "average_workflow_time": sum(w["execution_time"] for w in self.workflow_history) / len(self.workflow_history) if self.workflow_history else 0,
                "success_rate": sum(1 for w in self.workflow_history if w["success"]) / len(self.workflow_history) if self.workflow_history else 0,
                "agent_assignments": self.agent_assignments,
                "performance_metrics": await self.performance_monitor.get_current_metrics()
            }

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Monitor workflow timeouts
                await self._check_workflow_timeouts()

                # Update performance metrics
                await self._update_performance_metrics()

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                # Clean up completed workflows
                await self._cleanup_completed_workflows()

                # Clean up old performance cache
                await self._cleanup_performance_cache()

            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    async def _check_workflow_timeouts(self):
        """Check for workflow timeouts"""
        current_time = datetime.now()

        for workflow_id, workflow in list(self.active_workflows.items()):
            # Simple timeout check (could be enhanced with per-workflow timeouts)
            if (current_time - workflow.created_at).total_seconds() > self.default_timeout:
                self.logger.warning(f"Workflow {workflow_id} timed out")
                await self._cleanup_workflow(workflow_id)

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Update agent performance in service registry
            for agent_id, agent in self.service_registry.registered_agents.items():
                # Simple performance decay over time
                if agent.availability_score < 1.0:
                    agent.availability_score = min(1.0, agent.availability_score + 0.01)

                # Reduce load over time
                if agent.current_load > 0:
                    agent.current_load = max(0, agent.current_load - 1)

        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")

    async def _cleanup_completed_workflows(self):
        """Clean up completed workflows"""
        # Remove workflows older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)

        completed_workflows = [
            workflow_id for workflow_id, workflow in self.active_workflows.items()
            if workflow.created_at < cutoff_time
        ]

        for workflow_id in completed_workflows:
            await self._cleanup_workflow(workflow_id)

    async def _cleanup_performance_cache(self):
        """Clean up old performance cache entries"""
        # Simple cache cleanup - remove entries older than 1 hour
        cutoff_time = time.time() - 3600

        old_keys = [
            key for key, timestamp in self.performance_cache.items()
            if isinstance(timestamp, (int, float)) and timestamp < cutoff_time
        ]

        for key in old_keys:
            del self.performance_cache[key]

    async def _cleanup_workflow(self, workflow_id: str):
        """Clean up workflow resources"""
        try:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            if workflow_id in self.agent_assignments:
                # Reduce agent loads
                for agent_id in self.agent_assignments[workflow_id]:
                    if agent_id in self.service_registry.registered_agents:
                        agent = self.service_registry.registered_agents[agent_id]
                        agent.current_load = max(0, agent.current_load - 1)

                del self.agent_assignments[workflow_id]

            self.logger.debug(f"Cleaned up workflow {workflow_id}")

        except Exception as e:
            self.logger.error(f"Workflow cleanup failed: {e}")

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        try:
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()

            # Clean up active workflows
            for workflow_id in list(self.active_workflows.keys()):
                await self._cleanup_workflow(workflow_id)

            self.logger.info(f"Research Orchestrator Agent {self.agent_id} shutdown successfully")

        except Exception as e:
            self.logger.error(f"Orchestrator shutdown failed: {e}")


# Convenience functions for creating and managing orchestrator instances

_orchestrator_instance = None

async def get_orchestrator(config: Dict[str, Any] = None) -> ResearchOrchestratorAgent:
    """Get or create the global orchestrator instance"""
    global _orchestrator_instance

    if _orchestrator_instance is None:
        if config is None:
            config = {
                'max_concurrent_workflows': 10,
                'default_timeout': 3600,
                'consensus_threshold': 0.7,
                'max_agents_per_workflow': 5
            }

        _orchestrator_instance = ResearchOrchestratorAgent(config)
        await _orchestrator_instance.initialize()

    return _orchestrator_instance

async def create_research_request(objective: str, **kwargs) -> ResearchRequest:
    """Create a research request with default parameters"""
    return ResearchRequest(
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        objective=objective,
        **kwargs
    )

async def coordinate_research_simple(objective: str, **kwargs) -> ResearchResults:
    """Simple interface for research coordination"""
    orchestrator = await get_orchestrator()
    request = await create_research_request(objective, **kwargs)
    return await orchestrator.coordinate_research(request)