"""
Supervisor Agent Implementation

The Supervisor Agent coordinates multi-agent workflows, makes strategic decisions,
and orchestrates the overall research process in the enhanced AI-Scientist-v2 system.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import yaml
from ai_scientist.llm import create_client


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """Represents a research task within the system."""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    assigned_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = None
    context: Dict[str, Any] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.context is None:
            self.context = {}


@dataclass
class WorkflowPlan:
    """Represents a workflow execution plan."""
    plan_id: str
    tasks: List[ResearchTask]
    dependencies: Dict[str, List[str]]
    estimated_duration: Optional[int] = None
    resource_requirements: Optional[Dict[str, Any]] = None
    success_criteria: Optional[List[str]] = None


@dataclass
class SystemState:
    """Represents current state of the research system."""
    active_tasks: List[ResearchTask]
    completed_tasks: List[ResearchTask]
    available_agents: List[str]
    resource_usage: Dict[str, float]
    current_theory_version: Optional[str] = None
    last_update: Optional[datetime] = None


@dataclass
class ProgressEvaluation:
    """Evaluation of current research progress."""
    overall_progress: float  # 0.0 to 1.0
    stage_progress: Dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]
    estimated_completion: Optional[datetime] = None


@dataclass
class StrategyDecision:
    """Strategic decision made by supervisor."""
    decision_id: str
    decision_type: str
    action: str
    rationale: str
    confidence: float
    alternatives: List[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SupervisorAgent:
    """
    Enhanced Supervisor Agent for orchestrating multi-agent research workflows.
    
    This agent provides high-level coordination, strategic decision making,
    and workflow orchestration for the AI-Scientist-v2 system.
    """

    def __init__(self, config: Dict[str, Any], profile_manager=None, knowledge_manager=None):
        """
        Initialize the Supervisor Agent.
        
        Args:
            config: Configuration dictionary for the supervisor agent
            profile_manager: Agent profile manager instance
            knowledge_manager: Knowledge management system instance
        """
        self.config = config
        self.profile_manager = profile_manager
        self.knowledge_manager = knowledge_manager
        
        # Initialize LLM client
        supervisor_config = config.get("orchestration", {}).get("supervisor_agent", {})
        self.llm_client = create_client(
            model=supervisor_config.get("model", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            temperature=supervisor_config.get("temp", 0.7),
            max_tokens=supervisor_config.get("max_tokens", 8192)
        )
        
        # Configuration parameters
        self.decision_threshold = supervisor_config.get("decision_threshold", 0.8)
        self.coordination_mode = supervisor_config.get("coordination_mode", "hierarchical")
        
        # State management
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        self.task_queue: List[ResearchTask] = []
        self.agent_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.system_metrics: Dict[str, Any] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def coordinate_workflow(self, research_objective: str, context: Dict[str, Any] = None) -> WorkflowPlan:
        """
        Create and coordinate a comprehensive research workflow.
        
        Args:
            research_objective: High-level research objective
            context: Additional context and constraints
            
        Returns:
            WorkflowPlan: Structured workflow plan
        """
        self.logger.info(f"Coordinating workflow for objective: {research_objective}")
        
        # Analyze research objective and break it down
        analysis_prompt = f"""
        As a research supervisor, analyze this research objective and create a structured workflow plan:
        
        Research Objective: {research_objective}
        Context: {context or 'None provided'}
        
        Please provide:
        1. Key research phases and tasks
        2. Task dependencies and ordering
        3. Appropriate specialist agents for each task
        4. Resource requirements and constraints
        5. Success criteria and evaluation metrics
        
        Structure your response as a detailed workflow plan.
        """
        
        response = await self.llm_client.generate(analysis_prompt)
        
        # Parse response and create workflow plan
        workflow_plan = self._parse_workflow_response(response, research_objective)
        
        # Store and track workflow
        self.active_workflows[workflow_plan.plan_id] = workflow_plan
        
        return workflow_plan

    async def delegate_to_specialist(self, task: ResearchTask, specialist_type: str) -> Dict[str, Any]:
        """
        Delegate a task to an appropriate specialist agent.
        
        Args:
            task: Research task to delegate
            specialist_type: Type of specialist agent needed
            
        Returns:
            Dict containing task assignment details and initial response
        """
        self.logger.info(f"Delegating task {task.task_id} to {specialist_type} specialist")
        
        # Get specialist configuration
        specialist_config = self.config.get("orchestration", {}).get("specialist_agents", {}).get(specialist_type, {})
        
        if not specialist_config.get("enabled", False):
            raise ValueError(f"Specialist agent {specialist_type} is not enabled")
        
        # Get agent profile if available
        agent_profile = None
        if self.profile_manager:
            profile_name = specialist_config.get("profile")
            agent_profile = self.profile_manager.get_profile(profile_name) if profile_name else None
        
        # Create specialized prompt based on agent profile
        delegation_prompt = self._create_delegation_prompt(task, specialist_type, agent_profile)
        
        # Create specialist client
        specialist_client = create_client(
            model=specialist_config.get("model", "gpt-4o-2024-11-20"),
            temperature=specialist_config.get("temp", 0.7),
            max_tokens=specialist_config.get("max_tokens", 8192)
        )
        
        # Delegate task
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = specialist_type
        task.started_at = datetime.now()
        
        try:
            response = await specialist_client.generate(delegation_prompt)
            
            # Update task tracking
            self.agent_assignments[task.task_id] = specialist_type
            
            return {
                "task_id": task.task_id,
                "assigned_agent": specialist_type,
                "initial_response": response,
                "status": "delegated",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.logger.error(f"Failed to delegate task {task.task_id}: {e}")
            raise

    async def evaluate_progress(self, workflow_id: str) -> ProgressEvaluation:
        """
        Evaluate current progress of a research workflow.
        
        Args:
            workflow_id: ID of the workflow to evaluate
            
        Returns:
            ProgressEvaluation: Detailed progress assessment
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate task completion rates
        total_tasks = len(workflow.tasks)
        completed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for task in workflow.tasks if task.status == TaskStatus.FAILED)
        
        overall_progress = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Identify bottlenecks
        bottlenecks = []
        in_progress_tasks = [task for task in workflow.tasks if task.status == TaskStatus.IN_PROGRESS]
        
        for task in in_progress_tasks:
            if task.started_at and (datetime.now() - task.started_at).total_seconds() > 3600:  # 1 hour
                bottlenecks.append(f"Task {task.task_id} running over time")
        
        # Generate recommendations
        recommendations = []
        if failed_tasks > 0:
            recommendations.append("Review and retry failed tasks")
        if len(bottlenecks) > 0:
            recommendations.append("Address identified bottlenecks")
        if overall_progress < 0.5 and len(in_progress_tasks) < 2:
            recommendations.append("Consider parallel task execution")
        
        return ProgressEvaluation(
            overall_progress=overall_progress,
            stage_progress=self._calculate_stage_progress(workflow),
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            estimated_completion=self._estimate_completion_time(workflow)
        )

    async def make_strategic_decision(self, context: Dict[str, Any]) -> StrategyDecision:
        """
        Make a strategic decision based on current context and system state.
        
        Args:
            context: Decision context including options, constraints, and data
            
        Returns:
            StrategyDecision: Strategic decision with rationale
        """
        decision_prompt = f"""
        As a research supervisor, make a strategic decision based on the following context:
        
        Context: {json.dumps(context, indent=2)}
        System State: {self._get_system_state_summary()}
        
        Consider:
        1. Research objectives and progress
        2. Available resources and constraints
        3. Risk assessment and mitigation
        4. Alternative approaches
        5. Expected outcomes and confidence levels
        
        Provide a clear decision with detailed rationale.
        """
        
        response = await self.llm_client.generate(decision_prompt)
        
        # Parse decision response
        decision = self._parse_decision_response(response, context)
        
        # Log decision
        self.logger.info(f"Strategic decision made: {decision.action} (confidence: {decision.confidence})")
        
        return decision

    async def interface_with_theory_agent(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interface with the Theory Evolution Agent to correlate findings.
        
        Args:
            findings: Research findings to correlate with theory
            
        Returns:
            Dict containing theory correlation results
        """
        if not self.knowledge_manager:
            self.logger.warning("Knowledge manager not available for theory interface")
            return {"status": "unavailable", "message": "Theory agent not configured"}
        
        self.logger.info("Interfacing with Theory Evolution Agent")
        
        # Format findings for theory correlation
        correlation_request = {
            "findings": findings,
            "timestamp": datetime.now().isoformat(),
            "context": "supervisor_workflow",
            "correlation_threshold": self.config.get("theory_evolution", {}).get("correlation_threshold", 0.75)
        }
        
        try:
            # This would interface with the Theory Evolution Agent
            # For now, we'll return a structured response
            correlation_result = {
                "status": "processed",
                "correlation_score": 0.85,  # Placeholder
                "theory_updates": ["Added new evidence for hypothesis X"],
                "recommendations": ["Continue current research direction"],
                "timestamp": datetime.now().isoformat()
            }
            
            return correlation_result
            
        except Exception as e:
            self.logger.error(f"Error interfacing with theory agent: {e}")
            return {"status": "error", "message": str(e)}

    def _parse_workflow_response(self, response: str, objective: str) -> WorkflowPlan:
        """Parse LLM response into a structured workflow plan."""
        import uuid
        
        # This is a simplified parser - in practice, would use more sophisticated parsing
        tasks = []
        dependencies = {}
        
        # Create placeholder tasks based on typical research workflow
        task_templates = [
            ("ideation", "Generate and refine research ideas", TaskPriority.HIGH),
            ("literature_review", "Conduct comprehensive literature review", TaskPriority.MEDIUM),
            ("methodology", "Design experimental methodology", TaskPriority.HIGH),
            ("experiment", "Execute experiments and collect data", TaskPriority.CRITICAL),
            ("analysis", "Analyze experimental results", TaskPriority.HIGH),
            ("theory_integration", "Integrate findings with theory", TaskPriority.MEDIUM),
            ("writeup", "Write research paper", TaskPriority.HIGH),
            ("review", "Peer review and revision", TaskPriority.MEDIUM)
        ]
        
        for i, (task_type, description, priority) in enumerate(task_templates):
            task_id = f"task_{i+1}_{task_type}"
            task = ResearchTask(
                task_id=task_id,
                task_type=task_type,
                description=description,
                priority=priority,
                status=TaskStatus.PENDING,
                context={"objective": objective}
            )
            tasks.append(task)
            
            # Set up basic dependencies
            if i > 0:
                dependencies[task_id] = [tasks[i-1].task_id]
            else:
                dependencies[task_id] = []
        
        return WorkflowPlan(
            plan_id=str(uuid.uuid4()),
            tasks=tasks,
            dependencies=dependencies,
            estimated_duration=7200,  # 2 hours placeholder
            success_criteria=["All tasks completed successfully", "Theory integration achieved"]
        )

    def _create_delegation_prompt(self, task: ResearchTask, specialist_type: str, agent_profile: Dict = None) -> str:
        """Create a specialized prompt for task delegation."""
        base_prompt = f"""
        Task ID: {task.task_id}
        Task Type: {task.task_type}
        Description: {task.description}
        Priority: {task.priority.name}
        Context: {json.dumps(task.context, indent=2)}
        """
        
        if agent_profile:
            profile_prompt = agent_profile.get("prompting_style", {}).get("prefix", "")
            base_prompt = f"{profile_prompt}\n\n{base_prompt}"
        
        return base_prompt

    def _calculate_stage_progress(self, workflow: WorkflowPlan) -> Dict[str, float]:
        """Calculate progress for different workflow stages."""
        stage_progress = {}
        
        # Group tasks by type/stage
        task_stages = {}
        for task in workflow.tasks:
            stage = task.task_type
            if stage not in task_stages:
                task_stages[stage] = []
            task_stages[stage].append(task)
        
        # Calculate progress per stage
        for stage, tasks in task_stages.items():
            completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
            total = len(tasks)
            stage_progress[stage] = completed / total if total > 0 else 0.0
        
        return stage_progress

    def _estimate_completion_time(self, workflow: WorkflowPlan) -> Optional[datetime]:
        """Estimate workflow completion time."""
        if workflow.estimated_duration:
            return datetime.now() + datetime.timedelta(seconds=workflow.estimated_duration)
        return None

    def _get_system_state_summary(self) -> Dict[str, Any]:
        """Get summary of current system state."""
        return {
            "active_workflows": len(self.active_workflows),
            "queued_tasks": len(self.task_queue),
            "agent_assignments": len(self.agent_assignments),
            "timestamp": datetime.now().isoformat()
        }

    def _parse_decision_response(self, response: str, context: Dict[str, Any]) -> StrategyDecision:
        """Parse LLM decision response into structured decision."""
        import uuid
        
        # Simplified parsing - would be more sophisticated in practice
        return StrategyDecision(
            decision_id=str(uuid.uuid4()),
            decision_type="strategic",
            action="proceed_with_current_plan",
            rationale=response[:200] + "..." if len(response) > 200 else response,
            confidence=0.85,  # Placeholder
            alternatives=["alternative_approach_1", "alternative_approach_2"]
        )

    def export_state(self) -> Dict[str, Any]:
        """Export current supervisor state for persistence."""
        return {
            "active_workflows": {k: asdict(v) for k, v in self.active_workflows.items()},
            "task_queue": [asdict(task) for task in self.task_queue],
            "agent_assignments": self.agent_assignments,
            "system_metrics": self.system_metrics,
            "timestamp": datetime.now().isoformat()
        }

    def import_state(self, state_data: Dict[str, Any]) -> None:
        """Import supervisor state from persistence."""
        # Implementation would restore state from saved data
        self.logger.info("Importing supervisor state")
        pass