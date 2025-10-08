"""
Ethical Framework Integration Module

This module provides seamless integration between the Ethical Framework Agent
and the existing Research Orchestrator Agent, enabling real-time ethical
oversight of autonomous research activities.

Key Integration Features:
- Real-time ethical compliance checking in research workflows
- Ethical scoring and decision support for agent coordination
- Automated ethical constraint enforcement
- Multi-layered ethical oversight integration
- Human oversight workflow management
- Ethical reporting and audit trail generation
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

# Import ethical framework components
from .ethical_framework_agent import (
    EthicalFrameworkAgent, ResearchEthicsContext, EthicalAssessment,
    DecisionMode, ComplianceStatus, EthicalRiskLevel, get_ethical_framework
)

# Import orchestrator components
from ..orchestration.research_orchestrator_agent import (
    ResearchOrchestratorAgent, ResearchRequest, ResearchResults
)

# Import existing components
from ..security.security_manager import SecurityManager


class IntegrationMode(Enum):
    """Ethical integration modes"""
    PRE_CHECK = "pre_check"  # Ethical assessment before research starts
    REAL_TIME = "real_time"  # Continuous ethical monitoring during research
    POST_REVIEW = "post_review"  # Ethical review after research completion
    COMPREHENSIVE = "comprehensive"  # All three modes combined


class EthicalIntegrationLevel(Enum):
    """Levels of ethical integration"""
    MINIMAL = "minimal"  # Basic compliance checking
    STANDARD = "standard"  # Full ethical oversight
    ENHANCED = "enhanced"  # Advanced ethical governance with learning
    CRITICAL = "critical"  # Maximum oversight for sensitive research


@dataclass
class EthicalIntegrationConfig:
    """Configuration for ethical integration"""
    integration_mode: IntegrationMode = IntegrationMode.COMPREHENSIVE
    integration_level: EthicalIntegrationLevel = EthicalIntegrationLevel.STANDARD
    enable_real_time_monitoring: bool = True
    ethical_threshold: float = 0.8
    human_oversight_threshold: float = 0.7
    enable_adaptive_learning: bool = True
    enable_cultural_considerations: bool = True
    reporting_interval: int = 3600  # seconds
    audit_trail_enabled: bool = True
    constraint_enforcement_enabled: bool = True


@dataclass
class EthicalCheckpoint:
    """Ethical checkpoint in research workflow"""
    checkpoint_id: str
    workflow_stage: str
    ethical_requirements: List[str]
    assessment_required: bool
    blocking: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchEthicsState:
    """Ethical state tracking for research workflows"""
    workflow_id: str
    current_ethical_score: float
    compliance_status: ComplianceStatus
    active_constraints: List[str]
    checkpoint_history: List[EthicalAssessment]
    human_oversight_required: bool
    last_ethical_check: datetime
    violation_history: List[Dict[str, Any]]
    mitigation_actions: List[Dict[str, Any]]


class EthicalIntegrationManager:
    """
    Ethical Integration Manager

    Manages the integration between ethical framework and research orchestrator,
    providing seamless ethical oversight throughout the research lifecycle.
    """

    def __init__(self, config: EthicalIntegrationConfig = None):
        self.config = config or EthicalIntegrationConfig()
        self.manager_id = f"ethical_integration_{uuid.uuid4().hex[:8]}"

        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.manager_id}")

        # Initialize components
        self.ethical_framework: EthicalFrameworkAgent = None
        self.orchestrator: ResearchOrchestratorAgent = None
        # Initialize security manager
        self.security_manager = SecurityManager()

        # Integration state
        self.active_research_states: Dict[str, ResearchEthicsState] = {}
        self.ethical_checkpoints: List[EthicalCheckpoint] = []
        self.integration_metrics = {
            "total_assessments": 0,
            "blocked_research": 0,
            "human_oversight_requests": 0,
            "compliance_violations": 0,
            "successful_mitigations": 0
        }

        # Initialize default ethical checkpoints
        self._initialize_ethical_checkpoints()

        # Background tasks
        self._monitoring_task = None
        self._reporting_task = None

    def _initialize_ethical_checkpoints(self):
        """Initialize default ethical checkpoints for research workflow"""
        self.ethical_checkpoints = [
            EthicalCheckpoint(
                checkpoint_id="proposal_review",
                workflow_stage="research_proposal",
                ethical_requirements=["human_subjects_protection", "data_privacy", "bias_assessment"],
                assessment_required=True,
                blocking=True
            ),
            EthicalCheckpoint(
                checkpoint_id="data_collection",
                workflow_stage="data_acquisition",
                ethical_requirements=["data_privacy", "informed_consent", "data_security"],
                assessment_required=True,
                blocking=True
            ),
            EthicalCheckpoint(
                checkpoint_id="experiment_design",
                workflow_stage="methodology_design",
                ethical_requirements=["safety_protocols", "risk_assessment", "environmental_impact"],
                assessment_required=True,
                blocking=False
            ),
            EthicalCheckpoint(
                checkpoint_id="analysis_phase",
                workflow_stage="data_analysis",
                ethical_requirements=["bias_detection", "fairness_assessment", "transparency"],
                assessment_required=True,
                blocking=False
            ),
            EthicalCheckpoint(
                checkpoint_id="publication_review",
                workflow_stage="results_publication",
                ethical_requirements=["authorship_ethics", "plagiarism_check", "conflict_of_interest"],
                assessment_required=True,
                blocking=False
            )
        ]

    async def initialize(self, ethical_framework: EthicalFrameworkAgent,
                       orchestrator: ResearchOrchestratorAgent):
        """Initialize the integration manager with framework and orchestrator"""
        try:
            self.ethical_framework = ethical_framework
            self.orchestrator = orchestrator

            # Start background tasks
            if self.config.enable_real_time_monitoring:
                self._monitoring_task = asyncio.create_task(self._real_time_monitoring_loop())

            self._reporting_task = asyncio.create_task(self._periodic_reporting_loop())

            self.logger.info(f"Ethical Integration Manager {self.manager_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize integration manager: {e}")
            raise

    async def integrate_research_request(self, request: ResearchRequest) -> Dict[str, Any]:
        """
        Integrate ethical assessment into research request processing

        Args:
            request: Research request to process ethically

        Returns:
            Enhanced request with ethical assessment results
        """
        try:
            self.logger.info(f"Integrating ethical assessment for request {request.request_id}")

            # Create research ethics context from request
            ethics_context = self._create_ethics_context_from_request(request)

            # Perform ethical assessment
            assessment = await self.ethical_framework.assess_research_ethics(
                request.request_id,
                request.objective,
                ethics_context
            )

            # Update integration metrics
            self.integration_metrics["total_assessments"] += 1

            # Handle assessment results
            integration_result = await self._handle_ethical_assessment(request, assessment)

            # Create research ethics state
            research_state = ResearchEthicsState(
                workflow_id=request.request_id,
                current_ethical_score=assessment.overall_score,
                compliance_status=assessment.compliance_status,
                active_constraints=[v["constraint_id"] for v in assessment.constraint_violations],
                checkpoint_history=[assessment],
                human_oversight_required=assessment.requires_human_oversight,
                last_ethical_check=datetime.now(),
                violation_history=assessment.constraint_violations,
                mitigation_actions=[]
            )

            self.active_research_states[request.request_id] = research_state

            # Log integration
            await self._log_integration_event(request.request_id, assessment, integration_result)

            return integration_result

        except Exception as e:
            self.logger.error(f"Research request integration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "ethical_clearance": False,
                "requires_human_oversight": True
            }

    def _create_ethics_context_from_request(self, request: ResearchRequest) -> ResearchEthicsContext:
        """Create research ethics context from research request"""
        return ResearchEthicsContext(
            research_domain=self._extract_research_domain(request.objective),
            methodologies=self._extract_methodologies(request.objective),
            data_sources=request.context.get("data_sources", []),
            human_involvement=self._detect_human_involvement(request.objective),
            environmental_impact=self._detect_environmental_impact(request.objective),
            cultural_considerations=request.context.get("cultural_considerations", []),
            regulatory_requirements=request.context.get("regulatory_requirements", []),
            institutional_policies=request.context.get("institutional_policies", []),
            stakeholder_groups=request.context.get("stakeholder_groups", []),
            expected_outcomes=request.context.get("expected_outcomes", []),
            potential_risks=request.context.get("potential_risks", []),
            mitigation_strategies=request.context.get("mitigation_strategies", [])
        )

    def _extract_research_domain(self, objective: str) -> str:
        """Extract research domain from objective"""
        domain_keywords = {
            "medical": ["medical", "health", "clinical", "patient", "healthcare"],
            "psychology": ["psychology", "behavior", "mental", "cognitive", "emotion"],
            "biology": ["biology", "genetic", "molecular", "cell", "organism"],
            "computer_science": ["computer", "algorithm", "software", "ai", "machine_learning"],
            "physics": ["physics", "quantum", "particle", "energy", "matter"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "compound"],
            "social_science": ["social", "society", "cultural", "political", "economic"]
        }

        objective_lower = objective.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in objective_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_methodologies(self, objective: str) -> List[str]:
        """Extract research methodologies from objective"""
        methodology_keywords = {
            "experimental": ["experiment", "trial", "test", "manipulate"],
            "observational": ["observe", "monitor", "survey", "measure"],
            "computational": ["simulate", "model", "compute", "algorithm"],
            "theoretical": ["theory", "model", "hypothesis", "analyze"],
            "qualitative": ["interview", "focus_group", "ethnography", "narrative"],
            "quantitative": ["statistical", "numerical", "data", "measurement"]
        }

        objective_lower = objective.lower()
        methodologies = []

        for methodology, keywords in methodology_keywords.items():
            if any(keyword in objective_lower for keyword in keywords):
                methodologies.append(methodology)

        return methodologies or ["general_research"]

    def _detect_human_involvement(self, objective: str) -> bool:
        """Detect if research involves human subjects"""
        human_keywords = ["human", "subject", "participant", "patient", "user", "people", "person"]
        objective_lower = objective.lower()
        return any(keyword in objective_lower for keyword in human_keywords)

    def _detect_environmental_impact(self, objective: str) -> bool:
        """Detect if research has environmental impact"""
        environmental_keywords = ["environment", "ecology", "climate", "pollution", "sustainability"]
        objective_lower = objective.lower()
        return any(keyword in objective_lower for keyword in environmental_keywords)

    async def _handle_ethical_assessment(self, request: ResearchRequest,
                                      assessment: EthicalAssessment) -> Dict[str, Any]:
        """Handle ethical assessment results"""
        result = {
            "success": True,
            "request_id": request.request_id,
            "ethical_assessment": assessment,
            "ethical_clearance": True,
            "requires_human_oversight": assessment.requires_human_oversight,
            "decision_mode": assessment.decision_mode.value,
            "recommendations": assessment.recommendations,
            "constraint_violations": len(assessment.constraint_violations)
        }

        # Handle blocking violations
        if assessment.compliance_status in [ComplianceStatus.CRITICAL_VIOLATIONS]:
            result["ethical_clearance"] = False
            result["blocked_reason"] = "Critical ethical violations detected"
            self.integration_metrics["blocked_research"] += 1
            self.integration_metrics["compliance_violations"] += 1

        # Handle human oversight requirements
        if assessment.requires_human_oversight:
            result["requires_human_oversight"] = True
            self.integration_metrics["human_oversight_requests"] += 1

        # Handle minor violations
        if assessment.constraint_violations:
            self.integration_metrics["compliance_violations"] += 1

        return result

    async def monitor_research_workflow(self, workflow_id: str, workflow_stage: str,
                                     content: str) -> Dict[str, Any]:
        """
        Monitor research workflow at specific stages

        Args:
            workflow_id: Unique identifier for the research workflow
            workflow_stage: Current stage of the workflow
            content: Content to assess ethically

        Returns:
            Monitoring results with ethical assessment
        """
        try:
            # Find relevant ethical checkpoint
            checkpoint = None
            for cp in self.ethical_checkpoints:
                if cp.workflow_stage == workflow_stage:
                    checkpoint = cp
                    break

            if not checkpoint:
                return {"success": True, "message": "No ethical checkpoint for this stage"}

            # Get current research state
            research_state = self.active_research_states.get(workflow_id)
            if not research_state:
                return {"success": False, "error": "Research workflow not found"}

            # Create context for assessment
            ethics_context = ResearchEthicsContext(
                research_domain="ongoing_research",
                methodologies=["current_stage"],
                data_sources=[],
                human_involvement=research_state.human_oversight_required,
                environmental_impact=False,
                cultural_considerations=[],
                regulatory_requirements=[],
                institutional_policies=[],
                stakeholder_groups=[],
                expected_outcomes=[],
                potential_risks=[],
                mitigation_strategies=[]
            )

            # Perform stage-specific ethical assessment
            assessment = await self.ethical_framework.assess_research_ethics(
                f"{workflow_id}_{workflow_stage}",
                content,
                ethics_context
            )

            # Update research state
            research_state.checkpoint_history.append(assessment)
            research_state.current_ethical_score = assessment.overall_score
            research_state.compliance_status = assessment.compliance_status
            research_state.last_ethical_check = datetime.now()

            # Handle violations
            if assessment.constraint_violations:
                research_state.violation_history.extend(assessment.constraint_violations)

            # Check if blocking
            if checkpoint.blocking and assessment.compliance_status in [ComplianceStatus.CRITICAL_VIOLATIONS]:
                return {
                    "success": False,
                    "blocked": True,
                    "checkpoint": checkpoint.checkpoint_id,
                    "assessment": assessment,
                    "reason": "Blocking ethical checkpoint failed"
                }

            return {
                "success": True,
                "checkpoint": checkpoint.checkpoint_id,
                "assessment": assessment,
                "workflow_continues": True
            }

        except Exception as e:
            self.logger.error(f"Workflow monitoring failed: {e}")
            return {"success": False, "error": str(e)}

    async def enhance_orchestrator_decision(self, orchestrator_decision: Dict[str, Any],
                                          workflow_id: str) -> Dict[str, Any]:
        """
        Enhance orchestrator decision with ethical considerations

        Args:
            orchestrator_decision: Original decision from orchestrator
            workflow_id: Workflow identifier

        Returns:
            Ethically enhanced decision
        """
        try:
            research_state = self.active_research_states.get(workflow_id)
            if not research_state:
                return orchestrator_decision  # Return original if no state

            # Calculate ethical confidence adjustment
            ethical_confidence = research_state.current_ethical_score
            original_confidence = orchestrator_decision.get("confidence", 0.5)

            # Adjust confidence based on ethical considerations
            adjusted_confidence = original_confidence * ethical_confidence

            # Add ethical metadata
            enhanced_decision = orchestrator_decision.copy()
            enhanced_decision.update({
                "ethical_confidence": ethical_confidence,
                "adjusted_confidence": adjusted_confidence,
                "compliance_status": research_state.compliance_status.value,
                "human_oversight_required": research_state.human_oversight_required,
                "active_violations": len(research_state.violation_history),
                "ethical_recommendations": research_state.mitigation_actions
            })

            # Apply ethical constraints if needed
            if research_state.compliance_status in [ComplianceStatus.CRITICAL_VIOLATIONS]:
                enhanced_decision["ethically_blocked"] = True
                enhanced_decision["ethical_reason"] = "Critical ethical violations present"

            return enhanced_decision

        except Exception as e:
            self.logger.error(f"Decision enhancement failed: {e}")
            return orchestrator_decision

    async def process_human_oversight_response(self, oversight_request_id: str,
                                            decision: str, reasoning: str) -> Dict[str, Any]:
        """
        Process human oversight response and update research state accordingly

        Args:
            oversight_request_id: ID of the oversight request
            decision: Human decision (approve/deny/modify)
            reasoning: Human reasoning for the decision

        Returns:
            Processing result
        """
        try:
            # Process with ethical framework
            framework_result = await self.ethical_framework.process_human_oversight_decision(
                oversight_request_id, decision, reasoning
            )

            if not framework_result["success"]:
                return framework_result

            # Update relevant research state
            for workflow_id, state in self.active_research_states.items():
                if state.human_oversight_required:
                    # Find if this oversight request relates to this workflow
                    # (In a real implementation, we'd have better tracking)
                    if decision.lower() == "approve":
                        state.human_oversight_required = False
                        state.mitigation_actions.append({
                            "action": "human_oversight_approved",
                            "timestamp": datetime.now().isoformat(),
                            "reasoning": reasoning
                        })

            self.logger.info(f"Processed human oversight for {oversight_request_id}: {decision}")

            return {"success": True, "decision": decision, "oversight_request_id": oversight_request_id}

        except Exception as e:
            self.logger.error(f"Human oversight processing failed: {e}")
            return {"success": False, "error": str(e)}

    async def _real_time_monitoring_loop(self):
        """Background loop for real-time ethical monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                await self._monitor_active_research_states()

            except Exception as e:
                self.logger.error(f"Real-time monitoring loop error: {e}")

    async def _monitor_active_research_states(self):
        """Monitor all active research states for ethical compliance"""
        try:
            current_time = datetime.now()

            for workflow_id, state in self.active_research_states.items():
                # Check for stale research states
                time_since_check = (current_time - state.last_ethical_check).total_seconds()

                if time_since_check > 3600:  # 1 hour without ethical check
                    self.logger.warning(f"Research workflow {workflow_id} has not been ethically assessed recently")

                # Check for deteriorating ethical status
                if state.current_ethical_score < 0.5 and not state.human_oversight_required:
                    self.logger.warning(f"Research workflow {workflow_id} has low ethical score: {state.current_ethical_score}")

        except Exception as e:
            self.logger.error(f"Active states monitoring failed: {e}")

    async def _periodic_reporting_loop(self):
        """Background loop for periodic ethical reporting"""
        while True:
            try:
                await asyncio.sleep(self.config.reporting_interval)

                await self._generate_integration_report()

            except Exception as e:
                self.logger.error(f"Periodic reporting loop error: {e}")

    async def _generate_integration_report(self):
        """Generate periodic integration report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "integration_manager_id": self.manager_id,
                "metrics": self.integration_metrics,
                "active_workflows": len(self.active_research_states),
                "ethical_checkpoints": len(self.ethical_checkpoints),
                "system_status": await self.get_integration_status()
            }

            self.logger.info(f"Generated integration report with {len(self.active_research_states)} active workflows")

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")

    async def _log_integration_event(self, workflow_id: str, assessment: EthicalAssessment,
                                    result: Dict[str, Any]):
        """Log ethical integration event"""
        if self.config.audit_trail_enabled:
            event = {
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "assessment_id": assessment.assessment_id,
                "ethical_score": assessment.overall_score,
                "risk_level": assessment.risk_level.value,
                "compliance_status": assessment.compliance_status.value,
                "requires_human_oversight": assessment.requires_human_oversight,
                "ethical_clearance": result["ethical_clearance"],
                "decision_mode": assessment.decision_mode.value
            }

            self.logger.info(f"Ethical integration event logged: {event_type}")

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        try:
            active_states = list(self.active_research_states.values())

            return {
                "integration_manager_id": self.manager_id,
                "status": "active",
                "integration_mode": self.config.integration_mode.value,
                "integration_level": self.config.integration_level.value,
                "active_workflows": len(active_states),
                "ethical_checkpoints": len(self.ethical_checkpoints),
                "metrics": self.integration_metrics,
                "average_ethical_score": sum(s.current_ethical_score for s in active_states) / len(active_states) if active_states else 0,
                "workflows_requiring_oversight": sum(1 for s in active_states if s.human_oversight_required),
                "total_violations": sum(len(s.violation_history) for s in active_states),
                "successful_mitigations": sum(len(s.mitigation_actions) for s in active_states),
                "real_time_monitoring_enabled": self.config.enable_real_time_monitoring,
                "adaptive_learning_enabled": self.config.enable_adaptive_learning,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    async def generate_ethical_integration_report(self, workflow_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive ethical integration report"""
        try:
            if workflow_id:
                # Generate report for specific workflow
                state = self.active_research_states.get(workflow_id)
                if not state:
                    return {"success": False, "error": "Workflow not found"}

                report = {
                    "workflow_id": workflow_id,
                    "generated_at": datetime.now().isoformat(),
                    "ethical_state": {
                        "current_score": state.current_ethical_score,
                        "compliance_status": state.compliance_status.value,
                        "human_oversight_required": state.human_oversight_required,
                        "active_constraints": state.active_constraints
                    },
                    "checkpoint_history": [
                        {
                            "assessment_id": cp.assessment_id,
                            "timestamp": cp.timestamp.isoformat(),
                            "score": cp.overall_score,
                            "risk_level": cp.risk_level.value,
                            "compliance_status": cp.compliance_status.value
                        }
                        for cp in state.checkpoint_history
                    ],
                    "violation_history": state.violation_history,
                    "mitigation_actions": state.mitigation_actions
                }
            else:
                # Generate overall integration report
                report = {
                    "report_type": "integration_overview",
                    "generated_at": datetime.now().isoformat(),
                    "integration_status": await self.get_integration_status(),
                    "workflow_summary": {
                        "total_workflows": len(self.active_research_states),
                        "high_risk_workflows": sum(1 for s in self.active_research_states.values() if s.current_ethical_score < 0.6),
                        "compliant_workflows": sum(1 for s in self.active_research_states.values() if s.compliance_status == ComplianceStatus.FULLY_COMPLIANT),
                        "oversight_required": sum(1 for s in self.active_research_states.values() if s.human_oversight_required)
                    },
                    "ethical_checkpoint_effectiveness": await self._calculate_checkpoint_effectiveness(),
                    "integration_trends": await self._calculate_integration_trends()
                }

            return {"success": True, "report": report}

        except Exception as e:
            self.logger.error(f"Integration report generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _calculate_checkpoint_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness of ethical checkpoints"""
        effectiveness = {}

        for checkpoint in self.ethical_checkpoints:
            # Calculate effectiveness based on violations caught
            violations_caught = 0
            total_assessments = 0

            for state in self.active_research_states.values():
                for assessment in state.checkpoint_history:
                    if checkpoint.checkpoint_id in assessment.assessment_id:
                        total_assessments += 1
                        if assessment.constraint_violations:
                            violations_caught += 1

            effectiveness[checkpoint.checkpoint_id] = violations_caught / total_assessments if total_assessments > 0 else 0.0

        return effectiveness

    async def _calculate_integration_trends(self) -> Dict[str, Any]:
        """Calculate integration trends over time"""
        # This would analyze historical data to identify trends
        # For now, return placeholder data
        return {
            "ethical_scores_trend": "stable",
            "compliance_rates_trend": "improving",
            "human_oversight_requests_trend": "decreasing",
            "integration_effectiveness_trend": "improving"
        }

    async def add_custom_ethical_checkpoint(self, checkpoint: EthicalCheckpoint):
        """Add a custom ethical checkpoint"""
        try:
            self.ethical_checkpoints.append(checkpoint)
            self.logger.info(f"Added custom ethical checkpoint: {checkpoint.checkpoint_id}")
            return {"success": True, "checkpoint_id": checkpoint.checkpoint_id}

        except Exception as e:
            self.logger.error(f"Failed to add custom checkpoint: {e}")
            return {"success": False, "error": str(e)}

    async def update_integration_config(self, config_updates: Dict[str, Any]):
        """Update integration configuration"""
        try:
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            self.logger.info("Integration configuration updated")

            # Restart background tasks if needed
            if self._monitoring_task and not self.config.enable_real_time_monitoring:
                self._monitoring_task.cancel()
                self._monitoring_task = None

            return {"success": True, "updated_config": config_updates}

        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return {"success": False, "error": str(e)}

    async def shutdown(self):
        """Gracefully shutdown the integration manager"""
        try:
            # Cancel background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._reporting_task:
                self._reporting_task.cancel()

            # Generate final report
            await self._generate_integration_report()

            self.logger.info(f"Ethical Integration Manager {self.manager_id} shutdown successfully")

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")


# Integration wrapper for easy use with Research Orchestrator

class EthicalOrchestratorWrapper:
    """
    Wrapper class that integrates ethical oversight directly with Research Orchestrator
    """

    def __init__(self, orchestrator: ResearchOrchestratorAgent,
                 integration_config: EthicalIntegrationConfig = None):
        self.orchestrator = orchestrator
        self.integration_manager = EthicalIntegrationManager(integration_config)
        self.logger = logging.getLogger(f"{__name__}.EthicalOrchestratorWrapper")

    async def initialize(self):
        """Initialize the ethical orchestrator wrapper"""
        try:
            # Get ethical framework instance
            from .ethical_framework_agent import get_ethical_framework
            ethical_framework = await get_ethical_framework()

            # Initialize integration manager
            await self.integration_manager.initialize(ethical_framework, self.orchestrator)

            self.logger.info("Ethical Orchestrator Wrapper initialized successfully")

        except Exception as e:
            self.logger.error(f"Wrapper initialization failed: {e}")
            raise

    async def coordinate_research_with_ethics(self, request: ResearchRequest) -> ResearchResults:
        """
        Coordinate research with integrated ethical oversight

        Args:
            request: Research request to coordinate ethically

        Returns:
            Research results with ethical compliance information
        """
        try:
            # Step 1: Integrate ethical assessment
            ethical_integration = await self.integration_manager.integrate_research_request(request)

            # Step 2: Check for ethical clearance
            if not ethical_integration["ethical_clearance"]:
                return ResearchResults(
                    request_id=request.request_id,
                    success=False,
                    results={"error": ethical_integration.get("blocked_reason", "Ethical clearance denied")},
                    execution_time=0.0,
                    agent_contributions={},
                    confidence_score=0.0,
                    ethical_compliance=ethical_integration,
                    recommendations=ethical_integration.get("recommendations", [])
                )

            # Step 3: Proceed with normal orchestration
            research_results = await self.orchestrator.coordinate_research(request)

            # Step 4: Enhance results with ethical information
            enhanced_results = await self.integration_manager.enhance_orchestrator_decision(
                research_results.__dict__,
                request.request_id
            )

            # Convert back to ResearchResults with ethical enhancements
            ethical_results = ResearchResults(
                request_id=research_results.request_id,
                success=research_results.success,
                results=research_results.results,
                execution_time=research_results.execution_time,
                agent_contributions=research_results.agent_contributions,
                confidence_score=enhanced_results.get("adjusted_confidence", research_results.confidence_score),
                ethical_compliance=ethical_integration,
                recommendations=research_results.recommendations + ethical_integration.get("recommendations", [])
            )

            return ethical_results

        except Exception as e:
            self.logger.error(f"Ethical research coordination failed: {e}")
            return ResearchResults(
                request_id=request.request_id,
                success=False,
                results={"error": str(e)},
                execution_time=0.0,
                agent_contributions={},
                confidence_score=0.0,
                ethical_compliance={"error": str(e)},
                recommendations=["Research coordination failed due to ethical integration error"]
            )

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including both orchestration and ethical metrics"""
        try:
            orchestrator_status = await self.orchestrator.get_orchestrator_status()
            integration_status = await self.integration_manager.get_integration_status()

            return {
                "orchestrator_status": orchestrator_status,
                "ethical_integration_status": integration_status,
                "system_overall": {
                    "operational": True,
                    "ethical_compliance_enabled": True,
                    "last_integration_check": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Comprehensive status check failed: {e}")
            return {"error": str(e)}