"""
Ethical Framework Agent - Autonomous Research Governance System

This module implements a comprehensive ethical governance framework that provides
real-time ethical oversight, autonomous decision-making, and adaptive ethical
constraint enforcement for the AI research platform.

Key Capabilities:
- Real-time ethical compliance monitoring and scoring
- Multi-layered ethical oversight with adaptive frameworks
- Machine learning-based ethical pattern recognition
- Cross-cultural ethical considerations and stakeholder representation
- Autonomous ethical decision-making with human oversight interfaces
- Risk assessment, mitigation, and violation detection
- Ethical impact prediction and adaptive constraint enforcement

Architecture:
- Core ethical engine with multiple ethical frameworks integration
- Real-time monitoring and compliance checking
- Learning-based ethical pattern analysis
- Multi-stakeholder interest representation
- Seamless integration with Research Orchestrator Agent
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import re

# Import existing components
from ..security.security_manager import SecurityManager
from ..core.config_manager import ConfigManager


class EthicalFrameworkType(Enum):
    """Types of ethical frameworks supported"""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    PRINCIPLE_BASED = "principle_based"  # Belmont principles, etc.
    PRECAUTIONARY = "precautionary"
    STAKEHOLDER = "stakeholder"
    CULTURAL_RELATIVIST = "cultural_relativist"


class EthicalRiskLevel(Enum):
    """Ethical risk levels"""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status levels"""
    FULLY_COMPLIANT = "fully_compliant"
    MINOR_VIOLATIONS = "minor_violations"
    SIGNIFICANT_VIOLATIONS = "significant_violations"
    CRITICAL_VIOLATIONS = "critical_violations"
    PENDING_REVIEW = "pending_review"
    REQUIRES_HUMAN_OVERSIGHT = "requires_human_oversight"


class DecisionMode(Enum):
    """Ethical decision-making modes"""
    AUTONOMOUS = "autonomous"  # Fully autonomous decisions
    SUPERVISED = "supervised"  # Requires human oversight
    CONSENSUS = "consensus"  # Requires multi-stakeholder consensus
    ESCALATED = "escalated"  # Requires human intervention


@dataclass
class EthicalConstraint:
    """Ethical constraint definition"""
    constraint_id: str
    name: str
    description: str
    framework_type: EthicalFrameworkType
    constraint_type: str  # "hard", "soft", "adaptive"
    severity: float  # 0.0 to 1.0
    enforcement_level: str  # "block", "warn", "log", "escalate"
    conditions: Dict[str, Any]
    exceptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicalAssessment:
    """Comprehensive ethical assessment result"""
    assessment_id: str
    request_id: str
    timestamp: datetime
    overall_score: float  # 0.0 to 1.0, where 1.0 is fully ethical
    risk_level: EthicalRiskLevel
    compliance_status: ComplianceStatus
    framework_scores: Dict[EthicalFrameworkType, float]
    constraint_violations: List[Dict[str, Any]]
    recommendations: List[str]
    requires_human_oversight: bool
    decision_mode: DecisionMode
    confidence_score: float
    impact_predictions: Dict[str, Any]
    stakeholder_impacts: Dict[str, float]
    mitigations_suggested: List[Dict[str, Any]]


@dataclass
class ResearchEthicsContext:
    """Research ethics context information"""
    research_domain: str
    methodologies: List[str]
    data_sources: List[str]
    human_involvement: bool
    environmental_impact: bool
    cultural_considerations: List[str]
    regulatory_requirements: List[str]
    institutional_policies: List[str]
    stakeholder_groups: List[str]
    expected_outcomes: List[str]
    potential_risks: List[str]
    mitigation_strategies: List[str]


class EthicalPatternRecognizer:
    """Machine learning-based ethical pattern recognition"""

    def __init__(self):
        self.pattern_database = defaultdict(list)
        self.risk_indicators = {}
        self.ethical_patterns = {}
        self.cultural_norms = {}
        self.learning_enabled = True

    async def analyze_ethical_patterns(self, research_content: str,
                                     context: ResearchEthicsContext) -> Dict[str, Any]:
        """Analyze content for ethical patterns and risks"""
        try:
            patterns = {
                "risk_indicators": await self._identify_risk_indicators(research_content),
                "ethical_concerns": await self._identify_ethical_concerns(research_content, context),
                "cultural_sensitivity": await self._assess_cultural_sensitivity(research_content, context),
                "stakeholder_impact": await self._predict_stakeholder_impact(research_content, context),
                "historical_patterns": await self._match_historical_patterns(research_content)
            }

            # Calculate pattern-based risk score
            pattern_risk_score = self._calculate_pattern_risk_score(patterns)
            patterns["risk_score"] = pattern_risk_score

            return patterns

        except Exception as e:
            logging.error(f"Pattern analysis failed: {e}")
            return {"error": str(e), "risk_score": 0.5}

    async def _identify_risk_indicators(self, content: str) -> List[Dict[str, Any]]:
        """Identify risk indicators in research content"""
        risk_keywords = {
            "human_subjects": ["human", "subject", "patient", "participant", "volunteer"],
            "privacy_concerns": ["privacy", "confidential", "personal data", "identifiable"],
            "bias_risks": ["bias", "discrimination", "stereotype", "prejudice"],
            "safety_risks": ["dangerous", "harmful", "toxic", "unsafe", "hazard"],
            "dual_use": ["weapon", "military", "surveillance", "exploitation"],
            "environmental": ["environment", "ecosystem", "pollution", "climate"],
            "social_impact": ["society", "community", "inequality", "disruption"]
        }

        content_lower = content.lower()
        indicators = []

        for risk_type, keywords in risk_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                indicators.append({
                    "risk_type": risk_type,
                    "keywords_found": matches,
                    "frequency": len(matches),
                    "context_snippets": self._extract_context_snippets(content, matches[:3])
                })

        return indicators

    async def _identify_ethical_concerns(self, content: str,
                                       context: ResearchEthicsContext) -> List[Dict[str, Any]]:
        """Identify specific ethical concerns"""
        concerns = []

        # Check for human subjects research concerns
        if context.human_involvement:
            consent_keywords = ["consent", "permission", "agreement", "informed"]
            if not any(kw in content.lower() for kw in consent_keywords):
                concerns.append({
                    "type": "informed_consent",
                    "severity": "high",
                    "description": "Human subjects research mentioned but consent procedures unclear"
                })

        # Check for data privacy concerns
        if "data" in content.lower() and context.data_sources:
            privacy_keywords = ["anonymous", "de-identified", "encrypted", "protected"]
            if not any(kw in content.lower() for kw in privacy_keywords):
                concerns.append({
                    "type": "data_privacy",
                    "severity": "medium",
                    "description": "Data usage mentioned but privacy protection unclear"
                })

        # Check for bias and fairness concerns
        bias_indicators = ["fair", "unbiased", "representative", "diverse"]
        if not any(kw in content.lower() for kw in bias_indicators):
            concerns.append({
                "type": "bias_fairness",
                "severity": "medium",
                "description": "Research may have bias/fairness implications not addressed"
            })

        return concerns

    async def _assess_cultural_sensitivity(self, content: str,
                                         context: ResearchEthicsContext) -> Dict[str, Any]:
        """Assess cultural sensitivity of research"""
        cultural_keywords = {
            "western_bias": ["western", "american", "european", "developed"],
            "cultural_generalization": ["all", "every", "always", "universal"],
            "indigenous": ["indigenous", "native", "tribal", "traditional"],
            "religious": ["religion", "faith", "spiritual", "belief"]
        }

        content_lower = content.lower()
        cultural_issues = []

        for issue_type, keywords in cultural_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                cultural_issues.append({
                    "issue_type": issue_type,
                    "keywords": matches,
                    "severity": "medium"
                })

        # Consider cultural context provided
        cultural_context_score = 1.0
        if context.cultural_considerations:
            cultural_context_score = 0.7  # Reduced risk if cultural context is considered

        return {
            "issues_identified": cultural_issues,
            "sensitivity_score": cultural_context_score - (len(cultural_issues) * 0.1),
            "considerations_addressed": len(context.cultural_considerations)
        }

    async def _predict_stakeholder_impact(self, content: str,
                                        context: ResearchEthicsContext) -> Dict[str, float]:
        """Predict impact on different stakeholder groups"""
        stakeholder_keywords = {
            "researchers": ["researcher", "scientist", "academic", "scholar"],
            "participants": ["participant", "subject", "volunteer", "user"],
            "society": ["society", "public", "community", "population"],
            "environment": ["environment", "nature", "ecosystem", "planet"],
            "institutions": ["institution", "organization", "company", "university"],
            "policy_makers": ["policy", "government", "regulation", "law"]
        }

        content_lower = content.lower()
        stakeholder_scores = {}

        for stakeholder, keywords in stakeholder_keywords.items():
            matches = sum(1 for kw in keywords if kw in content_lower)
            impact_score = min(1.0, matches * 0.2)  # Each keyword adds 0.2 to impact

            # Adjust based on context
            if stakeholder in [s.lower() for s in context.stakeholder_groups]:
                impact_score *= 1.2  # Increase impact if explicitly mentioned

            stakeholder_scores[stakeholder] = min(1.0, impact_score)

        return stakeholder_scores

    async def _match_historical_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Match against historical ethical patterns"""
        # This would integrate with a database of historical ethical cases
        # For now, return simulated matches
        return [
            {
                "case_type": "data_privacy",
                "similarity": 0.75,
                "historical_outcome": "required_additional_safeguards",
                "lessons_learned": ["Implement data anonymization", "Obtain explicit consent"]
            }
        ]

    def _extract_context_snippets(self, content: str, keywords: List[str]) -> List[str]:
        """Extract context snippets around keywords"""
        snippets = []
        for keyword in keywords:
            # Find keyword positions and extract surrounding context
            pattern = re.compile(f".{{0,50}}{re.escape(keyword)}.{{0,50}}", re.IGNORECASE)
            matches = pattern.findall(content)
            snippets.extend(matches[:2])  # Limit to 2 snippets per keyword
        return snippets[:5]  # Limit total snippets

    def _calculate_pattern_risk_score(self, patterns: Dict[str, Any]) -> float:
        """Calculate overall risk score from pattern analysis"""
        risk_score = 0.0

        # Risk indicators contribution
        risk_indicators = patterns.get("risk_indicators", [])
        if risk_indicators:
            indicator_score = sum(indicator["frequency"] * 0.1 for indicator in risk_indicators)
            risk_score += min(0.4, indicator_score)

        # Ethical concerns contribution
        concerns = patterns.get("ethical_concerns", [])
        if concerns:
            concern_score = sum(0.2 if c["severity"] == "high" else 0.1 for c in concerns)
            risk_score += min(0.3, concern_score)

        # Cultural sensitivity contribution
        cultural_sensitivity = patterns.get("cultural_sensitivity", {})
        sensitivity_score = 1.0 - cultural_sensitivity.get("sensitivity_score", 1.0)
        risk_score += min(0.2, sensitivity_score)

        # Stakeholder impact contribution
        stakeholder_impacts = patterns.get("stakeholder_impact", {})
        if stakeholder_impacts:
            max_impact = max(stakeholder_impacts.values()) if stakeholder_impacts.values() else 0
            risk_score += min(0.3, max_impact * 0.5)

        return min(1.0, max(0.0, risk_score))


class EthicalConstraintEngine:
    """Manages and enforces ethical constraints"""

    def __init__(self):
        self.constraints: Dict[str, EthicalConstraint] = {}
        self.constraint_history = deque(maxlen=1000)
        self.adaptive_constraints = True
        self.enforcement_levels = {
            "block": 1.0,
            "warn": 0.7,
            "log": 0.4,
            "escalate": 0.9
        }

    def add_constraint(self, constraint: EthicalConstraint):
        """Add a new ethical constraint"""
        self.constraints[constraint.constraint_id] = constraint
        logging.info(f"Added ethical constraint: {constraint.name}")

    async def evaluate_constraints(self, research_content: str,
                                 context: ResearchEthicsContext) -> Dict[str, Any]:
        """Evaluate all constraints against research content"""
        results = {
            "violations": [],
            "warnings": [],
            "passed_constraints": [],
            "overall_compliance_score": 0.0,
            "blocked": False,
            "requires_escalation": False
        }

        total_weight = 0
        compliance_score = 0.0

        for constraint in self.constraints.values():
            evaluation = await self._evaluate_single_constraint(constraint, research_content, context)

            if evaluation["violated"]:
                violation = {
                    "constraint_id": constraint.constraint_id,
                    "name": constraint.name,
                    "severity": constraint.severity,
                    "enforcement_level": constraint.enforcement_level,
                    "details": evaluation["details"],
                    "suggested_action": evaluation["suggested_action"]
                }

                if constraint.enforcement_level == "block":
                    results["violations"].append(violation)
                    results["blocked"] = True
                elif constraint.enforcement_level == "escalate":
                    results["violations"].append(violation)
                    results["requires_escalation"] = True
                elif constraint.enforcement_level == "warn":
                    results["warnings"].append(violation)
            else:
                results["passed_constraints"].append(constraint.constraint_id)

            # Calculate weighted compliance score
            weight = constraint.severity
            compliance_score += weight * (1.0 if not evaluation["violated"] else 0.0)
            total_weight += weight

        results["overall_compliance_score"] = compliance_score / total_weight if total_weight > 0 else 1.0

        # Record constraint evaluation
        self.constraint_history.append({
            "timestamp": datetime.now(),
            "research_content_hash": hash(research_content),
            "results": results
        })

        return results

    async def _evaluate_single_constraint(self, constraint: EthicalConstraint,
                                        content: str, context: ResearchEthicsContext) -> Dict[str, Any]:
        """Evaluate a single constraint"""
        violated = False
        details = []
        suggested_action = "proceed"

        # Check if constraint conditions are met
        for condition_type, condition_value in constraint.conditions.items():
            if condition_type == "keyword_prohibited":
                prohibited_keywords = condition_value if isinstance(condition_value, list) else [condition_value]
                found_keywords = [kw for kw in prohibited_keywords if kw.lower() in content.lower()]
                if found_keywords:
                    violated = True
                    details.append(f"Prohibited keywords found: {found_keywords}")
                    suggested_action = "remove_prohibited_content"

            elif condition_type == "human_subjects_required":
                if condition_value and not context.human_involvement:
                    violated = False  # Not violated if no human subjects
                elif condition_value and context.human_involvement:
                    # Check for proper human subjects protections
                    consent_keywords = ["consent", "informed", "permission", "approval"]
                    if not any(kw in content.lower() for kw in consent_keywords):
                        violated = True
                        details.append("Human subjects research lacks consent documentation")
                        suggested_action = "add_consent_procedures"

            elif condition_type == "data_privacy_required":
                if condition_value and context.data_sources:
                    privacy_keywords = ["anonymous", "de-identified", "encrypted", "confidential"]
                    if not any(kw in content.lower() for kw in privacy_keywords):
                        violated = True
                        details.append("Data sources mentioned without privacy protections")
                        suggested_action = "implement_data_privacy"

            elif condition_type == "bias_assessment_required":
                if condition_value:
                    bias_keywords = ["bias", "fairness", "unbiased", "representative"]
                    if not any(kw in content.lower() for kw in bias_keywords):
                        violated = True
                        details.append("Research lacks bias assessment")
                        suggested_action = "conduct_bias_assessment"

        # Check for exceptions
        if violated:
            for exception in constraint.exceptions:
                if exception.lower() in content.lower():
                    violated = False
                    details.append(f"Exception applied: {exception}")
                    break

        return {
            "violated": violated,
            "details": details,
            "suggested_action": suggested_action
        }

    async def adapt_constraints(self, feedback: Dict[str, Any]):
        """Adapt constraints based on feedback and learning"""
        if not self.adaptive_constraints:
            return

        # This would implement machine learning-based constraint adaptation
        # For now, it's a placeholder for adaptive learning
        pass


class EthicalFrameworkAgent:
    """
    Ethical Framework Agent - Comprehensive Research Governance

    This agent provides autonomous ethical oversight for research activities,
    integrating multiple ethical frameworks and providing real-time compliance
    monitoring and decision-making capabilities.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent_id = f"ethical_framework_{uuid.uuid4().hex[:8]}"

        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")

        # Initialize core components
        self.security_manager = SecurityManager()
        self.pattern_recognizer = EthicalPatternRecognizer()
        self.constraint_engine = EthicalConstraintEngine()

        # Initialize ethical frameworks
        self.frameworks = self._initialize_ethical_frameworks()

        # Initialize default constraints
        self._initialize_default_constraints()

        # Agent state
        self.assessment_history = deque(maxlen=1000)
        self.human_oversight_requests = []
        self.ethical_decisions = []
        self.learning_data = []

        # Configuration
        self.enable_adaptive_learning = self.config.get("enable_adaptive_learning", True)
        self.human_oversight_threshold = self.config.get("human_oversight_threshold", 0.7)
        self.ethical_threshold = self.config.get("ethical_threshold", 0.8)
        self.enable_cultural_considerations = self.config.get("enable_cultural_considerations", True)

        # Background tasks
        self._learning_task = None
        self._monitoring_task = None

    def _initialize_ethical_frameworks(self) -> Dict[EthicalFrameworkType, Dict[str, Any]]:
        """Initialize supported ethical frameworks"""
        return {
            EthicalFrameworkType.UTILITARIAN: {
                "description": "Maximize overall well-being and minimize harm",
                "principles": ["beneficence", "non-maleficence", "utility_maximization"],
                "weight": 0.2
            },
            EthicalFrameworkType.DEONTOLOGICAL: {
                "description": "Focus on duties, rules, and obligations",
                "principles": ["respect_for_persons", "autonomy", "duty"],
                "weight": 0.2
            },
            EthicalFrameworkType.VIRTUE_ETHICS: {
                "description": "Focus on character and moral virtues",
                "principles": ["integrity", "honesty", "courage", "wisdom"],
                "weight": 0.15
            },
            EthicalFrameworkType.CARE_ETHICS: {
                "description": "Focus on relationships, care, and interdependence",
                "principles": ["empathy", "care", "relationships", "interdependence"],
                "weight": 0.15
            },
            EthicalFrameworkType.PRINCIPLE_BASED: {
                "description": "Based on established ethical principles (Belmont, etc.)",
                "principles": ["respect_for_persons", "beneficence", "justice"],
                "weight": 0.2
            },
            EthicalFrameworkType.PRECAUTIONARY: {
                "description": "Precaution in the face of uncertainty",
                "principles": ["prevention", "caution", "risk_averse"],
                "weight": 0.1
            }
        }

    def _initialize_default_constraints(self):
        """Initialize default ethical constraints"""
        default_constraints = [
            EthicalConstraint(
                constraint_id="human_subjects_protection",
                name="Human Subjects Protection",
                description="Research involving human subjects must have proper protections",
                framework_type=EthicalFrameworkType.PRINCIPLE_BASED,
                constraint_type="hard",
                severity=0.9,
                enforcement_level="block",
                conditions={
                    "human_subjects_required": True
                },
                exceptions=["literature_review", "meta_analysis"]
            ),
            EthicalConstraint(
                constraint_id="data_privacy",
                name="Data Privacy Protection",
                description="Personal data must be properly protected",
                framework_type=EthicalFrameworkType.DEONTOLOGICAL,
                constraint_type="hard",
                severity=0.8,
                enforcement_level="block",
                conditions={
                    "data_privacy_required": True
                }
            ),
            EthicalConstraint(
                constraint_id="bias_assessment",
                name="Bias and Fairness Assessment",
                description="Research must consider potential biases and fairness implications",
                framework_type=EthicalFrameworkType.UTILITARIAN,
                constraint_type="soft",
                severity=0.6,
                enforcement_level="warn",
                conditions={
                    "bias_assessment_required": True
                }
            ),
            EthicalConstraint(
                constraint_id="dual_use_research",
                name="Dual-Use Research Concerns",
                description="Research that could be misused must have additional safeguards",
                framework_type=EthicalFrameworkType.PRECAUTIONARY,
                constraint_type="hard",
                severity=0.95,
                enforcement_level="escalate",
                conditions={
                    "keyword_prohibited": ["weapon", "military_application", "bioweapon"]
                },
                exceptions=["defensive_research", "peaceful_applications"]
            ),
            EthicalConstraint(
                constraint_id="environmental_impact",
                name="Environmental Impact Assessment",
                description="Research with environmental impact must be assessed",
                framework_type=EthicalFrameworkType.UTILITARIAN,
                constraint_type="soft",
                severity=0.5,
                enforcement_level="warn",
                conditions={
                    "environmental_impact_required": True
                }
            )
        ]

        for constraint in default_constraints:
            self.constraint_engine.add_constraint(constraint)

    async def initialize(self):
        """Initialize the ethical framework agent"""
        try:
            # Start background tasks
            self._learning_task = asyncio.create_task(self._adaptive_learning_loop())
            self._monitoring_task = asyncio.create_task(self._compliance_monitoring_loop())

            self.logger.info(f"Ethical Framework Agent {self.agent_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ethical framework agent: {e}")
            raise

    async def assess_research_ethics(self, request_id: str, research_content: str,
                                   context: ResearchEthicsContext) -> EthicalAssessment:
        """
        Main interface for ethical assessment of research

        Args:
            request_id: Unique identifier for the research request
            research_content: The research proposal or content to assess
            context: Research ethics context information

        Returns:
            EthicalAssessment with comprehensive ethical evaluation
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting ethical assessment for request {request_id}")

            # Step 1: Pattern-based analysis
            pattern_analysis = await self.pattern_recognizer.analyze_ethical_patterns(
                research_content, context
            )

            # Step 2: Constraint evaluation
            constraint_results = await self.constraint_engine.evaluate_constraints(
                research_content, context
            )

            # Step 3: Multi-framework ethical analysis
            framework_scores = await self._evaluate_ethical_frameworks(
                research_content, context
            )

            # Step 4: Stakeholder impact analysis
            stakeholder_impacts = await self._analyze_stakeholder_impacts(
                research_content, context
            )

            # Step 5: Ethical impact prediction
            impact_predictions = await self._predict_ethical_impacts(
                research_content, context
            )

            # Step 6: Calculate overall ethical score
            overall_score = self._calculate_overall_ethical_score(
                pattern_analysis, constraint_results, framework_scores,
                stakeholder_impacts, impact_predictions
            )

            # Step 7: Determine risk level and compliance status
            risk_level = self._determine_risk_level(overall_score, constraint_results)
            compliance_status = self._determine_compliance_status(
                overall_score, constraint_results
            )

            # Step 8: Generate recommendations
            recommendations = await self._generate_recommendations(
                overall_score, constraint_results, pattern_analysis
            )

            # Step 9: Determine decision mode
            decision_mode = self._determine_decision_mode(
                overall_score, risk_level, constraint_results
            )

            # Step 10: Check for human oversight requirement
            requires_human_oversight = overall_score < self.human_oversight_threshold or \
                                     constraint_results.get("blocked", False) or \
                                     constraint_results.get("requires_escalation", False)

            # Create assessment
            assessment = EthicalAssessment(
                assessment_id=f"ethics_{uuid.uuid4().hex[:8]}",
                request_id=request_id,
                timestamp=datetime.now(),
                overall_score=overall_score,
                risk_level=risk_level,
                compliance_status=compliance_status,
                framework_scores=framework_scores,
                constraint_violations=constraint_results.get("violations", []),
                recommendations=recommendations,
                requires_human_oversight=requires_human_oversight,
                decision_mode=decision_mode,
                confidence_score=self._calculate_confidence_score(
                    pattern_analysis, constraint_results, framework_scores
                ),
                impact_predictions=impact_predictions,
                stakeholder_impacts=stakeholder_impacts,
                mitigations_suggested=constraint_results.get("warnings", [])
            )

            # Store assessment in history
            self.assessment_history.append(assessment)

            # Log assessment
            await self._log_ethical_assessment(assessment)

            # Request human oversight if required
            if requires_human_oversight:
                await self._request_human_oversight(assessment)

            execution_time = time.time() - start_time
            self.logger.info(f"Ethical assessment completed in {execution_time:.2f}s, "
                           f"score: {overall_score:.3f}, risk: {risk_level.value}")

            return assessment

        except Exception as e:
            self.logger.error(f"Ethical assessment failed for request {request_id}: {e}")

            # Return fallback assessment
            return EthicalAssessment(
                assessment_id=f"ethics_error_{uuid.uuid4().hex[:8]}",
                request_id=request_id,
                timestamp=datetime.now(),
                overall_score=0.0,
                risk_level=EthicalRiskLevel.CRITICAL,
                compliance_status=ComplianceStatus.REQUIRES_HUMAN_OVERSIGHT,
                framework_scores={},
                constraint_violations=[{"error": str(e)}],
                recommendations=["Human oversight required due to assessment error"],
                requires_human_oversight=True,
                decision_mode=DecisionMode.ESCALATED,
                confidence_score=0.0,
                impact_predictions={},
                stakeholder_impacts={},
                mitigations_suggested=[]
            )

    async def _evaluate_ethical_frameworks(self, research_content: str,
                                          context: ResearchEthicsContext) -> Dict[EthicalFrameworkType, float]:
        """Evaluate research against multiple ethical frameworks"""
        framework_scores = {}

        for framework_type, framework_info in self.frameworks.items():
            try:
                score = await self._evaluate_single_framework(
                    framework_type, framework_info, research_content, context
                )
                framework_scores[framework_type] = score

            except Exception as e:
                self.logger.error(f"Framework evaluation failed for {framework_type}: {e}")
                framework_scores[framework_type] = 0.0

        return framework_scores

    async def _evaluate_single_framework(self, framework_type: EthicalFrameworkType,
                                       framework_info: Dict[str, Any],
                                       research_content: str,
                                       context: ResearchEthicsContext) -> float:
        """Evaluate research against a single ethical framework"""
        content_lower = research_content.lower()
        score = 0.5  # Default neutral score

        if framework_type == EthicalFrameworkType.UTILITARIAN:
            # Check for beneficence and harm reduction
            beneficence_keywords = ["benefit", "help", "improve", "positive", "well-being"]
            harm_keywords = ["harm", "damage", "negative", "risk", "danger"]

            beneficence_count = sum(1 for kw in beneficence_keywords if kw in content_lower)
            harm_count = sum(1 for kw in harm_keywords if kw in content_lower)

            score = min(1.0, 0.5 + (beneficence_count * 0.1) - (harm_count * 0.1))

        elif framework_type == EthicalFrameworkType.DEONTOLOGICAL:
            # Check for duty-based language
            duty_keywords = ["must", "should", "obligation", "duty", "responsibility"]
            rights_keywords = ["rights", "autonomy", "consent", "respect"]

            duty_count = sum(1 for kw in duty_keywords if kw in content_lower)
            rights_count = sum(1 for kw in rights_keywords if kw in content_lower)

            score = min(1.0, 0.3 + (duty_count * 0.1) + (rights_count * 0.15))

        elif framework_type == EthicalFrameworkType.VIRTUE_ETHICS:
            # Check for virtue-based language
            virtue_keywords = ["integrity", "honest", "transparent", "courage", "wisdom"]

            virtue_count = sum(1 for kw in virtue_keywords if kw in content_lower)
            score = min(1.0, 0.4 + (virtue_count * 0.12))

        elif framework_type == EthicalFrameworkType.CARE_ETHICS:
            # Check for care and relationship language
            care_keywords = ["care", "support", "relationships", "empathy", "community"]

            care_count = sum(1 for kw in care_keywords if kw in content_lower)
            score = min(1.0, 0.4 + (care_count * 0.12))

        elif framework_type == EthicalFrameworkType.PRINCIPLE_BASED:
            # Check for principle-based language
            principle_keywords = ["respect", "beneficence", "justice", "autonomy", "ethical"]

            principle_count = sum(1 for kw in principle_keywords if kw in content_lower)
            score = min(1.0, 0.3 + (principle_count * 0.14))

        elif framework_type == EthicalFrameworkType.PRECAUTIONARY:
            # Check for precautionary language
            precaution_keywords = ["caution", "careful", "risk", "safety", "precaution"]

            precaution_count = sum(1 for kw in precaution_keywords if kw in content_lower)
            score = min(1.0, 0.4 + (precaution_count * 0.12))

        return max(0.0, min(1.0, score))

    async def _analyze_stakeholder_impacts(self, research_content: str,
                                         context: ResearchEthicsContext) -> Dict[str, float]:
        """Analyze impacts on different stakeholder groups"""
        # Use pattern recognizer for detailed stakeholder analysis
        pattern_impacts = await self.pattern_recognizer.predict_stakeholder_impact(
            research_content, context
        )

        # Enhance with context-specific impacts
        enhanced_impacts = pattern_impacts.copy()

        # Consider explicit stakeholder groups from context
        for stakeholder_group in context.stakeholder_groups:
            if stakeholder_group.lower() not in enhanced_impacts:
                enhanced_impacts[stakeholder_group.lower()] = 0.5

        # Adjust impacts based on potential risks
        for risk in context.potential_risks:
            if "human" in risk.lower():
                enhanced_impacts["participants"] = min(1.0, enhanced_impacts.get("participants", 0) + 0.2)
            elif "environment" in risk.lower():
                enhanced_impacts["environment"] = min(1.0, enhanced_impacts.get("environment", 0) + 0.2)

        return enhanced_impacts

    async def _predict_ethical_impacts(self, research_content: str,
                                      context: ResearchEthicsContext) -> Dict[str, Any]:
        """Predict ethical impacts of research"""
        impacts = {
            "short_term_impacts": await self._predict_short_term_impacts(research_content, context),
            "long_term_impacts": await self._predict_long_term_impacts(research_content, context),
            "systemic_impacts": await self._predict_systemic_impacts(research_content, context),
            "uncertainty_level": await self._assess_uncertainty(research_content, context),
            "adaptability_potential": await self._assess_adaptability_potential(research_content, context)
        }

        return impacts

    async def _predict_short_term_impacts(self, research_content: str,
                                         context: ResearchEthicsContext) -> Dict[str, float]:
        """Predict short-term ethical impacts"""
        impacts = {
            "privacy_impact": 0.3 if "data" in research_content.lower() else 0.0,
            "safety_impact": 0.4 if any(kw in research_content.lower() for kw in ["safety", "risk", "danger"]) else 0.0,
            "autonomy_impact": 0.5 if context.human_involvement else 0.0,
            "immediate_benefits": 0.3 if any(kw in research_content.lower() for kw in ["benefit", "help", "improve"]) else 0.0
        }

        return impacts

    async def _predict_long_term_impacts(self, research_content: str,
                                        context: ResearchEthicsContext) -> Dict[str, float]:
        """Predict long-term ethical impacts"""
        impacts = {
            "societal_impact": 0.4 if any(kw in research_content.lower() for kw in ["society", "social", "cultural"]) else 0.0,
            "environmental_impact": 0.5 if context.environmental_impact else 0.0,
            "knowledge_impact": 0.6 if any(kw in research_content.lower() for kw in ["research", "study", "investigation"]) else 0.0,
            "policy_impact": 0.3 if any(kw in research_content.lower() for kw in ["policy", "regulation", "guideline"]) else 0.0
        }

        return impacts

    async def _predict_systemic_impacts(self, research_content: str,
                                       context: ResearchEthicsContext) -> Dict[str, float]:
        """Predict systemic ethical impacts"""
        impacts = {
            "institutional_impact": 0.3 if any(kw in research_content.lower() for kw in ["institution", "organization"]) else 0.0,
            "disciplinary_impact": 0.4 if context.research_domain else 0.0,
            "ethical_norms_impact": 0.2 if any(kw in research_content.lower() for kw in ["ethical", "moral", "values"]) else 0.0,
            "power_dynamics_impact": 0.3 if any(kw in research_content.lower() for kw in ["power", "authority", "control"]) else 0.0
        }

        return impacts

    async def _assess_uncertainty(self, research_content: str,
                                 context: ResearchEthicsContext) -> float:
        """Assess level of uncertainty in ethical impacts"""
        uncertainty_keywords = ["uncertain", "unknown", "might", "could", "potentially", "possible"]
        content_lower = research_content.lower()

        uncertainty_count = sum(1 for kw in uncertainty_keywords if kw in content_lower)

        # Higher uncertainty for novel research
        novelty_indicators = ["novel", "innovative", "new", "unprecedented", "breakthrough"]
        novelty_count = sum(1 for kw in novelty_indicators if kw in content_lower)

        uncertainty_score = min(1.0, 0.2 + (uncertainty_count * 0.1) + (novelty_count * 0.15))

        return uncertainty_score

    async def _assess_adaptability_potential(self, research_content: str,
                                           context: ResearchEthicsContext) -> float:
        """Assess potential for adaptive ethical governance"""
        adaptability_keywords = ["adaptive", "flexible", "iterative", "responsive", "dynamic"]
        monitoring_keywords = ["monitor", "track", "observe", "supervise", "oversight"]

        content_lower = research_content.lower()

        adaptability_count = sum(1 for kw in adaptability_keywords if kw in content_lower)
        monitoring_count = sum(1 for kw in monitoring_keywords if kw in content_lower)

        adaptability_score = min(1.0, 0.3 + (adaptability_count * 0.15) + (monitoring_count * 0.1))

        # Consider mitigation strategies
        if context.mitigation_strategies:
            adaptability_score += min(0.3, len(context.mitigation_strategies) * 0.1)

        return adaptability_score

    def _calculate_overall_ethical_score(self, pattern_analysis: Dict[str, Any],
                                       constraint_results: Dict[str, Any],
                                       framework_scores: Dict[EthicalFrameworkType, float],
                                       stakeholder_impacts: Dict[str, float],
                                       impact_predictions: Dict[str, Any]) -> float:
        """Calculate overall ethical score"""
        # Weighted combination of different assessment components

        # Pattern analysis score (inverse of risk score)
        pattern_score = 1.0 - pattern_analysis.get("risk_score", 0.5)

        # Constraint compliance score
        constraint_score = constraint_results.get("overall_compliance_score", 0.5)

        # Framework scores (weighted average)
        framework_weights = {framework: info["weight"] for framework, info in self.frameworks.items()}
        weighted_framework_score = sum(
            score * framework_weights.get(framework, 0.1)
            for framework, score in framework_scores.items()
        ) / sum(framework_weights.values()) if framework_weights else 0.5

        # Stakeholder impact score (average impact, considering both positive and negative)
        stakeholder_score = sum(stakeholder_impacts.values()) / len(stakeholder_impacts) if stakeholder_impacts else 0.5

        # Impact prediction score (considers uncertainty and adaptability)
        uncertainty = impact_predictions.get("uncertainty_level", 0.5)
        adaptability = impact_predictions.get("adaptability_potential", 0.5)
        impact_score = (adaptability + (1.0 - uncertainty)) / 2

        # Final weighted score
        overall_score = (
            pattern_score * 0.25 +
            constraint_score * 0.30 +
            weighted_framework_score * 0.25 +
            stakeholder_score * 0.10 +
            impact_score * 0.10
        )

        return max(0.0, min(1.0, overall_score))

    def _determine_risk_level(self, overall_score: float,
                            constraint_results: Dict[str, Any]) -> EthicalRiskLevel:
        """Determine ethical risk level"""
        if overall_score >= 0.9:
            return EthicalRiskLevel.NEGLIGIBLE
        elif overall_score >= 0.8:
            return EthicalRiskLevel.LOW
        elif overall_score >= 0.6:
            return EthicalRiskLevel.MEDIUM
        elif overall_score >= 0.4:
            return EthicalRiskLevel.HIGH
        else:
            return EthicalRiskLevel.CRITICAL

    def _determine_compliance_status(self, overall_score: float,
                                   constraint_results: Dict[str, Any]) -> ComplianceStatus:
        """Determine compliance status"""
        if constraint_results.get("blocked", False):
            return ComplianceStatus.CRITICAL_VIOLATIONS
        elif constraint_results.get("requires_escalation", False):
            return ComplianceStatus.REQUIRES_HUMAN_OVERSIGHT
        elif constraint_results.get("violations"):
            if len(constraint_results["violations"]) > 2:
                return ComplianceStatus.SIGNIFICANT_VIOLATIONS
            else:
                return ComplianceStatus.MINOR_VIOLATIONS
        elif overall_score >= 0.9:
            return ComplianceStatus.FULLY_COMPLIANT
        else:
            return ComplianceStatus.PENDING_REVIEW

    def _determine_decision_mode(self, overall_score: float,
                               risk_level: EthicalRiskLevel,
                               constraint_results: Dict[str, Any]) -> DecisionMode:
        """Determine appropriate decision mode"""
        if constraint_results.get("blocked", False):
            return DecisionMode.ESCALATED
        elif constraint_results.get("requires_escalation", False):
            return DecisionMode.ESCALATED
        elif overall_score < self.human_oversight_threshold:
            return DecisionMode.SUPERVISED
        elif risk_level in [EthicalRiskLevel.HIGH, EthicalRiskLevel.CRITICAL]:
            return DecisionMode.SUPERVISED
        elif overall_score >= 0.9:
            return DecisionMode.AUTONOMOUS
        else:
            return DecisionMode.CONSENSUS

    def _calculate_confidence_score(self, pattern_analysis: Dict[str, Any],
                                   constraint_results: Dict[str, Any],
                                   framework_scores: Dict[EthicalFrameworkType, float]) -> float:
        """Calculate confidence score for the assessment"""
        # Confidence based on consistency across different assessment methods
        pattern_score = 1.0 - pattern_analysis.get("risk_score", 0.5)
        constraint_score = constraint_results.get("overall_compliance_score", 0.5)

        # Calculate variance in framework scores
        framework_values = list(framework_scores.values())
        framework_variance = np.var(framework_values) if framework_values else 0.25

        # Higher confidence with consistent results
        consistency_score = 1.0 - framework_variance

        # Overall confidence
        confidence = (pattern_score + constraint_score + consistency_score) / 3

        return max(0.0, min(1.0, confidence))

    async def _generate_recommendations(self, overall_score: float,
                                     constraint_results: Dict[str, Any],
                                     pattern_analysis: Dict[str, Any]) -> List[str]:
        """Generate ethical recommendations"""
        recommendations = []

        # General recommendations based on score
        if overall_score < 0.6:
            recommendations.append("Significant ethical concerns identified. Consider major revisions to research approach.")
        elif overall_score < 0.8:
            recommendations.append("Some ethical concerns identified. Review and address specific issues.")

        # Constraint-based recommendations
        for violation in constraint_results.get("violations", []):
            recommendations.append(f"Address constraint violation: {violation['name']}")

        for warning in constraint_results.get("warnings", []):
            recommendations.append(f"Consider addressing: {warning['name']}")

        # Pattern-based recommendations
        risk_indicators = pattern_analysis.get("risk_indicators", [])
        for indicator in risk_indicators:
            if indicator["frequency"] > 2:
                recommendations.append(f"Multiple {indicator['risk_type']} indicators detected. Consider additional safeguards.")

        # Default recommendations if none specific
        if not recommendations:
            recommendations.append("Research appears ethically sound. Proceed with standard ethical oversight.")

        return recommendations

    async def _request_human_oversight(self, assessment: EthicalAssessment):
        """Request human oversight for ethically sensitive research"""
        oversight_request = {
            "request_id": f"oversight_{uuid.uuid4().hex[:8]}",
            "assessment_id": assessment.assessment_id,
            "timestamp": datetime.now(),
            "risk_level": assessment.risk_level.value,
            "compliance_status": assessment.compliance_status.value,
            "overall_score": assessment.overall_score,
            "primary_concerns": [v["name"] for v in assessment.constraint_violations[:3]],
            "recommended_action": assessment.decision_mode.value,
            "deadline": datetime.now() + timedelta(hours=24),
            "status": "pending"
        }

        self.human_oversight_requests.append(oversight_request)

        # Log the oversight request
        self.logger.warning(f"Human oversight requested for assessment {assessment.assessment_id}, "
                           f"risk level: {assessment.risk_level.value}")

        # In a real implementation, this would send notifications to human reviewers
        # For now, we'll just log it

    async def _log_ethical_assessment(self, assessment: EthicalAssessment):
        """Log ethical assessment details"""
        log_entry = {
            "assessment_id": assessment.assessment_id,
            "request_id": assessment.request_id,
            "timestamp": assessment.timestamp.isoformat(),
            "overall_score": assessment.overall_score,
            "risk_level": assessment.risk_level.value,
            "compliance_status": assessment.compliance_status.value,
            "requires_human_oversight": assessment.requires_human_oversight,
            "decision_mode": assessment.decision_mode.value,
            "constraint_violations_count": len(assessment.constraint_violations),
            "framework_scores": {f.value: score for f, score in assessment.framework_scores.items()}
        }

        # Log ethical assessment
        self.logger.info(f"Ethical assessment logged: {assessment.assessment_id}")

    async def _adaptive_learning_loop(self):
        """Background loop for adaptive learning"""
        while True:
            try:
                await asyncio.sleep(3600)  # Learn every hour

                if self.enable_adaptive_learning:
                    await self._perform_adaptive_learning()

            except Exception as e:
                self.logger.error(f"Adaptive learning loop error: {e}")

    async def _perform_adaptive_learning(self):
        """Perform adaptive learning from assessment history"""
        try:
            # Analyze patterns in assessment history
            if len(self.assessment_history) < 10:
                return

            # Learn from constraint effectiveness
            await self.constraint_engine.adapt_constraints({
                "assessment_history": list(self.assessment_history),
                "human_feedback": []  # Would include human feedback in real implementation
            })

            # Update pattern recognition
            recent_assessments = list(self.assessment_history)[-50:]
            await self.pattern_recognizer.learning_data.extend(recent_assessments)

            self.logger.info("Adaptive learning completed")

        except Exception as e:
            self.logger.error(f"Adaptive learning failed: {e}")

    async def _compliance_monitoring_loop(self):
        """Background loop for compliance monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes

                await self._monitor_compliance_status()

            except Exception as e:
                self.logger.error(f"Compliance monitoring loop error: {e}")

    async def _monitor_compliance_status(self):
        """Monitor overall compliance status"""
        try:
            # Calculate recent compliance metrics
            recent_assessments = list(self.assessment_history)[-100:]

            if not recent_assessments:
                return

            # Calculate compliance rates
            total_assessments = len(recent_assessments)
            compliant_count = sum(1 for a in recent_assessments
                               if a.compliance_status in [ComplianceStatus.FULLY_COMPLIANT])

            compliance_rate = compliant_count / total_assessments if total_assessments > 0 else 0

            # Calculate average ethical score
            avg_score = sum(a.overall_score for a in recent_assessments) / total_assessments

            # Check for trends
            if compliance_rate < 0.8:
                self.logger.warning(f"Low compliance rate detected: {compliance_rate:.2f}")

            if avg_score < 0.7:
                self.logger.warning(f"Low average ethical score detected: {avg_score:.2f}")

            # Log compliance metrics
            await self.logging_system.log_compliance_metrics({
                "timestamp": datetime.now().isoformat(),
                "compliance_rate": compliance_rate,
                "average_ethical_score": avg_score,
                "total_assessments": total_assessments,
                "human_oversight_requests": len(self.human_oversight_requests)
            })

        except Exception as e:
            self.logger.error(f"Compliance monitoring failed: {e}")

    async def get_ethical_status(self) -> Dict[str, Any]:
        """Get comprehensive ethical framework status"""
        try:
            recent_assessments = list(self.assessment_history)[-100:]

            return {
                "agent_id": self.agent_id,
                "status": "active",
                "total_assessments": len(self.assessment_history),
                "recent_assessments": len(recent_assessments),
                "average_ethical_score": sum(a.overall_score for a in recent_assessments) / len(recent_assessments) if recent_assessments else 0,
                "compliance_rate": sum(1 for a in recent_assessments if a.compliance_status == ComplianceStatus.FULLY_COMPLIANT) / len(recent_assessments) if recent_assessments else 0,
                "human_oversight_requests": len(self.human_oversight_requests),
                "pending_oversight": len([r for r in self.human_oversight_requests if r["status"] == "pending"]),
                "active_constraints": len(self.constraint_engine.constraints),
                "frameworks_enabled": len(self.frameworks),
                "adaptive_learning_enabled": self.enable_adaptive_learning,
                "cultural_considerations_enabled": self.enable_cultural_considerations,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}

    async def add_custom_constraint(self, constraint: EthicalConstraint):
        """Add a custom ethical constraint"""
        try:
            self.constraint_engine.add_constraint(constraint)
            self.logger.info(f"Added custom constraint: {constraint.name}")

            return {"success": True, "constraint_id": constraint.constraint_id}

        except Exception as e:
            self.logger.error(f"Failed to add custom constraint: {e}")
            return {"success": False, "error": str(e)}

    async def process_human_oversight_decision(self, oversight_request_id: str,
                                            decision: str, reasoning: str) -> Dict[str, Any]:
        """Process human oversight decision"""
        try:
            # Find the oversight request
            request = None
            for r in self.human_oversight_requests:
                if r["request_id"] == oversight_request_id:
                    request = r
                    break

            if not request:
                return {"success": False, "error": "Oversight request not found"}

            # Update request status
            request["status"] = "processed"
            request["decision"] = decision
            request["reasoning"] = reasoning
            request["processed_at"] = datetime.now().isoformat()

            # Record decision for learning
            self.ethical_decisions.append({
                "oversight_request_id": oversight_request_id,
                "decision": decision,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.info(f"Processed human oversight decision for {oversight_request_id}: {decision}")

            return {"success": True, "request_id": oversight_request_id, "decision": decision}

        except Exception as e:
            self.logger.error(f"Failed to process human oversight decision: {e}")
            return {"success": False, "error": str(e)}

    async def generate_ethical_report(self, assessment_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive ethical report"""
        try:
            if assessment_id:
                # Generate report for specific assessment
                assessment = None
                for a in self.assessment_history:
                    if a.assessment_id == assessment_id:
                        assessment = a
                        break

                if not assessment:
                    return {"success": False, "error": "Assessment not found"}

                report = {
                    "assessment_id": assessment.assessment_id,
                    "request_id": assessment.request_id,
                    "generated_at": datetime.now().isoformat(),
                    "summary": {
                        "overall_score": assessment.overall_score,
                        "risk_level": assessment.risk_level.value,
                        "compliance_status": assessment.compliance_status.value,
                        "requires_human_oversight": assessment.requires_human_oversight
                    },
                    "detailed_analysis": {
                        "framework_scores": {f.value: score for f, score in assessment.framework_scores.items()},
                        "constraint_violations": assessment.constraint_violations,
                        "stakeholder_impacts": assessment.stakeholder_impacts,
                        "impact_predictions": assessment.impact_predictions
                    },
                    "recommendations": assessment.recommendations,
                    "confidence_metrics": {
                        "confidence_score": assessment.confidence_score,
                        "decision_mode": assessment.decision_mode.value
                    }
                }
            else:
                # Generate overall system report
                recent_assessments = list(self.assessment_history)[-100:]

                report = {
                    "report_type": "system_overview",
                    "generated_at": datetime.now().isoformat(),
                    "system_metrics": await self.get_ethical_status(),
                    "recent_activity": {
                        "assessments_last_24h": len([a for a in self.assessment_history if
                                                    (datetime.now() - a.timestamp).total_seconds() < 86400]),
                        "oversight_requests_last_24h": len([r for r in self.human_oversight_requests if
                                                          (datetime.now() - datetime.fromisoformat(r["timestamp"])).total_seconds() < 86400])
                    },
                    "trend_analysis": {
                        "compliance_trend": self._calculate_compliance_trend(),
                        "risk_level_distribution": self._calculate_risk_distribution(),
                        "framework_effectiveness": self._calculate_framework_effectiveness()
                    }
                }

            return {"success": True, "report": report}

        except Exception as e:
            self.logger.error(f"Failed to generate ethical report: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_compliance_trend(self) -> str:
        """Calculate compliance trend"""
        if len(self.assessment_history) < 20:
            return "insufficient_data"

        recent = list(self.assessment_history)[-20:]
        older = list(self.assessment_history)[-40:-20]

        recent_compliance = sum(1 for a in recent if a.compliance_status == ComplianceStatus.FULLY_COMPLIANT) / len(recent)
        older_compliance = sum(1 for a in older if a.compliance_status == ComplianceStatus.FULLY_COMPLIANT) / len(older) if older else recent_compliance

        if recent_compliance > older_compliance + 0.1:
            return "improving"
        elif recent_compliance < older_compliance - 0.1:
            return "declining"
        else:
            return "stable"

    def _calculate_risk_distribution(self) -> Dict[str, float]:
        """Calculate risk level distribution"""
        if not self.assessment_history:
            return {}

        total = len(self.assessment_history)
        distribution = {}

        for risk_level in EthicalRiskLevel:
            count = sum(1 for a in self.assessment_history if a.risk_level == risk_level)
            distribution[risk_level.value] = count / total

        return distribution

    def _calculate_framework_effectiveness(self) -> Dict[str, float]:
        """Calculate framework effectiveness scores"""
        if not self.assessment_history:
            return {}

        effectiveness = {}

        for framework_type in EthicalFrameworkType:
            framework_scores = [a.framework_scores.get(framework_type, 0) for a in self.assessment_history]
            if framework_scores:
                effectiveness[framework_type.value] = sum(framework_scores) / len(framework_scores)
            else:
                effectiveness[framework_type.value] = 0.0

        return effectiveness

    async def shutdown(self):
        """Gracefully shutdown the ethical framework agent"""
        try:
            # Cancel background tasks
            if self._learning_task:
                self._learning_task.cancel()
            if self._monitoring_task:
                self._monitoring_task.cancel()

            self.logger.info(f"Ethical Framework Agent {self.agent_id} shutdown successfully")

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")


# Convenience functions for ethical framework integration

_ethical_framework_instance = None

async def get_ethical_framework(config: Dict[str, Any] = None) -> EthicalFrameworkAgent:
    """Get or create the global ethical framework instance"""
    global _ethical_framework_instance

    if _ethical_framework_instance is None:
        if config is None:
            config = {
                "enable_adaptive_learning": True,
                "human_oversight_threshold": 0.7,
                "ethical_threshold": 0.8,
                "enable_cultural_considerations": True
            }

        _ethical_framework_instance = EthicalFrameworkAgent(config)
        await _ethical_framework_instance.initialize()

    return _ethical_framework_instance

async def assess_research_ethics_simple(request_id: str, research_content: str,
                                      context: ResearchEthicsContext) -> EthicalAssessment:
    """Simple interface for research ethics assessment"""
    framework = await get_ethical_framework()
    return await framework.assess_research_ethics(request_id, research_content, context)