"""
Theory Evolution Agent

The core agent responsible for evolving research theories based on new findings,
correlating information, and maintaining theoretical coherence.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

import numpy as np
from ai_scientist.llm import create_client


class CorrelationStrength(Enum):
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class UpdateDecision(Enum):
    REJECT = "reject"
    ACCEPT = "accept"
    MODIFY = "modify"
    DEFER = "defer"


@dataclass
class ResearchFindings:
    """Represents research findings to be correlated with theory."""
    finding_id: str
    title: str
    content: str
    context: Dict[str, Any]
    confidence: float
    source: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CorrelationResult:
    """Result of correlating findings with existing theory."""
    correlation_id: str
    finding_id: str
    correlation_strength: CorrelationStrength
    correlation_score: float
    related_concepts: List[str]
    integration_points: List[str]
    conflicts: List[str]
    rationale: str
    confidence: float
    timestamp: datetime
    
    # Additional analysis
    novelty_score: float = 0.0
    coherence_impact: float = 0.0
    theoretical_implications: List[str] = None

    def __post_init__(self):
        if self.theoretical_implications is None:
            self.theoretical_implications = []


@dataclass
class TheoryUpdate:
    """Represents an update to the theory."""
    update_id: str
    correlation_id: str
    decision: UpdateDecision
    changes: List[Dict[str, Any]]
    rationale: str
    confidence: float
    impact_assessment: Dict[str, Any]
    timestamp: datetime
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None


@dataclass
class TheoryResponse:
    """Response from theory query."""
    query_id: str
    query: str
    response: str
    relevant_sections: List[str]
    confidence: float
    sources: List[str]
    timestamp: datetime


@dataclass
class TheoryVersion:
    """Represents a version of the theory."""
    version_id: str
    version_number: str
    description: str
    changes: List[str]
    timestamp: datetime
    parent_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TheoryEvolutionAgent:
    """
    Enhanced Theory Evolution Agent that automatically correlates findings
    with existing theory and manages theory updates.
    """

    def __init__(self, config: Dict[str, Any], theory_storage=None, correlator=None, version_manager=None):
        """
        Initialize the Theory Evolution Agent.
        
        Args:
            config: Configuration dictionary
            theory_storage: Theory storage backend
            correlator: Theory correlation engine
            version_manager: Theory version management system
        """
        self.config = config
        self.theory_storage = theory_storage
        self.correlator = correlator
        self.version_manager = version_manager
        
        # Initialize LLM client for reasoning
        theory_config = config.get("theory_evolution", {})
        self.llm_client = create_client(
            model=config.get("rag_engine", {}).get("leann", {}).get("model", "google/gemma-3-4b-it"),
            temperature=0.7,
            max_tokens=4096
        )
        
        # Configuration parameters
        self.correlation_threshold = theory_config.get("correlation_threshold", 0.75)
        self.auto_update = theory_config.get("auto_update", True)
        self.update_frequency = theory_config.get("update_frequency", "per_experiment")
        
        # State management
        self.pending_correlations: Dict[str, CorrelationResult] = {}
        self.pending_updates: Dict[str, TheoryUpdate] = {}
        self.current_theory_version: Optional[str] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def correlate_findings(self, findings: ResearchFindings) -> CorrelationResult:
        """
        Correlate research findings with existing theory.
        
        Args:
            findings: Research findings to correlate
            
        Returns:
            CorrelationResult: Detailed correlation analysis
        """
        self.logger.info(f"Correlating findings: {findings.finding_id}")
        
        # Extract current theory content for correlation
        current_theory = await self._get_current_theory()
        
        # Use correlator if available, otherwise use LLM-based correlation
        if self.correlator:
            correlation_result = await self.correlator.correlate(findings, current_theory)
        else:
            correlation_result = await self._llm_based_correlation(findings, current_theory)
        
        # Store pending correlation
        self.pending_correlations[correlation_result.correlation_id] = correlation_result
        
        # Auto-trigger update if enabled and correlation is strong enough
        if (self.auto_update and 
            correlation_result.correlation_score >= self.correlation_threshold):
            await self._auto_trigger_update(correlation_result)
        
        return correlation_result

    async def _llm_based_correlation(self, findings: ResearchFindings, current_theory: str) -> CorrelationResult:
        """Perform LLM-based correlation analysis."""
        
        correlation_prompt = f"""
        Analyze the correlation between new research findings and existing theory.
        
        NEW FINDINGS:
        Title: {findings.title}
        Content: {findings.content}
        Confidence: {findings.confidence}
        Source: {findings.source}
        Context: {json.dumps(findings.context, indent=2)}
        
        EXISTING THEORY:
        {current_theory[:4000]}  # Truncate for context limits
        
        Please analyze:
        1. How well do these findings correlate with the existing theory? (0.0 to 1.0)
        2. What are the key integration points?
        3. Are there any conflicts or contradictions?
        4. What theoretical implications do these findings have?
        5. How novel are these findings in context of the theory?
        6. What would be the impact on theory coherence if integrated?
        
        Provide a structured analysis with specific scores and detailed reasoning.
        """
        
        response = await self.llm_client.generate(correlation_prompt)
        
        # Parse LLM response into structured result
        correlation_result = self._parse_correlation_response(response, findings)
        
        return correlation_result

    def _parse_correlation_response(self, response: str, findings: ResearchFindings) -> CorrelationResult:
        """Parse LLM correlation response into structured result."""
        
        # Extract correlation score (simplified parsing)
        correlation_score = self._extract_score_from_response(response)
        
        # Determine correlation strength
        if correlation_score >= 0.8:
            strength = CorrelationStrength.VERY_STRONG
        elif correlation_score >= 0.6:
            strength = CorrelationStrength.STRONG
        elif correlation_score >= 0.4:
            strength = CorrelationStrength.MODERATE
        elif correlation_score >= 0.2:
            strength = CorrelationStrength.WEAK
        else:
            strength = CorrelationStrength.NONE
        
        # Extract key components (simplified)
        related_concepts = self._extract_concepts_from_response(response)
        integration_points = self._extract_integration_points(response)
        conflicts = self._extract_conflicts(response)
        
        return CorrelationResult(
            correlation_id=str(uuid.uuid4()),
            finding_id=findings.finding_id,
            correlation_strength=strength,
            correlation_score=correlation_score,
            related_concepts=related_concepts,
            integration_points=integration_points,
            conflicts=conflicts,
            rationale=response[:500],  # Truncated rationale
            confidence=0.8,  # Placeholder
            timestamp=datetime.now(),
            novelty_score=self._calculate_novelty_score(response),
            coherence_impact=self._calculate_coherence_impact(response),
            theoretical_implications=self._extract_implications(response)
        )

    def _extract_score_from_response(self, response: str) -> float:
        """Extract correlation score from LLM response."""
        # Simplified extraction - would use more sophisticated parsing in practice
        import re
        
        # Look for patterns like "correlation: 0.85" or "score: 0.75"
        score_patterns = [
            r'correlation[:\s]+([0-9]\.[0-9]+)',
            r'score[:\s]+([0-9]\.[0-9]+)',
            r'([0-9]\.[0-9]+).*correlation'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Default score if none found
        return 0.5

    def _extract_concepts_from_response(self, response: str) -> List[str]:
        """Extract related concepts from response."""
        # Simplified extraction
        concepts = []
        if "machine learning" in response.lower():
            concepts.append("machine_learning")
        if "neural network" in response.lower():
            concepts.append("neural_networks")
        if "optimization" in response.lower():
            concepts.append("optimization")
        return concepts

    def _extract_integration_points(self, response: str) -> List[str]:
        """Extract integration points from response."""
        # Simplified extraction
        points = []
        if "methodology" in response.lower():
            points.append("methodological_framework")
        if "evidence" in response.lower():
            points.append("empirical_evidence")
        if "theory" in response.lower():
            points.append("theoretical_foundation")
        return points

    def _extract_conflicts(self, response: str) -> List[str]:
        """Extract conflicts from response."""
        conflicts = []
        if "contradict" in response.lower():
            conflicts.append("contradiction_identified")
        if "inconsistent" in response.lower():
            conflicts.append("inconsistency_found")
        return conflicts

    def _calculate_novelty_score(self, response: str) -> float:
        """Calculate novelty score from response."""
        if "novel" in response.lower() or "new" in response.lower():
            return 0.8
        return 0.5

    def _calculate_coherence_impact(self, response: str) -> float:
        """Calculate coherence impact from response."""
        if "strengthen" in response.lower() or "enhance" in response.lower():
            return 0.8
        elif "weaken" in response.lower() or "conflict" in response.lower():
            return -0.3
        return 0.0

    def _extract_implications(self, response: str) -> List[str]:
        """Extract theoretical implications from response."""
        implications = []
        if "extend" in response.lower():
            implications.append("theory_extension")
        if "refine" in response.lower():
            implications.append("theory_refinement")
        return implications

    async def update_theory(self, correlation: CorrelationResult, user_approval: bool = False) -> TheoryUpdate:
        """
        Update theory based on correlation result.
        
        Args:
            correlation: Correlation result to base update on
            user_approval: Whether user has approved the update
            
        Returns:
            TheoryUpdate: Details of the theory update
        """
        self.logger.info(f"Updating theory based on correlation: {correlation.correlation_id}")
        
        # Determine update decision
        if correlation.correlation_strength == CorrelationStrength.NONE:
            decision = UpdateDecision.REJECT
        elif correlation.conflicts:
            decision = UpdateDecision.DEFER  # Requires manual review
        elif correlation.correlation_score >= self.correlation_threshold:
            decision = UpdateDecision.ACCEPT
        else:
            decision = UpdateDecision.MODIFY
        
        # Generate update changes
        changes = await self._generate_theory_changes(correlation, decision)
        
        # Assess impact
        impact_assessment = await self._assess_update_impact(changes)
        
        theory_update = TheoryUpdate(
            update_id=str(uuid.uuid4()),
            correlation_id=correlation.correlation_id,
            decision=decision,
            changes=changes,
            rationale=f"Based on correlation analysis: {correlation.rationale[:200]}",
            confidence=correlation.confidence * 0.9,  # Slight discount for uncertainty
            impact_assessment=impact_assessment,
            timestamp=datetime.now()
        )
        
        # Apply update if decision is to accept and auto-update is enabled
        if decision == UpdateDecision.ACCEPT and self.auto_update and not user_approval:
            await self._apply_theory_update(theory_update)
        else:
            # Store for manual approval
            self.pending_updates[theory_update.update_id] = theory_update
        
        return theory_update

    async def _generate_theory_changes(self, correlation: CorrelationResult, decision: UpdateDecision) -> List[Dict[str, Any]]:
        """Generate specific changes to apply to theory."""
        changes = []
        
        if decision == UpdateDecision.ACCEPT:
            # Add new evidence or concepts
            if correlation.integration_points:
                for point in correlation.integration_points:
                    changes.append({
                        "type": "addition",
                        "section": point,
                        "content": f"New evidence supports: {correlation.rationale[:100]}",
                        "rationale": correlation.rationale
                    })
        
        elif decision == UpdateDecision.MODIFY:
            # Modify existing sections
            changes.append({
                "type": "modification",
                "section": "empirical_evidence",
                "content": "Updated with new findings (partial integration)",
                "rationale": "Moderate correlation requires careful integration"
            })
        
        return changes

    async def _assess_update_impact(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the impact of proposed theory updates."""
        return {
            "coherence_change": 0.1,  # Placeholder
            "completeness_change": 0.05,
            "affected_sections": [change.get("section", "unknown") for change in changes],
            "risk_level": "low",
            "reversible": True
        }

    async def _apply_theory_update(self, theory_update: TheoryUpdate) -> None:
        """Apply theory update to storage."""
        if self.theory_storage:
            await self.theory_storage.apply_update(theory_update)
        
        if self.version_manager:
            await self.version_manager.create_version(
                description=f"Updated based on findings: {theory_update.correlation_id}",
                changes=[change.get("content", "") for change in theory_update.changes]
            )
        
        self.logger.info(f"Applied theory update: {theory_update.update_id}")

    async def _auto_trigger_update(self, correlation: CorrelationResult) -> None:
        """Automatically trigger theory update for strong correlations."""
        if correlation.correlation_score >= self.correlation_threshold:
            update = await self.update_theory(correlation, user_approval=False)
            self.logger.info(f"Auto-triggered theory update: {update.update_id}")

    async def query_theory(self, question: str) -> TheoryResponse:
        """
        Query the current theory with a specific question.
        
        Args:
            question: Question to ask about the theory
            
        Returns:
            TheoryResponse: Response from theory analysis
        """
        self.logger.info(f"Querying theory: {question}")
        
        current_theory = await self._get_current_theory()
        
        query_prompt = f"""
        Based on the following theory, answer this question:
        
        QUESTION: {question}
        
        THEORY:
        {current_theory}
        
        Provide a comprehensive answer citing relevant sections and explaining the theoretical basis.
        """
        
        response = await self.llm_client.generate(query_prompt)
        
        return TheoryResponse(
            query_id=str(uuid.uuid4()),
            query=question,
            response=response,
            relevant_sections=["theory_main"],  # Placeholder
            confidence=0.8,
            sources=["current_theory"],
            timestamp=datetime.now()
        )

    async def version_theory(self, reason: str) -> TheoryVersion:
        """
        Create a new version of the theory.
        
        Args:
            reason: Reason for creating new version
            
        Returns:
            TheoryVersion: New theory version information
        """
        if self.version_manager:
            return await self.version_manager.create_version(
                description=reason,
                changes=[]
            )
        
        # Fallback implementation
        return TheoryVersion(
            version_id=str(uuid.uuid4()),
            version_number="1.0.0",
            description=reason,
            changes=[],
            timestamp=datetime.now()
        )

    async def rollback_theory(self, version_id: str) -> bool:
        """
        Rollback theory to a previous version.
        
        Args:
            version_id: ID of version to rollback to
            
        Returns:
            bool: Success status
        """
        if self.version_manager:
            return await self.version_manager.rollback_to_version(version_id)
        
        self.logger.warning("Version manager not available for rollback")
        return False

    async def _get_current_theory(self) -> str:
        """Get current theory content."""
        if self.theory_storage:
            return await self.theory_storage.get_current_theory()
        
        # Fallback to placeholder theory
        return """
        Current Research Theory:
        
        1. Core Principles
        - Machine learning models benefit from proper regularization
        - Data quality is fundamental to model performance
        - Systematic experimentation leads to reproducible results
        
        2. Supporting Evidence
        - Multiple studies confirm regularization benefits
        - Empirical evidence shows data preprocessing importance
        
        3. Theoretical Framework
        - Based on statistical learning theory
        - Incorporates principles of scientific method
        """

    def get_correlation_status(self, correlation_id: str) -> Optional[CorrelationResult]:
        """Get status of a specific correlation."""
        return self.pending_correlations.get(correlation_id)

    def get_pending_updates(self) -> List[TheoryUpdate]:
        """Get list of pending theory updates."""
        return list(self.pending_updates.values())

    def approve_update(self, update_id: str, approved_by: str) -> bool:
        """
        Approve a pending theory update.
        
        Args:
            update_id: ID of update to approve
            approved_by: Identifier of approver
            
        Returns:
            bool: Success status
        """
        if update_id not in self.pending_updates:
            return False
        
        update = self.pending_updates[update_id]
        update.approved_by = approved_by
        update.approval_timestamp = datetime.now()
        
        # Apply the update
        asyncio.create_task(self._apply_theory_update(update))
        
        # Remove from pending
        del self.pending_updates[update_id]
        
        return True

    def export_state(self) -> Dict[str, Any]:
        """Export current agent state."""
        return {
            "pending_correlations": {k: asdict(v) for k, v in self.pending_correlations.items()},
            "pending_updates": {k: asdict(v) for k, v in self.pending_updates.items()},
            "current_theory_version": self.current_theory_version,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }