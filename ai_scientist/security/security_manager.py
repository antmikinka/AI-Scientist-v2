"""
Security Manager for AI-Scientist-v2

This module provides security management capabilities including authentication,
authorization, and ethical compliance checking for research operations.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_ethical_compliance: bool = True
    api_key_validation: bool = True
    risk_threshold: float = 0.7


class SecurityManager:
    """
    Security Manager - Handles security and ethical compliance

    This class provides security management capabilities for the AI-Scientist-v2
    platform, including authentication, authorization, and ethical compliance checking.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.security_config = SecurityConfig(**self.config.get("security", {}))

        self.logger = logging.getLogger(f"{__name__}.SecurityManager")

    async def check_research_compliance(self, objective: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check research request for ethical compliance

        Args:
            objective: Research objective to check
            context: Additional context for compliance checking

        Returns:
            Compliance assessment result
        """
        try:
            if not self.security_config.enable_ethical_compliance:
                return {"approved": True, "reason": "Compliance checking disabled"}

            # Basic compliance checks
            risk_factors = self._assess_risk_factors(objective, context or {})
            risk_score = self._calculate_risk_score(risk_factors)

            # Check against threshold
            approved = risk_score <= self.security_config.risk_threshold

            result = {
                "approved": approved,
                "risk_score": risk_score,
                "risk_level": "low" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "high",
                "risk_factors": risk_factors,
                "timestamp": datetime.now().isoformat(),
                "compliance_score": 1.0 - risk_score
            }

            if not approved:
                result["reason"] = f"Risk score {risk_score:.2f} exceeds threshold {self.security_config.risk_threshold}"

            return result

        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            return {
                "approved": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _assess_risk_factors(self, objective: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk factors in research objective"""
        risk_factors = {}

        # Check for high-risk keywords
        high_risk_keywords = [
            "human", "subjects", "patients", "medical", "clinical",
            "weapons", "military", "surveillance", "manipulation",
            "exploitation", "harm", "dangerous", "toxic"
        ]

        objective_lower = objective.lower()
        risk_count = sum(1 for keyword in high_risk_keywords if keyword in objective_lower)

        risk_factors["keyword_risks"] = risk_count
        risk_factors["high_risk_keywords_found"] = [
            keyword for keyword in high_risk_keywords if keyword in objective_lower
        ]

        # Check context for safety measures
        safety_measures = context.get("safety_measures", [])
        ethical_overrides = context.get("ethical_requirements", {})

        risk_factors["safety_measures_present"] = len(safety_measures) > 0
        risk_factors["ethical_overrides_present"] = len(ethical_overrides) > 0

        # Check for human subjects research
        human_subjects = context.get("human_subjects", False)
        if human_subjects or "human" in objective_lower or "subjects" in objective_lower:
            risk_factors["human_subjects_research"] = True
            risk_factors["requires_irb_approval"] = True

        return risk_factors

    def _calculate_risk_score(self, risk_factors: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        score = 0.0

        # Keyword risks
        keyword_risks = risk_factors.get("keyword_risks", 0)
        score += min(0.4, keyword_risks * 0.1)  # Max 0.4 for keyword risks

        # Human subjects research
        if risk_factors.get("human_subjects_research", False):
            score += 0.5

        # Safety measures reduce risk
        if risk_factors.get("safety_measures_present", False):
            score -= 0.1

        # Ethical overrides reduce risk
        if risk_factors.get("ethical_overrides_present", False):
            score -= 0.1

        return max(0.0, min(1.0, score))

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        if not self.security_config.api_key_validation:
            return True

        # Simple validation - in production, use proper authentication
        return api_key and api_key.startswith("sk-") and len(api_key) > 20

    async def check_authorization(self, user_id: str, resource: str, action: str) -> bool:
        """Check user authorization"""
        if not self.security_config.enable_authorization:
            return True

        # Simple authorization - in production, use proper RBAC
        return True

    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> bool:
        """Log security event"""
        try:
            event = {
                "event_type": event_type,
                "details": details,
                "timestamp": datetime.now().isoformat(),
                "user": details.get("user", "anonymous"),
                "resource": details.get("resource", "unknown"),
                "action": details.get("action", "unknown")
            }

            self.logger.info(f"Security event: {event}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            return False

    async def get_security_status(self) -> Dict[str, Any]:
        """Get security system status"""
        return {
            "authentication_enabled": self.security_config.enable_authentication,
            "authorization_enabled": self.security_config.enable_authorization,
            "compliance_checking_enabled": self.security_config.enable_ethical_compliance,
            "risk_threshold": self.security_config.risk_threshold,
            "timestamp": datetime.now().isoformat()
        }