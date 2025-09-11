"""
Agent Profile Management System

Manages agent personalities, behaviors, and capabilities for specialist agents
in the enhanced AI-Scientist-v2 system.
"""

import json
import logging
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class ProfileAdaptationMode(Enum):
    STATIC = "static"
    ADAPTIVE = "adaptive"
    EVOLUTIONARY = "evolutionary"


@dataclass
class PersonalityTraits:
    """Represents personality traits for an agent profile."""
    traits: List[str]
    openness: float  # 0.0 to 1.0
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def __post_init__(self):
        # Validate trait scores
        for trait_name, trait_value in [
            ("openness", self.openness),
            ("conscientiousness", self.conscientiousness),
            ("extraversion", self.extraversion),
            ("agreeableness", self.agreeableness),
            ("neuroticism", self.neuroticism)
        ]:
            if not 0.0 <= trait_value <= 1.0:
                raise ValueError(f"{trait_name} must be between 0.0 and 1.0")


@dataclass
class PromptingStyle:
    """Defines how an agent should be prompted."""
    prefix: str
    emphasis: str
    constraints: str
    
    def format_prompt(self, base_prompt: str) -> str:
        """Format a base prompt with this prompting style."""
        formatted = f"{self.prefix}\n\n{base_prompt}"
        if self.emphasis:
            formatted += f"\n\nFocus particularly on: {self.emphasis}"
        if self.constraints:
            formatted += f"\n\nConstraints: {self.constraints}"
        return formatted


@dataclass
class InteractionPreferences:
    """Defines how an agent prefers to interact."""
    collaboration_style: str
    feedback_style: str
    communication_style: str


@dataclass
class AgentProfile:
    """Complete agent profile definition."""
    profile_id: str
    name: str
    description: str
    personality: PersonalityTraits
    strengths: List[str]
    focus_areas: List[str]
    decision_style: str
    risk_tolerance: float
    prompting_style: PromptingStyle
    interaction_preferences: InteractionPreferences
    
    # Performance tracking
    performance_history: List[Dict[str, Any]] = None
    adaptation_data: Optional[Dict[str, Any]] = None
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be between 0.0 and 1.0")


class AgentProfileManager:
    """
    Manages agent profiles including loading, adaptation, and performance tracking.
    """

    def __init__(self, config_path: Optional[str] = None, adaptation_mode: ProfileAdaptationMode = ProfileAdaptationMode.ADAPTIVE):
        """
        Initialize the Agent Profile Manager.
        
        Args:
            config_path: Path to agent profiles configuration file
            adaptation_mode: How profiles should adapt over time
        """
        self.config_path = config_path or "ai_scientist/config/agent_profiles.yaml"
        self.adaptation_mode = adaptation_mode
        self.profiles: Dict[str, AgentProfile] = {}
        self.assignment_rules: Dict[str, Dict[str, str]] = {}
        self.adaptation_config: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Load profiles from configuration
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load agent profiles from configuration file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"Profile config file not found: {self.config_path}")
                self._create_default_profiles()
                return
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load individual profiles
            for profile_name, profile_data in config.get("profiles", {}).items():
                try:
                    profile = self._parse_profile(profile_name, profile_data)
                    self.profiles[profile_name] = profile
                    self.logger.info(f"Loaded profile: {profile_name}")
                except Exception as e:
                    self.logger.error(f"Error loading profile {profile_name}: {e}")
            
            # Load assignment rules
            self.assignment_rules = config.get("assignment_rules", {})
            
            # Load adaptation configuration
            self.adaptation_config = config.get("adaptation", {})
            
            self.logger.info(f"Loaded {len(self.profiles)} agent profiles")
            
        except Exception as e:
            self.logger.error(f"Error loading profiles: {e}")
            self._create_default_profiles()

    def _parse_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> AgentProfile:
        """Parse profile data into AgentProfile object."""
        
        # Parse personality traits
        personality_data = profile_data.get("personality", {})
        personality = PersonalityTraits(
            traits=personality_data.get("traits", []),
            openness=personality_data.get("openness", 0.5),
            conscientiousness=personality_data.get("conscientiousness", 0.5),
            extraversion=personality_data.get("extraversion", 0.5),
            agreeableness=personality_data.get("agreeableness", 0.5),
            neuroticism=personality_data.get("neuroticism", 0.5)
        )
        
        # Parse prompting style
        prompting_data = profile_data.get("prompting_style", {})
        prompting_style = PromptingStyle(
            prefix=prompting_data.get("prefix", ""),
            emphasis=prompting_data.get("emphasis", ""),
            constraints=prompting_data.get("constraints", "")
        )
        
        # Parse interaction preferences
        interaction_data = profile_data.get("interaction_preferences", {})
        interaction_preferences = InteractionPreferences(
            collaboration_style=interaction_data.get("collaboration_style", "standard"),
            feedback_style=interaction_data.get("feedback_style", "standard"),
            communication_style=interaction_data.get("communication_style", "standard")
        )
        
        return AgentProfile(
            profile_id=profile_name,
            name=profile_name,
            description=profile_data.get("description", ""),
            personality=personality,
            strengths=profile_data.get("strengths", []),
            focus_areas=profile_data.get("focus_areas", []),
            decision_style=profile_data.get("decision_style", "balanced"),
            risk_tolerance=profile_data.get("risk_tolerance", 0.5),
            prompting_style=prompting_style,
            interaction_preferences=interaction_preferences
        )

    def _create_default_profiles(self) -> None:
        """Create default agent profiles if configuration is not available."""
        self.logger.info("Creating default agent profiles")
        
        # Create basic default profiles
        default_profiles = {
            "creative_researcher": {
                "description": "Creative and innovative researcher",
                "personality": {
                    "traits": ["creative", "innovative", "risk-taking"],
                    "openness": 0.9,
                    "conscientiousness": 0.7,
                    "extraversion": 0.8,
                    "agreeableness": 0.6,
                    "neuroticism": 0.3
                },
                "strengths": ["ideation", "novel_approaches"],
                "focus_areas": ["breakthrough_ideas"],
                "decision_style": "intuitive",
                "risk_tolerance": 0.8,
                "prompting_style": {
                    "prefix": "Think creatively and innovatively:",
                    "emphasis": "originality",
                    "constraints": "minimal"
                },
                "interaction_preferences": {
                    "collaboration_style": "brainstorming",
                    "feedback_style": "encouraging",
                    "communication_style": "enthusiastic"
                }
            }
        }
        
        for profile_name, profile_data in default_profiles.items():
            try:
                profile = self._parse_profile(profile_name, profile_data)
                self.profiles[profile_name] = profile
            except Exception as e:
                self.logger.error(f"Error creating default profile {profile_name}: {e}")

    def get_profile(self, profile_name: str) -> Optional[AgentProfile]:
        """
        Get an agent profile by name.
        
        Args:
            profile_name: Name of the profile to retrieve
            
        Returns:
            AgentProfile if found, None otherwise
        """
        return self.profiles.get(profile_name)

    def list_profiles(self) -> List[str]:
        """Get list of available profile names."""
        return list(self.profiles.keys())

    def get_profile_for_task(self, task_type: str) -> Optional[AgentProfile]:
        """
        Get the best profile for a specific task type.
        
        Args:
            task_type: Type of task (e.g., 'ideation', 'experiment', 'analysis')
            
        Returns:
            Best matching AgentProfile or None
        """
        # Check assignment rules first
        if task_type in self.assignment_rules:
            primary_profile = self.assignment_rules[task_type].get("primary")
            if primary_profile and primary_profile in self.profiles:
                return self.profiles[primary_profile]
        
        # Fallback to best match based on profile characteristics
        return self._find_best_profile_match(task_type)

    def _find_best_profile_match(self, task_type: str) -> Optional[AgentProfile]:
        """Find the best profile match for a task type."""
        # Simple matching logic - could be more sophisticated
        task_profile_mapping = {
            "ideation": ["creative_researcher", "theory_synthesizer"],
            "experiment": ["methodical_experimenter"],
            "analysis": ["analytical_thinker"],
            "review": ["critical_reviewer"],
            "theory": ["theory_synthesizer"]
        }
        
        preferred_profiles = task_profile_mapping.get(task_type, [])
        
        for profile_name in preferred_profiles:
            if profile_name in self.profiles:
                return self.profiles[profile_name]
        
        # Return first available profile as fallback
        if self.profiles:
            return next(iter(self.profiles.values()))
        
        return None

    def record_performance(self, profile_name: str, task_id: str, performance_data: Dict[str, Any]) -> None:
        """
        Record performance data for a profile.
        
        Args:
            profile_name: Name of the profile
            task_id: ID of the task
            performance_data: Performance metrics and feedback
        """
        if profile_name not in self.profiles:
            self.logger.warning(f"Profile {profile_name} not found for performance recording")
            return
        
        profile = self.profiles[profile_name]
        
        performance_record = {
            "task_id": task_id,
            "timestamp": performance_data.get("timestamp"),
            "metrics": performance_data.get("metrics", {}),
            "feedback": performance_data.get("feedback", ""),
            "success": performance_data.get("success", False)
        }
        
        profile.performance_history.append(performance_record)
        
        # Trigger adaptation if enabled
        if self.adaptation_mode != ProfileAdaptationMode.STATIC:
            self._adapt_profile(profile_name, performance_record)

    def _adapt_profile(self, profile_name: str, performance_record: Dict[str, Any]) -> None:
        """
        Adapt a profile based on performance feedback.
        
        Args:
            profile_name: Name of the profile to adapt
            performance_record: Recent performance data
        """
        if not self.adaptation_config.get("enabled", False):
            return
        
        profile = self.profiles[profile_name]
        adaptation_rate = self.adaptation_config.get("adaptation_rate", 0.1)
        
        # Simple adaptation logic - adjust risk tolerance based on success
        if performance_record.get("success", False):
            # Successful task - slightly increase risk tolerance
            profile.risk_tolerance = min(1.0, profile.risk_tolerance + adaptation_rate * 0.1)
        else:
            # Failed task - slightly decrease risk tolerance
            profile.risk_tolerance = max(0.0, profile.risk_tolerance - adaptation_rate * 0.1)
        
        profile.last_updated = performance_record.get("timestamp")
        
        self.logger.info(f"Adapted profile {profile_name}, new risk tolerance: {profile.risk_tolerance:.2f}")

    def create_specialized_prompt(self, profile_name: str, base_prompt: str, task_context: Dict[str, Any] = None) -> str:
        """
        Create a specialized prompt for an agent based on its profile.
        
        Args:
            profile_name: Name of the agent profile
            base_prompt: Base prompt to specialize
            task_context: Additional context for the task
            
        Returns:
            Specialized prompt string
        """
        profile = self.get_profile(profile_name)
        if not profile:
            self.logger.warning(f"Profile {profile_name} not found, using base prompt")
            return base_prompt
        
        # Format prompt with profile style
        specialized_prompt = profile.prompting_style.format_prompt(base_prompt)
        
        # Add personality-based modifications
        if profile.personality.openness > 0.8:
            specialized_prompt += "\n\nFeel free to explore unconventional approaches and creative solutions."
        
        if profile.personality.conscientiousness > 0.8:
            specialized_prompt += "\n\nEnsure thorough, systematic, and methodical approach to this task."
        
        # Add task context if provided
        if task_context:
            context_str = "\n\nTask Context:\n" + "\n".join([f"- {k}: {v}" for k, v in task_context.items()])
            specialized_prompt += context_str
        
        return specialized_prompt

    def get_compatibility_score(self, profile1_name: str, profile2_name: str) -> float:
        """
        Calculate compatibility score between two profiles for collaboration.
        
        Args:
            profile1_name: First profile name
            profile2_name: Second profile name
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        profile1 = self.get_profile(profile1_name)
        profile2 = self.get_profile(profile2_name)
        
        if not profile1 or not profile2:
            return 0.0
        
        # Calculate personality compatibility using Big Five traits
        trait_differences = [
            abs(profile1.personality.openness - profile2.personality.openness),
            abs(profile1.personality.conscientiousness - profile2.personality.conscientiousness),
            abs(profile1.personality.extraversion - profile2.personality.extraversion),
            abs(profile1.personality.agreeableness - profile2.personality.agreeableness),
            abs(profile1.personality.neuroticism - profile2.personality.neuroticism)
        ]
        
        # Lower differences mean higher compatibility
        avg_difference = sum(trait_differences) / len(trait_differences)
        personality_compatibility = 1.0 - avg_difference
        
        # Factor in complementary strengths
        shared_strengths = set(profile1.strengths) & set(profile2.strengths)
        complementary_strengths = set(profile1.strengths) | set(profile2.strengths)
        
        if len(complementary_strengths) > 0:
            strength_synergy = len(shared_strengths) / len(complementary_strengths)
        else:
            strength_synergy = 0.0
        
        # Combine factors
        compatibility_score = (personality_compatibility * 0.7) + (strength_synergy * 0.3)
        
        return min(1.0, max(0.0, compatibility_score))

    def suggest_profile_improvements(self, profile_name: str) -> List[str]:
        """
        Suggest improvements for a profile based on performance history.
        
        Args:
            profile_name: Name of the profile to analyze
            
        Returns:
            List of improvement suggestions
        """
        profile = self.get_profile(profile_name)
        if not profile or not profile.performance_history:
            return ["Insufficient performance data for analysis"]
        
        suggestions = []
        
        # Analyze success rate
        total_tasks = len(profile.performance_history)
        successful_tasks = sum(1 for record in profile.performance_history if record.get("success", False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        if success_rate < 0.6:
            suggestions.append("Consider reducing risk tolerance or increasing training")
        elif success_rate > 0.9:
            suggestions.append("Profile performing excellently - consider increasing challenge level")
        
        # Analyze task type performance
        task_performance = {}
        for record in profile.performance_history:
            task_type = record.get("task_type", "unknown")
            if task_type not in task_performance:
                task_performance[task_type] = []
            task_performance[task_type].append(record.get("success", False))
        
        for task_type, results in task_performance.items():
            task_success_rate = sum(results) / len(results) if results else 0.0
            if task_success_rate < 0.5:
                suggestions.append(f"Improve performance on {task_type} tasks")
        
        return suggestions if suggestions else ["Profile performing well overall"]

    def export_profiles(self, export_path: str) -> None:
        """Export all profiles to a file."""
        export_data = {
            "profiles": {},
            "assignment_rules": self.assignment_rules,
            "adaptation_config": self.adaptation_config
        }
        
        for profile_name, profile in self.profiles.items():
            export_data["profiles"][profile_name] = asdict(profile)
        
        with open(export_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False)
        
        self.logger.info(f"Exported {len(self.profiles)} profiles to {export_path}")

    def get_performance_summary(self, profile_name: str) -> Dict[str, Any]:
        """Get performance summary for a profile."""
        profile = self.get_profile(profile_name)
        if not profile or not profile.performance_history:
            return {"error": "No performance data available"}
        
        total_tasks = len(profile.performance_history)
        successful_tasks = sum(1 for record in profile.performance_history if record.get("success", False))
        success_rate = successful_tasks / total_tasks
        
        return {
            "profile_name": profile_name,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "last_performance": profile.performance_history[-1] if profile.performance_history else None,
            "risk_tolerance": profile.risk_tolerance,
            "last_updated": profile.last_updated
        }