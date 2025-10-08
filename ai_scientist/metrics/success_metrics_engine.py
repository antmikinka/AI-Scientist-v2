"""
Success Metrics Framework for AI-Scientist-v2

This module provides comprehensive metrics collection, analysis, and monitoring
for tracking progress toward revolutionizing scientific discovery through
responsible multi-agent AI systems.

Key Focus Areas:
- Life's Work Measurement (100x scientific acceleration, democratization, knowledge advancement)
- Quantifiable Success Indicators (research output, user adoption, system performance)
- Real-time Monitoring (KPI dashboards, automated analysis, predictive analytics)
- Multi-dimensional Assessment (technical, user, scientific, economic, ethical)
- Goal Alignment (progress tracking toward ultimate mission goals)

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

# Import existing system components
from ..core import get_logger, StructuredLogger
from ..monitoring.performance_monitor import PerformanceMonitor


class MetricCategory(Enum):
    """Categories of success metrics"""
    SCIENTIFIC_ACCELERATION = "scientific_acceleration"
    DEMOCRATIZATION = "democratization"
    KNOWLEDGE_ADVANCEMENT = "knowledge_advancement"
    IMPACT_MEASUREMENT = "impact_measurement"
    SYSTEM_EVOLUTION = "system_evolution"
    TECHNICAL_PERFORMANCE = "technical_performance"
    USER_SATISFACTION = "user_satisfaction"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    INNOVATION_FREQUENCY = "innovation_frequency"
    ECONOMIC_ACCESSIBILITY = "economic_accessibility"


class MetricType(Enum):
    """Types of metrics for different measurement approaches"""
    COUNTER = "counter"  # Cumulative count
    GAUGE = "gauge"  # Current value
    RATE = "rate"  # Rate over time
    PERCENTAGE = "percentage"  # Percentage calculation
    SCORE = "score"  # Calculated score (0-100)
    RATIO = "ratio"  # Ratio calculation
    INDEX = "index"  # Composite index
    BOOLEAN = "boolean"  # True/False achievement


class GoalStatus(Enum):
    """Status of goal achievement"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    APPROACHING = "approaching"
    ACHIEVED = "achieved"
    EXCEEDED = "exceeded"


@dataclass
class MetricDefinition:
    """Definition of a success metric"""
    name: str
    category: MetricCategory
    metric_type: MetricType
    description: str
    unit: str
    target_value: float
    weight: float = 1.0
    collection_frequency: str = "1h"  # ISO 8601 duration format
    calculation_method: str = "simple"  # simple, weighted, composite
    data_sources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    is_lifes_work_critical: bool = False


@dataclass
class MetricValue:
    """Recorded value of a metric at a specific time"""
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Confidence in the measurement (0-1)


@dataclass
class Goal:
    """Life's work goal with progress tracking"""
    name: str
    description: str
    category: str
    target_metrics: Dict[str, float]  # metric_name -> target_value
    current_metrics: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    weight: float = 1.0
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: float = 0.0
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)


@dataclass
class LifeWorkMetrics:
    """Core metrics for tracking life's work progress"""
    # Scientific Acceleration Metrics
    research_acceleration_factor: float = 1.0  # Current acceleration factor
    papers_generated_per_day: float = 0.0
    experiments_conducted_per_day: float = 0.0
    breakthrough_discovery_rate: float = 0.0  # Breakthroughs per month
    knowledge_generation_rate: float = 0.0  # Novel insights per day

    # Democratization Metrics
    global_users_count: int = 0
    geographies_reached: int = 0
    language_accessibility: float = 0.0  # 0-100 scale
    economic_accessibility_score: float = 0.0  # 0-100 scale
    user_diversity_index: float = 0.0  # Diversity of user base

    # Knowledge Advancement Metrics
    novel_hypotheses_generated: int = 0
    successful_experiments: int = 0
    peer_review_acceptance_rate: float = 0.0  # Percentage
    knowledge_graph_expansion: int = 0  # New connections per day
    interdisciplinary_insights: int = 0

    # Impact Measurement
    real_world_applications: int = 0
    scientific_citations_received: int = 0
    industry_adoption_count: int = 0
    policy_influence_count: int = 0
    education_integration_count: int = 0

    # System Evolution
    autonomous_improvement_count: int = 0
    self_optimization_rate: float = 0.0  # Improvements per week
    agent_capability_expansion: int = 0
    learning_velocity: float = 0.0  # Rate of capability improvement

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    collection_period: str = "daily"


class SuccessMetricsEngine:
    """
    Core engine for collecting, calculating, and analyzing success metrics

    This class provides the foundation for measuring progress toward the life's work
    of revolutionizing scientific discovery through responsible multi-agent AI systems.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.SuccessMetricsEngine")

        # Initialize components
        self.performance_monitor = PerformanceMonitor(config.get("performance_monitor", {}))
        self.metrics_registry: Dict[str, MetricDefinition] = {}
        self.metrics_history: deque = deque(maxlen=10000)
        self.current_metrics: LifeWorkMetrics = LifeWorkMetrics()

        # Goals and progress tracking
        self.lifes_work_goals: List[Goal] = []
        self.goal_progress: Dict[str, Dict[str, Any]] = {}

        # Real-time monitoring
        self.monitoring_active = False
        self._monitor_thread = None
        self._collection_interval = self.config.get("collection_interval", 300)  # 5 minutes

        # Analytics and prediction
        self.trend_analysis: Dict[str, Any] = {}
        self.predictions: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []

        # Initialize metric definitions
        self._initialize_metric_definitions()
        self._initialize_lifes_work_goals()

    def _initialize_metric_definitions(self):
        """Initialize all success metric definitions"""

        # Scientific Acceleration Metrics
        self.metrics_registry["research_acceleration_factor"] = MetricDefinition(
            name="research_acceleration_factor",
            category=MetricCategory.SCIENTIFIC_ACCELERATION,
            metric_type=MetricType.INDEX,
            description="Current research acceleration factor compared to traditional methods",
            unit="x",
            target_value=100.0,  # 100x goal
            weight=2.0,
            collection_frequency="1h",
            calculation_method="composite",
            is_lifes_work_critical=True
        )

        self.metrics_registry["papers_generated_per_day"] = MetricDefinition(
            name="papers_generated_per_day",
            category=MetricCategory.SCIENTIFIC_ACCELERATION,
            metric_type=MetricType.RATE,
            description="Number of scientific papers generated per day",
            unit="papers/day",
            target_value=50.0,
            collection_frequency="1d",
            is_lifes_work_critical=True
        )

        self.metrics_registry["breakthrough_discovery_rate"] = MetricDefinition(
            name="breakthrough_discovery_rate",
            category=MetricCategory.KNOWLEDGE_ADVANCEMENT,
            metric_type=MetricType.RATE,
            description="Number of breakthrough discoveries per month",
            unit="breakthroughs/month",
            target_value=5.0,
            weight=1.5,
            is_lifes_work_critical=True
        )

        # Democratization Metrics
        self.metrics_registry["global_users_count"] = MetricDefinition(
            name="global_users_count",
            category=MetricCategory.DEMOCRATIZATION,
            metric_type=MetricType.COUNTER,
            description="Total number of global users",
            unit="users",
            target_value=1000000,
            weight=1.2,
            collection_frequency="1d",
            is_lifes_work_critical=True
        )

        self.metrics_registry["geographies_reached"] = MetricDefinition(
            name="geographies_reached",
            category=MetricCategory.DEMOCRATIZATION,
            metric_type=MetricType.COUNTER,
            description="Number of countries/regions reached",
            unit="countries",
            target_value=195,  # All UN countries
            collection_frequency="1w"
        )

        # Impact Metrics
        self.metrics_registry["real_world_applications"] = MetricDefinition(
            name="real_world_applications",
            category=MetricCategory.IMPACT_MEASUREMENT,
            metric_type=MetricType.COUNTER,
            description="Number of real-world applications of generated research",
            unit="applications",
            target_value=100,
            collection_frequency="1w",
            is_lifes_work_critical=True
        )

        # System Evolution Metrics
        self.metrics_registry["autonomous_improvement_count"] = MetricDefinition(
            name="autonomous_improvement_count",
            category=MetricCategory.SYSTEM_EVOLUTION,
            metric_type=MetricType.COUNTER,
            description="Number of autonomous system improvements",
            unit="improvements",
            target_value=50,
            collection_frequency="1d"
        )

        # User Satisfaction
        self.metrics_registry["user_satisfaction_score"] = MetricDefinition(
            name="user_satisfaction_score",
            category=MetricCategory.USER_SATISFACTION,
            metric_type=MetricType.SCORE,
            description="Overall user satisfaction score",
            unit="score",
            target_value=90.0,
            collection_frequency="1d"
        )

        # Ethical Compliance
        self.metrics_registry["ethical_compliance_rate"] = MetricDefinition(
            name="ethical_compliance_rate",
            category=MetricCategory.ETHICAL_COMPLIANCE,
            metric_type=MetricType.PERCENTAGE,
            description="Rate of ethical compliance across all operations",
            unit="%",
            target_value=99.5,
            weight=1.8,
            collection_frequency="1h",
            is_lifes_work_critical=True
        )

        # Innovation Frequency
        self.metrics_registry["innovation_frequency"] = MetricDefinition(
            name="innovation_frequency",
            category=MetricCategory.INNOVATION_FREQUENCY,
            metric_type=MetricType.RATE,
            description="Frequency of innovative breakthroughs",
            unit="innovations/week",
            target_value=3.0,
            collection_frequency="1w",
            is_lifes_work_critical=True
        )

        self.logger.info(f"Initialized {len(self.metrics_registry)} metric definitions")

    def _initialize_lifes_work_goals(self):
        """Initialize the primary life's work goals"""

        # Primary Goal: 100x Scientific Acceleration
        acceleration_goal = Goal(
            name="100x Scientific Acceleration",
            description="Achieve 100x acceleration in scientific research output and discovery",
            category="scientific_acceleration",
            target_metrics={
                "research_acceleration_factor": 100.0,
                "papers_generated_per_day": 50.0,
                "breakthrough_discovery_rate": 5.0
            },
            deadline=datetime(2028, 12, 31),  # 3-year target
            weight=3.0,
            status=GoalStatus.IN_PROGRESS,
            milestones=[
                {"name": "10x acceleration", "target": 10.0, "deadline": datetime(2025, 6, 30)},
                {"name": "25x acceleration", "target": 25.0, "deadline": datetime(2026, 6, 30)},
                {"name": "50x acceleration", "target": 50.0, "deadline": datetime(2027, 6, 30)},
                {"name": "100x acceleration", "target": 100.0, "deadline": datetime(2028, 12, 31)}
            ],
            risks=[
                "Technical limitations in AI model capabilities",
                "Resistance from traditional research community",
                "Ethical concerns about autonomous research"
            ],
            opportunities=[
                "Advances in AI technology",
                "Growing acceptance of AI in research",
                "Increasing global scientific challenges requiring faster solutions"
            ]
        )

        # Global Democratization Goal
        democratization_goal = Goal(
            name="Global Research Democratization",
            description="Democratize access to advanced research capabilities globally",
            category="democratization",
            target_metrics={
                "global_users_count": 1000000,
                "geographies_reached": 195,
                "economic_accessibility_score": 85.0
            },
            deadline=datetime(2027, 12, 31),
            weight=2.5,
            status=GoalStatus.IN_PROGRESS,
            milestones=[
                {"name": "100K users", "target": 100000, "deadline": datetime(2025, 12, 31)},
                {"name": "500K users", "target": 500000, "deadline": datetime(2026, 12, 31)},
                {"name": "1M users", "target": 1000000, "deadline": datetime(2027, 12, 31)}
            ],
            risks=[
                "Economic barriers to access",
                "Language and cultural barriers",
                "Infrastructure limitations in developing regions"
            ],
            opportunities=[
                "Mobile technology penetration",
                "Open source movement growth",
                "Global education initiatives"
            ]
        )

        # Knowledge Advancement Goal
        knowledge_goal = Goal(
            name="Knowledge Advancement Revolution",
            description="Generate unprecedented rates of novel scientific insights and breakthroughs",
            category="knowledge_advancement",
            target_metrics={
                "novel_hypotheses_generated": 10000,
                "successful_experiments": 5000,
                "peer_review_acceptance_rate": 80.0,
                "interdisciplinary_insights": 1000
            },
            deadline=datetime(2028, 6, 30),
            weight=2.8,
            status=GoalStatus.IN_PROGRESS,
            milestones=[
                {"name": "First 1000 hypotheses", "target": 1000, "deadline": datetime(2025, 6, 30)},
                {"name": "First 5000 hypotheses", "target": 5000, "deadline": datetime(2026, 6, 30)},
                {"name": "10K hypotheses achieved", "target": 10000, "deadline": datetime(2028, 6, 30)}
            ],
            risks=[
                "Quality vs quantity trade-offs",
                "Validation challenges for novel hypotheses",
                "Publication bottlenecks"
            ],
            opportunities=[
                "New publishing models",
                "AI-powered validation systems",
                "Collaborative research platforms"
            ]
        )

        self.lifes_work_goals = [acceleration_goal, democratization_goal, knowledge_goal]
        self.logger.info(f"Initialized {len(self.lifes_work_goals)} life's work goals")

    async def initialize(self):
        """Initialize the Success Metrics Engine"""
        try:
            # Initialize performance monitor
            await self.performance_monitor.initialize()

            # Start metrics collection
            self.monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
            self._monitor_thread.start()

            # Load historical data if available
            await self._load_historical_data()

            self.logger.info("Success Metrics Engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Success Metrics Engine: {e}")
            raise

    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a specific metric value"""
        try:
            if metric_name not in self.metrics_registry:
                self.logger.warning(f"Unknown metric: {metric_name}")
                return

            # Create metric value record
            metric_value = MetricValue(
                metric_name=metric_name,
                value=value,
                metadata=metadata or {},
                timestamp=datetime.now()
            )

            # Store in history
            self.metrics_history.append(metric_value)

            # Update current metrics
            self._update_current_metrics(metric_name, value)

            # Log the recording
            self.logger.info(f"Recorded metric {metric_name}: {value}")

        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name}: {e}")

    def _update_current_metrics(self, metric_name: str, value: float):
        """Update the current metrics structure"""
        try:
            if hasattr(self.current_metrics, metric_name):
                setattr(self.current_metrics, metric_name, value)
            else:
                # Handle custom metrics
                self.logger.debug(f"Custom metric {metric_name} not in current metrics structure")

        except Exception as e:
            self.logger.error(f"Error updating current metrics: {e}")

    def _metrics_collection_loop(self):
        """Background loop for automatic metrics collection"""
        while self.monitoring_active:
            try:
                # Collect automated metrics
                asyncio.run(self._collect_automated_metrics())

                # Update goal progress
                asyncio.run(self._update_goal_progress())

                # Perform trend analysis
                asyncio.run(self._perform_trend_analysis())

                # Check for alerts
                asyncio.run(self._check_alerts())

                # Sleep for next collection interval
                time.sleep(self._collection_interval)

            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self._collection_interval)

    async def _collect_automated_metrics(self):
        """Collect metrics automatically from system sources"""
        try:
            # Get performance metrics
            perf_metrics = await self.performance_monitor.get_current_metrics()

            # Update system performance metrics
            if "performance_metrics" in perf_metrics:
                self.record_metric(
                    "system_throughput",
                    perf_metrics["performance_metrics"].get("throughput", 0.0),
                    {"source": "performance_monitor"}
                )

                self.record_metric(
                    "system_error_rate",
                    perf_metrics["performance_metrics"].get("error_rate", 0.0),
                    {"source": "performance_monitor"}
                )

            # Calculate research acceleration factor (simplified calculation)
            acceleration_factor = self._calculate_acceleration_factor()
            self.record_metric(
                "research_acceleration_factor",
                acceleration_factor,
                {"calculation": "automated"}
            )

            # Update user metrics (placeholder - would integrate with actual user tracking)
            user_count = self._estimate_user_count()
            self.record_metric(
                "global_users_count",
                user_count,
                {"source": "estimated"}
            )

        except Exception as e:
            self.logger.error(f"Error collecting automated metrics: {e}")

    def _calculate_acceleration_factor(self) -> float:
        """Calculate current research acceleration factor"""
        try:
            # This is a simplified calculation - in practice, this would involve
            # comparing current research output to baseline traditional methods

            # Get recent metrics
            recent_papers = getattr(self.current_metrics, 'papers_generated_per_day', 0)
            baseline_papers = 0.5  # Traditional research baseline

            if baseline_papers > 0:
                acceleration = recent_papers / baseline_papers
                return min(acceleration, 1000.0)  # Cap at 1000x for sanity

            return 1.0

        except Exception as e:
            self.logger.error(f"Error calculating acceleration factor: {e}")
            return 1.0

    def _estimate_user_count(self) -> int:
        """Estimate current user count"""
        try:
            # This would integrate with actual user tracking systems
            # For now, return a placeholder based on active workflows
            active_workflows = getattr(self.current_metrics, 'active_workflows', 0)
            estimated_users = max(1, active_workflows * 2)  # Rough estimate

            return int(estimated_users)

        except Exception as e:
            self.logger.error(f"Error estimating user count: {e}")
            return 0

    async def _update_goal_progress(self):
        """Update progress toward life's work goals"""
        try:
            for goal in self.lifes_work_goals:
                progress = self._calculate_goal_progress(goal)
                self.goal_progress[goal.name] = progress

                # Update goal status based on progress
                goal.progress_percentage = progress["overall_progress"]

                if progress["overall_progress"] >= 100:
                    goal.status = GoalStatus.ACHIEVED
                elif progress["overall_progress"] >= 90:
                    goal.status = GoalStatus.APPROACHING
                elif progress["overall_progress"] > 0:
                    goal.status = GoalStatus.IN_PROGRESS
                else:
                    goal.status = GoalStatus.NOT_STARTED

        except Exception as e:
            self.logger.error(f"Error updating goal progress: {e}")

    def _calculate_goal_progress(self, goal: Goal) -> Dict[str, Any]:
        """Calculate progress toward a specific goal"""
        try:
            metric_progress = {}
            total_weight = 0
            weighted_progress = 0

            for metric_name, target_value in goal.target_metrics.items():
                current_value = getattr(self.current_metrics, metric_name, 0.0)

                # Calculate progress for this metric
                if target_value > 0:
                    metric_progress_pct = min(100.0, (current_value / target_value) * 100)
                else:
                    metric_progress_pct = 0.0

                # Get metric weight
                metric_def = self.metrics_registry.get(metric_name)
                weight = metric_def.weight if metric_def else 1.0

                metric_progress[metric_name] = {
                    "current_value": current_value,
                    "target_value": target_value,
                    "progress_percentage": metric_progress_pct,
                    "weight": weight
                }

                weighted_progress += metric_progress_pct * weight
                total_weight += weight

            # Calculate overall progress
            overall_progress = weighted_progress / total_weight if total_weight > 0 else 0

            # Check milestone progress
            milestone_progress = []
            current_time = datetime.now()

            for milestone in goal.milestones:
                if current_time <= milestone["deadline"]:
                    milestone_target = milestone["target"]
                    current_metric_value = next(
                        (getattr(self.current_metrics, metric, 0.0)
                         for metric in goal.target_metrics.keys()),
                        0.0
                    )

                    if milestone_target > 0:
                        milestone_progress_pct = min(100.0, (current_metric_value / milestone_target) * 100)
                    else:
                        milestone_progress_pct = 0.0

                    milestone_progress.append({
                        "milestone": milestone["name"],
                        "target": milestone_target,
                        "current": current_metric_value,
                        "progress": milestone_progress_pct,
                        "deadline": milestone["deadline"].isoformat(),
                        "days_remaining": (milestone["deadline"] - current_time).days
                    })

            return {
                "overall_progress": overall_progress,
                "metric_progress": metric_progress,
                "milestone_progress": milestone_progress,
                "last_updated": datetime.now().isoformat(),
                "status": goal.status.value
            }

        except Exception as e:
            self.logger.error(f"Error calculating goal progress: {e}")
            return {"overall_progress": 0.0, "error": str(e)}

    async def _perform_trend_analysis(self):
        """Perform trend analysis on historical metrics"""
        try:
            if len(self.metrics_history) < 2:
                return

            # Analyze trends for each metric
            for metric_name in self.metrics_registry.keys():
                recent_values = [
                    mv for mv in self.metrics_history
                    if mv.metric_name == metric_name and
                    mv.timestamp > datetime.now() - timedelta(days=7)
                ]

                if len(recent_values) >= 2:
                    # Calculate trend
                    values = [mv.value for mv in recent_values]
                    timestamps = [(mv.timestamp - recent_values[0].timestamp).total_seconds() for mv in recent_values]

                    # Simple linear regression for trend
                    if len(values) >= 2:
                        trend_slope = self._calculate_trend_slope(values, timestamps)
                        trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"

                        self.trend_analysis[metric_name] = {
                            "trend": trend_direction,
                            "slope": trend_slope,
                            "recent_values": values[-5:],  # Last 5 values
                            "confidence": self._calculate_trend_confidence(values, trend_slope)
                        }

        except Exception as e:
            self.logger.error(f"Error performing trend analysis: {e}")

    def _calculate_trend_slope(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate trend slope using simple linear regression"""
        try:
            if len(values) != len(timestamps) or len(values) < 2:
                return 0.0

            n = len(values)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)

            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0.0

            return slope

        except Exception as e:
            self.logger.error(f"Error calculating trend slope: {e}")
            return 0.0

    def _calculate_trend_confidence(self, values: List[float], slope: float) -> float:
        """Calculate confidence in trend analysis"""
        try:
            if len(values) < 3:
                return 0.0

            # Calculate R-squared as confidence measure
            n = len(values)
            if n == 0:
                return 0.0

            mean_y = sum(values) / n
            total_sum_squares = sum((y - mean_y) ** 2 for y in values)

            if total_sum_squares == 0:
                return 1.0  # Perfect fit for constant values

            # This is a simplified confidence calculation
            confidence = min(1.0, max(0.0, abs(slope) / (total_sum_squares / n) ** 0.5))

            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating trend confidence: {e}")
            return 0.0

    async def _check_alerts(self):
        """Check for metric alerts and threshold violations"""
        try:
            new_alerts = []
            current_time = datetime.now()

            for metric_name, metric_def in self.metrics_registry.items():
                # Get recent values
                recent_values = [
                    mv for mv in self.metrics_history
                    if mv.metric_name == metric_name and
                    mv.timestamp > current_time - timedelta(hours=1)
                ]

                if recent_values:
                    latest_value = recent_values[-1].value

                    # Check target threshold
                    if metric_def.target_value > 0:
                        current_pct = (latest_value / metric_def.target_value) * 100

                        # Alert if critically below target
                        if current_pct < 20 and metric_def.is_lifes_work_critical:
                            alert = {
                                "type": "critical",
                                "metric": metric_name,
                                "current_value": latest_value,
                                "target_value": metric_def.target_value,
                                "percentage_of_target": current_pct,
                                "message": f"Critical metric {metric_name} is only {current_pct:.1f}% of target",
                                "timestamp": current_time.isoformat(),
                                "severity": "high"
                            }
                            new_alerts.append(alert)

                        # Alert if approaching/exceeding target
                        elif current_pct >= 95:
                            alert = {
                                "type": "achievement",
                                "metric": metric_name,
                                "current_value": latest_value,
                                "target_value": metric_def.target_value,
                                "percentage_of_target": current_pct,
                                "message": f"Metric {metric_name} approaching target ({current_pct:.1f}%)",
                                "timestamp": current_time.isoformat(),
                                "severity": "medium"
                            }
                            new_alerts.append(alert)

            # Add new alerts
            self.alerts.extend(new_alerts)

            # Keep only recent alerts (last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.alerts = [alert for alert in self.alerts
                          if datetime.fromisoformat(alert["timestamp"]) > cutoff_time]

            # Log critical alerts
            for alert in new_alerts:
                if alert["severity"] == "high":
                    self.logger.warning(f"CRITICAL ALERT: {alert['message']}")
                elif alert["severity"] == "medium":
                    self.logger.info(f"Alert: {alert['message']}")

        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")

    async def _load_historical_data(self):
        """Load historical metrics data from persistent storage"""
        try:
            # Placeholder for loading from database/file
            # In a real implementation, this would load from persistent storage
            self.logger.info("Historical data loading would happen here")

        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current success metrics snapshot"""
        try:
            return {
                "lifes_work_metrics": asdict(self.current_metrics),
                "goals_progress": {goal.name: self.goal_progress.get(goal.name, {})
                                 for goal in self.lifes_work_goals},
                "trend_analysis": self.trend_analysis,
                "active_alerts": self.alerts[-10:],  # Last 10 alerts
                "system_status": "healthy" if len(self.alerts) < 5 else "warning",
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {"error": str(e)}

    async def get_goal_progress(self, goal_name: str = None) -> Dict[str, Any]:
        """Get detailed progress for specific goal or all goals"""
        try:
            if goal_name:
                goal = next((g for g in self.lifes_work_goals if g.name == goal_name), None)
                if goal:
                    return {
                        "goal": {
                            "name": goal.name,
                            "description": goal.description,
                            "category": goal.category,
                            "deadline": goal.deadline.isoformat() if goal.deadline else None,
                            "weight": goal.weight,
                            "status": goal.status.value
                        },
                        "progress": self.goal_progress.get(goal.name, {})
                    }
                else:
                    return {"error": f"Goal '{goal_name}' not found"}
            else:
                return {
                    "goals": [
                        {
                            "name": goal.name,
                            "description": goal.description,
                            "category": goal.category,
                            "progress_percentage": goal.progress_percentage,
                            "status": goal.status.value,
                            "deadline": goal.deadline.isoformat() if goal.deadline else None
                        }
                        for goal in self.lifes_work_goals
                    ],
                    "overall_progress": sum(goal.progress_percentage for goal in self.lifes_work_goals) / len(self.lifes_work_goals)
                }

        except Exception as e:
            self.logger.error(f"Error getting goal progress: {e}")
            return {"error": str(e)}

    async def get_metric_history(self, metric_name: str, duration_hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=duration_hours)

            history = [
                {
                    "timestamp": mv.timestamp.isoformat(),
                    "value": mv.value,
                    "metadata": mv.metadata,
                    "confidence": mv.confidence
                }
                for mv in self.metrics_history
                if mv.metric_name == metric_name and mv.timestamp > cutoff_time
            ]

            return sorted(history, key=lambda x: x["timestamp"])

        except Exception as e:
            self.logger.error(f"Error getting metric history: {e}")
            return []

    async def get_predictions(self) -> Dict[str, Any]:
        """Get predictions and forecasts for goal achievement"""
        try:
            predictions = {}

            for goal in self.lifes_work_goals:
                goal_predictions = []

                for metric_name, target_value in goal.target_metrics.items():
                    # Get historical data
                    history = await self.get_metric_history(metric_name, 168)  # Last week

                    if len(history) >= 2:
                        # Calculate trend and predict
                        values = [h["value"] for h in history]
                        timestamps = [(datetime.fromisoformat(h["timestamp"]) - datetime.now()).total_seconds() for h in history]

                        trend_slope = self._calculate_trend_slope(values, timestamps)

                        # Predict time to reach target
                        if trend_slope > 0:
                            current_value = values[-1] if values else 0
                            time_to_target = (target_value - current_value) / trend_slope if trend_slope > 0 else float('inf')

                            predicted_date = datetime.now() + timedelta(seconds=time_to_target)

                            goal_predictions.append({
                                "metric": metric_name,
                                "current_value": current_value,
                                "target_value": target_value,
                                "trend_slope": trend_slope,
                                "predicted_achievement_date": predicted_date.isoformat() if time_to_target != float('inf') else None,
                                "confidence": self._calculate_prediction_confidence(history, trend_slope, target_value)
                            })

                predictions[goal.name] = goal_predictions

            return {
                "predictions": predictions,
                "overall_assessment": self._generate_overall_assessment(predictions),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return {"error": str(e)}

    def _calculate_prediction_confidence(self, history: List[Dict], trend_slope: float, target_value: float) -> float:
        """Calculate confidence in prediction"""
        try:
            if len(history) < 3:
                return 0.0

            # Factors affecting confidence
            factors = {
                "data_points": min(1.0, len(history) / 10),  # More data = more confidence
                "trend_consistency": min(1.0, abs(trend_slope) / 0.1),  # Stronger trend = more confidence
                "target_proximity": min(1.0, target_value / (history[-1]["value"] * 2)) if history[-1]["value"] > 0 else 0.0
            }

            # Weighted confidence score
            confidence = (
                factors["data_points"] * 0.4 +
                factors["trend_consistency"] * 0.4 +
                factors["target_proximity"] * 0.2
            )

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating prediction confidence: {e}")
            return 0.0

    def _generate_overall_assessment(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of goal achievement likelihood"""
        try:
            assessments = {}

            for goal_name, goal_predictions in predictions.items():
                if not goal_predictions:
                    assessments[goal_name] = {
                        "likelihood": "unknown",
                        "confidence": 0.0,
                        "key_factors": []
                    }
                    continue

                # Analyze predictions
                positive_predictions = sum(1 for p in goal_predictions if p.get("predicted_achievement_date"))
                avg_confidence = sum(p.get("confidence", 0.0) for p in goal_predictions) / len(goal_predictions)

                # Determine likelihood
                if positive_predictions == len(goal_predictions) and avg_confidence > 0.7:
                    likelihood = "high"
                elif positive_predictions >= len(goal_predictions) * 0.5 and avg_confidence > 0.5:
                    likelihood = "medium"
                else:
                    likelihood = "low"

                assessments[goal_name] = {
                    "likelihood": likelihood,
                    "confidence": avg_confidence,
                    "key_factors": [
                        f"{positive_predictions}/{len(goal_predictions)} metrics on track",
                        f"Average prediction confidence: {avg_confidence:.1%}"
                    ]
                }

            return assessments

        except Exception as e:
            self.logger.error(f"Error generating overall assessment: {e}")
            return {}

    async def generate_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive success metrics report"""
        try:
            current_time = datetime.now()

            # Gather all data
            current_metrics = await self.get_current_metrics()
            goal_progress = await self.get_goal_progress()
            predictions = await self.get_predictions()

            # Calculate overall success score
            overall_score = self._calculate_overall_success_score()

            report = {
                "report_metadata": {
                    "generated_at": current_time.isoformat(),
                    "report_type": report_type,
                    "version": "1.0"
                },
                "executive_summary": {
                    "overall_success_score": overall_score,
                    "primary_achievement": self._identify_primary_achievement(),
                    "critical_challenges": self._identify_critical_challenges(),
                    "strategic_recommendations": self._generate_strategic_recommendations()
                },
                "current_metrics": current_metrics,
                "goal_progress": goal_progress,
                "predictions": predictions,
                "detailed_analysis": {
                    "metric_trends": self.trend_analysis,
                    "recent_alerts": self.alerts,
                    "milestone_analysis": self._analyze_milestones()
                }
            }

            if report_type == "comprehensive":
                report["comprehensive_analysis"] = {
                    "life_work_impact": self._analyze_life_work_impact(),
                    "system_health": await self._analyze_system_health(),
                    "future_outlook": self._generate_future_outlook()
                }

            return report

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return {"error": str(e)}

    def _calculate_overall_success_score(self) -> float:
        """Calculate overall success score across all dimensions"""
        try:
            scores = {
                "scientific_acceleration": self._calculate_category_score(MetricCategory.SCIENTIFIC_ACCELERATION),
                "democratization": self._calculate_category_score(MetricCategory.DEMOCRATIZATION),
                "knowledge_advancement": self._calculate_category_score(MetricCategory.KNOWLEDGE_ADVANCEMENT),
                "impact_measurement": self._calculate_category_score(MetricCategory.IMPACT_MEASUREMENT),
                "system_evolution": self._calculate_category_score(MetricCategory.SYSTEM_EVOLUTION),
                "ethical_compliance": self._calculate_category_score(MetricCategory.ETHICAL_COMPLIANCE)
            }

            # Weighted overall score
            weights = {
                "scientific_acceleration": 0.25,
                "democratization": 0.20,
                "knowledge_advancement": 0.20,
                "impact_measurement": 0.15,
                "system_evolution": 0.10,
                "ethical_compliance": 0.10
            }

            overall_score = sum(scores[category] * weights[category] for category in scores)

            return min(100.0, max(0.0, overall_score))

        except Exception as e:
            self.logger.error(f"Error calculating overall success score: {e}")
            return 0.0

    def _calculate_category_score(self, category: MetricCategory) -> float:
        """Calculate score for a specific metric category"""
        try:
            category_metrics = [m for m in self.metrics_registry.values() if m.category == category]

            if not category_metrics:
                return 0.0

            total_score = 0.0
            total_weight = 0.0

            for metric in category_metrics:
                current_value = getattr(self.current_metrics, metric.name, 0.0)

                if metric.target_value > 0:
                    metric_score = min(100.0, (current_value / metric.target_value) * 100)
                else:
                    metric_score = 0.0

                total_score += metric_score * metric.weight
                total_weight += metric.weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating category score: {e}")
            return 0.0

    def _identify_primary_achievement(self) -> str:
        """Identify the primary achievement based on current metrics"""
        try:
            best_achievement = ""
            best_score = 0.0

            for category in MetricCategory:
                score = self._calculate_category_score(category)
                if score > best_score:
                    best_score = score
                    best_achievement = f"Strong performance in {category.value.replace('_', ' ').title()}"

            return best_achievement if best_achievement else "Steady progress across all dimensions"

        except Exception as e:
            self.logger.error(f"Error identifying primary achievement: {e}")
            return "Achievement analysis unavailable"

    def _identify_critical_challenges(self) -> List[str]:
        """Identify critical challenges based on current metrics"""
        try:
            challenges = []

            # Check for low-performing categories
            for category in MetricCategory:
                score = self._calculate_category_score(category)
                if score < 30.0:  # Below 30%
                    challenges.append(f"Low performance in {category.value.replace('_', ' ').title()}")

            # Check for high-priority alerts
            high_alerts = [alert for alert in self.alerts if alert.get("severity") == "high"]
            if high_alerts:
                challenges.append(f"Multiple critical alerts ({len(high_alerts)} urgent issues)")

            return challenges if challenges else ["No critical challenges identified"]

        except Exception as e:
            self.logger.error(f"Error identifying critical challenges: {e}")
            return ["Challenge analysis unavailable"]

    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on current performance"""
        try:
            recommendations = []

            # Analyze each goal
            for goal in self.lifes_work_goals:
                progress = self.goal_progress.get(goal.name, {})
                overall_progress = progress.get("overall_progress", 0.0)

                if overall_progress < 25:
                    recommendations.append(f"Accelerate efforts for {goal.name} - current progress only {overall_progress:.1f}%")
                elif overall_progress > 90:
                    recommendations.append(f"Prepare for achievement of {goal.name} - nearing target")

                # Check milestone progress
                milestones = progress.get("milestone_progress", [])
                for milestone in milestones:
                    if milestone.get("days_remaining", 0) < 30 and milestone.get("progress", 0) < 80:
                        recommendations.append(f"Urgent: {milestone['milestone']} milestone at risk")

            return recommendations if recommendations else ["Continue current strategic direction"]

        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {e}")
            return ["Recommendation generation unavailable"]

    def _analyze_milestones(self) -> Dict[str, Any]:
        """Analyze progress toward key milestones"""
        try:
            milestone_analysis = {}

            for goal in self.lifes_work_goals:
                progress = self.goal_progress.get(goal.name, {})
                milestones = progress.get("milestone_progress", [])

                milestone_analysis[goal.name] = {
                    "total_milestones": len(goal.milestones),
                    "completed_milestones": len([m for m in milestones if m.get("progress", 0) >= 100]),
                    "at_risk_milestones": len([m for m in milestones if m.get("days_remaining", 0) < 30 and m.get("progress", 0) < 80]),
                    "next_milestone": self._find_next_milestone(milestones)
                }

            return milestone_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing milestones: {e}")
            return {}

    def _find_next_milestone(self, milestones: List[Dict]) -> Optional[Dict]:
        """Find the next upcoming milestone"""
        try:
            upcoming = [m for m in milestones if m.get("progress", 0) < 100]

            if upcoming:
                # Sort by days remaining
                upcoming.sort(key=lambda m: m.get("days_remaining", float('inf')))
                return upcoming[0]

            return None

        except Exception as e:
            self.logger.error(f"Error finding next milestone: {e}")
            return None

    def _analyze_life_work_impact(self) -> Dict[str, Any]:
        """Analyze overall life work impact"""
        try:
            impact_metrics = {
                "scientific_acceleration": self._calculate_category_score(MetricCategory.SCIENTIFIC_ACCELERATION),
                "democratization_score": self._calculate_category_score(MetricCategory.DEMOCRATIZATION),
                "knowledge_generation": self._calculate_category_score(MetricCategory.KNOWLEDGE_ADVANCEMENT),
                "real_world_impact": self._calculate_category_score(MetricCategory.IMPACT_MEASUREMENT)
            }

            # Calculate life work progress
            life_work_progress = sum(impact_metrics.values()) / len(impact_metrics)

            return {
                "overall_impact_score": life_work_progress,
                "impact_dimensions": impact_metrics,
                "key_achievements": [
                    f"Research acceleration factor: {getattr(self.current_metrics, 'research_acceleration_factor', 0.0):.1f}x",
                    f"Global users reached: {getattr(self.current_metrics, 'global_users_count', 0):,}",
                    f"Breakthrough discoveries: {getattr(self.current_metrics, 'breakthrough_discovery_rate', 0.0):.1f}/month"
                ],
                "progress_toward_100x_goal": f"{life_work_progress:.1f}% complete"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing life work impact: {e}")
            return {"error": str(e)}

    async def _analyze_system_health(self) -> Dict[str, Any]:
        """Analyze overall system health"""
        try:
            # Get performance metrics
            perf_metrics = await self.performance_monitor.get_performance_summary()

            # Calculate system health score
            system_score = 0.0
            factors = 0

            if "system_metrics" in perf_metrics:
                system_metrics = perf_metrics["system_metrics"]
                cpu_health = max(0, 100 - system_metrics.get("cpu_usage", 100))
                memory_health = max(0, 100 - system_metrics.get("memory_usage", 100))
                system_score += (cpu_health + memory_health) / 2
                factors += 1

            # Add error rate health
            error_rate = getattr(self.current_metrics, 'system_error_rate', 0.0)
            error_health = max(0, 100 - (error_rate * 100))
            system_score += error_health
            factors += 1

            # Add ethical compliance health
            ethical_compliance = getattr(self.current_metrics, 'ethical_compliance_rate', 0.0)
            system_score += ethical_compliance
            factors += 1

            overall_health = system_score / factors if factors > 0 else 0

            return {
                "overall_health_score": overall_health,
                "health_status": "excellent" if overall_health > 90 else "good" if overall_health > 70 else "fair" if overall_health > 50 else "poor",
                "performance_health": perf_metrics,
                "alert_count": len(self.alerts),
                "uptime_estimate": "99.9%" if overall_health > 90 else "99.0%" if overall_health > 70 else "95.0%"
            }

        except Exception as e:
            self.logger.error(f"Error analyzing system health: {e}")
            return {"error": str(e)}

    def _generate_future_outlook(self) -> Dict[str, Any]:
        """Generate future outlook based on trends and predictions"""
        try:
            # Get predictions
            predictions_data = asyncio.run(self.get_predictions())
            predictions = predictions_data.get("predictions", {})
            assessment = predictions_data.get("overall_assessment", {})

            # Analyze trends
            positive_trends = sum(1 for trend in self.trend_analysis.values() if trend.get("trend") == "increasing")
            total_trends = len(self.trend_analysis)

            # Generate outlook
            if positive_trends / total_trends > 0.7 if total_trends > 0 else 0:
                outlook = "very positive"
                confidence = "high"
            elif positive_trends / total_trends > 0.4 if total_trends > 0 else 0:
                outlook = "positive"
                confidence = "medium"
            else:
                outlook = "cautious"
                confidence = "low"

            return {
                "overall_outlook": outlook,
                "confidence_level": confidence,
                "key_opportunities": [
                    "Continued AI technology advancement",
                    "Growing acceptance of AI in research",
                    "Expanding global research challenges"
                ],
                "potential_risks": [
                    "Technical limitations in AI capabilities",
                    "Ethical and regulatory challenges",
                    "Market competition and adoption barriers"
                ],
                "strategic_focus": self._recommend_strategic_focus(),
                "next_6_months": self._project_6month_outlook()
            }

        except Exception as e:
            self.logger.error(f"Error generating future outlook: {e}")
            return {"error": str(e)}

    def _recommend_strategic_focus(self) -> str:
        """Recommend strategic focus based on current performance"""
        try:
            # Find lowest performing category
            category_scores = {}
            for category in MetricCategory:
                score = self._calculate_category_score(category)
                category_scores[category] = score

            lowest_category = min(category_scores, key=category_scores.get)

            focus_areas = {
                MetricCategory.SCIENTIFIC_ACCELERATION: "Research output optimization and AI model enhancement",
                MetricCategory.DEMOCRATIZATION: "Global accessibility and user acquisition",
                MetricCategory.KNOWLEDGE_ADVANCEMENT: "Breakthrough discovery acceleration",
                MetricCategory.IMPACT_MEASUREMENT: "Real-world application and industry partnership development",
                MetricCategory.SYSTEM_EVOLUTION: "Autonomous improvement and self-learning capabilities",
                MetricCategory.ETHICAL_COMPLIANCE: "Ethical framework enhancement and compliance optimization"
            }

            return focus_areas.get(lowest_category, "Balanced development across all areas")

        except Exception as e:
            self.logger.error(f"Error recommending strategic focus: {e}")
            return "Balanced development strategy"

    def _project_6month_outlook(self) -> Dict[str, Any]:
        """Project expected outcomes for next 6 months"""
        try:
            projections = {}

            for goal in self.lifes_work_goals:
                progress = self.goal_progress.get(goal.name, {})
                current_progress = progress.get("overall_progress", 0.0)

                # Project based on current trajectory
                monthly_growth_rate = self._estimate_growth_rate(goal.name)

                projected_progress = min(100.0, current_progress + (monthly_growth_rate * 6))

                projections[goal.name] = {
                    "current_progress": current_progress,
                    "projected_progress": projected_progress,
                    "monthly_growth_rate": monthly_growth_rate,
                    "expected_achievement_date": self._estimate_achievement_date(goal.name, current_progress, monthly_growth_rate)
                }

            return projections

        except Exception as e:
            self.logger.error(f"Error projecting 6-month outlook: {e}")
            return {}

    def _estimate_growth_rate(self, goal_name: str) -> float:
        """Estimate monthly growth rate for a goal"""
        try:
            # This is a simplified estimation
            # In practice, this would use historical trend analysis
            return 5.0  # 5% monthly growth as default

        except Exception as e:
            self.logger.error(f"Error estimating growth rate: {e}")
            return 0.0

    def _estimate_achievement_date(self, goal_name: str, current_progress: float, growth_rate: float) -> Optional[str]:
        """Estimate when a goal will be achieved"""
        try:
            if growth_rate <= 0 or current_progress >= 100:
                return None

            months_needed = (100 - current_progress) / growth_rate
            achievement_date = datetime.now() + timedelta(days=months_needed * 30)

            return achievement_date.isoformat()

        except Exception as e:
            self.logger.error(f"Error estimating achievement date: {e}")
            return None

    async def shutdown(self):
        """Shutdown the Success Metrics Engine"""
        try:
            self.monitoring_active = False

            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

            await self.performance_monitor.shutdown()

            self.logger.info("Success Metrics Engine shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down Success Metrics Engine: {e}")


# Global instance for easy access
_success_metrics_engine: Optional[SuccessMetricsEngine] = None


def get_success_metrics_engine() -> SuccessMetricsEngine:
    """Get the global Success Metrics Engine instance"""
    global _success_metrics_engine
    if _success_metrics_engine is None:
        _success_metrics_engine = SuccessMetricsEngine()
    return _success_metrics_engine


def initialize_success_metrics(config: Dict[str, Any] = None) -> SuccessMetricsEngine:
    """Initialize the global Success Metrics Engine"""
    global _success_metrics_engine
    _success_metrics_engine = SuccessMetricsEngine(config)
    return _success_metrics_engine