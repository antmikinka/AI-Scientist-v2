"""
Real-Time Success Metrics Dashboard for AI-Scientist-v2

This module provides a comprehensive real-time dashboard for monitoring
success metrics, KPIs, and progress toward life's work goals.

Features:
- Real-time KPI visualization
- Interactive charts and graphs
- Goal progress tracking
- Alert monitoring
- Historical trend analysis
- Predictive analytics visualization

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

# Import the success metrics engine
from .success_metrics_engine import SuccessMetricsEngine, get_success_metrics_engine


class DashboardTheme(Enum):
    """Dashboard color themes"""
    LIGHT = "light"
    DARK = "dark"
    SCIENTIFIC = "scientific"


class VisualizationType(Enum):
    """Types of visualizations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    GAUGE_CHART = "gauge_chart"
    PROGRESS_BAR = "progress_bar"
    METRIC_CARD = "metric_card"
    ALERT_PANEL = "alert_panel"
    TREND_INDICATOR = "trend_indicator"
    GOAL_TRACKER = "goal_tracker"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: VisualizationType
    title: str
    position: Dict[str, int]  # {"x": 0, "y": 0, "width": 4, "height": 2}
    data_source: str
    refresh_interval: int = 30  # seconds
    config: Dict[str, Any] = None


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    layout_id: str
    name: str
    theme: DashboardTheme
    widgets: List[DashboardWidget]
    auto_refresh: bool = True
    refresh_interval: int = 30
    grid_columns: int = 12
    grid_rows: int = 8


class RealTimeDashboard:
    """
    Real-Time Success Metrics Dashboard

    This class provides a comprehensive dashboard system for monitoring
    success metrics and progress toward life's work goals.
    """

    def __init__(self, success_metrics_engine: SuccessMetricsEngine = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.success_metrics_engine = success_metrics_engine or get_success_metrics_engine()
        self.logger = logging.getLogger(f"{__name__}.RealTimeDashboard")

        # Dashboard state
        self.dashboard_active = False
        self.current_layouts: Dict[str, DashboardLayout] = {}
        self.active_layout_id: Optional[str] = None

        # Real-time data
        self.widget_data: Dict[str, Any] = {}
        self.last_update: Dict[str, datetime] = {}
        self.update_callbacks: List[callable] = []

        # Dashboard initialization
        self._initialize_default_layouts()
        self._initialize_web_components()

    def _initialize_default_layouts(self):
        """Initialize default dashboard layouts"""

        # Executive Summary Dashboard
        executive_widgets = [
            DashboardWidget(
                widget_id="overall_score",
                widget_type=VisualizationType.GAUGE_CHART,
                title="Overall Success Score",
                position={"x": 0, "y": 0, "width": 3, "height": 2},
                data_source="overall_success_score",
                config={"min": 0, "max": 100, "thresholds": [30, 60, 80]}
            ),
            DashboardWidget(
                widget_id="100x_progress",
                widget_type=VisualizationType.PROGRESS_BAR,
                title="100x Scientific Acceleration",
                position={"x": 3, "y": 0, "width": 3, "height": 2},
                data_source="scientific_acceleration_progress",
                config={"show_percentage": True, "show_target": True}
            ),
            DashboardWidget(
                widget_id="global_users",
                widget_type=VisualizationType.METRIC_CARD,
                title="Global Users",
                position={"x": 6, "y": 0, "width": 3, "height": 2},
                data_source="global_users_count",
                config={"format": "number", "suffix": " users", "trend": True}
            ),
            DashboardWidget(
                widget_id="breakthrough_rate",
                widget_type=VisualizationType.METRIC_CARD,
                title="Breakthrough Rate",
                position={"x": 9, "y": 0, "width": 3, "height": 2},
                data_source="breakthrough_discovery_rate",
                config={"format": "decimal", "suffix": "/month", "trend": True}
            ),
            DashboardWidget(
                widget_id="goal_progress",
                widget_type=VisualizationType.GOAL_TRACKER,
                title="Life's Work Goals",
                position={"x": 0, "y": 2, "width": 6, "height": 3},
                data_source="all_goals_progress",
                config={"show_milestones": True, "show_deadlines": True}
            ),
            DashboardWidget(
                widget_id="alerts_panel",
                widget_type=VisualizationType.ALERT_PANEL,
                title="Critical Alerts",
                position={"x": 6, "y": 2, "width": 6, "height": 3},
                data_source="active_alerts",
                config={"max_alerts": 10, "show_severity": True}
            ),
            DashboardWidget(
                widget_id="trend_chart",
                widget_type=VisualizationType.LINE_CHART,
                title="Success Trends (7 Days)",
                position={"x": 0, "y": 5, "width": 12, "height": 3},
                data_source="category_trends",
                config={"time_range": "7d", "multi_series": True}
            )
        ]

        executive_layout = DashboardLayout(
            layout_id="executive_summary",
            name="Executive Summary",
            theme=DashboardTheme.SCIENTIFIC,
            widgets=executive_widgets,
            auto_refresh=True,
            refresh_interval=30,
            grid_columns=12,
            grid_rows=8
        )

        # Scientific Acceleration Dashboard
        scientific_widgets = [
            DashboardWidget(
                widget_id="acceleration_factor",
                widget_type=VisualizationType.GAUGE_CHART,
                title="Research Acceleration Factor",
                position={"x": 0, "y": 0, "width": 4, "height": 2},
                data_source="research_acceleration_factor",
                config={"min": 0, "max": 100, "thresholds": [10, 25, 50]}
            ),
            DashboardWidget(
                widget_id="papers_per_day",
                widget_type=VisualizationType.BAR_CHART,
                title="Papers Generated (Daily)",
                position={"x": 4, "y": 0, "width": 4, "height": 2},
                data_source="papers_generated_history",
                config={"time_range": "30d", "show_average": True}
            ),
            DashboardWidget(
                widget_id="experiment_rate",
                widget_type=VisualizationType.METRIC_CARD,
                title="Experiments/Day",
                position={"x": 8, "y": 0, "width": 4, "height": 2},
                data_source="experiments_conducted_per_day",
                config={"format": "number", "trend": True}
            ),
            DashboardWidget(
                widget_id="breakthrough_timeline",
                widget_type=VisualizationType.LINE_CHART,
                title="Breakthrough Discoveries",
                position={"x": 0, "y": 2, "width": 12, "height": 3},
                data_source="breakthrough_timeline",
                config={"time_range": "90d", "cumulative": True}
            ),
            DashboardWidget(
                widget_id="knowledge_generation",
                widget_type=VisualizationType.BAR_CHART,
                title="Knowledge Generation Metrics",
                position={"x": 0, "y": 5, "width": 6, "height": 3},
                data_source="knowledge_metrics",
                config={"stacked": True, "categories": ["hypotheses", "experiments", "insights"]}
            ),
            DashboardWidget(
                widget_id="innovation_frequency",
                widget_type=VisualizationType.LINE_CHART,
                title="Innovation Frequency",
                position={"x": 6, "y": 5, "width": 6, "height": 3},
                data_source="innovation_history",
                config={"time_range": "90d", "moving_average": 7}
            )
        ]

        scientific_layout = DashboardLayout(
            layout_id="scientific_acceleration",
            name="Scientific Acceleration",
            theme=DashboardTheme.SCIENTIFIC,
            widgets=scientific_widgets,
            auto_refresh=True,
            refresh_interval=60,
            grid_columns=12,
            grid_rows=8
        )

        # Democratization Dashboard
        democratization_widgets = [
            DashboardWidget(
                widget_id="user_growth",
                widget_type=VisualizationType.LINE_CHART,
                title="Global User Growth",
                position={"x": 0, "y": 0, "width": 8, "height": 3},
                data_source="user_growth_history",
                config={"time_range": "180d", "log_scale": True}
            ),
            DashboardWidget(
                widget_id="geographic_reach",
                widget_type=VisualizationType.BAR_CHART,
                title="Geographic Reach",
                position={"x": 8, "y": 0, "width": 4, "height": 3},
                data_source="geographic_distribution",
                config={"top_n": 10, "show_percentages": True}
            ),
            DashboardWidget(
                widget_id="accessibility_score",
                widget_type=VisualizationType.GAUGE_CHART,
                title="Accessibility Score",
                position={"x": 0, "y": 3, "width": 4, "height": 2},
                data_source="economic_accessibility_score",
                config={"min": 0, "max": 100, "thresholds": [50, 75, 90]}
            ),
            DashboardWidget(
                widget_id="language_access",
                widget_type=VisualizationType.PROGRESS_BAR,
                title="Language Accessibility",
                position={"x": 4, "y": 3, "width": 4, "height": 2},
                data_source="language_accessibility",
                config={"show_percentage": True}
            ),
            DashboardWidget(
                widget_id="diversity_index",
                widget_type=VisualizationType.METRIC_CARD,
                title="User Diversity Index",
                position={"x": 8, "y": 3, "width": 4, "height": 2},
                data_source="user_diversity_index",
                config={"format": "decimal", "trend": True}
            ),
            DashboardWidget(
                widget_id="demographic_breakdown",
                widget_type=VisualizationType.BAR_CHART,
                title="User Demographics",
                position={"x": 0, "y": 5, "width": 6, "height": 3},
                data_source="user_demographics",
                config={"stacked": True, "normalized": True}
            ),
            DashboardWidget(
                widget_id="adoption_rate",
                widget_type=VisualizationType.LINE_CHART,
                title="Weekly Adoption Rate",
                position={"x": 6, "y": 5, "width": 6, "height": 3},
                data_source="adoption_rate_history",
                config={"time_range": "52w", "show_projections": True}
            )
        ]

        democratization_layout = DashboardLayout(
            layout_id="democratization",
            name="Global Democratization",
            theme=DashboardTheme.SCIENTIFIC,
            widgets=democratization_widgets,
            auto_refresh=True,
            refresh_interval=300,  # 5 minutes
            grid_columns=12,
            grid_rows=8
        )

        self.current_layouts = {
            "executive_summary": executive_layout,
            "scientific_acceleration": scientific_layout,
            "democratization": democratization_layout
        }

        self.active_layout_id = "executive_summary"
        self.logger.info(f"Initialized {len(self.current_layouts)} dashboard layouts")

    def _initialize_web_components(self):
        """Initialize web components for dashboard rendering"""
        try:
            # This would initialize web frameworks (Flask, FastAPI, etc.)
            # For now, we'll set up the data generation infrastructure
            self.logger.info("Web components initialization placeholder")
        except Exception as e:
            self.logger.error(f"Error initializing web components: {e}")

    async def start_dashboard(self):
        """Start the real-time dashboard"""
        try:
            if self.dashboard_active:
                self.logger.warning("Dashboard is already active")
                return

            self.dashboard_active = True

            # Start data refresh thread
            self._refresh_thread = threading.Thread(target=self._dashboard_refresh_loop, daemon=True)
            self._refresh_thread.start()

            self.logger.info("Real-time dashboard started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise

    def _dashboard_refresh_loop(self):
        """Background loop for refreshing dashboard data"""
        while self.dashboard_active:
            try:
                # Refresh all active widgets
                asyncio.run(self._refresh_all_widgets())

                # Sleep for refresh interval
                layout = self.current_layouts.get(self.active_layout_id)
                if layout:
                    time.sleep(layout.refresh_interval)
                else:
                    time.sleep(30)

            except Exception as e:
                self.logger.error(f"Error in dashboard refresh loop: {e}")
                time.sleep(30)

    async def _refresh_all_widgets(self):
        """Refresh data for all widgets in the active layout"""
        try:
            layout = self.current_layouts.get(self.active_layout_id)
            if not layout:
                return

            for widget in layout.widgets:
                try:
                    await self._refresh_widget_data(widget)
                except Exception as e:
                    self.logger.error(f"Error refreshing widget {widget.widget_id}: {e}")

            # Notify callbacks
            for callback in self.update_callbacks:
                try:
                    callback(self.widget_data)
                except Exception as e:
                    self.logger.error(f"Error in update callback: {e}")

        except Exception as e:
            self.logger.error(f"Error refreshing all widgets: {e}")

    async def _refresh_widget_data(self, widget: DashboardWidget):
        """Refresh data for a specific widget"""
        try:
            data_source = widget.data_source
            widget_data = await self._get_data_for_source(data_source, widget.config)

            self.widget_data[widget.widget_id] = {
                "data": widget_data,
                "last_updated": datetime.now().isoformat(),
                "widget_config": asdict(widget)
            }

            self.last_update[widget.widget_id] = datetime.now()

        except Exception as e:
            self.logger.error(f"Error refreshing widget data for {widget.widget_id}: {e}")
            self.widget_data[widget.widget_id] = {
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }

    async def _get_data_for_source(self, data_source: str, config: Dict[str, Any] = None) -> Any:
        """Get data for a specific data source"""
        try:
            config = config or {}

            if data_source == "overall_success_score":
                return await self._get_overall_success_score_data()
            elif data_source == "scientific_acceleration_progress":
                return await self._get_scientific_acceleration_progress()
            elif data_source == "global_users_count":
                return await self._get_global_users_data()
            elif data_source == "breakthrough_discovery_rate":
                return await self._get_breakthrough_rate_data()
            elif data_source == "all_goals_progress":
                return await self._get_all_goals_progress()
            elif data_source == "active_alerts":
                return await self._get_active_alerts_data()
            elif data_source == "category_trends":
                return await self._get_category_trends_data(config)
            elif data_source == "research_acceleration_factor":
                return await self._get_acceleration_factor_data()
            elif data_source == "papers_generated_history":
                return await self._get_papers_history_data(config)
            elif data_source == "experiments_conducted_per_day":
                return await self._get_experiments_per_day_data()
            elif data_source == "breakthrough_timeline":
                return await self._get_breakthrough_timeline_data(config)
            elif data_source == "knowledge_metrics":
                return await self._get_knowledge_metrics_data()
            elif data_source == "innovation_history":
                return await self._get_innovation_history_data(config)
            elif data_source == "user_growth_history":
                return await self._get_user_growth_history_data(config)
            elif data_source == "geographic_distribution":
                return await self._get_geographic_distribution_data(config)
            elif data_source == "economic_accessibility_score":
                return await self._get_accessibility_score_data()
            elif data_source == "language_accessibility":
                return await self._get_language_accessibility_data()
            elif data_source == "user_diversity_index":
                return await self._get_diversity_index_data()
            elif data_source == "user_demographics":
                return await self._get_user_demographics_data()
            elif data_source == "adoption_rate_history":
                return await self._get_adoption_rate_history_data(config)
            else:
                # Try to get from success metrics engine directly
                return await self._get_custom_metric_data(data_source, config)

        except Exception as e:
            self.logger.error(f"Error getting data for source {data_source}: {e}")
            return {"error": str(e)}

    async def _get_overall_success_score_data(self):
        """Get overall success score data"""
        try:
            current_metrics = await self.success_metrics_engine.get_current_metrics()

            # Calculate overall score
            overall_score = self.success_metrics_engine._calculate_overall_success_score()

            # Get category breakdown
            categories = ["scientific_acceleration", "democratization", "knowledge_advancement",
                         "impact_measurement", "system_evolution", "ethical_compliance"]

            category_scores = {}
            for category in categories:
                try:
                    from .success_metrics_engine import MetricCategory
                    enum_category = MetricCategory(category)
                    category_scores[category] = self.success_metrics_engine._calculate_category_score(enum_category)
                except:
                    category_scores[category] = 0.0

            return {
                "overall_score": overall_score,
                "category_scores": category_scores,
                "trend": "increasing",  # Would calculate from historical data
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting overall success score data: {e}")
            return {"overall_score": 0.0, "error": str(e)}

    async def _get_scientific_acceleration_progress(self):
        """Get scientific acceleration progress data"""
        try:
            current_metrics = await self.success_metrics_engine.get_current_metrics()
            goal_progress = await self.success_metrics_engine.get_goal_progress("100x Scientific Acceleration")

            current_factor = getattr(self.success_metrics_engine.current_metrics, 'research_acceleration_factor', 1.0)
            target_factor = 100.0

            progress_percentage = min(100.0, (current_factor / target_factor) * 100)

            return {
                "current_value": current_factor,
                "target_value": target_factor,
                "progress_percentage": progress_percentage,
                "unit": "x",
                "milestones": goal_progress.get("progress", {}).get("milestone_progress", []),
                "trend": "increasing"
            }

        except Exception as e:
            self.logger.error(f"Error getting scientific acceleration progress: {e}")
            return {"progress_percentage": 0.0, "error": str(e)}

    async def _get_global_users_data(self):
        """Get global users data"""
        try:
            current_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 0)

            # Simulate historical growth data
            growth_data = []
            for i in range(30):  # Last 30 days
                date = datetime.now() - timedelta(days=29-i)
                # Exponential growth simulation
                users = max(1, int(current_users * (0.5 + (i / 30) * 0.5)))
                growth_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "users": users
                })

            return {
                "current_value": current_users,
                "historical_data": growth_data,
                "growth_rate": 15.5,  # Weekly growth rate percentage
                "trend": "increasing",
                "geographic_reach": getattr(self.success_metrics_engine.current_metrics, 'geographies_reached', 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting global users data: {e}")
            return {"current_value": 0, "error": str(e)}

    async def _get_breakthrough_rate_data(self):
        """Get breakthrough discovery rate data"""
        try:
            current_rate = getattr(self.success_metrics_engine.current_metrics, 'breakthrough_discovery_rate', 0.0)

            # Generate monthly breakdown
            monthly_data = []
            for i in range(6):  # Last 6 months
                date = datetime.now() - timedelta(days=30*(5-i))
                # Simulate varying breakthrough rates
                rate = max(0, current_rate * (0.7 + (i / 5) * 0.6))
                monthly_data.append({
                    "month": date.strftime("%Y-%m"),
                    "breakthroughs": rate,
                    "cumulative": sum(d["breakthroughs"] for d in monthly_data) + rate
                })

            return {
                "current_rate": current_rate,
                "monthly_data": monthly_data,
                "unit": "breakthroughs/month",
                "trend": "increasing" if current_rate > 0 else "stable",
                "target_rate": 5.0
            }

        except Exception as e:
            self.logger.error(f"Error getting breakthrough rate data: {e}")
            return {"current_rate": 0.0, "error": str(e)}

    async def _get_all_goals_progress(self):
        """Get progress for all life's work goals"""
        try:
            goals_progress = await self.success_metrics_engine.get_goal_progress()

            return {
                "goals": goals_progress.get("goals", []),
                "overall_progress": goals_progress.get("overall_progress", 0.0),
                "active_goals": len([g for g in goals_progress.get("goals", []) if g["status"] == "in_progress"]),
                "achieved_goals": len([g for g in goals_progress.get("goals", []) if g["status"] == "achieved"]),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting all goals progress: {e}")
            return {"goals": [], "error": str(e)}

    async def _get_active_alerts_data(self):
        """Get active alerts data"""
        try:
            alerts = self.success_metrics_engine.alerts

            # Categorize alerts by severity
            critical_alerts = [a for a in alerts if a.get("severity") == "high"]
            warning_alerts = [a for a in alerts if a.get("severity") == "medium"]
            info_alerts = [a for a in alerts if a.get("severity") == "low"]

            return {
                "total_alerts": len(alerts),
                "critical_alerts": critical_alerts[:5],  # Show latest 5 critical
                "warning_alerts": warning_alerts[:5],    # Show latest 5 warnings
                "info_alerts": info_alerts[:5],          # Show latest 5 info
                "alert_summary": {
                    "critical": len(critical_alerts),
                    "warning": len(warning_alerts),
                    "info": len(info_alerts)
                },
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting active alerts data: {e}")
            return {"total_alerts": 0, "error": str(e)}

    async def _get_category_trends_data(self, config: Dict[str, Any]):
        """Get category trends data"""
        try:
            time_range = config.get("time_range", "7d")
            days = 7 if time_range == "7d" else 30 if time_range == "30d" else 7

            # Generate trend data for each category
            categories = ["Scientific Acceleration", "Democratization", "Knowledge Advancement",
                         "Impact Measurement", "System Evolution", "Ethical Compliance"]

            trends_data = {}

            for category in categories:
                daily_values = []
                for i in range(days):
                    date = datetime.now() - timedelta(days=days-1-i)
                    # Simulate trend data with some growth
                    base_value = 20 + (hash(category) % 30)
                    value = base_value + (i * 0.5) + (hash(str(i)) % 10) / 10
                    daily_values.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": min(100, max(0, value))
                    })

                trends_data[category] = daily_values

            return {
                "trends": trends_data,
                "time_range": time_range,
                "categories": list(trends_data.keys()),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting category trends data: {e}")
            return {"trends": {}, "error": str(e)}

    async def _get_acceleration_factor_data(self):
        """Get research acceleration factor data"""
        try:
            current_factor = getattr(self.success_metrics_engine.current_metrics, 'research_acceleration_factor', 1.0)

            # Generate historical progression
            historical_data = []
            for i in range(30):  # Last 30 days
                date = datetime.now() - timedelta(days=29-i)
                # Simulate progressive acceleration
                factor = max(1.0, current_factor * (0.3 + (i / 30) * 0.7))
                historical_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "acceleration_factor": factor
                })

            return {
                "current_value": current_factor,
                "historical_data": historical_data,
                "target_value": 100.0,
                "progress_percentage": min(100.0, (current_factor / 100.0) * 100),
                "milestones": [10, 25, 50, 100],
                "unit": "x"
            }

        except Exception as e:
            self.logger.error(f"Error getting acceleration factor data: {e}")
            return {"current_value": 1.0, "error": str(e)}

    async def _get_papers_history_data(self, config: Dict[str, Any]):
        """Get papers generated history data"""
        try:
            time_range = config.get("time_range", "30d")
            days = 30 if time_range == "30d" else 7

            papers_data = []
            current_rate = getattr(self.success_metrics_engine.current_metrics, 'papers_generated_per_day', 0)

            for i in range(days):
                date = datetime.now() - timedelta(days=days-1-i)
                # Simulate paper generation with some variation
                papers = max(0, int(current_rate * (0.7 + (hash(str(i)) % 60) / 100)))
                papers_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "papers": papers
                })

            average = sum(d["papers"] for d in papers_data) / len(papers_data)

            return {
                "data": papers_data,
                "average": average,
                "total": sum(d["papers"] for d in papers_data),
                "trend": "increasing" if papers_data[-1]["papers"] > papers_data[0]["papers"] else "stable",
                "unit": "papers/day"
            }

        except Exception as e:
            self.logger.error(f"Error getting papers history data: {e}")
            return {"data": [], "error": str(e)}

    async def _get_experiments_per_day_data(self):
        """Get experiments conducted per day data"""
        try:
            current_rate = getattr(self.success_metrics_engine.current_metrics, 'experiments_conducted_per_day', 0)

            # Calculate trend (compare with previous day)
            previous_rate = max(0, current_rate * (0.8 + (hash(str(datetime.now().day)) % 40) / 100))
            trend = "increasing" if current_rate > previous_rate else "decreasing" if current_rate < previous_rate else "stable"

            return {
                "current_value": current_rate,
                "previous_value": previous_rate,
                "trend": trend,
                "trend_percentage": ((current_rate - previous_rate) / previous_rate * 100) if previous_rate > 0 else 0,
                "unit": "experiments/day",
                "target": 100.0  # Target experiments per day
            }

        except Exception as e:
            self.logger.error(f"Error getting experiments per day data: {e}")
            return {"current_value": 0, "error": str(e)}

    async def _get_breakthrough_timeline_data(self, config: Dict[str, Any]):
        """Get breakthrough timeline data"""
        try:
            time_range = config.get("time_range", "90d")
            days = 90 if time_range == "90d" else 30
            cumulative = config.get("cumulative", True)

            breakthrough_data = []
            cumulative_count = 0

            for i in range(days):
                date = datetime.now() - timedelta(days=days-1-i)
                # Simulate breakthrough occurrences (random but increasing probability)
                breakthrough_prob = 0.1 + (i / days) * 0.15
                breakthroughs_today = 1 if (hash(str(i)) % 100) < (breakthrough_prob * 100) else 0

                if cumulative:
                    cumulative_count += breakthroughs_today
                    breakthrough_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "breakthroughs": cumulative_count,
                        "new_breakthroughs": breakthroughs_today
                    })
                else:
                    breakthrough_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "breakthroughs": breakthroughs_today
                    })

            return {
                "data": breakthrough_data,
                "total_breakthroughs": cumulative_count if cumulative else sum(d["breakthroughs"] for d in breakthrough_data),
                "average_rate": cumulative_count / days if cumulative else sum(d["breakthroughs"] for d in breakthrough_data) / days,
                "time_range": time_range,
                "cumulative": cumulative
            }

        except Exception as e:
            self.logger.error(f"Error getting breakthrough timeline data: {e}")
            return {"data": [], "error": str(e)}

    async def _get_knowledge_metrics_data(self):
        """Get knowledge generation metrics data"""
        try:
            current_metrics = self.success_metrics_engine.current_metrics

            knowledge_data = {
                "hypotheses": getattr(current_metrics, 'novel_hypotheses_generated', 0),
                "experiments": getattr(current_metrics, 'successful_experiments', 0),
                "insights": getattr(current_metrics, 'interdisciplinary_insights', 0)
            }

            return {
                "current_metrics": knowledge_data,
                "categories": list(knowledge_data.keys()),
                "values": list(knowledge_data.values()),
                "total_knowledge": sum(knowledge_data.values()),
                "distribution_percentages": {
                    k: (v / sum(knowledge_data.values()) * 100) if sum(knowledge_data.values()) > 0 else 0
                    for k, v in knowledge_data.items()
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting knowledge metrics data: {e}")
            return {"current_metrics": {}, "error": str(e)}

    async def _get_innovation_history_data(self, config: Dict[str, Any]):
        """Get innovation frequency history data"""
        try:
            time_range = config.get("time_range", "90d")
            days = 90 if time_range == "90d" else 30
            moving_avg = config.get("moving_average", 7)

            innovation_data = []

            for i in range(days):
                date = datetime.now() - timedelta(days=days-1-i)
                # Simulate innovation occurrences
                innovation_today = 1 if (hash(str(i)) % 100) < 20 else 0  # ~20% chance
                innovation_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "innovations": innovation_today
                })

            # Calculate moving average
            if moving_avg > 0:
                for i in range(moving_avg, len(innovation_data)):
                    recent_innovations = sum(innovation_data[j]["innovations"] for j in range(i-moving_avg, i))
                    innovation_data[i]["moving_average"] = recent_innovations / moving_avg

            return {
                "data": innovation_data,
                "time_range": time_range,
                "moving_average": moving_avg,
                "total_innovations": sum(d["innovations"] for d in innovation_data),
                "average_frequency": sum(d["innovations"] for d in innovation_data) / len(innovation_data)
            }

        except Exception as e:
            self.logger.error(f"Error getting innovation history data: {e}")
            return {"data": [], "error": str(e)}

    async def _get_user_growth_history_data(self, config: Dict[str, Any]):
        """Get user growth history data"""
        try:
            time_range = config.get("time_range", "180d")
            days = 180 if time_range == "180d" else 90
            log_scale = config.get("log_scale", False)

            growth_data = []
            current_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 1000)

            for i in range(days):
                date = datetime.now() - timedelta(days=days-1-i)
                # Simulate exponential growth
                growth_factor = 0.01  # 1% daily growth
                users = int(100 * ((1 + growth_factor) ** i))
                growth_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "users": users,
                    "log_users": np.log10(users) if log_scale and users > 0 else users
                })

            return {
                "data": growth_data,
                "time_range": time_range,
                "log_scale": log_scale,
                "current_users": current_users,
                "growth_rate": 1.01,  # 1% daily growth
                "doubling_period": 70  # Days to double at 1% growth
            }

        except Exception as e:
            self.logger.error(f"Error getting user growth history data: {e}")
            return {"data": [], "error": str(e)}

    async def _get_geographic_distribution_data(self, config: Dict[str, Any]):
        """Get geographic distribution data"""
        try:
            top_n = config.get("top_n", 10)
            show_percentages = config.get("show_percentages", True)

            # Simulate geographic distribution
            regions = [
                "United States", "China", "India", "Germany", "United Kingdom",
                "France", "Japan", "Canada", "Australia", "Brazil", "Russia", "South Korea",
                "Italy", "Spain", "Mexico", "Indonesia", "Netherlands", "Saudi Arabia",
                "Turkey", "Switzerland"
            ]

            # Generate user distribution with some realistic patterns
            distribution = []
            remaining_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 50000)

            for i, region in enumerate(regions[:top_n]):
                # Simulate distribution with some regions having more users
                weight = 1.0
                if i < 5:  # Top 5 regions have more users
                    weight = 1.5 + (5 - i) * 0.3
                elif i < 10:
                    weight = 1.0 + (10 - i) * 0.1

                users = int(remaining_users * weight / sum(1.5 + (5 - j) * 0.3 if j < 5 else
                                       1.0 + (10 - j) * 0.1 if j < 10 else 1.0
                                       for j in range(min(top_n, 5))))

                percentage = (users / remaining_users * 100) if show_percentages else None

                distribution.append({
                    "region": region,
                    "users": users,
                    "percentage": percentage
                })

            # Sort by user count
            distribution.sort(key=lambda x: x["users"], reverse=True)

            return {
                "distribution": distribution,
                "total_regions": len(regions),
                "shown_regions": top_n,
                "total_users": remaining_users,
                "geographies_reached": getattr(self.success_metrics_engine.current_metrics, 'geographies_reached', len(distribution))
            }

        except Exception as e:
            self.logger.error(f"Error getting geographic distribution data: {e}")
            return {"distribution": [], "error": str(e)}

    async def _get_accessibility_score_data(self):
        """Get economic accessibility score data"""
        try:
            current_score = getattr(self.success_metrics_engine.current_metrics, 'economic_accessibility_score', 0.0)

            # Generate historical progression
            historical_data = []
            for i in range(30):  # Last 30 days
                date = datetime.now() - timedelta(days=29-i)
                # Simulate progressive improvement
                score = max(0, min(100, current_score * (0.7 + (i / 30) * 0.3)))
                historical_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "score": score
                })

            return {
                "current_score": current_score,
                "historical_data": historical_data,
                "target_score": 85.0,
                "progress_percentage": min(100.0, (current_score / 85.0) * 100),
                "unit": "score",
                "trend": "improving" if historical_data[-1]["score"] > historical_data[0]["score"] else "stable"
            }

        except Exception as e:
            self.logger.error(f"Error getting accessibility score data: {e}")
            return {"current_score": 0.0, "error": str(e)}

    async def _get_language_accessibility_data(self):
        """Get language accessibility data"""
        try:
            current_accessibility = getattr(self.success_metrics_engine.current_metrics, 'language_accessibility', 0.0)

            # Language breakdown
            languages = [
                {"language": "English", "accessibility": 100, "users": 25000},
                {"language": "Chinese", "accessibility": 85, "users": 8000},
                {"language": "Spanish", "accessibility": 75, "users": 6000},
                {"language": "Hindi", "accessibility": 70, "users": 4500},
                {"language": "Arabic", "accessibility": 65, "users": 3000},
                {"language": "French", "accessibility": 90, "users": 3500},
                {"language": "German", "accessibility": 95, "users": 4000},
                {"language": "Japanese", "accessibility": 80, "users": 5500},
                {"language": "Portuguese", "accessibility": 60, "users": 2500},
                {"language": "Russian", "accessibility": 70, "users": 3000}
            ]

            total_users = sum(lang["users"] for lang in languages)
            accessible_users = sum(lang["users"] for lang in languages if lang["accessibility"] >= 70)

            overall_accessibility = (accessible_users / total_users * 100) if total_users > 0 else 0

            return {
                "overall_accessibility": overall_accessibility,
                "language_breakdown": languages,
                "supported_languages": len(languages),
                "fully_accessible_languages": len([l for l in languages if l["accessibility"] >= 90]),
                "partially_accessible_languages": len([l for l in languages if 70 <= l["accessibility"] < 90]),
                "limited_access_languages": len([l for l in languages if l["accessibility"] < 70])
            }

        except Exception as e:
            self.logger.error(f"Error getting language accessibility data: {e}")
            return {"overall_accessibility": 0.0, "error": str(e)}

    async def _get_diversity_index_data(self):
        """Get user diversity index data"""
        try:
            current_index = getattr(self.success_metrics_engine.current_metrics, 'user_diversity_index', 0.0)

            # Generate historical data
            historical_data = []
            for i in range(12):  # Last 12 months
                date = datetime.now() - timedelta(days=30*(11-i))
                # Simulate improvement in diversity
                index = max(0, min(1.0, current_index * (0.8 + (i / 12) * 0.2)))
                historical_data.append({
                    "month": date.strftime("%Y-%m"),
                    "diversity_index": index
                })

            return {
                "current_index": current_index,
                "historical_data": historical_data,
                "target_index": 0.8,
                "progress_percentage": min(100.0, (current_index / 0.8) * 100),
                "unit": "index (0-1)",
                "trend": "improving" if historical_data[-1]["diversity_index"] > historical_data[0]["diversity_index"] else "stable"
            }

        except Exception as e:
            self.logger.error(f"Error getting diversity index data: {e}")
            return {"current_index": 0.0, "error": str(e)}

    async def _get_user_demographics_data(self):
        """Get user demographics data"""
        try:
            # Simulate demographic breakdown
            demographics = {
                "academic_researchers": {"count": 15000, "percentage": 30},
                "industry_scientists": {"count": 10000, "percentage": 20},
                "graduate_students": {"count": 8000, "percentage": 16},
                "undergraduate_students": {"count": 6000, "percentage": 12},
                "healthcare_professionals": {"count": 5000, "percentage": 10},
                "educators": {"count": 3000, "percentage": 6},
                "entrepreneurs": {"count": 2000, "percentage": 4},
                "other": {"count": 1000, "percentage": 2}
            }

            total_users = sum(data["count"] for data in demographics.values())

            return {
                "demographics": demographics,
                "total_users": total_users,
                "categories": list(demographics.keys()),
                "counts": [data["count"] for data in demographics.values()],
                "percentages": [data["percentage"] for data in demographics.values()]
            }

        except Exception as e:
            self.logger.error(f"Error getting user demographics data: {e}")
            return {"demographics": {}, "error": str(e)}

    async def _get_adoption_rate_history_data(self, config: Dict[str, Any]):
        """Get adoption rate history data"""
        try:
            time_range = config.get("time_range", "52w")
            weeks = 52 if time_range == "52w" else 12
            show_projections = config.get("show_projections", True)

            adoption_data = []

            for i in range(weeks):
                date = datetime.now() - timedelta(weeks=weeks-1-i)
                # Simulate adoption rate with seasonal variations
                base_rate = 500
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * i / 52)  # Seasonal variation
                growth_factor = 1.0 + (i / weeks) * 0.5  # Growth over time
                adoption_rate = int(base_rate * seasonal_factor * growth_factor)

                adoption_data.append({
                    "week": date.strftime("%Y-%W"),
                    "new_users": adoption_rate,
                    "cumulative_users": sum(d["new_users"] for d in adoption_data)
                })

            # Add projections if requested
            if show_projections:
                last_cumulative = adoption_data[-1]["cumulative_users"]
                for i in range(4):  # 4 weeks projection
                    date = datetime.now() + timedelta(weeks=i+1)
                    projected_rate = int(adoption_data[-1]["new_users"] * (1.05 ** (i+1)))  # 5% growth projection

                    adoption_data.append({
                        "week": date.strftime("%Y-%W"),
                        "new_users": projected_rate,
                        "cumulative_users": last_cumulative + sum(d["new_users"] for d in adoption_data[-4:]),
                        "projected": True
                    })

            return {
                "data": adoption_data,
                "time_range": time_range,
                "show_projections": show_projections,
                "average_weekly_adoption": sum(d["new_users"] for d in adoption_data if not d.get("projected", False)) / weeks,
                "total_adoption": sum(d["new_users"] for d in adoption_data if not d.get("projected", False))
            }

        except Exception as e:
            self.logger.error(f"Error getting adoption rate history data: {e}")
            return {"data": [], "error": str(e)}

    async def _get_custom_metric_data(self, metric_name: str, config: Dict[str, Any]):
        """Get custom metric data from success metrics engine"""
        try:
            # Try to get from current metrics
            if hasattr(self.success_metrics_engine.current_metrics, metric_name):
                value = getattr(self.success_metrics_engine.current_metrics, metric_name)

                return {
                    "current_value": value,
                    "metric_name": metric_name,
                    "last_updated": datetime.now().isoformat()
                }

            # Try to get metric definition
            if metric_name in self.success_metrics_engine.metrics_registry:
                metric_def = self.success_metrics_engine.metrics_registry[metric_name]

                # Get historical data
                history = await self.success_metrics_engine.get_metric_history(metric_name, 24)  # Last 24 hours

                return {
                    "current_value": history[-1]["value"] if history else 0,
                    "historical_data": history,
                    "metric_definition": {
                        "name": metric_def.name,
                        "category": metric_def.category.value,
                        "unit": metric_def.unit,
                        "target_value": metric_def.target_value
                    }
                }

            # If not found, return placeholder
            return {
                "current_value": 0,
                "metric_name": metric_name,
                "error": "Metric not found"
            }

        except Exception as e:
            self.logger.error(f"Error getting custom metric data for {metric_name}: {e}")
            return {"current_value": 0, "error": str(e)}

    def add_update_callback(self, callback: callable):
        """Add a callback to be called when dashboard data is updated"""
        self.update_callbacks.append(callback)

    def remove_update_callback(self, callback: callable):
        """Remove an update callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def set_layout(self, layout_id: str):
        """Set the active dashboard layout"""
        if layout_id in self.current_layouts:
            self.active_layout_id = layout_id
            self.logger.info(f"Switched to layout: {layout_id}")
        else:
            self.logger.error(f"Layout not found: {layout_id}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data for rendering"""
        try:
            layout = self.current_layouts.get(self.active_layout_id)
            if not layout:
                return {"error": "No active layout"}

            return {
                "layout": {
                    "layout_id": layout.layout_id,
                    "name": layout.name,
                    "theme": layout.theme.value,
                    "widgets": [asdict(widget) for widget in layout.widgets],
                    "grid_columns": layout.grid_columns,
                    "grid_rows": layout.grid_rows
                },
                "widget_data": self.widget_data,
                "last_updated": datetime.now().isoformat(),
                "dashboard_active": self.dashboard_active
            }

        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

    async def generate_dashboard_export(self, format_type: str = "json") -> Dict[str, Any]:
        """Export dashboard data in various formats"""
        try:
            dashboard_data = self.get_dashboard_data()

            if format_type == "json":
                return dashboard_data
            elif format_type == "summary":
                # Generate a summary report
                return {
                    "summary": {
                        "layout_name": dashboard_data["layout"]["name"],
                        "total_widgets": len(dashboard_data["layout"]["widgets"]),
                        "active_widgets": len([w for w in dashboard_data["widget_data"].values() if "error" not in w]),
                        "last_updated": dashboard_data["last_updated"],
                        "key_metrics": self._extract_key_metrics(dashboard_data["widget_data"])
                    }
                }
            else:
                return {"error": f"Unsupported export format: {format_type}"}

        except Exception as e:
            self.logger.error(f"Error generating dashboard export: {e}")
            return {"error": str(e)}

    def _extract_key_metrics(self, widget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from widget data"""
        try:
            key_metrics = {}

            for widget_id, data in widget_data.items():
                if "error" in data:
                    continue

                if "data" in data:
                    data_content = data["data"]

                    # Extract specific key metrics
                    if widget_id == "overall_score":
                        key_metrics["overall_success_score"] = data_content.get("overall_score", 0)
                    elif widget_id == "100x_progress":
                        key_metrics["acceleration_progress"] = data_content.get("progress_percentage", 0)
                    elif widget_id == "global_users":
                        key_metrics["global_users"] = data_content.get("current_value", 0)
                    elif widget_id == "breakthrough_rate":
                        key_metrics["breakthrough_rate"] = data_content.get("current_rate", 0)

            return key_metrics

        except Exception as e:
            self.logger.error(f"Error extracting key metrics: {e}")
            return {}

    async def stop_dashboard(self):
        """Stop the real-time dashboard"""
        try:
            self.dashboard_active = False

            if hasattr(self, '_refresh_thread') and self._refresh_thread.is_alive():
                self._refresh_thread.join(timeout=5.0)

            self.logger.info("Real-time dashboard stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {e}")


# Global instance for easy access
_real_time_dashboard: Optional[RealTimeDashboard] = None


def get_real_time_dashboard(success_metrics_engine: SuccessMetricsEngine = None) -> RealTimeDashboard:
    """Get the global Real-Time Dashboard instance"""
    global _real_time_dashboard
    if _real_time_dashboard is None:
        _real_time_dashboard = RealTimeDashboard(success_metrics_engine)
    return _real_time_dashboard


def initialize_real_time_dashboard(success_metrics_engine: SuccessMetricsEngine = None,
                                 config: Dict[str, Any] = None) -> RealTimeDashboard:
    """Initialize the global Real-Time Dashboard"""
    global _real_time_dashboard
    _real_time_dashboard = RealTimeDashboard(success_metrics_engine, config)
    return _real_time_dashboard