"""
Performance Monitor for AI-Scientist-v2

This module provides performance monitoring capabilities for the AI-Scientist-v2
platform, including metrics collection, analysis, and real-time monitoring.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import asyncio


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    average_response_time: float = 0.0
    agent_loads: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0
    error_rate: float = 0.0


class PerformanceMonitor:
    """
    Performance Monitor - System performance tracking and analysis

    This class provides comprehensive performance monitoring for the AI-Scientist-v2
    platform, including real-time metrics collection, historical data analysis,
    and performance optimization recommendations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_history: deque = deque(maxlen=self.config.get("history_size", 1000))
        self.current_metrics: PerformanceMetrics = PerformanceMetrics()

        # Performance tracking
        self.workflow_start_times: Dict[str, float] = {}
        self.workflow_results: List[Dict[str, Any]] = []
        self.agent_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0,
            "success_rate": 0.0
        })

        # Monitoring state
        self.monitoring_active = False
        self._monitor_thread = None
        self._monitor_interval = self.config.get("monitor_interval", 5.0)  # seconds

        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")

    async def initialize(self):
        """Initialize the performance monitor"""
        try:
            self.monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()

            self.logger.info("Performance monitor initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            raise

    def record_workflow_start(self, workflow_id: str):
        """Record workflow start time"""
        self.workflow_start_times[workflow_id] = time.time()
        self.current_metrics.active_workflows += 1

    def record_workflow_completion(self, workflow_id: str, success: bool, execution_time: float):
        """Record workflow completion"""
        # Update workflow statistics
        if workflow_id in self.workflow_start_times:
            del self.workflow_start_times[workflow_id]

        self.current_metrics.active_workflows -= 1

        if success:
            self.current_metrics.completed_workflows += 1
        else:
            self.current_metrics.failed_workflows += 1

        # Update average response time
        self._update_average_response_time(execution_time)

        # Store result for analysis
        self.workflow_results.append({
            "workflow_id": workflow_id,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })

    def record_agent_task(self, agent_id: str, success: bool, response_time: float):
        """Record agent task performance"""
        agent_stats = self.agent_performance[agent_id]

        agent_stats["total_tasks"] += 1
        agent_stats["total_response_time"] += response_time

        if success:
            agent_stats["successful_tasks"] += 1
        else:
            agent_stats["failed_tasks"] += 1

        # Update calculated metrics
        agent_stats["average_response_time"] = (
            agent_stats["total_response_time"] / agent_stats["total_tasks"]
        )
        agent_stats["success_rate"] = (
            agent_stats["successful_tasks"] / agent_stats["total_tasks"]
        )

        # Update current metrics
        self.current_metrics.agent_loads[agent_id] = (
            agent_stats["total_tasks"] - agent_stats["successful_tasks"]
        )

    def _update_average_response_time(self, execution_time: float):
        """Update average response time"""
        total_completed = self.current_metrics.completed_workflows + self.current_metrics.failed_workflows

        if total_completed > 0:
            # Calculate weighted average
            current_total = self.current_metrics.average_response_time * (total_completed - 1)
            self.current_metrics.average_response_time = (current_total + execution_time) / total_completed

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                self._collect_system_metrics()

                # Calculate derived metrics
                self._calculate_derived_metrics()

                # Store in history
                self.metrics_history.append(self.current_metrics)
                self.current_metrics = PerformanceMetrics()  # Reset for next interval

                # Sleep for monitoring interval
                time.sleep(self._monitor_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self._monitor_interval)

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # System metrics (simplified for this implementation)
            import psutil

            # CPU usage
            self.current_metrics.cpu_usage = psutil.cpu_percent()

            # Memory usage
            memory = psutil.virtual_memory()
            self.current_metrics.memory_usage = memory.percent

            # Active workflows from start times
            self.current_metrics.active_workflows = len(self.workflow_start_times)

        except ImportError:
            # psutil not available, use placeholder values
            self.current_metrics.cpu_usage = 0.0
            self.current_metrics.memory_usage = 0.0

    def _calculate_derived_metrics(self):
        """Calculate derived performance metrics"""
        try:
            # Calculate throughput (workflows per minute)
            recent_results = [
                r for r in self.workflow_results
                if datetime.fromisoformat(r["timestamp"]) > datetime.now() - timedelta(minutes=1)
            ]
            self.current_metrics.throughput = len(recent_results)

            # Calculate error rate
            total_recent = len(recent_results)
            if total_recent > 0:
                failed_recent = sum(1 for r in recent_results if not r["success"])
                self.current_metrics.error_rate = failed_recent / total_recent
            else:
                self.current_metrics.error_rate = 0.0

        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {e}")

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            return {
                "system_metrics": {
                    "cpu_usage": self.current_metrics.cpu_usage,
                    "memory_usage": self.current_metrics.memory_usage,
                    "timestamp": datetime.now().isoformat()
                },
                "workflow_metrics": {
                    "active_workflows": self.current_metrics.active_workflows,
                    "completed_workflows": self.current_metrics.completed_workflows,
                    "failed_workflows": self.current_metrics.failed_workflows,
                    "average_response_time": self.current_metrics.average_response_time
                },
                "performance_metrics": {
                    "throughput": self.current_metrics.throughput,
                    "error_rate": self.current_metrics.error_rate
                },
                "agent_metrics": dict(self.current_metrics.agent_loads),
                "agent_performance": dict(self.agent_performance)
            }

        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {"error": str(e)}

    async def get_historical_metrics(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get historical performance metrics"""
        try:
            cutoff_time = time.time() - (duration_minutes * 60)

            historical_data = []
            for metrics in self.metrics_history:
                if metrics.timestamp >= cutoff_time:
                    historical_data.append({
                        "timestamp": metrics.timestamp,
                        "cpu_usage": metrics.cpu_usage,
                        "memory_usage": metrics.memory_usage,
                        "active_workflows": metrics.active_workflows,
                        "throughput": metrics.throughput,
                        "error_rate": metrics.error_rate
                    })

            return historical_data

        except Exception as e:
            self.logger.error(f"Error getting historical metrics: {e}")
            return []

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and recommendations"""
        try:
            current = await self.get_current_metrics()

            # Analyze performance trends
            summary = {
                "current_status": "healthy",
                "alerts": [],
                "recommendations": [],
                "key_metrics": current
            }

            # Check for performance issues
            if current["system_metrics"]["cpu_usage"] > 80:
                summary["alerts"].append("High CPU usage detected")
                summary["recommendations"].append("Consider scaling horizontally or optimizing workflows")

            if current["system_metrics"]["memory_usage"] > 80:
                summary["alerts"].append("High memory usage detected")
                summary["recommendations"].append("Check for memory leaks or increase system memory")

            if current["performance_metrics"]["error_rate"] > 0.1:
                summary["alerts"].append("High error rate detected")
                summary["recommendations"].append("Review recent failed workflows and agent performance")

            if current["workflow_metrics"]["average_response_time"] > 300:  # 5 minutes
                summary["alerts"].append("Slow response times detected")
                summary["recommendations"].append("Optimize workflow parallelization or agent allocation")

            # Overall status
            if summary["alerts"]:
                summary["current_status"] = "degraded"
                if len(summary["alerts"]) > 2:
                    summary["current_status"] = "critical"

            return summary

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

    async def get_agent_performance_report(self) -> Dict[str, Any]:
        """Get detailed agent performance report"""
        try:
            report = {
                "agent_summary": {},
                "top_performers": [],
                "underperformers": [],
                "recommendations": []
            }

            # Analyze each agent
            for agent_id, stats in self.agent_performance.items():
                if stats["total_tasks"] > 0:
                    agent_summary = {
                        "agent_id": agent_id,
                        "total_tasks": stats["total_tasks"],
                        "success_rate": stats["success_rate"],
                        "average_response_time": stats["average_response_time"],
                        "performance_score": self._calculate_agent_performance_score(stats)
                    }
                    report["agent_summary"][agent_id] = agent_summary

                    # Categorize performance
                    if agent_summary["performance_score"] > 0.8:
                        report["top_performers"].append(agent_id)
                    elif agent_summary["performance_score"] < 0.5:
                        report["underperformers"].append(agent_id)

            # Generate recommendations
            if report["underperformers"]:
                report["recommendations"].append(
                    f"Consider retraining or replacing underperforming agents: {', '.join(report['underperformers'])}"
                )

            if report["top_performers"]:
                report["recommendations"].append(
                    f"Consider increasing workload for top performers: {', '.join(report['top_performers'])}"
                )

            return report

        except Exception as e:
            self.logger.error(f"Error getting agent performance report: {e}")
            return {"error": str(e)}

    def _calculate_agent_performance_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall performance score for an agent"""
        try:
            # Weighted score based on success rate and response time
            success_weight = 0.7
            response_time_weight = 0.3

            # Normalize success rate (0-1)
            success_score = stats["success_rate"]

            # Normalize response time (faster is better, cap at 60 seconds)
            response_time_score = max(0, 1 - (stats["average_response_time"] / 60))

            # Calculate weighted score
            overall_score = (success_weight * success_score) + (response_time_weight * response_time_score)

            return overall_score

        except Exception as e:
            self.logger.error(f"Error calculating agent performance score: {e}")
            return 0.0

    async def shutdown(self):
        """Shutdown the performance monitor"""
        try:
            self.monitoring_active = False

            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

            self.logger.info("Performance monitor shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down performance monitor: {e}")

    def reset_metrics(self):
        """Reset all performance metrics"""
        try:
            self.metrics_history.clear()
            self.workflow_start_times.clear()
            self.workflow_results.clear()
            self.agent_performance.clear()
            self.current_metrics = PerformanceMetrics()

            self.logger.info("Performance metrics reset")

        except Exception as e:
            self.logger.error(f"Error resetting metrics: {e}")