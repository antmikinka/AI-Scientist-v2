"""
Automated Metrics Collection System for AI-Scientist-v2

This module provides automated collection, aggregation, and analysis of success metrics
from multiple data sources including system performance, user interactions, research output,
and external APIs.

Features:
- Automated data collection from multiple sources
- Real-time metric aggregation and calculation
- Integration with existing orchestration and monitoring systems
- Historical data storage and analysis
- Data quality validation and cleaning
- Performance-optimized collection pipelines
- Configurable collection schedules and priorities

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing system components
from .success_metrics_engine import SuccessMetricsEngine, get_success_metrics_engine
from .real_time_dashboard import RealTimeDashboard, get_real_time_dashboard
from .reporting_alert_system import ReportingAlertSystem, get_reporting_alert_system
from ..monitoring.performance_monitor import PerformanceMonitor
from ..orchestration.research_orchestrator_agent import ResearchOrchestratorAgent
from ..ethical.ethical_framework_agent import EthicalFrameworkAgent


class DataSource(Enum):
    """Types of data sources for metrics collection"""
    SYSTEM_METRICS = "system_metrics"
    USER_INTERACTIONS = "user_interactions"
    RESEARCH_OUTPUT = "research_output"
    AGENT_PERFORMANCE = "agent_performance"
    EXTERNAL_APIS = "external_apis"
    DATABASE = "database"
    LOG_FILES = "log_files"
    FILE_SYSTEM = "file_system"
    NETWORK_METRICS = "network_metrics"
    BUSINESS_INTELLIGENCE = "business_intelligence"


class CollectionPriority(Enum):
    """Collection priority levels"""
    CRITICAL = 1  # Real-time, immediate collection
    HIGH = 2      # Every 1-5 minutes
    NORMAL = 3    # Every 15-30 minutes
    LOW = 4       # Every 1-6 hours
    BATCH = 5     # Daily or less frequent


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = 5  # Complete, accurate, timely
    GOOD = 4      # Minor quality issues
    FAIR = 3      # Some gaps or inaccuracies
    POOR = 2      # Significant quality issues
    UNUSABLE = 1  # Data cannot be used for analysis


@dataclass
class CollectionTask:
    """Individual metric collection task"""
    task_id: str
    metric_name: str
    data_source: DataSource
    collection_method: str
    priority: CollectionPriority
    frequency: str  # ISO 8601 duration
    enabled: bool = True
    data_validators: List[str] = field(default_factory=list)
    data_transformers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 30  # seconds
    retry_count: int = 3
    backoff_factor: float = 2.0
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    average_execution_time: float = 0.0


@dataclass
class CollectionResult:
    """Result of a metric collection task"""
    task_id: str
    metric_name: str
    data_source: DataSource
    timestamp: datetime
    value: Union[float, int, str, Dict, List]
    data_quality: DataQuality
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    data_hash: Optional[str] = None


@dataclass
class CollectionPipeline:
    """Pipeline for collecting and processing metrics"""
    pipeline_id: str
    name: str
    description: str
    tasks: List[CollectionTask]
    execution_order: List[str]  # task_id order
    max_concurrent_tasks: int = 5
    timeout: int = 300  # seconds
    enabled: bool = True
    schedule: str = "continuous"  # continuous, scheduled, on_demand
    data_aggregation: Dict[str, str] = field(default_factory=dict)  # aggregation rules


class AutomatedMetricsCollector:
    """
    Automated Metrics Collection System

    This class provides comprehensive automated collection, validation, and
    processing of success metrics from multiple data sources.
    """

    def __init__(self, success_metrics_engine: SuccessMetricsEngine = None,
                 performance_monitor: PerformanceMonitor = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.success_metrics_engine = success_metrics_engine or get_success_metrics_engine()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.logger = logging.getLogger(f"{__name__}.AutomatedMetricsCollector")

        # Collection system state
        self.collection_active = False
        self.pipelines: Dict[str, CollectionPipeline] = {}
        self.active_tasks: Dict[str, CollectionTask] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.result_buffer: List[CollectionResult] = []
        self.collection_stats: Dict[str, Any] = {}

        # Thread pools and workers
        self.collection_workers: List[threading.Thread] = []
        self.processing_workers: List[threading.Thread] = []
        self.executor: Optional[ThreadPoolExecutor] = None

        # Data storage
        self.data_cache: Dict[str, Dict[str, Any]] = {}  # metric_name -> cached data
        self.historical_data: Dict[str, List[CollectionResult]] = {}  # metric_name -> historical results
        self.max_cache_size = self.config.get("max_cache_size", 10000)
        self.max_history_size = self.config.get("max_history_size", 100000)

        # Collection configuration
        self.default_timeout = self.config.get("default_timeout", 30)
        self.max_workers = self.config.get("max_workers", 10)
        self.collection_interval = self.config.get("collection_interval", 60)  # seconds
        self.batch_size = self.config.get("batch_size", 100)

        # Data quality configuration
        self.quality_thresholds = self.config.get("quality_thresholds", {
            "completeness": 0.95,
            "accuracy": 0.90,
            "timeliness": 0.95,
            "consistency": 0.85
        })

        # Initialize collection pipelines
        self._initialize_collection_pipelines()
        self._initialize_data_collectors()

    def _initialize_collection_pipelines(self):
        """Initialize default collection pipelines"""
        try:
            # System Performance Pipeline
            system_tasks = [
                CollectionTask(
                    task_id="cpu_usage",
                    metric_name="system_cpu_usage",
                    data_source=DataSource.SYSTEM_METRICS,
                    collection_method="psutil_cpu",
                    priority=CollectionPriority.HIGH,
                    frequency="PT5M",
                    timeout=10,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="memory_usage",
                    metric_name="system_memory_usage",
                    data_source=DataSource.SYSTEM_METRICS,
                    collection_method="psutil_memory",
                    priority=CollectionPriority.HIGH,
                    frequency="PT5M",
                    timeout=10,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="disk_usage",
                    metric_name="system_disk_usage",
                    data_source=DataSource.SYSTEM_METRICS,
                    collection_method="psutil_disk",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT15M",
                    timeout=15,
                    retry_count=2
                ),
                CollectionTask(
                    task_id="network_metrics",
                    metric_name="system_network_metrics",
                    data_source=DataSource.NETWORK_METRICS,
                    collection_method="psutil_network",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT15M",
                    timeout=15,
                    retry_count=2
                )
            ]

            system_pipeline = CollectionPipeline(
                pipeline_id="system_performance",
                name="System Performance Metrics",
                description="Collect system performance and resource utilization metrics",
                tasks=system_tasks,
                execution_order=[task.task_id for task in system_tasks],
                max_concurrent_tasks=4,
                timeout=120,
                enabled=True,
                schedule="continuous"
            )

            # Research Output Pipeline
            research_tasks = [
                CollectionTask(
                    task_id="papers_generated",
                    metric_name="papers_generated_per_day",
                    data_source=DataSource.RESEARCH_OUTPUT,
                    collection_method="paper_count",
                    priority=CollectionPriority.HIGH,
                    frequency="PT1H",
                    timeout=30,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="experiments_conducted",
                    metric_name="experiments_conducted_per_day",
                    data_source=DataSource.RESEARCH_OUTPUT,
                    collection_method="experiment_count",
                    priority=CollectionPriority.HIGH,
                    frequency="PT1H",
                    timeout=30,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="hypotheses_generated",
                    metric_name="novel_hypotheses_generated",
                    data_source=DataSource.RESEARCH_OUTPUT,
                    collection_method="hypothesis_count",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT2H",
                    timeout=45,
                    retry_count=2
                ),
                CollectionTask(
                    task_id="breakthrough_discoveries",
                    metric_name="breakthrough_discovery_rate",
                    data_source=DataSource.RESEARCH_OUTPUT,
                    collection_method="breakthrough_analysis",
                    priority=CollectionPriority.HIGH,
                    frequency="PT4H",
                    timeout=60,
                    retry_count=3
                )
            ]

            research_pipeline = CollectionPipeline(
                pipeline_id="research_output",
                name="Research Output Metrics",
                description="Collect research output and scientific discovery metrics",
                tasks=research_tasks,
                execution_order=[task.task_id for task in research_tasks],
                max_concurrent_tasks=3,
                timeout=180,
                enabled=True,
                schedule="continuous"
            )

            # User Interaction Pipeline
            user_tasks = [
                CollectionTask(
                    task_id="user_count",
                    metric_name="global_users_count",
                    data_source=DataSource.USER_INTERACTIONS,
                    collection_method="user_database",
                    priority=CollectionPriority.HIGH,
                    frequency="PT30M",
                    timeout=20,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="geographic_distribution",
                    metric_name="geographic_distribution",
                    data_source=DataSource.USER_INTERACTIONS,
                    collection_method="user_geography",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT1H",
                    timeout=30,
                    retry_count=2
                ),
                CollectionTask(
                    task_id="user_sessions",
                    metric_name="user_session_metrics",
                    data_source=DataSource.USER_INTERACTIONS,
                    collection_method="session_tracking",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT15M",
                    timeout=25,
                    retry_count=2
                ),
                CollectionTask(
                    task_id="user_satisfaction",
                    metric_name="user_satisfaction_score",
                    data_source=DataSource.USER_INTERACTIONS,
                    collection_method="satisfaction_survey",
                    priority=CollectionPriority.LOW,
                    frequency="PT6H",
                    timeout=60,
                    retry_count=2
                )
            ]

            user_pipeline = CollectionPipeline(
                pipeline_id="user_interactions",
                name="User Interaction Metrics",
                description="Collect user engagement and satisfaction metrics",
                tasks=user_tasks,
                execution_order=[task.task_id for task in user_tasks],
                max_concurrent_tasks=3,
                timeout=150,
                enabled=True,
                schedule="continuous"
            )

            # Agent Performance Pipeline
            agent_tasks = [
                CollectionTask(
                    task_id="agent_throughput",
                    metric_name="agent_throughput_metrics",
                    data_source=DataSource.AGENT_PERFORMANCE,
                    collection_method="throughput_analysis",
                    priority=CollectionPriority.HIGH,
                    frequency="PT5M",
                    timeout=20,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="agent_success_rate",
                    metric_name="agent_success_rate",
                    data_source=DataSource.AGENT_PERFORMANCE,
                    collection_method="success_rate_calculation",
                    priority=CollectionPriority.HIGH,
                    frequency="PT10M",
                    timeout=30,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="agent_response_time",
                    metric_name="agent_response_time",
                    data_source=DataSource.AGENT_PERFORMANCE,
                    collection_method="response_time_analysis",
                    priority=CollectionPriority.NORMAL,
                    frequency="PT5M",
                    timeout=15,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="agent_errors",
                    metric_name="agent_error_rate",
                    data_source=DataSource.AGENT_PERFORMANCE,
                    collection_method="error_tracking",
                    priority=CollectionPriority.HIGH,
                    frequency="PT5M",
                    timeout=15,
                    retry_count=3
                )
            ]

            agent_pipeline = CollectionPipeline(
                pipeline_id="agent_performance",
                name="Agent Performance Metrics",
                description="Collect multi-agent system performance metrics",
                tasks=agent_tasks,
                execution_order=[task.task_id for task in agent_tasks],
                max_concurrent_tasks=4,
                timeout=120,
                enabled=True,
                schedule="continuous"
            )

            # External Data Pipeline
            external_tasks = [
                CollectionTask(
                    task_id="scientific_publications",
                    metric_name="scientific_citations_received",
                    data_source=DataSource.EXTERNAL_APIS,
                    collection_method="api_scrape",
                    priority=CollectionPriority.LOW,
                    frequency="P1D",
                    timeout=120,
                    retry_count=3
                ),
                CollectionTask(
                    task_id="industry_adoption",
                    metric_name="industry_adoption_count",
                    data_source=DataSource.BUSINESS_INTELLIGENCE,
                    collection_method="market_research",
                    priority=CollectionPriority.LOW,
                    frequency="P7D",
                    timeout=300,
                    retry_count=2
                ),
                CollectionTask(
                    task_id="policy_impact",
                    metric_name="policy_influence_count",
                    data_source=DataSource.BUSINESS_INTELLIGENCE,
                    collection_method="policy_monitoring",
                    priority=CollectionPriority.LOW,
                    frequency="P14D",
                    timeout=600,
                    retry_count=2
                )
            ]

            external_pipeline = CollectionPipeline(
                pipeline_id="external_data",
                name="External Data Sources",
                description="Collect external data from APIs and business intelligence sources",
                tasks=external_tasks,
                execution_order=[task.task_id for task in external_tasks],
                max_concurrent_tasks=2,
                timeout=900,
                enabled=True,
                schedule="scheduled"
            )

            self.pipelines = {
                "system_performance": system_pipeline,
                "research_output": research_pipeline,
                "user_interactions": user_pipeline,
                "agent_performance": agent_pipeline,
                "external_data": external_pipeline
            }

            self.logger.info(f"Initialized {len(self.pipelines)} collection pipelines")

        except Exception as e:
            self.logger.error(f"Error initializing collection pipelines: {e}")

    def _initialize_data_collectors(self):
        """Initialize specific data collection methods"""
        try:
            # Register collection methods
            self.collection_methods = {
                # System metrics collectors
                "psutil_cpu": self._collect_cpu_usage,
                "psutil_memory": self._collect_memory_usage,
                "psutil_disk": self._collect_disk_usage,
                "psutil_network": self._collect_network_metrics,

                # Research output collectors
                "paper_count": self._collect_paper_count,
                "experiment_count": self._collect_experiment_count,
                "hypothesis_count": self._collect_hypothesis_count,
                "breakthrough_analysis": self._collect_breakthrough_analysis,

                # User interaction collectors
                "user_database": self._collect_user_count,
                "user_geography": self._collect_geographic_distribution,
                "session_tracking": self._collect_session_metrics,
                "satisfaction_survey": self._collect_satisfaction_score,

                # Agent performance collectors
                "throughput_analysis": self._collect_agent_throughput,
                "success_rate_calculation": self._collect_agent_success_rate,
                "response_time_analysis": self._collect_agent_response_time,
                "error_tracking": self._collect_agent_error_rate,

                # External data collectors
                "api_scrape": self._collect_external_api_data,
                "market_research": self._collect_market_research_data,
                "policy_monitoring": self._collect_policy_data
            }

            self.logger.info(f"Initialized {len(self.collection_methods)} collection methods")

        except Exception as e:
            self.logger.error(f"Error initializing data collectors: {e}")

    async def initialize(self):
        """Initialize the automated metrics collector"""
        try:
            # Start collection system
            self.collection_active = True

            # Initialize thread pool
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # Start collection workers
            for i in range(self.max_workers):
                worker = threading.Thread(target=self._collection_worker_loop, daemon=True)
                worker.start()
                self.collection_workers.append(worker)

            # Start processing workers
            for i in range(3):  # 3 processing workers
                worker = threading.Thread(target=self._processing_worker_loop, daemon=True)
                worker.start()
                self.processing_workers.append(worker)

            # Start monitoring thread
            self._monitor_thread = threading.Thread(target=self._collection_monitoring_loop, daemon=True)
            self._monitor_thread.start()

            # Schedule pipeline executions
            self._schedule_pipeline_executions()

            self.logger.info("Automated Metrics Collector initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Automated Metrics Collector: {e}")
            raise

    def _collection_worker_loop(self):
        """Worker thread for executing collection tasks"""
        while self.collection_active:
            try:
                # Get task from queue
                task = None
                try:
                    task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if task:
                    # Execute collection task
                    result = self._execute_collection_task(task)

                    # Add result to processing queue
                    if result:
                        self.result_buffer.append(result)

                    # Mark task as completed
                    self.task_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in collection worker loop: {e}")
                time.sleep(5)

    def _processing_worker_loop(self):
        """Worker thread for processing collection results"""
        while self.collection_active:
            try:
                # Process results from buffer
                if self.result_buffer:
                    # Process batch of results
                    batch = self.result_buffer[:self.batch_size]
                    self.result_buffer = self.result_buffer[self.batch_size:]

                    for result in batch:
                        asyncio.run(self._process_collection_result(result))

                time.sleep(1)  # Process every second

            except Exception as e:
                self.logger.error(f"Error in processing worker loop: {e}")
                time.sleep(5)

    def _collection_monitoring_loop(self):
        """Background loop for monitoring collection system"""
        while self.collection_active:
            try:
                # Update collection statistics
                self._update_collection_stats()

                # Schedule overdue tasks
                self._schedule_overdue_tasks()

                # Clean up old data
                self._cleanup_old_data()

                # Check system health
                self._check_collection_health()

                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute

            except Exception as e:
                self.logger.error(f"Error in collection monitoring loop: {e}")
                time.sleep(60)

    def _schedule_pipeline_executions(self):
        """Schedule execution of all collection pipelines"""
        try:
            for pipeline_id, pipeline in self.pipelines.items():
                if not pipeline.enabled:
                    continue

                for task in pipeline.tasks:
                    if not task.enabled:
                        continue

                    # Calculate next execution time
                    next_execution = self._calculate_next_execution(task.frequency)

                    # Schedule task
                    if next_execution:
                        self._schedule_task(task, next_execution)

                    # Add to active tasks
                    self.active_tasks[task.task_id] = task

        except Exception as e:
            self.logger.error(f"Error scheduling pipeline executions: {e}")

    def _calculate_next_execution(self, frequency: str) -> Optional[datetime]:
        """Calculate next execution time based on frequency"""
        try:
            # Parse ISO 8601 duration
            if frequency.startswith("PT"):
                # Duration format: PT5M = 5 minutes, PT1H = 1 hour
                if "M" in frequency:
                    minutes = int(frequency.split("M")[0].replace("PT", ""))
                    interval = timedelta(minutes=minutes)
                elif "H" in frequency:
                    hours = int(frequency.split("H")[0].replace("PT", ""))
                    interval = timedelta(hours=hours)
                else:
                    return None
            elif frequency.startswith("P"):
                # Period format: P1D = 1 day, P7D = 7 days
                if "D" in frequency:
                    days = int(frequency.split("D")[0].replace("P", ""))
                    interval = timedelta(days=days)
                else:
                    return None
            else:
                return None

            return datetime.now() + interval

        except Exception as e:
            self.logger.error(f"Error calculating next execution for frequency {frequency}: {e}")
            return None

    def _schedule_task(self, task: CollectionTask, execution_time: datetime):
        """Schedule a task for execution"""
        try:
            task.next_execution = execution_time

            # Calculate delay
            delay = max(0, (execution_time - datetime.now()).total_seconds())

            # Schedule task execution
            if delay > 0:
                timer = threading.Timer(delay, self._queue_task, args=[task.task_id])
                timer.daemon = True
                timer.start()
            else:
                # Execute immediately
                self._queue_task(task.task_id)

        except Exception as e:
            self.logger.error(f"Error scheduling task {task.task_id}: {e}")

    def _queue_task(self, task_id: str):
        """Queue a task for execution"""
        try:
            task = self.active_tasks.get(task_id)
            if not task:
                return

            # Check if task should be executed
            if not task.enabled:
                return

            # Add to task queue
            self.task_queue.put(task)

            # Update task statistics
            task.last_execution = datetime.now()
            task.execution_count += 1

            # Calculate next execution
            next_execution = self._calculate_next_execution(task.frequency)
            if next_execution:
                task.next_execution = next_execution
                self._schedule_task(task, next_execution)

        except Exception as e:
            self.logger.error(f"Error queuing task {task_id}: {e}")

    def _execute_collection_task(self, task: CollectionTask) -> Optional[CollectionResult]:
        """Execute a single collection task"""
        try:
            start_time = time.time()
            collection_method = self.collection_methods.get(task.collection_method)

            if not collection_method:
                return CollectionResult(
                    task_id=task.task_id,
                    metric_name=task.metric_name,
                    data_source=task.data_source,
                    timestamp=datetime.now(),
                    value=None,
                    data_quality=DataQuality.UNUSABLE,
                    confidence_score=0.0,
                    execution_time=0.0,
                    error_message=f"Collection method '{task.collection_method}' not found"
                )

            # Execute collection with retries
            result = self._execute_with_retry(collection_method, task)

            execution_time = time.time() - start_time

            if result:
                result.execution_time = execution_time

                # Update task statistics
                if result.error_message:
                    task.error_count += 1
                else:
                    task.success_count += 1

                # Update average execution time
                if task.execution_count > 0:
                    task.average_execution_time = (
                        (task.average_execution_time * (task.execution_count - 1) + execution_time) /
                        task.execution_count
                    )

            return result

        except Exception as e:
            self.logger.error(f"Error executing collection task {task.task_id}: {e}")
            return CollectionResult(
                task_id=task.task_id,
                metric_name=task.metric_name,
                data_source=task.data_source,
                timestamp=datetime.now(),
                value=None,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                execution_time=0.0,
                error_message=str(e)
            )

    def _execute_with_retry(self, collection_method: Callable, task: CollectionTask) -> Optional[CollectionResult]:
        """Execute collection method with retry logic"""
        try:
            last_error = None

            for attempt in range(task.retry_count + 1):
                try:
                    # Execute collection method
                    result = collection_method()

                    if result:
                        return result

                except Exception as e:
                    last_error = e
                    if attempt < task.retry_count:
                        # Calculate backoff delay
                        backoff_delay = task.backoff_factor ** attempt
                        time.sleep(backoff_delay)

            # All attempts failed
            return CollectionResult(
                task_id=task.task_id,
                metric_name=task.metric_name,
                data_source=task.data_source,
                timestamp=datetime.now(),
                value=None,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=f"All {task.retry_count + 1} attempts failed. Last error: {str(last_error)}"
            )

        except Exception as e:
            return CollectionResult(
                task_id=task.task_id,
                metric_name=task.metric_name,
                data_source=task.data_source,
                timestamp=datetime.now(),
                value=None,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=f"Retry execution failed: {str(e)}"
            )

    async def _process_collection_result(self, result: CollectionResult):
        """Process a collection result"""
        try:
            # Validate data quality
            validated_result = self._validate_data_quality(result)

            # Apply data transformations
            transformed_result = self._apply_data_transformations(validated_result)

            # Update data cache
            self._update_data_cache(transformed_result)

            # Update historical data
            self._update_historical_data(transformed_result)

            # Forward to success metrics engine
            if transformed_result.value is not None and transformed_result.error_message is None:
                if isinstance(transformed_result.value, (int, float)):
                    self.success_metrics_engine.record_metric(
                        transformed_result.metric_name,
                        float(transformed_result.value),
                        {
                            "source": "automated_collection",
                            "data_quality": transformed_result.data_quality.value,
                            "confidence_score": transformed_result.confidence_score,
                            "collection_time": transformed_result.timestamp.isoformat()
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error processing collection result: {e}")

    def _validate_data_quality(self, result: CollectionResult) -> CollectionResult:
        """Validate data quality of collection result"""
        try:
            quality_score = 0.0
            warnings = []

            # Check completeness
            if result.value is None:
                warnings.append("No data collected")
                quality_score += 0.0
            else:
                quality_score += 0.25

            # Check timeliness
            age = (datetime.now() - result.timestamp).total_seconds()
            if age < 300:  # Less than 5 minutes old
                quality_score += 0.25
            elif age < 3600:  # Less than 1 hour old
                quality_score += 0.15
                warnings.append("Data is more than 5 minutes old")
            else:
                warnings.append("Data is more than 1 hour old")

            # Check confidence
            if result.confidence_score > 0.8:
                quality_score += 0.25
            elif result.confidence_score > 0.5:
                quality_score += 0.15
                warnings.append("Low confidence in data accuracy")

            # Check consistency
            if self._check_data_consistency(result):
                quality_score += 0.25
            else:
                warnings.append("Data inconsistent with historical patterns")

            # Determine overall data quality
            if quality_score >= 0.9:
                data_quality = DataQuality.EXCELLENT
            elif quality_score >= 0.7:
                data_quality = DataQuality.GOOD
            elif quality_score >= 0.5:
                data_quality = DataQuality.FAIR
            elif quality_score >= 0.3:
                data_quality = DataQuality.POOR
            else:
                data_quality = DataQuality.UNUSABLE

            # Update result
            result.data_quality = data_quality
            result.confidence_score = quality_score
            result.warnings = warnings

            return result

        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            result.data_quality = DataQuality.UNUSABLE
            result.confidence_score = 0.0
            result.warnings = ["Data quality validation failed"]
            return result

    def _check_data_consistency(self, result: CollectionResult) -> bool:
        """Check if data is consistent with historical patterns"""
        try:
            # Get historical data for this metric
            historical = self.historical_data.get(result.metric_name, [])
            if len(historical) < 5:
                return True  # Not enough historical data to check

            # Get recent values
            recent_values = [r.value for r in historical[-10:] if r.value is not None]
            if not recent_values:
                return True

            # Check if current value is within reasonable bounds
            if not isinstance(result.value, (int, float)):
                return True

            # Calculate statistics (manual implementation)
            if np:
                mean_value = np.mean(recent_values)
                std_value = np.std(recent_values)

                # Check if within 3 standard deviations
                if std_value > 0:
                    z_score = abs(result.value - mean_value) / std_value
                    return z_score <= 3.0  # Within 3 standard deviations
            else:
                # Simple statistical check without numpy
                if recent_values:
                    mean_value = sum(recent_values) / len(recent_values)
                    variance = sum((x - mean_value) ** 2 for x in recent_values) / len(recent_values)
                    std_value = variance ** 0.5

                    if std_value > 0:
                        z_score = abs(result.value - mean_value) / std_value
                        return z_score <= 3.0

            return True

        except Exception as e:
            self.logger.error(f"Error checking data consistency: {e}")
            return True  # Assume consistent if check fails

    def _apply_data_transformations(self, result: CollectionResult) -> CollectionResult:
        """Apply data transformations to collection result"""
        try:
            # Apply metric-specific transformations
            if result.metric_name == "research_acceleration_factor":
                # Ensure acceleration factor is reasonable
                if isinstance(result.value, (int, float)):
                    result.value = max(1.0, min(1000.0, float(result.value)))  # Clamp between 1x and 1000x

            elif result.metric_name == "ethical_compliance_rate":
                # Ensure compliance rate is percentage
                if isinstance(result.value, (int, float)):
                    result.value = max(0.0, min(100.0, float(result.value)))

            elif result.metric_name == "global_users_count":
                # Ensure user count is reasonable
                if isinstance(result.value, (int, float)):
                    result.value = max(0, int(result.value))

            # Calculate data hash for deduplication
            result.data_hash = self._calculate_data_hash(result)

            return result

        except Exception as e:
            self.logger.error(f"Error applying data transformations: {e}")
            return result

    def _calculate_data_hash(self, result: CollectionResult) -> str:
        """Calculate hash of result data for deduplication"""
        try:
            data_string = f"{result.metric_name}_{result.value}_{result.timestamp.isoformat()}"
            return hashlib.md5(data_string.encode()).hexdigest()
        except Exception:
            return ""

    def _update_data_cache(self, result: CollectionResult):
        """Update data cache with collection result"""
        try:
            if result.metric_name not in self.data_cache:
                self.data_cache[result.metric_name] = {}

            self.data_cache[result.metric_name][result.timestamp.isoformat()] = {
                "value": result.value,
                "quality": result.data_quality.value,
                "confidence": result.confidence_score,
                "source": result.data_source.value
            }

            # Limit cache size
            if len(self.data_cache[result.metric_name]) > self.max_cache_size:
                # Remove oldest entries
                sorted_keys = sorted(self.data_cache[result.metric_name].keys())
                keys_to_remove = sorted_keys[:len(self.data_cache[result.metric_name]) - self.max_cache_size]
                for key in keys_to_remove:
                    del self.data_cache[result.metric_name][key]

        except Exception as e:
            self.logger.error(f"Error updating data cache: {e}")

    def _update_historical_data(self, result: CollectionResult):
        """Update historical data with collection result"""
        try:
            if result.metric_name not in self.historical_data:
                self.historical_data[result.metric_name] = []

            self.historical_data[result.metric_name].append(result)

            # Limit history size
            if len(self.historical_data[result.metric_name]) > self.max_history_size:
                self.historical_data[result.metric_name] = self.historical_data[result.metric_name][-self.max_history_size:]

        except Exception as e:
            self.logger.error(f"Error updating historical data: {e}")

    def _update_collection_stats(self):
        """Update collection statistics"""
        try:
            total_tasks = len(self.active_tasks)
            active_tasks = len([t for t in self.active_tasks.values() if t.enabled])
            successful_executions = sum(t.success_count for t in self.active_tasks.values())
            total_executions = sum(t.execution_count for t in self.active_tasks.values())

            self.collection_stats = {
                "total_tasks": total_tasks,
                "active_tasks": active_tasks,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "total_executions": total_executions,
                "queue_size": self.task_queue.qsize(),
                "result_buffer_size": len(self.result_buffer),
                "cache_entries": sum(len(cache) for cache in self.data_cache.values()),
                "history_entries": sum(len(history) for history in self.historical_data.values()),
                "collection_health": self._calculate_collection_health()
            }

        except Exception as e:
            self.logger.error(f"Error updating collection stats: {e}")

    def _calculate_collection_health(self) -> str:
        """Calculate overall collection system health"""
        try:
            if not self.collection_stats:
                return "unknown"

            success_rate = self.collection_stats.get("success_rate", 0)
            queue_size = self.collection_stats.get("queue_size", 0)

            if success_rate > 95 and queue_size < 10:
                return "excellent"
            elif success_rate > 85 and queue_size < 50:
                return "good"
            elif success_rate > 70 and queue_size < 100:
                return "fair"
            elif success_rate > 50:
                return "poor"
            else:
                return "critical"

        except Exception as e:
            self.logger.error(f"Error calculating collection health: {e}")
            return "unknown"

    def _schedule_overdue_tasks(self):
        """Schedule tasks that are overdue for execution"""
        try:
            current_time = datetime.now()

            for task in self.active_tasks.values():
                if not task.enabled:
                    continue

                if task.next_execution and task.next_execution < current_time:
                    # Task is overdue, reschedule immediately
                    self._queue_task(task.task_id)

        except Exception as e:
            self.logger.error(f"Error scheduling overdue tasks: {e}")

    def _cleanup_old_data(self):
        """Clean up old cached and historical data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days of data

            # Clean up cache
            for metric_name, cache in self.data_cache.items():
                keys_to_remove = [key for key, data in cache.items()
                                if datetime.fromisoformat(key) < cutoff_time]
                for key in keys_to_remove:
                    del cache[key]

            # Clean up history
            for metric_name, history in self.historical_data.items():
                self.historical_data[metric_name] = [
                    result for result in history
                    if result.timestamp > cutoff_time
                ]

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def _check_collection_health(self):
        """Check collection system health and log warnings"""
        try:
            health = self.collection_stats.get("collection_health", "unknown")
            success_rate = self.collection_stats.get("success_rate", 0)

            if health == "critical":
                self.logger.critical(f"Collection system health is CRITICAL - success rate: {success_rate:.1f}%")
            elif health == "poor":
                self.logger.warning(f"Collection system health is POOR - success rate: {success_rate:.1f}%")

        except Exception as e:
            self.logger.error(f"Error checking collection health: {e}")

    # Collection methods for different data sources
    def _collect_cpu_usage(self) -> CollectionResult:
        """Collect CPU usage metrics"""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)

            return CollectionResult(
                task_id="cpu_usage",
                metric_name="system_cpu_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=cpu_usage,
                data_quality=DataQuality.GOOD,
                confidence_score=0.95,
                metadata={"cores": psutil.cpu_count(), "interval": 1}
            )

        except ImportError:
            return CollectionResult(
                task_id="cpu_usage",
                metric_name="system_cpu_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.POOR,
                confidence_score=0.1,
                error_message="psutil not available"
            )
        except Exception as e:
            return CollectionResult(
                task_id="cpu_usage",
                metric_name="system_cpu_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_memory_usage(self) -> CollectionResult:
        """Collect memory usage metrics"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            return CollectionResult(
                task_id="memory_usage",
                metric_name="system_memory_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=memory_usage,
                data_quality=DataQuality.GOOD,
                confidence_score=0.95,
                metadata={"total_gb": memory.total / (1024**3), "available_gb": memory.available / (1024**3)}
            )

        except ImportError:
            return CollectionResult(
                task_id="memory_usage",
                metric_name="system_memory_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.POOR,
                confidence_score=0.1,
                error_message="psutil not available"
            )
        except Exception as e:
            return CollectionResult(
                task_id="memory_usage",
                metric_name="system_memory_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_disk_usage(self) -> CollectionResult:
        """Collect disk usage metrics"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100

            return CollectionResult(
                task_id="disk_usage",
                metric_name="system_disk_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=disk_usage,
                data_quality=DataQuality.GOOD,
                confidence_score=0.95,
                metadata={"total_gb": disk.total / (1024**3), "free_gb": disk.free / (1024**3)}
            )

        except ImportError:
            return CollectionResult(
                task_id="disk_usage",
                metric_name="system_disk_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.POOR,
                confidence_score=0.1,
                error_message="psutil not available"
            )
        except Exception as e:
            return CollectionResult(
                task_id="disk_usage",
                metric_name="system_disk_usage",
                data_source=DataSource.SYSTEM_METRICS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_network_metrics(self) -> CollectionResult:
        """Collect network metrics"""
        try:
            import psutil
            network = psutil.net_io_counters()

            return CollectionResult(
                task_id="network_metrics",
                metric_name="system_network_metrics",
                data_source=DataSource.NETWORK_METRICS,
                timestamp=datetime.now(),
                value={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"interfaces": len(psutil.net_if_addrs())}
            )

        except ImportError:
            return CollectionResult(
                task_id="network_metrics",
                metric_name="system_network_metrics",
                data_source=DataSource.NETWORK_METRICS,
                timestamp=datetime.now(),
                value={"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0},
                data_quality=DataQuality.POOR,
                confidence_score=0.1,
                error_message="psutil not available"
            )
        except Exception as e:
            return CollectionResult(
                task_id="network_metrics",
                metric_name="system_network_metrics",
                data_source=DataSource.NETWORK_METRICS,
                timestamp=datetime.now(),
                value={"bytes_sent": 0, "bytes_recv": 0, "packets_sent": 0, "packets_recv": 0},
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_paper_count(self) -> CollectionResult:
        """Collect paper generation count"""
        try:
            # This would integrate with actual research output tracking
            # For now, simulate based on current metrics
            current_papers = getattr(self.success_metrics_engine.current_metrics, 'papers_generated_per_day', 0)

            # Simulate some variation
            papers_today = max(0, int(current_papers * (0.8 + (hash(str(datetime.now().day)) % 40) / 100)))

            return CollectionResult(
                task_id="papers_generated",
                metric_name="papers_generated_per_day",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=papers_today,
                data_quality=DataQuality.GOOD,
                confidence_score=0.8,
                metadata={"simulation": True, "base_rate": current_papers}
            )

        except Exception as e:
            return CollectionResult(
                task_id="papers_generated",
                metric_name="papers_generated_per_day",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_experiment_count(self) -> CollectionResult:
        """Collect experiment count"""
        try:
            # This would integrate with actual experiment tracking
            # For now, simulate based on current metrics
            current_experiments = getattr(self.success_metrics_engine.current_metrics, 'experiments_conducted_per_day', 0)

            # Simulate some variation
            experiments_today = max(0, int(current_experiments * (0.8 + (hash(str(datetime.now().day)) % 40) / 100)))

            return CollectionResult(
                task_id="experiments_conducted",
                metric_name="experiments_conducted_per_day",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=experiments_today,
                data_quality=DataQuality.GOOD,
                confidence_score=0.8,
                metadata={"simulation": True, "base_rate": current_experiments}
            )

        except Exception as e:
            return CollectionResult(
                task_id="experiments_conducted",
                metric_name="experiments_conducted_per_day",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_hypothesis_count(self) -> CollectionResult:
        """Collect hypothesis generation count"""
        try:
            # This would integrate with actual hypothesis tracking
            # For now, simulate based on papers generated
            papers = getattr(self.success_metrics_engine.current_metrics, 'papers_generated_per_day', 0)
            hypotheses = max(0, int(papers * 2.5))  # ~2.5 hypotheses per paper

            return CollectionResult(
                task_id="hypotheses_generated",
                metric_name="novel_hypotheses_generated",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=hypotheses,
                data_quality=DataQuality.FAIR,
                confidence_score=0.6,
                metadata={"simulation": True, "papers_today": papers}
            )

        except Exception as e:
            return CollectionResult(
                task_id="hypotheses_generated",
                metric_name="novel_hypotheses_generated",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_breakthrough_analysis(self) -> CollectionResult:
        """Collect breakthrough discovery analysis"""
        try:
            # This would integrate with actual breakthrough detection
            # For now, simulate based on research activity
            papers = getattr(self.success_metrics_engine.current_metrics, 'papers_generated_per_day', 0)
            experiments = getattr(self.success_metrics_engine.current_metrics, 'experiments_conducted_per_day', 0)

            # Simulate breakthrough probability based on activity
            breakthrough_prob = min(0.3, (papers + experiments) / 100)  # Max 30% probability
            breakthrough_today = 1 if (hash(str(datetime.now().date())) % 100) < (breakthrough_prob * 100) else 0

            return CollectionResult(
                task_id="breakthrough_discoveries",
                metric_name="breakthrough_discovery_rate",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=breakthrough_today * 30,  # Convert to monthly rate
                data_quality=DataQuality.FAIR,
                confidence_score=0.5,
                metadata={"simulation": True, "probability": breakthrough_prob, "papers": papers, "experiments": experiments}
            )

        except Exception as e:
            return CollectionResult(
                task_id="breakthrough_discoveries",
                metric_name="breakthrough_discovery_rate",
                data_source=DataSource.RESEARCH_OUTPUT,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_user_count(self) -> CollectionResult:
        """Collect global user count"""
        try:
            # This would integrate with actual user tracking systems
            # For now, simulate based on system activity
            current_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 1000)

            # Simulate growth
            growth_rate = 0.01  # 1% daily growth
            users_today = int(current_users * (1 + growth_rate))

            return CollectionResult(
                task_id="user_count",
                metric_name="global_users_count",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value=users_today,
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"simulation": True, "growth_rate": growth_rate, "previous_count": current_users}
            )

        except Exception as e:
            return CollectionResult(
                task_id="user_count",
                metric_name="global_users_count",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_geographic_distribution(self) -> CollectionResult:
        """Collect geographic distribution data"""
        try:
            # This would integrate with actual user location tracking
            # For now, simulate distribution
            current_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 1000)

            # Simulate geographic distribution
            regions = [
                ("North America", 0.35),
                ("Europe", 0.25),
                ("Asia", 0.20),
                ("South America", 0.08),
                ("Africa", 0.05),
                ("Oceania", 0.04),
                ("Other", 0.03)
            ]

            distribution = {}
            for region, percentage in regions:
                distribution[region] = int(current_users * percentage)

            return CollectionResult(
                task_id="geographic_distribution",
                metric_name="geographic_distribution",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value=distribution,
                data_quality=DataQuality.FAIR,
                confidence_score=0.7,
                metadata={"simulation": True, "total_users": current_users, "regions_covered": len(regions)}
            )

        except Exception as e:
            return CollectionResult(
                task_id="geographic_distribution",
                metric_name="geographic_distribution",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value={},
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_session_metrics(self) -> CollectionResult:
        """Collect user session metrics"""
        try:
            # This would integrate with actual session tracking
            # For now, simulate based on user count
            current_users = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 1000)

            # Simulate session metrics
            active_sessions = int(current_users * 0.15)  # 15% of users active
            average_session_duration = 25.5  # minutes
            sessions_today = int(current_users * 0.8)  # 80% of users had sessions today

            return CollectionResult(
                task_id="user_sessions",
                metric_name="user_session_metrics",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value={
                    "active_sessions": active_sessions,
                    "average_session_duration": average_session_duration,
                    "sessions_today": sessions_today
                },
                data_quality=DataQuality.FAIR,
                confidence_score=0.6,
                metadata={"simulation": True, "current_users": current_users}
            )

        except Exception as e:
            return CollectionResult(
                task_id="user_sessions",
                metric_name="user_session_metrics",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value={"active_sessions": 0, "average_session_duration": 0, "sessions_today": 0},
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_satisfaction_score(self) -> CollectionResult:
        """Collect user satisfaction score"""
        try:
            # This would integrate with actual satisfaction surveys
            # For now, simulate based on system performance
            system_health = self.collection_stats.get("collection_health", "good")

            # Simulate satisfaction score based on system health
            health_scores = {
                "excellent": 92,
                "good": 85,
                "fair": 72,
                "poor": 58,
                "critical": 35
            }

            satisfaction_score = health_scores.get(system_health, 75)
            # Add some variation
            satisfaction_score += (hash(str(datetime.now().date())) % 21) - 10  # ±10 variation
            satisfaction_score = max(0, min(100, satisfaction_score))

            return CollectionResult(
                task_id="user_satisfaction",
                metric_name="user_satisfaction_score",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value=satisfaction_score,
                data_quality=DataQuality.FAIR,
                confidence_score=0.6,
                metadata={"simulation": True, "system_health": system_health}
            )

        except Exception as e:
            return CollectionResult(
                task_id="user_satisfaction",
                metric_name="user_satisfaction_score",
                data_source=DataSource.USER_INTERACTIONS,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_agent_throughput(self) -> CollectionResult:
        """Collect agent throughput metrics"""
        try:
            # Get performance metrics from performance monitor
            perf_metrics = asyncio.run(self.performance_monitor.get_current_metrics())

            throughput = perf_metrics.get("performance_metrics", {}).get("throughput", 0)

            return CollectionResult(
                task_id="agent_throughput",
                metric_name="agent_throughput_metrics",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=throughput,
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"source": "performance_monitor"}
            )

        except Exception as e:
            return CollectionResult(
                task_id="agent_throughput",
                metric_name="agent_throughput_metrics",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_agent_success_rate(self) -> CollectionResult:
        """Collect agent success rate metrics"""
        try:
            # Get agent performance report
            agent_report = asyncio.run(self.performance_monitor.get_agent_performance_report())

            # Calculate overall success rate
            total_tasks = sum(stats.get("total_tasks", 0) for stats in agent_report.get("agent_summary", {}).values())
            successful_tasks = sum(stats.get("successful_tasks", 0) for stats in agent_report.get("agent_summary", {}).values())

            success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

            return CollectionResult(
                task_id="agent_success_rate",
                metric_name="agent_success_rate",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=success_rate,
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"total_tasks": total_tasks, "successful_tasks": successful_tasks}
            )

        except Exception as e:
            return CollectionResult(
                task_id="agent_success_rate",
                metric_name="agent_success_rate",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_agent_response_time(self) -> CollectionResult:
        """Collect agent response time metrics"""
        try:
            # Get agent performance report
            agent_report = asyncio.run(self.performance_monitor.get_agent_performance_report())

            # Calculate average response time
            response_times = [stats.get("average_response_time", 0) for stats in agent_report.get("agent_summary", {}).values()]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            return CollectionResult(
                task_id="agent_response_time",
                metric_name="agent_response_time",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=avg_response_time,
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"agents_analyzed": len(response_times), "response_times": response_times}
            )

        except Exception as e:
            return CollectionResult(
                task_id="agent_response_time",
                metric_name="agent_response_time",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_agent_error_rate(self) -> CollectionResult:
        """Collect agent error rate metrics"""
        try:
            # Get performance metrics from performance monitor
            perf_metrics = asyncio.run(self.performance_monitor.get_current_metrics())

            error_rate = perf_metrics.get("performance_metrics", {}).get("error_rate", 0)

            return CollectionResult(
                task_id="agent_errors",
                metric_name="agent_error_rate",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=error_rate * 100,  # Convert to percentage
                data_quality=DataQuality.GOOD,
                confidence_score=0.9,
                metadata={"source": "performance_monitor"}
            )

        except Exception as e:
            return CollectionResult(
                task_id="agent_errors",
                metric_name="agent_error_rate",
                data_source=DataSource.AGENT_PERFORMANCE,
                timestamp=datetime.now(),
                value=0.0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_external_api_data(self) -> CollectionResult:
        """Collect external API data (placeholder implementation)"""
        try:
            # This would integrate with external APIs like:
            # - Scientific publication databases
            # - Citation tracking services
            # - Academic research platforms

            return CollectionResult(
                task_id="scientific_publications",
                metric_name="scientific_citations_received",
                data_source=DataSource.EXTERNAL_APIS,
                timestamp=datetime.now(),
                value=0,  # Placeholder - would be actual citation count
                data_quality=DataQuality.POOR,
                confidence_score=0.2,
                metadata={"implementation": "placeholder", "data_source": "external_apis"}
            )

        except Exception as e:
            return CollectionResult(
                task_id="scientific_publications",
                metric_name="scientific_citations_received",
                data_source=DataSource.EXTERNAL_APIS,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_market_research_data(self) -> CollectionResult:
        """Collect market research data (placeholder implementation)"""
        try:
            # This would integrate with market research services
            # to track industry adoption and business impact

            return CollectionResult(
                task_id="industry_adoption",
                metric_name="industry_adoption_count",
                data_source=DataSource.BUSINESS_INTELLIGENCE,
                timestamp=datetime.now(),
                value=0,  # Placeholder - would be actual adoption count
                data_quality=DataQuality.POOR,
                confidence_score=0.2,
                metadata={"implementation": "placeholder", "data_source": "market_research"}
            )

        except Exception as e:
            return CollectionResult(
                task_id="industry_adoption",
                metric_name="industry_adoption_count",
                data_source=DataSource.BUSINESS_INTELLIGENCE,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def _collect_policy_data(self) -> CollectionResult:
        """Collect policy monitoring data (placeholder implementation)"""
        try:
            # This would integrate with policy monitoring services
            # to track policy influence and government adoption

            return CollectionResult(
                task_id="policy_impact",
                metric_name="policy_influence_count",
                data_source=DataSource.BUSINESS_INTELLIGENCE,
                timestamp=datetime.now(),
                value=0,  # Placeholder - would be actual policy count
                data_quality=DataQuality.POOR,
                confidence_score=0.2,
                metadata={"implementation": "placeholder", "data_source": "policy_monitoring"}
            )

        except Exception as e:
            return CollectionResult(
                task_id="policy_impact",
                metric_name="policy_influence_count",
                data_source=DataSource.BUSINESS_INTELLIGENCE,
                timestamp=datetime.now(),
                value=0,
                data_quality=DataQuality.UNUSABLE,
                confidence_score=0.0,
                error_message=str(e)
            )

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection system status"""
        try:
            return {
                "system_active": self.collection_active,
                "statistics": self.collection_stats,
                "pipelines": {
                    pipeline_id: {
                        "name": pipeline.name,
                        "enabled": pipeline.enabled,
                        "tasks": len(pipeline.tasks),
                        "active_tasks": len([t for t in pipeline.tasks if t.enabled])
                    }
                    for pipeline_id, pipeline in self.pipelines.items()
                },
                "data_sources": {
                    source.value: len([t for t in self.active_tasks.values() if t.data_source == source])
                    for source in DataSource
                },
                "quality_metrics": {
                    "data_quality_distribution": self._get_data_quality_distribution(),
                    "confidence_distribution": self._get_confidence_distribution(),
                    "error_rate": self._get_error_rate()
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting collection status: {e}")
            return {"error": str(e)}

    def _get_data_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of data quality levels"""
        try:
            distribution = {quality.value: 0 for quality in DataQuality}

            for history in self.historical_data.values():
                for result in history:
                    distribution[result.data_quality.value] += 1

            return distribution

        except Exception as e:
            self.logger.error(f"Error getting data quality distribution: {e}")
            return {quality.value: 0 for quality in DataQuality}

    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores"""
        try:
            distribution = {
                "high": 0,    # 0.8 - 1.0
                "medium": 0,  # 0.5 - 0.8
                "low": 0      # 0.0 - 0.5
            }

            for history in self.historical_data.values():
                for result in history:
                    if result.confidence_score >= 0.8:
                        distribution["high"] += 1
                    elif result.confidence_score >= 0.5:
                        distribution["medium"] += 1
                    else:
                        distribution["low"] += 1

            return distribution

        except Exception as e:
            self.logger.error(f"Error getting confidence distribution: {e}")
            return {"high": 0, "medium": 0, "low": 0}

    def _get_error_rate(self) -> float:
        """Get current error rate"""
        try:
            total_executions = sum(t.execution_count for t in self.active_tasks.values())
            error_executions = sum(t.error_count for t in self.active_tasks.values())

            return (error_executions / total_executions * 100) if total_executions > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error getting error rate: {e}")
            return 0.0

    async def shutdown(self):
        """Shutdown the automated metrics collector"""
        try:
            self.collection_active = False

            # Shutdown thread pool
            if self.executor:
                self.executor.shutdown(wait=True)

            # Wait for threads to finish
            for worker in self.collection_workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            for worker in self.processing_workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

            self.logger.info("Automated Metrics Collector shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down Automated Metrics Collector: {e}")


# Global instance for easy access
_automated_metrics_collector: Optional[AutomatedMetricsCollector] = None


def get_automated_metrics_collector(success_metrics_engine: SuccessMetricsEngine = None,
                                   performance_monitor: PerformanceMonitor = None) -> AutomatedMetricsCollector:
    """Get the global Automated Metrics Collector instance"""
    global _automated_metrics_collector
    if _automated_metrics_collector is None:
        _automated_metrics_collector = AutomatedMetricsCollector(success_metrics_engine, performance_monitor)
    return _automated_metrics_collector


def initialize_automated_metrics_collector(success_metrics_engine: SuccessMetricsEngine = None,
                                         performance_monitor: PerformanceMonitor = None,
                                         config: Dict[str, Any] = None) -> AutomatedMetricsCollector:
    """Initialize the global Automated Metrics Collector"""
    global _automated_metrics_collector
    _automated_metrics_collector = AutomatedMetricsCollector(success_metrics_engine, performance_monitor, config)
    return _automated_metrics_collector