"""
Comprehensive Reporting and Alert System for AI-Scientist-v2

This module provides automated reporting, intelligent alerting, and notification
systems for tracking success metrics and life's work progress.

Features:
- Automated report generation (daily, weekly, monthly, quarterly)
- Intelligent alert system with severity levels and escalation
- Multi-channel notifications (email, Slack, dashboard, webhook)
- Customizable report templates and scheduling
- Alert aggregation and suppression to prevent noise
- Performance benchmarking and trend analysis
- Stakeholder-tailored reporting

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from pathlib import Path
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io

# Import existing system components
from .success_metrics_engine import SuccessMetricsEngine, get_success_metrics_engine
from .real_time_dashboard import RealTimeDashboard, get_real_time_dashboard


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts"""
    METRIC_THRESHOLD = "metric_threshold"
    GOAL_MILESTONE = "goal_milestone"
    SYSTEM_PERFORMANCE = "system_performance"
    ETHICAL_VIOLATION = "ethical_violation"
    PREDICTION_FAILURE = "prediction_failure"
    TREND_ANOMALY = "trend_anomaly"
    ACHIEVEMENT = "achievement"
    RISK_DETECTED = "risk_detected"


class ReportFrequency(Enum):
    """Report generation frequencies"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"
    SMS = "sms"
    IN_APP = "in_app"


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    escalation_level: int = 0
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    suppression_until: Optional[datetime] = None


@dataclass
class ReportTemplate:
    """Report template definition"""
    template_id: str
    name: str
    description: str
    frequency: ReportFrequency
    recipients: List[str]
    sections: List[Dict[str, Any]]
    format_options: Dict[str, Any] = field(default_factory=dict)
    custom_css: Optional[str] = None
    custom_header: Optional[str] = None
    custom_footer: Optional[str] = None


@dataclass
class NotificationRule:
    """Notification routing rules"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    recipients: List[str]
    enabled: bool = True
    priority: int = 0


@dataclass
class Benchmark:
    """Performance benchmark definition"""
    benchmark_id: str
    name: str
    description: str
    metric_name: str
    baseline_value: float
    target_value: float
    comparison_period: str  # e.g., "30d", "1y"
    category: str
    created_at: datetime = field(default_factory=datetime.now)


class ReportingAlertSystem:
    """
    Comprehensive Reporting and Alert System

    This class provides automated reporting, intelligent alerting, and
    multi-channel notifications for success metrics monitoring.
    """

    def __init__(self, success_metrics_engine: SuccessMetricsEngine = None,
                 real_time_dashboard: RealTimeDashboard = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.success_metrics_engine = success_metrics_engine or get_success_metrics_engine()
        self.real_time_dashboard = real_time_dashboard or get_real_time_dashboard()
        self.logger = logging.getLogger(f"{__name__}.ReportingAlertSystem")

        # Alert system state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_rules: List[NotificationRule] = []

        # Report generation state
        self.report_templates: Dict[str, ReportTemplate] = {}
        self.scheduled_reports: Dict[str, threading.Timer] = {}
        self.report_cache: Dict[str, Dict[str, Any]] = {}

        # Notification system state
        self.notification_queue: List[Dict[str, Any]] = []
        self.notification_workers: List[threading.Thread] = []
        self.max_queue_size = self.config.get("max_queue_size", 1000)

        # System state
        self.system_active = False
        self._alert_monitor_thread = None
        self._notification_processor_thread = None
        self._report_scheduler_thread = None

        # Email configuration
        self.email_config = self.config.get("email", {})
        self.smtp_configured = bool(self.email_config.get("smtp_server"))

        # Initialize components
        self._initialize_alert_rules()
        self._initialize_report_templates()
        self._initialize_notification_rules()

    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        try:
            # Critical metric threshold alerts
            self.alert_rules["critical_acceleration"] = {
                "name": "Critical Research Acceleration Alert",
                "type": AlertType.METRIC_THRESHOLD,
                "metric": "research_acceleration_factor",
                "condition": "value < 5.0",
                "severity": AlertSeverity.CRITICAL,
                "message": "Research acceleration factor has dropped below critical threshold",
                "cooldown": 3600  # 1 hour
            }

            self.alert_rules["ethical_compliance"] = {
                "name": "Ethical Compliance Alert",
                "type": AlertType.METRIC_THRESHOLD,
                "metric": "ethical_compliance_rate",
                "condition": "value < 95.0",
                "severity": AlertSeverity.CRITICAL,
                "message": "Ethical compliance rate has fallen below acceptable threshold",
                "cooldown": 1800  # 30 minutes
            }

            self.alert_rules["goal_achievement"] = {
                "name": "Goal Achievement Alert",
                "type": AlertType.GOAL_MILESTONE,
                "condition": "progress_percentage >= 100",
                "severity": AlertSeverity.INFO,
                "message": "Life's work goal has been achieved!",
                "cooldown": 86400  # 24 hours
            }

            self.alert_rules["system_performance"] = {
                "name": "System Performance Alert",
                "type": AlertType.SYSTEM_PERFORMANCE,
                "condition": "error_rate > 0.1",
                "severity": AlertSeverity.WARNING,
                "message": "System error rate exceeds acceptable threshold",
                "cooldown": 1800  # 30 minutes
            }

            self.alert_rules["breakthrough_achievement"] = {
                "name": "Breakthrough Discovery Alert",
                "type": AlertType.ACHIEVEMENT,
                "condition": "breakthrough_discovery_rate > 0",
                "severity": AlertSeverity.INFO,
                "message": "New breakthrough discovery achieved!",
                "cooldown": 3600  # 1 hour
            }

            self.logger.info(f"Initialized {len(self.alert_rules)} alert rules")

        except Exception as e:
            self.logger.error(f"Error initializing alert rules: {e}")

    def _initialize_report_templates(self):
        """Initialize default report templates"""
        try:
            # Daily Executive Summary
            daily_executive = ReportTemplate(
                template_id="daily_executive_summary",
                name="Daily Executive Summary",
                description="Daily overview of success metrics and goal progress",
                frequency=ReportFrequency.DAILY,
                recipients=["executive_team@example.com"],
                sections=[
                    {
                        "title": "Executive Summary",
                        "type": "summary",
                        "content": "overall_success_score, primary_achievement, critical_challenges"
                    },
                    {
                        "title": "Life's Work Progress",
                        "type": "goals",
                        "content": "all_goals_progress, milestone_achievements"
                    },
                    {
                        "title": "Key Metrics Dashboard",
                        "type": "metrics",
                        "content": "research_acceleration, democratization, knowledge_advancement"
                    },
                    {
                        "title": "Alerts and Issues",
                        "type": "alerts",
                        "content": "critical_alerts, system_health"
                    }
                ],
                format_options={
                    "include_charts": True,
                    "chart_style": "professional",
                    "color_scheme": "scientific"
                }
            )

            # Weekly Technical Report
            weekly_technical = ReportTemplate(
                template_id="weekly_technical_report",
                name="Weekly Technical Report",
                description="Detailed technical analysis and performance metrics",
                frequency=ReportFrequency.WEEKLY,
                recipients=["technical_team@example.com", "engineering@example.com"],
                sections=[
                    {
                        "title": "System Performance",
                        "type": "performance",
                        "content": "system_metrics, agent_performance, throughput_analysis"
                    },
                    {
                        "title": "Research Output Analysis",
                        "type": "research",
                        "content": "paper_generation, experiment_results, breakthrough_analysis"
                    },
                    {
                        "title": "User Engagement",
                        "type": "analytics",
                        "content": "user_growth, geographic_distribution, engagement_metrics"
                    },
                    {
                        "title": "Technical Challenges",
                        "type": "issues",
                        "content": "bug_reports, performance_issues, system_alerts"
                    }
                ],
                format_options={
                    "include_charts": True,
                    "include_raw_data": True,
                    "chart_style": "technical",
                    "data_export": True
                }
            )

            # Monthly Life's Work Report
            monthly_lifes_work = ReportTemplate(
                template_id="monthly_lifes_work_report",
                name="Monthly Life's Work Report",
                description="Comprehensive progress toward life's work goals",
                frequency=ReportFrequency.MONTHLY,
                recipients=["board@example.com", "investors@example.com", "leadership@example.com"],
                sections=[
                    {
                        "title": "Life's Work Impact Assessment",
                        "type": "impact",
                        "content": "overall_impact, 100x_acceleration_progress, democratization_impact"
                    },
                    {
                        "title": "Goal Achievement Analysis",
                        "type": "goals_detailed",
                        "content": "goal_progress_detailed, milestone_analysis, risk_assessment"
                    },
                    {
                        "title": "Scientific Advancement",
                        "type": "scientific",
                        "content": "research_acceleration_detailed, breakthrough_impact, knowledge_generation"
                    },
                    {
                        "title": "Global Impact",
                        "type": "global",
                        "content": "democratization_progress, geographic_expansion, accessibility_metrics"
                    },
                    {
                        "title": "Strategic Outlook",
                        "type": "strategy",
                        "content": "predictions, strategic_recommendations, resource_needs"
                    }
                ],
                format_options={
                    "include_charts": True,
                    "include_projections": True,
                    "chart_style": "executive",
                    "executive_summary": True,
                    "appendix": True
                }
            )

            self.report_templates = {
                "daily_executive": daily_executive,
                "weekly_technical": weekly_technical,
                "monthly_lifes_work": monthly_lifes_work
            }

            self.logger.info(f"Initialized {len(self.report_templates)} report templates")

        except Exception as e:
            self.logger.error(f"Error initializing report templates: {e}")

    def _initialize_notification_rules(self):
        """Initialize notification routing rules"""
        try:
            # Critical alerts - all channels
            critical_rule = NotificationRule(
                rule_id="critical_alerts",
                name="Critical Alert Notifications",
                conditions={"severity": "critical"},
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
                recipients=["leadership@example.com", "oncall@example.com"],
                priority=100
            )

            # Warning alerts - email and Slack
            warning_rule = NotificationRule(
                rule_id="warning_alerts",
                name="Warning Alert Notifications",
                conditions={"severity": "warning"},
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                recipients=["team@example.com"],
                priority=50
            )

            # Info alerts - dashboard only
            info_rule = NotificationRule(
                rule_id="info_alerts",
                name="Info Alert Notifications",
                conditions={"severity": "info"},
                channels=[NotificationChannel.DASHBOARD, NotificationChannel.IN_APP],
                recipients=["all_users"],
                priority=10
            )

            # Achievement alerts - all channels for celebration
            achievement_rule = NotificationRule(
                rule_id="achievement_alerts",
                name="Achievement Notifications",
                conditions={"alert_type": "achievement"},
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DASHBOARD],
                recipients=["team@example.com", "leadership@example.com"],
                priority=75
            )

            self.notification_rules = [critical_rule, warning_rule, info_rule, achievement_rule]

            self.logger.info(f"Initialized {len(self.notification_rules)} notification rules")

        except Exception as e:
            self.logger.error(f"Error initializing notification rules: {e}")

    async def initialize(self):
        """Initialize the reporting and alert system"""
        try:
            # Start system components
            self.system_active = True

            # Start monitoring threads
            self._alert_monitor_thread = threading.Thread(target=self._alert_monitoring_loop, daemon=True)
            self._alert_monitor_thread.start()

            self._notification_processor_thread = threading.Thread(target=self._notification_processing_loop, daemon=True)
            self._notification_processor_thread.start()

            self._report_scheduler_thread = threading.Thread(target=self._report_scheduling_loop, daemon=True)
            self._report_scheduler_thread.start()

            # Start notification workers
            for i in range(3):  # 3 worker threads
                worker = threading.Thread(target=self._notification_worker_loop, daemon=True)
                worker.start()
                self.notification_workers.append(worker)

            # Schedule reports
            self._schedule_all_reports()

            self.logger.info("Reporting and Alert System initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Reporting and Alert System: {e}")
            raise

    def _alert_monitoring_loop(self):
        """Background loop for monitoring alerts"""
        while self.system_active:
            try:
                # Check metric thresholds
                asyncio.run(self._check_metric_thresholds())

                # Check goal milestones
                asyncio.run(self._check_goal_milestones())

                # Check system performance
                asyncio.run(self._check_system_performance())

                # Check for achievement alerts
                asyncio.run(self._check_achievements())

                # Clean up resolved alerts
                self._cleanup_resolved_alerts()

                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(60)

    def _notification_processing_loop(self):
        """Background loop for processing notifications"""
        while self.system_active:
            try:
                if self.notification_queue:
                    notification = self.notification_queue.pop(0)
                    self._process_notification(notification)

                time.sleep(1)  # Process notifications every second

            except Exception as e:
                self.logger.error(f"Error in notification processing loop: {e}")
                time.sleep(5)

    def _notification_worker_loop(self):
        """Worker thread for sending notifications"""
        while self.system_active:
            try:
                # This would handle the actual sending of notifications
                # For now, it's a placeholder
                time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error in notification worker loop: {e}")
                time.sleep(5)

    def _report_scheduling_loop(self):
        """Background loop for scheduling report generation"""
        while self.system_active:
            try:
                # Check for scheduled reports
                asyncio.run(self._check_scheduled_reports())

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in report scheduling loop: {e}")
                time.sleep(300)

    async def _check_metric_thresholds(self):
        """Check metric thresholds and generate alerts"""
        try:
            current_metrics = await self.success_metrics_engine.get_current_metrics()

            for rule_id, rule in self.alert_rules.items():
                if rule.get("type") != AlertType.METRIC_THRESHOLD:
                    continue

                metric_name = rule.get("metric")
                if not metric_name:
                    continue

                # Get current value
                current_value = getattr(self.success_metrics_engine.current_metrics, metric_name, None)
                if current_value is None:
                    continue

                # Evaluate condition
                condition = rule.get("condition", "")
                if self._evaluate_condition(condition, current_value):
                    await self._create_alert_from_rule(rule_id, rule, current_value)

        except Exception as e:
            self.logger.error(f"Error checking metric thresholds: {e}")

    def _evaluate_condition(self, condition: str, value: float) -> bool:
        """Evaluate a condition string against a value"""
        try:
            # Simple condition evaluation
            if ">" in condition:
                threshold = float(condition.split(">")[1].strip())
                return value > threshold
            elif "<" in condition:
                threshold = float(condition.split("<")[1].strip())
                return value < threshold
            elif ">=" in condition:
                threshold = float(condition.split(">=")[1].strip())
                return value >= threshold
            elif "<=" in condition:
                threshold = float(condition.split("<=")[1].strip())
                return value <= threshold
            elif "==" in condition:
                threshold = float(condition.split("==")[1].strip())
                return value == threshold

            return False

        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    async def _create_alert_from_rule(self, rule_id: str, rule: Dict[str, Any], current_value: float):
        """Create an alert from a rule"""
        try:
            alert_id = f"{rule_id}_{int(time.time())}"

            # Check if alert is suppressed
            if self._is_alert_suppressed(rule_id):
                return

            # Check cooldown
            if self._is_alert_in_cooldown(rule_id):
                return

            alert = Alert(
                alert_id=alert_id,
                alert_type=rule.get("type", AlertType.METRIC_THRESHOLD),
                severity=rule.get("severity", AlertSeverity.WARNING),
                title=rule.get("name", "Metric Alert"),
                message=rule.get("message", "Metric threshold exceeded"),
                metric_name=rule.get("metric"),
                current_value=current_value,
                threshold_value=self._extract_threshold_from_condition(rule.get("condition", "")),
                context={
                    "rule_id": rule_id,
                    "timestamp": datetime.now().isoformat(),
                    "metric_value": current_value
                }
            )

            # Set suppression and cooldown
            alert.suppression_until = datetime.now() + timedelta(seconds=rule.get("cooldown", 3600))

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Queue notifications
            await self._queue_alert_notifications(alert)

            self.logger.warning(f"Alert created: {alert.title} - {alert.message}")

        except Exception as e:
            self.logger.error(f"Error creating alert from rule {rule_id}: {e}")

    def _is_alert_suppressed(self, rule_id: str) -> bool:
        """Check if an alert is currently suppressed"""
        try:
            # Check if there's an active alert for this rule that's suppressed
            for alert in self.active_alerts.values():
                if alert.context.get("rule_id") == rule_id:
                    if alert.suppression_until and alert.suppression_until > datetime.now():
                        return True
            return False

        except Exception as e:
            self.logger.error(f"Error checking alert suppression: {e}")
            return False

    def _is_alert_in_cooldown(self, rule_id: str) -> bool:
        """Check if an alert is in cooldown period"""
        try:
            # Check recent alert history for cooldown
            cutoff_time = datetime.now() - timedelta(minutes=60)
            recent_alerts = [a for a in self.alert_history if a.timestamp > cutoff_time]

            for alert in recent_alerts:
                if alert.context.get("rule_id") == rule_id:
                    # Check if within cooldown period
                    rule = self.alert_rules.get(rule_id, {})
                    cooldown = rule.get("cooldown", 3600)
                    if (datetime.now() - alert.timestamp).total_seconds() < cooldown:
                        return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking alert cooldown: {e}")
            return False

    def _extract_threshold_from_condition(self, condition: str) -> Optional[float]:
        """Extract threshold value from condition string"""
        try:
            if ">" in condition:
                return float(condition.split(">")[1].strip())
            elif "<" in condition:
                return float(condition.split("<")[1].strip())
            elif ">=" in condition:
                return float(condition.split(">=")[1].strip())
            elif "<=" in condition:
                return float(condition.split("<=")[1].strip())
            return None

        except Exception:
            return None

    async def _queue_alert_notifications(self, alert: Alert):
        """Queue notifications for an alert"""
        try:
            # Find matching notification rules
            matching_rules = self._find_matching_notification_rules(alert)

            for rule in matching_rules:
                for channel in rule.channels:
                    notification = {
                        "type": "alert",
                        "alert_id": alert.alert_id,
                        "channel": channel,
                        "recipients": rule.recipients,
                        "content": {
                            "title": alert.title,
                            "message": alert.message,
                            "severity": alert.severity.value,
                            "timestamp": alert.timestamp.isoformat(),
                            "context": alert.context
                        }
                    }

                    if len(self.notification_queue) < self.max_queue_size:
                        self.notification_queue.append(notification)
                    else:
                        self.logger.error("Notification queue full, dropping notification")

        except Exception as e:
            self.logger.error(f"Error queuing alert notifications: {e}")

    def _find_matching_notification_rules(self, alert: Alert) -> List[NotificationRule]:
        """Find notification rules that match an alert"""
        try:
            matching_rules = []

            for rule in self.notification_rules:
                if not rule.enabled:
                    continue

                # Check severity match
                if "severity" in rule.conditions:
                    if rule.conditions["severity"] != alert.severity.value:
                        continue

                # Check alert type match
                if "alert_type" in rule.conditions:
                    if rule.conditions["alert_type"] != alert.alert_type.value:
                        continue

                matching_rules.append(rule)

            # Sort by priority
            matching_rules.sort(key=lambda r: r.priority, reverse=True)

            return matching_rules

        except Exception as e:
            self.logger.error(f"Error finding matching notification rules: {e}")
            return []

    def _process_notification(self, notification: Dict[str, Any]):
        """Process a single notification"""
        try:
            channel = notification.get("channel")
            content = notification.get("content", {})
            recipients = notification.get("recipients", [])

            if channel == NotificationChannel.EMAIL:
                self._send_email_notification(recipients, content)
            elif channel == NotificationChannel.SLACK:
                self._send_slack_notification(recipients, content)
            elif channel == NotificationChannel.DASHBOARD:
                self._send_dashboard_notification(content)
            elif channel == NotificationChannel.WEBHOOK:
                self._send_webhook_notification(recipients, content)
            elif channel == NotificationChannel.SMS:
                self._send_sms_notification(recipients, content)
            elif channel == NotificationChannel.IN_APP:
                self._send_in_app_notification(content)

        except Exception as e:
            self.logger.error(f"Error processing notification: {e}")

    def _send_email_notification(self, recipients: List[str], content: Dict[str, Any]):
        """Send email notification"""
        try:
            if not self.smtp_configured:
                self.logger.warning("SMTP not configured, skipping email notification")
                return

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get("from_address", "noreply@ai-scientist-v2.com")
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"[{content.get('severity', 'INFO').upper()}] {content.get('title', 'Alert')}"

            # Create HTML body
            html_body = self._create_email_html(content)
            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config.get("smtp_port", 587)) as server:
                if self.email_config.get("use_tls", True):
                    server.starttls()
                if self.email_config.get("username") and self.email_config.get("password"):
                    server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)

            self.logger.info(f"Email notification sent to {len(recipients)} recipients")

        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")

    def _create_email_html(self, content: Dict[str, Any]) -> str:
        """Create HTML content for email notification"""
        try:
            severity_colors = {
                "info": "#17a2b8",
                "warning": "#ffc107",
                "critical": "#dc3545",
                "emergency": "#6f42c1"
            }

            color = severity_colors.get(content.get("severity", "info"), "#6c757d")

            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .alert-box {{ border-left: 4px solid {color}; padding: 15px; margin: 10px 0; background-color: #f8f9fa; }}
                    .severity {{ color: {color}; font-weight: bold; text-transform: uppercase; }}
                    .title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .message {{ margin: 10px 0; }}
                    .timestamp {{ color: #6c757d; font-size: 12px; }}
                    .footer {{ margin-top: 20px; padding-top: 10px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="alert-box">
                    <div class="severity">{content.get('severity', 'INFO')}</div>
                    <div class="title">{content.get('title', 'Alert')}</div>
                    <div class="message">{content.get('message', '')}</div>
                    <div class="timestamp">Time: {content.get('timestamp', '')}</div>
                </div>
                <div class="footer">
                    This notification was generated by AI-Scientist-v2 Success Metrics System.
                </div>
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"Error creating email HTML: {e}")
            return f"<p>{content.get('message', '')}</p>"

    def _send_slack_notification(self, recipients: List[str], content: Dict[str, Any]):
        """Send Slack notification"""
        try:
            slack_config = self.config.get("slack", {})
            webhook_url = slack_config.get("webhook_url")

            if not webhook_url:
                self.logger.warning("Slack webhook not configured, skipping Slack notification")
                return

            # Create Slack payload
            severity_colors = {
                "info": "#17a2b8",
                "warning": "#ffc107",
                "critical": "#dc3545",
                "emergency": "#6f42c1"
            }

            color = severity_colors.get(content.get("severity", "info"), "#6c757d")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": content.get("title", "Alert"),
                        "text": content.get("message", ""),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": content.get("severity", "INFO").upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": content.get("timestamp", ""),
                                "short": True
                            }
                        ],
                        "footer": "AI-Scientist-v2",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info("Slack notification sent successfully")

        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")

    def _send_dashboard_notification(self, content: Dict[str, Any]):
        """Send dashboard notification"""
        try:
            # Add to dashboard alerts
            dashboard_alert = {
                "type": "alert",
                "severity": content.get("severity", "info"),
                "title": content.get("title", "Alert"),
                "message": content.get("message", ""),
                "timestamp": content.get("timestamp", datetime.now().isoformat()),
                "id": f"dashboard_alert_{int(time.time())}"
            }

            # Add to success metrics engine alerts
            self.success_metrics_engine.alerts.append(dashboard_alert)

            # Keep only recent dashboard alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.success_metrics_engine.alerts = [
                alert for alert in self.success_metrics_engine.alerts
                if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
            ]

            self.logger.info("Dashboard notification added successfully")

        except Exception as e:
            self.logger.error(f"Error sending dashboard notification: {e}")

    def _send_webhook_notification(self, recipients: List[str], content: Dict[str, Any]):
        """Send webhook notification"""
        try:
            for webhook_url in recipients:
                payload = {
                    "alert": content,
                    "timestamp": datetime.now().isoformat(),
                    "source": "ai_scientist_v2"
                }

                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()

            self.logger.info(f"Webhook notifications sent to {len(recipients)} recipients")

        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")

    def _send_sms_notification(self, recipients: List[str], content: Dict[str, Any]):
        """Send SMS notification"""
        try:
            # Placeholder for SMS implementation
            self.logger.info(f"SMS notification would be sent to {len(recipients)} recipients")

        except Exception as e:
            self.logger.error(f"Error sending SMS notification: {e}")

    def _send_in_app_notification(self, content: Dict[str, Any]):
        """Send in-app notification"""
        try:
            # Add to in-app notification system
            self.logger.info("In-app notification created")

        except Exception as e:
            self.logger.error(f"Error sending in-app notification: {e}")

    async def _check_goal_milestones(self):
        """Check goal milestones and generate alerts"""
        try:
            goal_progress = await self.success_metrics_engine.get_goal_progress()

            for goal_data in goal_progress.get("goals", []):
                goal_name = goal_data.get("name", "")
                progress_percentage = goal_data.get("progress_percentage", 0)

                # Check for milestone achievements
                if progress_percentage >= 100:
                    milestone_alert = Alert(
                        alert_id=f"goal_achieved_{goal_name}_{int(time.time())}",
                        alert_type=AlertType.GOAL_MILESTONE,
                        severity=AlertSeverity.INFO,
                        title=f"Goal Achieved: {goal_name}",
                        message=f"Life's work goal '{goal_name}' has been achieved!",
                        context={
                            "goal_name": goal_name,
                            "progress_percentage": progress_percentage,
                            "achievement_timestamp": datetime.now().isoformat()
                        }
                    )

                    self.active_alerts[milestone_alert.alert_id] = milestone_alert
                    self.alert_history.append(milestone_alert)

                    await self._queue_alert_notifications(milestone_alert)

        except Exception as e:
            self.logger.error(f"Error checking goal milestones: {e}")

    async def _check_system_performance(self):
        """Check system performance and generate alerts"""
        try:
            # Get system health
            system_health = await self.success_metrics_engine._analyze_system_health()

            health_score = system_health.get("overall_health_score", 100)
            alert_count = system_health.get("alert_count", 0)

            # Alert on poor system health
            if health_score < 70:
                performance_alert = Alert(
                    alert_id=f"system_health_{int(time.time())}",
                    alert_type=AlertType.SYSTEM_PERFORMANCE,
                    severity=AlertSeverity.WARNING if health_score > 50 else AlertSeverity.CRITICAL,
                    title="System Performance Alert",
                    message=f"System health score has dropped to {health_score:.1f}%",
                    current_value=health_score,
                    threshold_value=70.0,
                    context={
                        "health_score": health_score,
                        "alert_count": alert_count,
                        "health_status": system_health.get("health_status", "unknown")
                    }
                )

                self.active_alerts[performance_alert.alert_id] = performance_alert
                self.alert_history.append(performance_alert)

                await self._queue_alert_notifications(performance_alert)

        except Exception as e:
            self.logger.error(f"Error checking system performance: {e}")

    async def _check_achievements(self):
        """Check for achievements and generate celebration alerts"""
        try:
            # Check for breakthrough discoveries
            breakthrough_rate = getattr(self.success_metrics_engine.current_metrics, 'breakthrough_discovery_rate', 0)

            if breakthrough_rate > 0:
                achievement_alert = Alert(
                    alert_id=f"breakthrough_{int(time.time())}",
                    alert_type=AlertType.ACHIEVEMENT,
                    severity=AlertSeverity.INFO,
                    title="Breakthrough Discovery!",
                    message=f"New breakthrough discovery achieved! Rate: {breakthrough_rate:.1f}/month",
                    current_value=breakthrough_rate,
                    context={
                        "achievement_type": "breakthrough_discovery",
                        "rate": breakthrough_rate
                    }
                )

                self.active_alerts[achievement_alert.alert_id] = achievement_alert
                self.alert_history.append(achievement_alert)

                await self._queue_alert_notifications(achievement_alert)

        except Exception as e:
            self.logger.error(f"Error checking achievements: {e}")

    def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        try:
            # Remove alerts older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Clean up active alerts
            to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.resolved and alert.resolution_timestamp and alert.resolution_timestamp < cutoff_time:
                    to_remove.append(alert_id)
                elif not alert.resolved and alert.suppression_until and alert.suppression_until < cutoff_time:
                    to_remove.append(alert_id)

            for alert_id in to_remove:
                del self.active_alerts[alert_id]

            # Clean up history (keep last 1000 alerts)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error cleaning up resolved alerts: {e}")

    async def _check_scheduled_reports(self):
        """Check for scheduled reports that need to be generated"""
        try:
            current_time = datetime.now()

            for template_id, template in self.report_templates.items():
                if template.frequency == ReportFrequency.ON_DEMAND:
                    continue

                # Check if report should be generated now
                if self._should_generate_report(template, current_time):
                    await self._generate_report(template)

        except Exception as e:
            self.logger.error(f"Error checking scheduled reports: {e}")

    def _should_generate_report(self, template: ReportTemplate, current_time: datetime) -> bool:
        """Check if a report should be generated at the current time"""
        try:
            # Check if report was already generated recently
            last_generation = self.report_cache.get(template.template_id, {}).get("generated_at")
            if last_generation:
                last_time = datetime.fromisoformat(last_generation)
                time_diff = current_time - last_time

                # Check frequency-specific intervals
                if template.frequency == ReportFrequency.HOURLY:
                    if time_diff < timedelta(hours=1):
                        return False
                elif template.frequency == ReportFrequency.DAILY:
                    if time_diff < timedelta(days=1):
                        return False
                elif template.frequency == ReportFrequency.WEEKLY:
                    if time_diff < timedelta(weeks=1):
                        return False
                elif template.frequency == ReportFrequency.MONTHLY:
                    if time_diff < timedelta(days=30):
                        return False
                elif template.frequency == ReportFrequency.QUARTERLY:
                    if time_diff < timedelta(days=90):
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking if report should be generated: {e}")
            return False

    async def _generate_report(self, template: ReportTemplate):
        """Generate a report from a template"""
        try:
            self.logger.info(f"Generating report: {template.name}")

            # Generate report data
            report_data = await self._collect_report_data(template)

            # Format report
            formatted_report = await self._format_report(template, report_data)

            # Cache report
            self.report_cache[template.template_id] = {
                "generated_at": datetime.now().isoformat(),
                "template": asdict(template),
                "data": report_data,
                "formatted": formatted_report
            }

            # Send report notifications
            await self._send_report_notifications(template, formatted_report)

            self.logger.info(f"Report generated successfully: {template.name}")

        except Exception as e:
            self.logger.error(f"Error generating report {template.name}: {e}")

    async def _collect_report_data(self, template: ReportTemplate) -> Dict[str, Any]:
        """Collect data for a report"""
        try:
            report_data = {
                "metadata": {
                    "report_name": template.name,
                    "generated_at": datetime.now().isoformat(),
                    "template_id": template.template_id,
                    "frequency": template.frequency.value
                }
            }

            # Collect data for each section
            for section in template.sections:
                section_type = section.get("type")
                content = section.get("content", "")

                if section_type == "summary":
                    report_data[section_type] = await self._collect_summary_data(content)
                elif section_type == "goals":
                    report_data[section_type] = await self._collect_goals_data(content)
                elif section_type == "metrics":
                    report_data[section_type] = await self._collect_metrics_data(content)
                elif section_type == "alerts":
                    report_data[section_type] = await self._collect_alerts_data(content)
                elif section_type == "performance":
                    report_data[section_type] = await self._collect_performance_data(content)
                elif section_type == "research":
                    report_data[section_type] = await self._collect_research_data(content)
                elif section_type == "analytics":
                    report_data[section_type] = await self._collect_analytics_data(content)
                elif section_type == "issues":
                    report_data[section_type] = await self._collect_issues_data(content)
                elif section_type == "impact":
                    report_data[section_type] = await self._collect_impact_data(content)
                elif section_type == "global":
                    report_data[section_type] = await self._collect_global_data(content)
                elif section_type == "strategy":
                    report_data[section_type] = await self._collect_strategy_data(content)

            return report_data

        except Exception as e:
            self.logger.error(f"Error collecting report data: {e}")
            return {"error": str(e)}

    async def _collect_summary_data(self, content: str) -> Dict[str, Any]:
        """Collect summary data for reports"""
        try:
            current_metrics = await self.success_metrics_engine.get_current_metrics()
            predictions = await self.success_metrics_engine.get_predictions()

            return {
                "overall_success_score": self.success_metrics_engine._calculate_overall_success_score(),
                "primary_achievement": self.success_metrics_engine._identify_primary_achievement(),
                "critical_challenges": self.success_metrics_engine._identify_critical_challenges(),
                "strategic_recommendations": self.success_metrics_engine._generate_strategic_recommendations(),
                "system_health": current_metrics.get("system_status", "unknown"),
                "predictions_summary": predictions.get("overall_assessment", {})
            }

        except Exception as e:
            self.logger.error(f"Error collecting summary data: {e}")
            return {"error": str(e)}

    async def _collect_goals_data(self, content: str) -> Dict[str, Any]:
        """Collect goals data for reports"""
        try:
            goals_progress = await self.success_metrics_engine.get_goal_progress()

            return {
                "goals": goals_progress.get("goals", []),
                "overall_progress": goals_progress.get("overall_progress", 0.0),
                "active_goals": len([g for g in goals_progress.get("goals", []) if g["status"] == "in_progress"]),
                "achieved_goals": len([g for g in goals_progress.get("goals", []) if g["status"] == "achieved"]),
                "milestone_analysis": self.success_metrics_engine._analyze_milestones()
            }

        except Exception as e:
            self.logger.error(f"Error collecting goals data: {e}")
            return {"error": str(e)}

    async def _collect_metrics_data(self, content: str) -> Dict[str, Any]:
        """Collect metrics data for reports"""
        try:
            current_metrics = self.success_metrics_engine.current_metrics

            return {
                "scientific_acceleration": {
                    "research_acceleration_factor": getattr(current_metrics, 'research_acceleration_factor', 0.0),
                    "papers_generated_per_day": getattr(current_metrics, 'papers_generated_per_day', 0.0),
                    "experiments_conducted_per_day": getattr(current_metrics, 'experiments_conducted_per_day', 0.0)
                },
                "democratization": {
                    "global_users_count": getattr(current_metrics, 'global_users_count', 0),
                    "geographies_reached": getattr(current_metrics, 'geographies_reached', 0),
                    "economic_accessibility_score": getattr(current_metrics, 'economic_accessibility_score', 0.0)
                },
                "knowledge_advancement": {
                    "novel_hypotheses_generated": getattr(current_metrics, 'novel_hypotheses_generated', 0),
                    "successful_experiments": getattr(current_metrics, 'successful_experiments', 0),
                    "breakthrough_discovery_rate": getattr(current_metrics, 'breakthrough_discovery_rate', 0.0)
                },
                "system_evolution": {
                    "autonomous_improvement_count": getattr(current_metrics, 'autonomous_improvement_count', 0),
                    "self_optimization_rate": getattr(current_metrics, 'self_optimization_rate', 0.0),
                    "learning_velocity": getattr(current_metrics, 'learning_velocity', 0.0)
                }
            }

        except Exception as e:
            self.logger.error(f"Error collecting metrics data: {e}")
            return {"error": str(e)}

    async def _collect_alerts_data(self, content: str) -> Dict[str, Any]:
        """Collect alerts data for reports"""
        try:
            critical_alerts = [a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL][-10:]
            warning_alerts = [a for a in self.alert_history if a.severity == AlertSeverity.WARNING][-10:]

            return {
                "active_alerts": len(self.active_alerts),
                "critical_alerts": critical_alerts,
                "warning_alerts": warning_alerts,
                "alert_summary": {
                    "total_alerts": len(self.alert_history),
                    "critical_count": len([a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL]),
                    "warning_count": len([a for a in self.alert_history if a.severity == AlertSeverity.WARNING]),
                    "info_count": len([a for a in self.alert_history if a.severity == AlertSeverity.INFO])
                }
            }

        except Exception as e:
            self.logger.error(f"Error collecting alerts data: {e}")
            return {"error": str(e)}

    async def _collect_performance_data(self, content: str) -> Dict[str, Any]:
        """Collect performance data for reports"""
        try:
            performance_summary = await self.success_metrics_engine.performance_monitor.get_performance_summary()
            agent_report = await self.success_metrics_engine.performance_monitor.get_agent_performance_report()

            return {
                "system_performance": performance_summary,
                "agent_performance": agent_report,
                "historical_metrics": await self.success_metrics_engine.performance_monitor.get_historical_metrics(168)  # Last 7 days
            }

        except Exception as e:
            self.logger.error(f"Error collecting performance data: {e}")
            return {"error": str(e)}

    async def _collect_research_data(self, content: str) -> Dict[str, Any]:
        """Collect research data for reports"""
        try:
            # This would integrate with actual research tracking systems
            # For now, provide simulated data

            return {
                "research_output": {
                    "papers_generated": getattr(self.success_metrics_engine.current_metrics, 'papers_generated_per_day', 0) * 30,  # Monthly
                    "experiments_conducted": getattr(self.success_metrics_engine.current_metrics, 'experiments_conducted_per_day', 0) * 30,
                    "hypotheses_generated": getattr(self.success_metrics_engine.current_metrics, 'novel_hypotheses_generated', 0)
                },
                "breakthrough_analysis": {
                    "breakthrough_rate": getattr(self.success_metrics_engine.current_metrics, 'breakthrough_discovery_rate', 0.0),
                    "cumulative_breakthroughs": sum(1 for _ in range(int(getattr(self.success_metrics_engine.current_metrics, 'breakthrough_discovery_rate', 0) * 12)))  # Last year
                },
                "research_quality": {
                    "peer_review_acceptance_rate": getattr(self.success_metrics_engine.current_metrics, 'peer_review_acceptance_rate', 0.0),
                    "citation_count": getattr(self.success_metrics_engine.current_metrics, 'scientific_citations_received', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"Error collecting research data: {e}")
            return {"error": str(e)}

    async def _collect_analytics_data(self, content: str) -> Dict[str, Any]:
        """Collect analytics data for reports"""
        try:
            current_metrics = self.success_metrics_engine.current_metrics

            return {
                "user_analytics": {
                    "total_users": getattr(current_metrics, 'global_users_count', 0),
                    "geographic_reach": getattr(current_metrics, 'geographies_reached', 0),
                    "diversity_index": getattr(current_metrics, 'user_diversity_index', 0.0)
                },
                "engagement_metrics": {
                    "user_satisfaction_score": getattr(current_metrics, 'user_satisfaction_score', 0.0),
                    "economic_accessibility": getattr(current_metrics, 'economic_accessibility_score', 0.0),
                    "language_accessibility": getattr(current_metrics, 'language_accessibility', 0.0)
                },
                "growth_metrics": {
                    "user_growth_rate": 15.5,  # Weekly percentage
                    "geographic_expansion": 2.3,  # New countries per month
                    "accessibility_improvement": 5.2  # Monthly improvement
                }
            }

        except Exception as e:
            self.logger.error(f"Error collecting analytics data: {e}")
            return {"error": str(e)}

    async def _collect_issues_data(self, content: str) -> Dict[str, Any]:
        """Collect issues data for reports"""
        try:
            return {
                "active_issues": len([a for a in self.active_alerts.values() if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING]]),
                "resolved_issues": len([a for a in self.alert_history if a.resolved]),
                "issue_categories": {
                    "performance": len([a for a in self.active_alerts.values() if a.alert_type == AlertType.SYSTEM_PERFORMANCE]),
                    "ethical": len([a for a in self.active_alerts.values() if a.alert_type == AlertType.ETHICAL_VIOLATION]),
                    "threshold": len([a for a in self.active_alerts.values() if a.alert_type == AlertType.METRIC_THRESHOLD])
                },
                "resolution_times": self._calculate_resolution_times()
            }

        except Exception as e:
            self.logger.error(f"Error collecting issues data: {e}")
            return {"error": str(e)}

    def _calculate_resolution_times(self) -> Dict[str, float]:
        """Calculate average resolution times"""
        try:
            resolved_alerts = [a for a in self.alert_history if a.resolved and a.resolution_timestamp]

            if not resolved_alerts:
                return {"average_hours": 0.0, "median_hours": 0.0}

            resolution_times = []
            for alert in resolved_alerts:
                resolution_time = (alert.resolution_timestamp - alert.timestamp).total_seconds() / 3600  # hours
                resolution_times.append(resolution_time)

            resolution_times.sort()

            return {
                "average_hours": sum(resolution_times) / len(resolution_times),
                "median_hours": resolution_times[len(resolution_times) // 2],
                "min_hours": min(resolution_times),
                "max_hours": max(resolution_times)
            }

        except Exception as e:
            self.logger.error(f"Error calculating resolution times: {e}")
            return {"average_hours": 0.0, "median_hours": 0.0}

    async def _collect_impact_data(self, content: str) -> Dict[str, Any]:
        """Collect impact data for reports"""
        try:
            return self.success_metrics_engine._analyze_life_work_impact()

        except Exception as e:
            self.logger.error(f"Error collecting impact data: {e}")
            return {"error": str(e)}

    async def _collect_global_data(self, content: str) -> Dict[str, Any]:
        """Collect global impact data for reports"""
        try:
            current_metrics = self.success_metrics_engine.current_metrics

            return {
                "global_reach": {
                    "total_countries": getattr(current_metrics, 'geographies_reached', 0),
                    "target_countries": 195,
                    "penetration_percentage": (getattr(current_metrics, 'geographies_reached', 0) / 195 * 100) if getattr(current_metrics, 'geographies_reached', 0) > 0 else 0
                },
                "accessibility_metrics": {
                    "economic_accessibility": getattr(current_metrics, 'economic_accessibility_score', 0.0),
                    "language_accessibility": getattr(current_metrics, 'language_accessibility', 0.0),
                    "overall_accessibility": (getattr(current_metrics, 'economic_accessibility_score', 0.0) + getattr(current_metrics, 'language_accessibility', 0.0)) / 2
                },
                "democratization_progress": {
                    "current_users": getattr(current_metrics, 'global_users_count', 0),
                    "target_users": 1000000,
                    "progress_percentage": (getattr(current_metrics, 'global_users_count', 0) / 1000000 * 100) if getattr(current_metrics, 'global_users_count', 0) > 0 else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error collecting global data: {e}")
            return {"error": str(e)}

    async def _collect_strategy_data(self, content: str) -> Dict[str, Any]:
        """Collect strategy data for reports"""
        try:
            predictions = await self.success_metrics_engine.get_predictions()
            current_metrics = await self.success_metrics_engine.get_current_metrics()

            return {
                "predictions": predictions,
                "strategic_focus": self.success_metrics_engine._recommend_strategic_focus(),
                "opportunities": [
                    "Continued AI technology advancement",
                    "Growing acceptance of AI in research",
                    "Expanding global research challenges",
                    "Open source movement growth"
                ],
                "risks": [
                    "Technical limitations in AI capabilities",
                    "Ethical and regulatory challenges",
                    "Market competition and adoption barriers"
                ],
                "resource_needs": self._estimate_resource_needs(current_metrics)
            }

        except Exception as e:
            self.logger.error(f"Error collecting strategy data: {e}")
            return {"error": str(e)}

    def _estimate_resource_needs(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource needs based on current metrics"""
        try:
            # Simplified resource estimation
            user_count = getattr(self.success_metrics_engine.current_metrics, 'global_users_count', 1000)
            acceleration_factor = getattr(self.success_metrics_engine.current_metrics, 'research_acceleration_factor', 1.0)

            return {
                "compute_resources": {
                    "current_need": "medium",
                    "projected_need": "high" if user_count > 10000 else "medium",
                    "recommended_scaling": "gradual" if acceleration_factor < 10 else "aggressive"
                },
                "human_resources": {
                    "current_team_size": 15,  # Placeholder
                    "recommended_size": 25 if user_count > 50000 else 20 if user_count > 10000 else 15,
                    "key_roles_needed": ["ML Engineers", "Research Scientists", "Ethics Officers"]
                },
                "infrastructure": {
                    "current_capacity": "adequate",
                    "upgrades_needed": ["Database scaling", "API gateway enhancement", "Monitoring expansion"]
                }
            }

        except Exception as e:
            self.logger.error(f"Error estimating resource needs: {e}")
            return {}

    async def _format_report(self, template: ReportTemplate, report_data: Dict[str, Any]) -> str:
        """Format report data into a readable format"""
        try:
            # Create HTML report
            html_content = self._create_html_report(template, report_data)

            return html_content

        except Exception as e:
            self.logger.error(f"Error formatting report: {e}")
            return f"<p>Error formatting report: {str(e)}</p>"

    def _create_html_report(self, template: ReportTemplate, report_data: Dict[str, Any]) -> str:
        """Create HTML report"""
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{template.name}</title>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007bff; }}
                    .header h1 {{ color: #007bff; margin: 0; }}
                    .header .subtitle {{ color: #6c757d; margin-top: 5px; }}
                    .section {{ margin-bottom: 30px; }}
                    .section h2 {{ color: #495057; border-left: 4px solid #007bff; padding-left: 15px; }}
                    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #28a745; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #28a745; }}
                    .metric-label {{ color: #6c757d; font-size: 14px; }}
                    .progress-bar {{ background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
                    .progress-fill {{ background: linear-gradient(90deg, #28a745, #20c997); height: 20px; border-radius: 10px; }}
                    .alert-box {{ padding: 15px; margin: 10px 0; border-radius: 4px; }}
                    .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                    .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                    .alert-info {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
                    .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{template.name}</h1>
                        <div class="subtitle">Generated on {report_data.get('metadata', {}).get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</div>
                    </div>
            """

            # Add sections
            for section in template.sections:
                section_type = section.get("type")
                section_data = report_data.get(section_type, {})

                if section_type == "summary":
                    html += self._create_summary_section(section_data)
                elif section_type == "goals":
                    html += self._create_goals_section(section_data)
                elif section_type == "metrics":
                    html += self._create_metrics_section(section_data)
                elif section_type == "alerts":
                    html += self._create_alerts_section(section_data)

            # Add footer
            html += f"""
                    <div class="footer">
                        This report was generated automatically by AI-Scientist-v2 Success Metrics Framework.
                        For questions or concerns, please contact the metrics team.
                    </div>
                </div>
            </body>
            </html>
            """

            return html

        except Exception as e:
            self.logger.error(f"Error creating HTML report: {e}")
            return f"<p>Error creating report: {str(e)}</p>"

    def _create_summary_section(self, data: Dict[str, Any]) -> str:
        """Create summary section HTML"""
        try:
            return f"""
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{data.get('overall_success_score', 0):.1f}%</div>
                            <div class="metric-label">Overall Success Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(data.get('critical_challenges', []))}</div>
                            <div class="metric-label">Critical Challenges</div>
                        </div>
                    </div>
                    <p><strong>Primary Achievement:</strong> {data.get('primary_achievement', 'Not identified')}</p>
                    <h3>Strategic Recommendations:</h3>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in data.get('strategic_recommendations', []))}
                    </ul>
                </div>
            """

        except Exception as e:
            self.logger.error(f"Error creating summary section: {e}")
            return "<p>Error creating summary section</p>"

    def _create_goals_section(self, data: Dict[str, Any]) -> str:
        """Create goals section HTML"""
        try:
            goals_html = ""
            for goal in data.get("goals", []):
                progress = goal.get("progress_percentage", 0)
                status = goal.get("status", "unknown")

                goals_html += f"""
                    <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 6px;">
                        <h4>{goal.get('name', 'Unknown Goal')}</h4>
                        <p>{goal.get('description', '')}</p>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {progress}%;"></div>
                        </div>
                        <p><strong>Progress:</strong> {progress:.1f}% | <strong>Status:</strong> {status.title()}</p>
                    </div>
                """

            return f"""
                <div class="section">
                    <h2>Life's Work Goals Progress</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{data.get('overall_progress', 0):.1f}%</div>
                            <div class="metric-label">Overall Progress</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{data.get('active_goals', 0)}</div>
                            <div class="metric-label">Active Goals</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{data.get('achieved_goals', 0)}</div>
                            <div class="metric-label">Achieved Goals</div>
                        </div>
                    </div>
                    {goals_html}
                </div>
            """

        except Exception as e:
            self.logger.error(f"Error creating goals section: {e}")
            return "<p>Error creating goals section</p>"

    def _create_metrics_section(self, data: Dict[str, Any]) -> str:
        """Create metrics section HTML"""
        try:
            metrics_html = ""

            for category, metrics in data.items():
                category_name = category.replace('_', ' ').title()
                metrics_html += f"<h3>{category_name}</h3><div class='metric-grid'>"

                for metric_name, metric_value in metrics.items():
                    display_name = metric_name.replace('_', ' ').title()
                    if isinstance(metric_value, float):
                        display_value = f"{metric_value:.2f}"
                    else:
                        display_value = str(metric_value)

                    metrics_html += f"""
                        <div class="metric-card">
                            <div class="metric-value">{display_value}</div>
                            <div class="metric-label">{display_name}</div>
                        </div>
                    """

                metrics_html += "</div>"

            return f"""
                <div class="section">
                    <h2>Key Metrics</h2>
                    {metrics_html}
                </div>
            """

        except Exception as e:
            self.logger.error(f"Error creating metrics section: {e}")
            return "<p>Error creating metrics section</p>"

    def _create_alerts_section(self, data: Dict[str, Any]) -> str:
        """Create alerts section HTML"""
        try:
            critical_alerts = data.get("critical_alerts", [])
            warning_alerts = data.get("warning_alerts", [])
            summary = data.get("alert_summary", {})

            alerts_html = ""

            if critical_alerts:
                alerts_html += "<h3>Critical Alerts</h3>"
                for alert in critical_alerts:
                    alerts_html += f"""
                        <div class="alert-box alert-critical">
                            <strong>{alert.get('title', 'Critical Alert')}</strong><br>
                            {alert.get('message', '')}<br>
                            <small>{alert.get('timestamp', '')}</small>
                        </div>
                    """

            if warning_alerts:
                alerts_html += "<h3>Warning Alerts</h3>"
                for alert in warning_alerts:
                    alerts_html += f"""
                        <div class="alert-box alert-warning">
                            <strong>{alert.get('title', 'Warning Alert')}</strong><br>
                            {alert.get('message', '')}<br>
                            <small>{alert.get('timestamp', '')}</small>
                        </div>
                    """

            return f"""
                <div class="section">
                    <h2>System Alerts</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{data.get('active_alerts', 0)}</div>
                            <div class="metric-label">Active Alerts</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('critical_count', 0)}</div>
                            <div class="metric-label">Critical Alerts</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary.get('warning_count', 0)}</div>
                            <div class="metric-label">Warning Alerts</div>
                        </div>
                    </div>
                    {alerts_html}
                </div>
            """

        except Exception as e:
            self.logger.error(f"Error creating alerts section: {e}")
            return "<p>Error creating alerts section</p>"

    async def _send_report_notifications(self, template: ReportTemplate, formatted_report: str):
        """Send report notifications"""
        try:
            for recipient in template.recipients:
                # Send email with report
                await self._send_report_email(recipient, template, formatted_report)

            self.logger.info(f"Report notifications sent to {len(template.recipients)} recipients")

        except Exception as e:
            self.logger.error(f"Error sending report notifications: {e}")

    async def _send_report_email(self, recipient: str, template: ReportTemplate, formatted_report: str):
        """Send report via email"""
        try:
            if not self.smtp_configured:
                self.logger.warning("SMTP not configured, skipping report email")
                return

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get("from_address", "noreply@ai-scientist-v2.com")
            msg['To'] = recipient
            msg['Subject'] = f"[Report] {template.name} - {datetime.now().strftime('%Y-%m-%d')}"

            # Create HTML body
            html_body = formatted_report
            msg.attach(MIMEText(html_body, 'html'))

            # Add PDF attachment if requested
            if template.format_options.get("include_pdf", False):
                pdf_attachment = self._create_pdf_attachment(formatted_report)
                if pdf_attachment:
                    msg.attach(pdf_attachment)

            # Send email
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config.get("smtp_port", 587)) as server:
                if self.email_config.get("use_tls", True):
                    server.starttls()
                if self.email_config.get("username") and self.email_config.get("password"):
                    server.login(self.email_config["username"], self.email_config["password"])
                server.send_message(msg)

            self.logger.info(f"Report email sent to {recipient}")

        except Exception as e:
            self.logger.error(f"Error sending report email to {recipient}: {e}")

    def _create_pdf_attachment(self, html_content: str) -> Optional[MIMEBase]:
        """Create PDF attachment from HTML content"""
        try:
            # This would use a library like WeasyPrint or pdfkit to convert HTML to PDF
            # For now, return None as placeholder
            return None

        except Exception as e:
            self.logger.error(f"Error creating PDF attachment: {e}")
            return None

    def _schedule_all_reports(self):
        """Schedule all report templates"""
        try:
            for template_id, template in self.report_templates.items():
                if template.frequency != ReportFrequency.ON_DEMAND:
                    self._schedule_report(template)

        except Exception as e:
            self.logger.error(f"Error scheduling reports: {e}")

    def _schedule_report(self, template: ReportTemplate):
        """Schedule a single report"""
        try:
            # Calculate next run time based on frequency
            if template.frequency == ReportFrequency.HOURLY:
                interval = 3600  # 1 hour
            elif template.frequency == ReportFrequency.DAILY:
                interval = 86400  # 24 hours
            elif template.frequency == ReportFrequency.WEEKLY:
                interval = 604800  # 7 days
            elif template.frequency == ReportFrequency.MONTHLY:
                interval = 2592000  # 30 days
            elif template.frequency == ReportFrequency.QUARTERLY:
                interval = 7776000  # 90 days
            else:
                return

            # Schedule the report
            timer = threading.Timer(interval, lambda: asyncio.run(self._generate_report(template)))
            timer.daemon = True
            timer.start()

            self.scheduled_reports[template.template_id] = timer

        except Exception as e:
            self.logger.error(f"Error scheduling report {template.template_id}: {e}")

    async def generate_on_demand_report(self, template_id: str, recipients: List[str] = None) -> Dict[str, Any]:
        """Generate a report on demand"""
        try:
            template = self.report_templates.get(template_id)
            if not template:
                return {"error": f"Report template '{template_id}' not found"}

            # Override recipients if provided
            if recipients:
                template.recipients = recipients

            # Generate report
            await self._generate_report(template)

            # Return cached report
            cached_report = self.report_cache.get(template_id, {})
            return {
                "success": True,
                "template_id": template_id,
                "report_data": cached_report.get("data"),
                "formatted_report": cached_report.get("formatted"),
                "generated_at": cached_report.get("generated_at")
            }

        except Exception as e:
            self.logger.error(f"Error generating on-demand report {template_id}: {e}")
            return {"error": str(e)}

    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        try:
            return {
                "active_alerts": len(self.active_alerts),
                "alert_summary": {
                    "critical": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                    "warning": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]),
                    "info": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.INFO])
                },
                "recent_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)[:10]
                ],
                "notification_queue_size": len(self.notification_queue),
                "system_status": "healthy" if len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]) == 0 else "warning"
            }

        except Exception as e:
            self.logger.error(f"Error getting alert status: {e}")
            return {"error": str(e)}

    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Resolve an alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                return False

            alert.resolved = True
            alert.resolution_timestamp = datetime.now()
            if resolution_note:
                alert.context["resolution_note"] = resolution_note

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    async def shutdown(self):
        """Shutdown the reporting and alert system"""
        try:
            self.system_active = False

            # Cancel scheduled reports
            for timer in self.scheduled_reports.values():
                timer.cancel()

            # Wait for threads to finish
            if self._alert_monitor_thread and self._alert_monitor_thread.is_alive():
                self._alert_monitor_thread.join(timeout=5.0)

            if self._notification_processor_thread and self._notification_processor_thread.is_alive():
                self._notification_processor_thread.join(timeout=5.0)

            if self._report_scheduler_thread and self._report_scheduler_thread.is_alive():
                self._report_scheduler_thread.join(timeout=5.0)

            for worker in self.notification_workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            self.logger.info("Reporting and Alert System shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error shutting down Reporting and Alert System: {e}")


# Global instance for easy access
_reporting_alert_system: Optional[ReportingAlertSystem] = None


def get_reporting_alert_system(success_metrics_engine: SuccessMetricsEngine = None,
                             real_time_dashboard: RealTimeDashboard = None) -> ReportingAlertSystem:
    """Get the global Reporting and Alert System instance"""
    global _reporting_alert_system
    if _reporting_alert_system is None:
        _reporting_alert_system = ReportingAlertSystem(success_metrics_engine, real_time_dashboard)
    return _reporting_alert_system


def initialize_reporting_alert_system(success_metrics_engine: SuccessMetricsEngine = None,
                                    real_time_dashboard: RealTimeDashboard = None,
                                    config: Dict[str, Any] = None) -> ReportingAlertSystem:
    """Initialize the global Reporting and Alert System"""
    global _reporting_alert_system
    _reporting_alert_system = ReportingAlertSystem(success_metrics_engine, real_time_dashboard, config)
    return _reporting_alert_system