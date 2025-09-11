"""
Cost Tracking and Budget Management for OpenRouter

Comprehensive cost tracking system that monitors API usage,
provides budget alerts, and optimizes costs across different models.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Types of budget alerts"""
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded" 
    RATE_LIMIT_APPROACHING = "rate_limit_approaching"
    UNUSUAL_SPENDING = "unusual_spending"

@dataclass
class UsageRecord:
    """Individual API usage record"""
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    cached_tokens: int = 0
    cache_savings: float = 0.0
    request_id: Optional[str] = None
    stage: Optional[str] = None  # Pipeline stage (ideation, experiment, etc.)

@dataclass
class BudgetAlert:
    """Budget alert record"""
    alert_type: AlertType
    message: str
    timestamp: float
    current_usage: float
    budget_limit: Optional[float] = None
    threshold_percentage: Optional[float] = None

@dataclass
class CostOptimizationSuggestion:
    """Cost optimization suggestion"""
    suggestion_type: str
    description: str
    potential_savings: float
    current_model: str
    suggested_model: Optional[str] = None
    implementation_notes: str = ""

class CostTracker:
    """
    Comprehensive cost tracking and budget management system.
    
    Features:
    - Real-time cost tracking
    - Budget alerts and notifications
    - Cost optimization suggestions
    - Usage analytics and reporting
    - Model cost comparison
    """
    
    def __init__(self, storage_dir: str = "~/.ai_scientist/cost_tracking"):
        """
        Initialize cost tracker.
        
        Args:
            storage_dir: Directory to store cost tracking data
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.usage_file = self.storage_dir / "usage_records.json"
        self.budget_file = self.storage_dir / "budget_config.json"
        self.alerts_file = self.storage_dir / "alerts.json"
        
        # In-memory data
        self.usage_records: List[UsageRecord] = []
        self.budget_config: Dict[str, Any] = {}
        self.alerts: List[BudgetAlert] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing data
        self._load_data()
        
        # Model pricing (simplified - would be updated from API)
        self.model_pricing = {
            "openai/gpt-4o": {"input": 0.005, "output": 0.015},
            "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "openai/o1-preview": {"input": 0.015, "output": 0.06},
            "openai/o1-mini": {"input": 0.003, "output": 0.012},
            "openai/o1": {"input": 0.06, "output": 0.24},
            "openai/o3-mini": {"input": 0.003, "output": 0.012},
            "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
            "anthropic/claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
            "google/gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
            "google/gemini-1.5-pro": {"input": 0.001, "output": 0.002},
            "deepseek/deepseek-v3": {"input": 0.00014, "output": 0.00028},
            "x-ai/grok-2": {"input": 0.002, "output": 0.01},
            "meta-llama/llama-3.1-405b": {"input": 0.002, "output": 0.002},
        }
        
        logger.info(f"Cost tracker initialized with storage at {self.storage_dir}")
    
    def set_budget(self, budget_type: str, amount: float, period: str = "monthly") -> None:
        """
        Set budget limits.
        
        Args:
            budget_type: Type of budget ("total", "per_request", "daily", "monthly")
            amount: Budget amount in USD
            period: Budget period (daily, weekly, monthly)
        """
        with self._lock:
            self.budget_config[budget_type] = {
                "amount": amount,
                "period": period,
                "set_date": time.time()
            }
            self._save_budget_config()
            
        logger.info(f"Set {budget_type} budget: ${amount:.4f} per {period}")
    
    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int,
                    cached_tokens: int = 0, request_id: Optional[str] = None,
                    stage: Optional[str] = None) -> float:
        """
        Record API usage and calculate cost.
        
        Args:
            model: Model identifier
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cached_tokens: Number of cached tokens
            request_id: Optional request ID
            stage: Pipeline stage name
            
        Returns:
            Total cost for this request
        """
        with self._lock:
            # Calculate cost
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
            cache_savings = self._calculate_cache_savings(model, cached_tokens)
            
            # Create usage record
            record = UsageRecord(
                timestamp=time.time(),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost=cost,
                cached_tokens=cached_tokens,
                cache_savings=cache_savings,
                request_id=request_id,
                stage=stage
            )
            
            self.usage_records.append(record)
            
            # Check budget alerts
            self._check_budget_alerts()
            
            # Save data periodically
            if len(self.usage_records) % 10 == 0:
                self._save_usage_records()
            
            logger.debug(f"Recorded usage: {model} - ${cost:.6f} (tokens: {prompt_tokens + completion_tokens})")
            return cost
    
    def get_current_usage(self, period: str = "monthly") -> Dict[str, Any]:
        """
        Get current usage statistics for a period.
        
        Args:
            period: Time period (daily, weekly, monthly, all)
            
        Returns:
            Usage statistics dictionary
        """
        with self._lock:
            now = time.time()
            
            # Calculate period start time
            if period == "daily":
                start_time = now - 24 * 3600
            elif period == "weekly":
                start_time = now - 7 * 24 * 3600
            elif period == "monthly":
                start_time = now - 30 * 24 * 3600
            else:  # all
                start_time = 0
            
            # Filter records
            relevant_records = [r for r in self.usage_records if r.timestamp >= start_time]
            
            if not relevant_records:
                return {
                    "period": period,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_requests": 0,
                    "models_used": {},
                    "stages_breakdown": {},
                    "cache_savings": 0.0
                }
            
            # Calculate statistics
            total_cost = sum(r.cost for r in relevant_records)
            total_tokens = sum(r.total_tokens for r in relevant_records)
            total_cache_savings = sum(r.cache_savings for r in relevant_records)
            
            # Model breakdown
            models_used = {}
            for record in relevant_records:
                if record.model not in models_used:
                    models_used[record.model] = {
                        "requests": 0,
                        "cost": 0.0,
                        "tokens": 0
                    }
                models_used[record.model]["requests"] += 1
                models_used[record.model]["cost"] += record.cost
                models_used[record.model]["tokens"] += record.total_tokens
            
            # Stage breakdown
            stages_breakdown = {}
            for record in relevant_records:
                stage = record.stage or "unknown"
                if stage not in stages_breakdown:
                    stages_breakdown[stage] = {
                        "requests": 0,
                        "cost": 0.0,
                        "tokens": 0
                    }
                stages_breakdown[stage]["requests"] += 1
                stages_breakdown[stage]["cost"] += record.cost
                stages_breakdown[stage]["tokens"] += record.total_tokens
            
            return {
                "period": period,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "total_requests": len(relevant_records),
                "models_used": models_used,
                "stages_breakdown": stages_breakdown,
                "cache_savings": total_cache_savings,
                "average_cost_per_request": total_cost / len(relevant_records),
                "start_time": start_time,
                "end_time": now
            }
    
    def get_optimization_suggestions(self) -> List[CostOptimizationSuggestion]:
        """
        Get cost optimization suggestions based on usage patterns.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Get recent usage (last 7 days)
        recent_usage = self.get_current_usage("weekly")
        
        if not recent_usage["models_used"]:
            return suggestions
        
        # Suggest cheaper alternatives for expensive models
        expensive_models = [
            ("openai/o1", "openai/o1-mini", 5.0),
            ("openai/o1-preview", "openai/o1-mini", 3.0),
            ("anthropic/claude-3-opus", "anthropic/claude-3.5-sonnet", 4.0),
            ("openai/gpt-4o", "openai/gpt-4o-mini", 8.0),
        ]
        
        for expensive, cheaper, savings_ratio in expensive_models:
            if expensive in recent_usage["models_used"]:
                current_cost = recent_usage["models_used"][expensive]["cost"]
                potential_savings = current_cost * (1 - 1/savings_ratio)
                
                suggestions.append(CostOptimizationSuggestion(
                    suggestion_type="model_substitution",
                    description=f"Replace {expensive} with {cheaper} for routine tasks",
                    potential_savings=potential_savings,
                    current_model=expensive,
                    suggested_model=cheaper,
                    implementation_notes=f"Could save ~{savings_ratio:.1f}x on cost while maintaining good quality"
                ))
        
        # Suggest enabling caching
        if recent_usage["cache_savings"] / max(recent_usage["total_cost"], 0.01) < 0.1:
            suggestions.append(CostOptimizationSuggestion(
                suggestion_type="enable_caching",
                description="Enable prompt caching to reduce costs",
                potential_savings=recent_usage["total_cost"] * 0.4,  # Estimate 40% savings
                current_model="all",
                implementation_notes="Enable automatic or ephemeral caching for repeated prompts"
            ))
        
        # Suggest batch processing
        if recent_usage["total_requests"] > 50:
            avg_tokens_per_request = recent_usage["total_tokens"] / recent_usage["total_requests"]
            if avg_tokens_per_request < 500:  # Small requests
                suggestions.append(CostOptimizationSuggestion(
                    suggestion_type="batch_processing",
                    description="Batch small requests together",
                    potential_savings=recent_usage["total_cost"] * 0.15,  # Estimate 15% savings
                    current_model="all",
                    implementation_notes="Combine multiple small requests into single larger ones"
                ))
        
        return suggestions
    
    def generate_cost_report(self, period: str = "monthly") -> Dict[str, Any]:
        """
        Generate comprehensive cost report.
        
        Args:
            period: Reporting period
            
        Returns:
            Detailed cost report
        """
        usage_stats = self.get_current_usage(period)
        optimization_suggestions = self.get_optimization_suggestions()
        
        # Budget analysis
        budget_analysis = {}
        for budget_type, config in self.budget_config.items():
            if config["period"] == period:
                budget_analysis[budget_type] = {
                    "budget_limit": config["amount"],
                    "current_usage": usage_stats["total_cost"],
                    "remaining": config["amount"] - usage_stats["total_cost"],
                    "utilization_percentage": (usage_stats["total_cost"] / config["amount"]) * 100
                }
        
        # Cost trends (simplified)
        cost_trends = self._calculate_cost_trends()
        
        report = {
            "report_period": period,
            "generated_at": datetime.now().isoformat(),
            "usage_statistics": usage_stats,
            "budget_analysis": budget_analysis,
            "optimization_suggestions": [
                {
                    "type": s.suggestion_type,
                    "description": s.description,
                    "potential_savings": s.potential_savings,
                    "current_model": s.current_model,
                    "suggested_model": s.suggested_model,
                    "notes": s.implementation_notes
                }
                for s in optimization_suggestions
            ],
            "cost_trends": cost_trends,
            "alerts": [
                {
                    "type": a.alert_type.value,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in self.alerts[-10:]  # Last 10 alerts
            ]
        }
        
        return report
    
    def export_usage_data(self, format: str = "json", file_path: Optional[str] = None) -> str:
        """
        Export usage data for external analysis.
        
        Args:
            format: Export format ("json", "csv")
            file_path: Optional file path to save
            
        Returns:
            File path where data was saved
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.storage_dir / f"usage_export_{timestamp}.{format}"
        
        file_path = Path(file_path)
        
        if format == "json":
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_records": len(self.usage_records),
                "usage_records": [
                    {
                        "timestamp": r.timestamp,
                        "datetime": datetime.fromtimestamp(r.timestamp).isoformat(),
                        "model": r.model,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "total_tokens": r.total_tokens,
                        "cost": r.cost,
                        "cached_tokens": r.cached_tokens,
                        "cache_savings": r.cache_savings,
                        "stage": r.stage
                    }
                    for r in self.usage_records
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == "csv":
            import csv
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "datetime", "model", "prompt_tokens", 
                    "completion_tokens", "total_tokens", "cost", 
                    "cached_tokens", "cache_savings", "stage"
                ])
                
                for r in self.usage_records:
                    writer.writerow([
                        r.timestamp,
                        datetime.fromtimestamp(r.timestamp).isoformat(),
                        r.model,
                        r.prompt_tokens,
                        r.completion_tokens,
                        r.total_tokens,
                        r.cost,
                        r.cached_tokens,
                        r.cache_savings,
                        r.stage
                    ])
        
        logger.info(f"Usage data exported to {file_path}")
        return str(file_path)
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        """Calculate cost for API usage"""
        pricing = self.model_pricing.get(model, {"input": 0.002, "output": 0.008})
        
        # Base cost calculation
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        
        # Apply caching discount (typical 50-90% savings)
        cache_savings = (cached_tokens / 1000) * pricing["input"] * 0.75  # 75% savings
        
        return max(0.0, prompt_cost + completion_cost - cache_savings)
    
    def _calculate_cache_savings(self, model: str, cached_tokens: int) -> float:
        """Calculate savings from prompt caching"""
        if cached_tokens <= 0:
            return 0.0
        
        pricing = self.model_pricing.get(model, {"input": 0.002, "output": 0.008})
        return (cached_tokens / 1000) * pricing["input"] * 0.75  # 75% savings rate
    
    def _check_budget_alerts(self) -> None:
        """Check for budget alerts"""
        current_usage = self.get_current_usage("monthly")
        
        for budget_type, config in self.budget_config.items():
            budget_limit = config["amount"]
            usage_amount = current_usage["total_cost"]
            utilization = usage_amount / budget_limit
            
            # Budget warning (80% threshold)
            if utilization >= 0.8 and utilization < 1.0:
                alert = BudgetAlert(
                    alert_type=AlertType.BUDGET_WARNING,
                    message=f"{budget_type} budget at {utilization:.1%} ({usage_amount:.4f}/{budget_limit:.4f})",
                    timestamp=time.time(),
                    current_usage=usage_amount,
                    budget_limit=budget_limit,
                    threshold_percentage=80.0
                )
                self._add_alert(alert)
            
            # Budget exceeded
            elif utilization >= 1.0:
                alert = BudgetAlert(
                    alert_type=AlertType.BUDGET_EXCEEDED,
                    message=f"{budget_type} budget exceeded! ({usage_amount:.4f}/{budget_limit:.4f})",
                    timestamp=time.time(),
                    current_usage=usage_amount,
                    budget_limit=budget_limit,
                    threshold_percentage=100.0
                )
                self._add_alert(alert)
    
    def _add_alert(self, alert: BudgetAlert) -> None:
        """Add alert if not duplicate"""
        # Check for recent duplicate alerts (within last hour)
        recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]
        duplicate = any(
            a.alert_type == alert.alert_type and 
            abs(a.current_usage - alert.current_usage) < 0.001
            for a in recent_alerts
        )
        
        if not duplicate:
            self.alerts.append(alert)
            logger.warning(f"Budget Alert: {alert.message}")
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            self._save_alerts()
    
    def _calculate_cost_trends(self) -> Dict[str, Any]:
        """Calculate cost trends (simplified implementation)"""
        now = time.time()
        
        # Get data for last 7 days, grouped by day
        daily_costs = {}
        for record in self.usage_records:
            if now - record.timestamp <= 7 * 24 * 3600:  # Last 7 days
                day = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d")
                daily_costs[day] = daily_costs.get(day, 0.0) + record.cost
        
        if len(daily_costs) < 2:
            return {"trend": "insufficient_data", "daily_costs": daily_costs}
        
        # Simple trend calculation
        costs = list(daily_costs.values())
        if len(costs) >= 2:
            recent_avg = sum(costs[-3:]) / min(3, len(costs))  # Last 3 days average
            earlier_avg = sum(costs[:-3]) / max(1, len(costs) - 3)  # Earlier days average
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "daily_costs": daily_costs,
            "recent_average": recent_avg if len(costs) >= 2 else costs[0] if costs else 0,
            "earlier_average": earlier_avg if len(costs) >= 2 else costs[0] if costs else 0
        }
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        # Load usage records
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    self.usage_records = [
                        UsageRecord(**record) for record in data
                    ]
                logger.info(f"Loaded {len(self.usage_records)} usage records")
            except Exception as e:
                logger.warning(f"Failed to load usage records: {e}")
                self.usage_records = []
        
        # Load budget config
        if self.budget_file.exists():
            try:
                with open(self.budget_file, 'r') as f:
                    self.budget_config = json.load(f)
                logger.info(f"Loaded budget configuration")
            except Exception as e:
                logger.warning(f"Failed to load budget config: {e}")
                self.budget_config = {}
        
        # Load alerts
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    self.alerts = [
                        BudgetAlert(
                            alert_type=AlertType(alert["alert_type"]),
                            message=alert["message"],
                            timestamp=alert["timestamp"],
                            current_usage=alert["current_usage"],
                            budget_limit=alert.get("budget_limit"),
                            threshold_percentage=alert.get("threshold_percentage")
                        )
                        for alert in data
                    ]
                logger.info(f"Loaded {len(self.alerts)} alerts")
            except Exception as e:
                logger.warning(f"Failed to load alerts: {e}")
                self.alerts = []
    
    def _save_usage_records(self) -> None:
        """Save usage records to file"""
        try:
            data = [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "cost": r.cost,
                    "cached_tokens": r.cached_tokens,
                    "cache_savings": r.cache_savings,
                    "request_id": r.request_id,
                    "stage": r.stage
                }
                for r in self.usage_records
            ]
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save usage records: {e}")
    
    def _save_budget_config(self) -> None:
        """Save budget configuration to file"""
        try:
            with open(self.budget_file, 'w') as f:
                json.dump(self.budget_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save budget config: {e}")
    
    def _save_alerts(self) -> None:
        """Save alerts to file"""
        try:
            data = [
                {
                    "alert_type": a.alert_type.value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "current_usage": a.current_usage,
                    "budget_limit": a.budget_limit,
                    "threshold_percentage": a.threshold_percentage
                }
                for a in self.alerts
            ]
            
            with open(self.alerts_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old usage records.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Number of records removed
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self._lock:
            initial_count = len(self.usage_records)
            self.usage_records = [r for r in self.usage_records if r.timestamp >= cutoff_time]
            removed_count = initial_count - len(self.usage_records)
            
            if removed_count > 0:
                self._save_usage_records()
                logger.info(f"Cleaned up {removed_count} old usage records")
            
            return removed_count

# Global cost tracker instance
_global_cost_tracker: Optional[CostTracker] = None

def get_global_cost_tracker() -> CostTracker:
    """Get or create global cost tracker instance"""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker

def initialize_cost_tracker(storage_dir: Optional[str] = None) -> CostTracker:
    """Initialize global cost tracker"""
    global _global_cost_tracker
    _global_cost_tracker = CostTracker(storage_dir or "~/.ai_scientist/cost_tracking")
    return _global_cost_tracker