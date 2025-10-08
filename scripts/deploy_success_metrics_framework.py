#!/usr/bin/env python3
"""
Deploy Success Metrics Framework - AI-Scientist-v2

This script demonstrates the deployment and usage of the comprehensive Success Metrics Framework
for tracking progress toward revolutionizing scientific discovery through responsible multi-agent AI systems.

Usage:
    python scripts/deploy_success_metrics_framework.py

The script will:
1. Initialize the complete Success Metrics Framework
2. Start automated metrics collection
3. Launch real-time monitoring dashboard
4. Generate comprehensive reports
5. Demonstrate goal tracking and analysis
6. Show predictive analytics capabilities

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_scientist.metrics import (
    initialize_success_metrics_framework,
    get_success_metrics_framework,
    SuccessMetricsFramework
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('success_metrics_deployment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class SuccessMetricsDeployment:
    """
    Deployment orchestrator for the Success Metrics Framework

    This class handles the complete deployment process including initialization,
    configuration, testing, and demonstration of all framework capabilities.
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_configuration()
        self.framework: SuccessMetricsFramework = None
        self.deployment_start_time = datetime.now()

    def _load_configuration(self) -> dict:
        """Load configuration from file or use defaults"""
        try:
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                # Use default configuration
                config = {
                    "collection": {
                        "max_workers": 10,
                        "collection_interval": 60,
                        "max_cache_size": 10000,
                        "max_history_size": 100000
                    },
                    "dashboard": {
                        "refresh_interval": 30,
                        "auto_refresh": True
                    },
                    "reporting": {
                        "email": {
                            "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                            "smtp_port": int(os.getenv("SMTP_PORT", 587)),
                            "username": os.getenv("SMTP_USERNAME", ""),
                            "password": os.getenv("SMTP_PASSWORD", ""),
                            "from_address": os.getenv("FROM_ADDRESS", "noreply@ai-scientist-v2.com")
                        },
                        "slack": {
                            "webhook_url": os.getenv("SLACK_WEBHOOK_URL", "")
                        }
                    },
                    "alerts": {
                        "critical_thresholds": {
                            "ethical_compliance": 95.0,
                            "system_error_rate": 0.1,
                            "research_acceleration_factor": 5.0
                        }
                    },
                    "goals": {
                        "100x_acceleration": {
                            "target": 100.0,
                            "deadline": "2028-12-31"
                        },
                        "global_democratization": {
                            "target_users": 1000000,
                            "target_geographies": 195,
                            "deadline": "2027-12-31"
                        }
                    }
                }
                logger.info("Using default configuration")
                return config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    async def deploy_framework(self):
        """Deploy the complete Success Metrics Framework"""
        try:
            logger.info("🚀 Starting Success Metrics Framework Deployment")
            logger.info(f"Deployment started at: {self.deployment_start_time}")

            # Step 1: Initialize the framework
            logger.info("📊 Step 1: Initializing Success Metrics Framework...")
            await self._initialize_framework()

            # Step 2: Verify system health
            logger.info("🏥 Step 2: Verifying system health...")
            await self._verify_system_health()

            # Step 3: Demonstrate metrics collection
            logger.info("📈 Step 3: Demonstrating automated metrics collection...")
            await self._demonstrate_metrics_collection()

            # Step 4: Show real-time dashboard
            logger.info("🖥️  Step 4: Launching real-time monitoring dashboard...")
            await self._launch_dashboard()

            # Step 5: Generate comprehensive reports
            logger.info("📋 Step 5: Generating comprehensive reports...")
            await self._generate_reports()

            # Step 6: Execute life's work assessment
            logger.info("🎯 Step 6: Executing comprehensive life's work assessment...")
            await self._execute_lifes_work_assessment()

            # Step 7: Show predictive analytics
            logger.info("🔮 Step 7: Demonstrating predictive analytics...")
            await self._show_predictive_analytics()

            # Step 8: Performance validation
            logger.info("⚡ Step 8: Validating system performance...")
            await self._validate_performance()

            logger.info("✅ Success Metrics Framework deployment completed successfully!")
            logger.info(f"Total deployment time: {datetime.now() - self.deployment_start_time}")

            # Keep the system running for demonstration
            await self._run_demonstration()

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise

    async def _initialize_framework(self):
        """Initialize the Success Metrics Framework"""
        try:
            self.framework = await initialize_success_metrics_framework(self.config)
            logger.info("✅ Framework initialized successfully")

            # Get comprehensive status
            status = await self.framework.get_comprehensive_status()
            logger.info(f"📊 Framework Status: {status['framework_status']['overall_health']}")

            # Show component status
            components = status['framework_status']['components']
            for component, comp_status in components.items():
                logger.info(f"   {component}: {comp_status}")

        except Exception as e:
            logger.error(f"❌ Framework initialization failed: {e}")
            raise

    async def _verify_system_health(self):
        """Verify system health and all components"""
        try:
            # Get comprehensive status
            status = await self.framework.get_comprehensive_status()

            overall_health = status.get('overall_health', 'unknown')
            logger.info(f"🏥 Overall System Health: {overall_health.upper()}")

            # Check individual components
            collection_health = status['collection']['statistics']['collection_health']
            logger.info(f"   Collection System: {collection_health}")

            active_alerts = status['alerts']['active_alerts']
            critical_alerts = status['alerts']['alert_summary']['critical']
            logger.info(f"   Active Alerts: {active_alerts} (Critical: {critical_alerts})")

            # Verify metrics collection
            success_rate = status['collection']['statistics']['success_rate']
            logger.info(f"   Collection Success Rate: {success_rate:.1f}%")

            if success_rate < 90:
                logger.warning("⚠️  Collection success rate is below 90%")

            if critical_alerts > 0:
                logger.warning(f"⚠️  {critical_alerts} critical alerts detected")

        except Exception as e:
            logger.error(f"❌ System health verification failed: {e}")
            raise

    async def _demonstrate_metrics_collection(self):
        """Demonstrate automated metrics collection"""
        try:
            logger.info("📈 Demonstrating metrics collection from multiple sources...")

            # Wait a bit for initial collection
            await asyncio.sleep(10)

            # Get collection status
            collection_status = self.framework.automated_metrics_collector.get_collection_status()

            logger.info(f"   Active Collection Tasks: {collection_status['statistics']['active_tasks']}")
            logger.info(f"   Total Executions: {collection_status['statistics']['total_executions']}")
            logger.info(f"   Success Rate: {collection_status['statistics']['success_rate']:.1f}%")

            # Show data sources
            data_sources = collection_status['data_sources']
            logger.info("   Active Data Sources:")
            for source, count in data_sources.items():
                if count > 0:
                    logger.info(f"      {source}: {count} tasks")

            # Show data quality metrics
            quality_metrics = collection_status['quality_metrics']
            logger.info("   Data Quality Distribution:")
            for quality, count in quality_metrics['data_quality_distribution'].items():
                if count > 0:
                    logger.info(f"      {quality}: {count} records")

        except Exception as e:
            logger.error(f"❌ Metrics collection demonstration failed: {e}")

    async def _launch_dashboard(self):
        """Launch and demonstrate real-time dashboard"""
        try:
            logger.info("🖥️  Real-Time Dashboard Status:")

            # Get dashboard data
            dashboard_data = self.framework.real_time_dashboard.get_dashboard_data()

            logger.info(f"   Active Layout: {dashboard_data['layout']['name']}")
            logger.info(f"   Total Widgets: {len(dashboard_data['layout']['widgets'])}")
            logger.info(f"   Dashboard Active: {dashboard_data['dashboard_active']}")
            logger.info(f"   Last Updated: {dashboard_data['last_updated']}")

            # Show available layouts
            layouts = list(self.framework.real_time_dashboard.current_layouts.keys())
            logger.info(f"   Available Layouts: {', '.join(layouts)}")

            # Demonstrate widget data
            widget_data = dashboard_data['widget_data']
            active_widgets = len([w for w in widget_data.values() if 'error' not in w])
            logger.info(f"   Active Widgets: {active_widgets}/{len(widget_data)}")

            # Sample some key metrics
            key_widgets = ['overall_score', '100x_progress', 'global_users', 'breakthrough_rate']
            for widget_id in key_widgets:
                if widget_id in widget_data and 'data' in widget_data[widget_id]:
                    data = widget_data[widget_id]['data']
                    if isinstance(data, dict):
                        if 'overall_score' in data:
                            logger.info(f"   Overall Success Score: {data['overall_score']:.1f}%")
                        elif 'progress_percentage' in data:
                            logger.info(f"   100x Acceleration Progress: {data['progress_percentage']:.1f}%")
                        elif 'current_value' in data:
                            logger.info(f"   {widget_id}: {data['current_value']}")

        except Exception as e:
            logger.error(f"❌ Dashboard launch demonstration failed: {e}")

    async def _generate_reports(self):
        """Generate comprehensive reports"""
        try:
            logger.info("📋 Generating comprehensive reports...")

            # Generate executive summary
            executive_report = await self.framework.reporting_alert_system.generate_on_demand_report(
                "daily_executive",
                recipients=["demo@ai-scientist-v2.com"]
            )

            if executive_report.get('success'):
                logger.info("✅ Daily Executive Report generated successfully")
                logger.info(f"   Report generated at: {executive_report['generated_at']}")
            else:
                logger.warning("⚠️  Executive report generation failed")

            # Generate comprehensive framework report
            comprehensive_report = await self.framework.generate_comprehensive_report("executive")

            logger.info("✅ Comprehensive Framework Report generated")
            logger.info(f"   Report Type: {comprehensive_report['report_metadata']['report_type']}")
            logger.info(f"   Framework Health: {comprehensive_report['executive_summary']['overall_framework_health']}")

            # Show key metrics from report
            success_metrics = comprehensive_report.get('success_metrics_analysis', {})
            if 'lifes_work_metrics' in success_metrics:
                metrics = success_metrics['lifes_work_metrics']
                logger.info("   Key Life's Work Metrics:")
                logger.info(f"      Research Acceleration: {metrics.get('research_acceleration_factor', 0):.1f}x")
                logger.info(f"      Global Users: {metrics.get('global_users_count', 0):,}")
                logger.info(f"      Breakthrough Rate: {metrics.get('breakthrough_discovery_rate', 0):.1f}/month")

            # Show goal progress
            goals_progress = comprehensive_report.get('goals_and_progress', {})
            if 'overall_progress' in goals_progress:
                logger.info(f"   Overall Goal Progress: {goals_progress['overall_progress']:.1f}%")

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")

    async def _execute_lifes_work_assessment(self):
        """Execute comprehensive life's work assessment"""
        try:
            logger.info("🎯 Executing comprehensive life's work assessment...")

            assessment = await self.framework.execute_lifes_work_assessment()

            logger.info("✅ Life's Work Assessment completed")
            logger.info(f"   Assessment Type: {assessment['assessment_metadata']['assessment_type']}")

            # Show overall assessment
            overall = assessment['overall_assessment']
            logger.info("   Overall Assessment:")
            logger.info(f"      Life's Work Progress: {overall['life_work_progress_percentage']:.1f}%")
            logger.info(f"      Primary Focus Area: {overall['primary_focus_area']}")
            logger.info(f"      Achievement Status: {overall['achievement_status']}")

            # Show critical priorities
            priorities = overall.get('next_critical_priorities', [])
            if priorities:
                logger.info("   Critical Priorities:")
                for i, priority in enumerate(priorities[:3], 1):
                    logger.info(f"      {i}. {priority}")

            # Show success probabilities
            success_prob = assessment['success_probability']
            logger.info("   Success Probabilities:")
            logger.info(f"      100x Acceleration: {success_prob['100x_goal_probability']:.1f}%")
            logger.info(f"      Democratization: {success_prob['democratization_goal_probability']:.1f}%")
            logger.info(f"      Overall Life's Work: {success_prob['overall_lifes_work_probability']:.1f}%")

        except Exception as e:
            logger.error(f"❌ Life's work assessment failed: {e}")

    async def _show_predictive_analytics(self):
        """Demonstrate predictive analytics capabilities"""
        try:
            logger.info("🔮 Demonstrating predictive analytics...")

            # Get predictions from success metrics engine
            predictions = await self.framework.success_metrics_engine.get_predictions()

            logger.info("✅ Predictive Analytics Results:")

            # Show overall assessment
            if 'overall_assessment' in predictions:
                assessment = predictions['overall_assessment']
                logger.info("   Goal Achievement Predictions:")
                for goal_name, goal_assessment in assessment.items():
                    likelihood = goal_assessment.get('likelihood', 'unknown')
                    confidence = goal_assessment.get('confidence', 0)
                    logger.info(f"      {goal_name}: {likelihood} (confidence: {confidence:.1%})")

            # Show specific goal predictions
            if 'predictions' in predictions:
                for goal_name, goal_predictions in predictions['predictions'].items():
                    logger.info(f"   {goal_name} Predictions:")
                    for pred in goal_predictions[:2]:  # Show top 2 predictions
                        metric = pred.get('metric', 'unknown')
                        current = pred.get('current_value', 0)
                        target = pred.get('target_value', 0)
                        predicted_date = pred.get('predicted_achievement_date')
                        confidence = pred.get('confidence', 0)

                        logger.info(f"      {metric}: {current:.1f} → {target:.1f}")
                        if predicted_date:
                            logger.info(f"         Predicted: {predicted_date[:10]} (confidence: {confidence:.1%})")

        except Exception as e:
            logger.error(f"❌ Predictive analytics demonstration failed: {e}")

    async def _validate_performance(self):
        """Validate system performance"""
        try:
            logger.info("⚡ Validating system performance...")

            # Get final status
            final_status = await self.framework.get_comprehensive_status()

            # Performance metrics
            collection_stats = final_status['collection']['statistics']
            logger.info("   Performance Metrics:")
            logger.info(f"      Collection Success Rate: {collection_stats['success_rate']:.1f}%")
            logger.info(f"      Total Executions: {collection_stats['total_executions']}")
            logger.info(f"      Queue Size: {collection_stats['queue_size']}")
            logger.info(f"      Cache Entries: {collection_stats['cache_entries']}")

            # Data quality
            quality_metrics = collection_stats.get('quality_metrics', {})
            confidence_dist = quality_metrics.get('confidence_distribution', {})
            logger.info("   Data Quality:")
            logger.info(f"      High Confidence: {confidence_dist.get('high', 0)} records")
            logger.info(f"      Medium Confidence: {confidence_dist.get('medium', 0)} records")
            logger.info(f"      Low Confidence: {confidence_dist.get('low', 0)} records")

            # Alert system performance
            alert_stats = final_status['alerts']
            logger.info("   Alert System:")
            logger.info(f"      Active Alerts: {alert_stats['active_alerts']}")
            logger.info(f"      Critical Alerts: {alert_stats['alert_summary']['critical']}")
            logger.info(f"      Notification Queue: {alert_stats['notification_queue_size']}")

            # Overall health
            overall_health = final_status.get('overall_health', 'unknown')
            logger.info(f"   Overall System Health: {overall_health.upper()}")

            # Deployment summary
            deployment_time = datetime.now() - self.deployment_start_time
            logger.info(f"   Total Deployment Time: {deployment_time}")

            if overall_health in ['excellent', 'good']:
                logger.info("✅ Performance validation PASSED")
            else:
                logger.warning("⚠️  Performance validation completed with warnings")

        except Exception as e:
            logger.error(f"❌ Performance validation failed: {e}")

    async def _run_demonstration(self):
        """Run continuous demonstration"""
        try:
            logger.info("🔄 Starting continuous demonstration (Press Ctrl+C to stop)...")

            # Run for demonstration period
            demo_duration = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < demo_duration:
                # Update metrics every 30 seconds
                await asyncio.sleep(30)

                # Show current status
                current_time = datetime.now().strftime("%H:%M:%S")
                status = await self.framework.get_comprehensive_status()

                # Show key metrics
                metrics = status.get('success_metrics', {}).get('lifes_work_metrics', {})
                acceleration = metrics.get('research_acceleration_factor', 0)
                users = metrics.get('global_users_count', 0)

                logger.info(f"📊 [{current_time}] Acceleration: {acceleration:.1f}x | Users: {users:,} | Health: {status.get('overall_health', 'unknown')}")

            logger.info("🏁 Demonstration period completed")

        except KeyboardInterrupt:
            logger.info("⏹️  Demonstration stopped by user")
        except Exception as e:
            logger.error(f"❌ Demonstration error: {e}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.framework:
                logger.info("🧹 Cleaning up Success Metrics Framework...")
                await self.framework.shutdown()
                logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


async def main():
    """Main deployment function"""
    deployment = None

    try:
        print("🚀 AI-Scientist-v2 Success Metrics Framework Deployment")
        print("=" * 60)
        print("This deployment will demonstrate the complete Success Metrics Framework")
        print("for tracking progress toward revolutionizing scientific discovery.")
        print("=" * 60)

        # Create deployment instance
        deployment = SuccessMetricsDeployment()

        # Deploy the framework
        await deployment.deploy_framework()

    except KeyboardInterrupt:
        logger.info("⏹️  Deployment stopped by user")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return 1
    finally:
        # Clean up
        if deployment:
            await deployment.cleanup()

    return 0


if __name__ == "__main__":
    # Run the deployment
    exit_code = asyncio.run(main())
    sys.exit(exit_code)