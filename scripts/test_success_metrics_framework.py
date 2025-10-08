#!/usr/bin/env python3
"""
Test Success Metrics Framework - AI-Scientist-v2

This script tests the complete Success Metrics Framework to ensure all components
are working correctly and providing accurate measurements for tracking progress
toward life's work goals.

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_scientist.metrics import (
    SuccessMetricsEngine,
    RealTimeDashboard,
    ReportingAlertSystem,
    AutomatedMetricsCollector,
    initialize_success_metrics_framework,
    MetricCategory,
    MetricType,
    GoalStatus,
    DataQuality,
    AlertSeverity
)

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestSuccessMetricsFramework(unittest.TestCase):
    """Test suite for Success Metrics Framework"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "collection": {
                "max_workers": 3,
                "collection_interval": 5,
                "max_cache_size": 100,
                "max_history_size": 1000
            },
            "dashboard": {
                "refresh_interval": 5,
                "auto_refresh": True
            },
            "alerts": {
                "critical_thresholds": {
                    "ethical_compliance": 95.0,
                    "system_error_rate": 0.1
                }
            }
        }

    async def async_setUp(self):
        """Async set up"""
        self.framework = await initialize_success_metrics_framework(self.config)
        await self.framework.initialize()

    async def async_tearDown(self):
        """Async tear down"""
        if self.framework:
            await self.framework.shutdown()

    def test_success_metrics_engine_initialization(self):
        """Test Success Metrics Engine initialization"""
        async def test():
            engine = SuccessMetricsEngine(self.config)
            await engine.initialize()

            # Test metric definitions
            self.assertTrue(len(engine.metrics_registry) > 0)
            self.assertIn("research_acceleration_factor", engine.metrics_registry)
            self.assertIn("global_users_count", engine.metrics_registry)

            # Test goal definitions
            self.assertTrue(len(engine.lifes_work_goals) > 0)
            self.assertTrue(any("100x" in goal.name for goal in engine.lifes_work_goals))

            # Test current metrics structure
            self.assertIsNotNone(engine.current_metrics)
            self.assertIsInstance(engine.current_metrics.research_acceleration_factor, (int, float))

            await engine.shutdown()

        asyncio.run(test())

    def test_metric_recording_and_calculation(self):
        """Test metric recording and calculation"""
        async def test():
            await self.async_setUp()

            engine = self.framework.success_metrics_engine

            # Record test metrics
            engine.record_metric("research_acceleration_factor", 15.5)
            engine.record_metric("global_users_count", 5000)
            engine.record_metric("breakthrough_discovery_rate", 2.3)

            # Wait for processing
            await asyncio.sleep(2)

            # Verify metrics were recorded
            self.assertEqual(engine.current_metrics.research_acceleration_factor, 15.5)
            self.assertEqual(engine.current_metrics.global_users_count, 5000)
            self.assertEqual(engine.current_metrics.breakthrough_discovery_rate, 2.3)

            # Test overall success score calculation
            score = engine._calculate_overall_success_score()
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

            await self.async_tearDown()

        asyncio.run(test())

    def test_goal_progress_tracking(self):
        """Test goal progress tracking"""
        async def test():
            await self.async_setUp()

            engine = self.framework.success_metrics_engine

            # Set some progress metrics
            engine.current_metrics.research_acceleration_factor = 25.0  # 25x out of 100x
            engine.current_metrics.global_users_count = 250000  # 250k out of 1M

            # Update goal progress
            await engine._update_goal_progress()

            # Check progress calculations
            progress = engine.goal_progress.get("100x Scientific Acceleration", {})
            self.assertIsNotNone(progress)
            self.assertGreater(progress.get("overall_progress", 0), 0)

            await self.async_tearDown()

        asyncio.run(test())

    def test_real_time_dashboard(self):
        """Test real-time dashboard functionality"""
        async def test():
            await self.async_setUp()

            dashboard = self.framework.real_time_dashboard

            # Test dashboard data retrieval
            dashboard_data = dashboard.get_dashboard_data()
            self.assertIsNotNone(dashboard_data)
            self.assertIn("layout", dashboard_data)
            self.assertIn("widget_data", dashboard_data)

            # Test widget data
            widget_data = dashboard_data["widget_data"]
            self.assertIsInstance(widget_data, dict)

            # Test layout switching
            original_layout = dashboard.active_layout_id
            dashboard.set_layout("scientific_acceleration")
            self.assertEqual(dashboard.active_layout_id, "scientific_acceleration")

            # Switch back
            dashboard.set_layout(original_layout)

            await self.async_tearDown()

        asyncio.run(test())

    def test_automated_metrics_collection(self):
        """Test automated metrics collection"""
        async def test():
            await self.async_setUp()

            collector = self.framework.automated_metrics_collector

            # Wait for initial collection
            await asyncio.sleep(10)

            # Test collection status
            status = collector.get_collection_status()
            self.assertTrue(status["collection_active"])
            self.assertIn("statistics", status)
            self.assertIn("pipelines", status)

            # Test that some metrics were collected
            self.assertGreater(status["statistics"]["total_executions"], 0)

            # Test data cache
            self.assertGreater(len(collector.data_cache), 0)

            # Test historical data
            for metric_name, history in collector.historical_data.items():
                self.assertGreater(len(history), 0)
                break  # Just check that at least one metric has history

            await self.async_tearDown()

        asyncio.run(test())

    def test_reporting_and_alert_system(self):
        """Test reporting and alert system"""
        async def test():
            await self.async_setUp()

            alerts = self.framework.reporting_alert_system

            # Test alert status
            alert_status = alerts.get_alert_status()
            self.assertIn("active_alerts", alert_status)
            self.assertIn("alert_summary", alert_status)

            # Test report generation
            report = await alerts.generate_on_demand_report("daily_executive")
            self.assertTrue(report.get("success", False))

            # Test manual alert creation
            test_alert = alerts.alert_rules["critical_acceleration"]
            await alerts._create_alert_from_rule("test_rule", test_alert, 2.0)

            # Check that alert was created
            self.assertGreater(len(alerts.active_alerts), 0)

            # Test alert resolution
            if alerts.active_alerts:
                alert_id = list(alerts.active_alerts.keys())[0]
                resolved = alerts.resolve_alert(alert_id, "Test resolution")
                self.assertTrue(resolved)

            await self.async_tearDown()

        asyncio.run(test())

    def test_framework_integration(self):
        """Test complete framework integration"""
        async def test():
            await self.async_setUp()

            # Test comprehensive status
            status = await self.framework.get_comprehensive_status()
            self.assertIn("framework_status", status)
            self.assertIn("success_metrics", status)
            self.assertIn("dashboard", status)
            self.assertIn("alerts", status)
            self.assertIn("collection", status)

            # Test overall health calculation
            overall_health = status.get("overall_health", "unknown")
            self.assertIn(overall_health, ["excellent", "good", "fair", "poor", "critical", "unknown"])

            # Test comprehensive report generation
            report = await self.framework.generate_comprehensive_report()
            self.assertIn("report_metadata", report)
            self.assertIn("executive_summary", report)
            self.assertIn("success_metrics_analysis", report)
            self.assertIn("goals_and_progress", report)

            # Test life's work assessment
            assessment = await self.framework.execute_lifes_work_assessment()
            self.assertIn("assessment_metadata", assessment)
            self.assertIn("overall_assessment", assessment)
            self.assertIn("success_probability", assessment)

            await self.async_tearDown()

        asyncio.run(test())

    def test_data_quality_validation(self):
        """Test data quality validation"""
        async def test():
            await self.async_setUp()

            collector = self.framework.automated_metrics_collector

            # Wait for some data collection
            await asyncio.sleep(10)

            # Check that data quality is being assessed
            status = collector.get_collection_status()
            quality_metrics = status.get("quality_metrics", {})
            self.assertIn("data_quality_distribution", quality_metrics)
            self.assertIn("confidence_distribution", quality_metrics)

            # Test data quality levels
            quality_dist = quality_metrics["data_quality_distribution"]
            total_records = sum(quality_dist.values())
            self.assertGreater(total_records, 0)

            await self.async_tearDown()

        asyncio.run(test())

    def test_predictive_analytics(self):
        """Test predictive analytics capabilities"""
        async def test():
            await self.async_setUp()

            engine = self.framework.success_metrics_engine

            # Set up some historical data by recording metrics
            for i in range(10):
                engine.record_metric("research_acceleration_factor", 5.0 + i * 2)
                engine.record_metric("global_users_count", 1000 + i * 500)
                await asyncio.sleep(0.1)

            # Get predictions
            predictions = await engine.get_predictions()
            self.assertIn("predictions", predictions)
            self.assertIn("overall_assessment", predictions)

            # Test prediction structure
            for goal_name, goal_predictions in predictions["predictions"].items():
                self.assertIsInstance(goal_predictions, list)
                if goal_predictions:
                    prediction = goal_predictions[0]
                    self.assertIn("metric", prediction)
                    self.assertIn("current_value", prediction)
                    self.assertIn("target_value", prediction)

            await self.async_tearDown()

        asyncio.run(test())

    def test_error_handling(self):
        """Test error handling and resilience"""
        async def test():
            await self.async_setUp()

            # Test invalid metric recording
            engine = self.framework.success_metrics_engine
            engine.record_metric("invalid_metric", 100)  # Should handle gracefully

            # Test with invalid data types
            engine.record_metric("research_acceleration_factor", "invalid")  # Should handle gracefully

            # Test system continues operating after errors
            await asyncio.sleep(5)

            # Verify system is still operational
            status = await self.framework.get_comprehensive_status()
            self.assertTrue(status["framework_status"]["initialized"])
            self.assertTrue(status["framework_status"]["active"])

            await self.async_tearDown()

        asyncio.run(test())


class TestPerformanceValidation(unittest.TestCase):
    """Performance validation tests"""

    def setUp(self):
        self.config = {
            "collection": {
                "max_workers": 5,
                "collection_interval": 2,
                "max_cache_size": 1000,
                "max_history_size": 10000
            }
        }

    async def async_setUp(self):
        """Async set up"""
        self.framework = await initialize_success_metrics_framework(self.config)
        await self.framework.initialize()

    async def async_tearDown(self):
        """Async tear down"""
        if self.framework:
            await self.framework.shutdown()

    def test_collection_performance(self):
        """Test metrics collection performance"""
        async def test():
            await self.async_setUp()

            collector = self.framework.automated_metrics_collector

            # Measure collection performance
            start_time = time.time()
            await asyncio.sleep(30)  # Run for 30 seconds
            end_time = time.time()

            status = collector.get_collection_status()
            stats = status["statistics"]

            total_executions = stats["total_executions"]
            success_rate = stats["success_rate"]
            duration = end_time - start_time

            executions_per_minute = (total_executions / duration) * 60

            logger.info(f"Collection Performance:")
            logger.info(f"  Total Executions: {total_executions}")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Executions/Minute: {executions_per_minute:.1f}")

            # Performance assertions
            self.assertGreater(success_rate, 80)  # Should have at least 80% success rate
            self.assertGreater(executions_per_minute, 10)  # Should execute at least 10 per minute

            await self.async_tearDown()

        asyncio.run(test())

    def test_dashboard_performance(self):
        """Test dashboard performance"""
        async def test():
            await self.async_setUp()

            dashboard = self.framework.real_time_dashboard

            # Measure dashboard response times
            response_times = []
            for _ in range(10):
                start_time = time.time()
                dashboard_data = dashboard.get_dashboard_data()
                end_time = time.time()

                response_times.append(end_time - start_time)
                self.assertIsNotNone(dashboard_data)

            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)

            logger.info(f"Dashboard Performance:")
            logger.info(f"  Average Response Time: {avg_response_time:.3f}s")
            logger.info(f"  Max Response Time: {max_response_time:.3f}s")

            # Performance assertions
            self.assertLess(avg_response_time, 0.1)  # Should respond in under 100ms
            self.assertLess(max_response_time, 0.5)  # Max response under 500ms

            await self.async_tearDown()

        asyncio.run(test())


async def run_validation_tests():
    """Run all validation tests"""
    logger.info("🧪 Starting Success Metrics Framework Validation Tests")
    logger.info("=" * 60)

    test_suite = unittest.TestSuite()

    # Add core functionality tests
    test_suite.addTest(TestSuccessMetricsFramework('test_success_metrics_engine_initialization'))
    test_suite.addTest(TestSuccessMetricsFramework('test_metric_recording_and_calculation'))
    test_suite.addTest(TestSuccessMetricsFramework('test_goal_progress_tracking'))
    test_suite.addTest(TestSuccessMetricsFramework('test_real_time_dashboard'))
    test_suite.addTest(TestSuccessMetricsFramework('test_automated_metrics_collection'))
    test_suite.addTest(TestSuccessMetricsFramework('test_reporting_and_alert_system'))
    test_suite.addTest(TestSuccessMetricsFramework('test_framework_integration'))
    test_suite.addTest(TestSuccessMetricsFramework('test_data_quality_validation'))
    test_suite.addTest(TestSuccessMetricsFramework('test_predictive_analytics'))
    test_suite.addTest(TestSuccessMetricsFramework('test_error_handling'))

    # Add performance tests
    test_suite.addTest(TestPerformanceValidation('test_collection_performance'))
    test_suite.addTest(TestPerformanceValidation('test_dashboard_performance'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Summary
    logger.info("=" * 60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")

    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")

    return result.wasSuccessful()


def main():
    """Main test function"""
    try:
        print("🧪 AI-Scientist-v2 Success Metrics Framework Validation")
        print("=" * 60)
        print("Running comprehensive validation tests for the Success Metrics Framework...")
        print("=" * 60)

        # Run validation tests
        success = asyncio.run(run_validation_tests())

        if success:
            print("\n✅ All tests passed! Success Metrics Framework is ready for deployment.")
            return 0
        else:
            print("\n❌ Some tests failed. Please review the errors above.")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Tests stopped by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)