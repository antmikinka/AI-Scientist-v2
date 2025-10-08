#!/usr/bin/env python3
"""
Simple Success Metrics Framework Test - AI-Scientist-v2

This script performs basic validation tests for the Success Metrics Framework
to ensure core functionality is working correctly.

Author: Jordan Blake - Principal Software Engineer & Technical Lead
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_success_metrics_engine():
    """Test the Success Metrics Engine"""
    try:
        logger.info("🧪 Testing Success Metrics Engine...")

        from ai_scientist.metrics.success_metrics_engine import SuccessMetricsEngine

        # Initialize the engine
        config = {
            "collection_interval": 10,
            "history_size": 100
        }

        engine = SuccessMetricsEngine(config)
        await engine.initialize()

        # Test metric definitions
        logger.info(f"   ✅ Loaded {len(engine.metrics_registry)} metric definitions")
        assert "research_acceleration_factor" in engine.metrics_registry
        assert "global_users_count" in engine.metrics_registry

        # Test goal definitions
        logger.info(f"   ✅ Loaded {len(engine.lifes_work_goals)} life's work goals")
        assert len(engine.lifes_work_goals) > 0

        # Test metric recording
        engine.record_metric("research_acceleration_factor", 15.5)
        engine.record_metric("global_users_count", 5000)
        engine.record_metric("breakthrough_discovery_rate", 2.3)

        logger.info("   ✅ Metric recording successful")

        # Wait for processing
        await asyncio.sleep(2)

        # Verify metrics were recorded
        assert engine.current_metrics.research_acceleration_factor == 15.5
        assert engine.current_metrics.global_users_count == 5000
        assert engine.current_metrics.breakthrough_discovery_rate == 2.3

        logger.info("   ✅ Metrics verified in current state")

        # Test overall success score calculation
        score = engine._calculate_overall_success_score()
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

        logger.info(f"   ✅ Overall success score: {score:.1f}%")

        # Test goal progress tracking
        await engine._update_goal_progress()
        progress = engine.goal_progress.get("100x Scientific Acceleration", {})
        assert progress is not None

        logger.info(f"   ✅ Goal progress tracking working")

        # Test predictions
        predictions = await engine.get_predictions()
        assert "predictions" in predictions
        assert "overall_assessment" in predictions

        logger.info("   ✅ Predictive analytics working")

        # Test comprehensive report
        report = await engine.generate_report("comprehensive")
        assert "report_metadata" in report
        assert "executive_summary" in report

        logger.info("   ✅ Comprehensive report generation working")

        # Shutdown
        await engine.shutdown()
        logger.info("✅ Success Metrics Engine test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Success Metrics Engine test FAILED: {e}")
        return False


async def test_automated_collection():
    """Test automated metrics collection"""
    try:
        logger.info("🧪 Testing Automated Metrics Collection...")

        from ai_scientist.metrics.automated_collection import AutomatedMetricsCollector
        from ai_scientist.metrics.success_metrics_engine import SuccessMetricsEngine

        # Initialize components
        config = {
            "max_workers": 3,
            "collection_interval": 5,
            "max_cache_size": 100
        }

        engine = SuccessMetricsEngine(config)
        await engine.initialize()

        collector = AutomatedMetricsCollector(engine, engine.performance_monitor, config)
        await collector.initialize()

        # Wait for initial collection
        await asyncio.sleep(10)

        # Test collection status
        status = collector.get_collection_status()
        assert status["collection_active"] == True
        assert "statistics" in status
        assert "pipelines" in status

        logger.info(f"   ✅ Collection system active")
        logger.info(f"   ✅ Active tasks: {status['statistics']['active_tasks']}")
        logger.info(f"   ✅ Success rate: {status['statistics']['success_rate']:.1f}%")

        # Test data cache
        assert len(collector.data_cache) > 0
        logger.info(f"   ✅ Data cache populated with {len(collector.data_cache)} metrics")

        # Test historical data
        has_history = False
        for metric_name, history in collector.historical_data.items():
            if len(history) > 0:
                has_history = True
                break

        assert has_history == True
        logger.info("   ✅ Historical data collection working")

        # Shutdown
        await collector.shutdown()
        await engine.shutdown()
        logger.info("✅ Automated Metrics Collection test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Automated Metrics Collection test FAILED: {e}")
        return False


async def test_real_time_dashboard():
    """Test real-time dashboard functionality"""
    try:
        logger.info("🧪 Testing Real-Time Dashboard...")

        from ai_scientist.metrics.real_time_dashboard import RealTimeDashboard
        from ai_scientist.metrics.success_metrics_engine import SuccessMetricsEngine

        # Initialize components
        config = {"refresh_interval": 5}

        engine = SuccessMetricsEngine(config)
        await engine.initialize()

        dashboard = RealTimeDashboard(engine, config)
        await dashboard.start_dashboard()

        # Test dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        assert "layout" in dashboard_data
        assert "widget_data" in dashboard_data
        assert dashboard_data["dashboard_active"] == True

        logger.info(f"   ✅ Dashboard active with layout: {dashboard_data['layout']['name']}")
        logger.info(f"   ✅ Total widgets: {len(dashboard_data['layout']['widgets'])}")

        # Test layout switching
        original_layout = dashboard.active_layout_id
        dashboard.set_layout("scientific_acceleration")
        assert dashboard.active_layout_id == "scientific_acceleration"

        # Switch back
        dashboard.set_layout(original_layout)
        logger.info("   ✅ Layout switching working")

        # Test widget data
        widget_data = dashboard_data["widget_data"]
        active_widgets = len([w for w in widget_data.values() if "error" not in w])
        logger.info(f"   ✅ Active widgets: {active_widgets}/{len(widget_data)}")

        # Test key metrics
        if "overall_score" in widget_data and "data" in widget_data["overall_score"]:
            overall_score = widget_data["overall_score"]["data"].get("overall_score", 0)
            logger.info(f"   ✅ Overall success score: {overall_score:.1f}%")

        # Shutdown
        await dashboard.stop_dashboard()
        await engine.shutdown()
        logger.info("✅ Real-Time Dashboard test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Real-Time Dashboard test FAILED: {e}")
        return False


async def test_framework_integration():
    """Test complete framework integration"""
    try:
        logger.info("🧪 Testing Complete Framework Integration...")

        from ai_scientist.metrics import initialize_success_metrics_framework

        # Initialize the complete framework
        config = {
            "collection": {
                "max_workers": 3,
                "collection_interval": 5,
                "max_cache_size": 100
            },
            "dashboard": {
                "refresh_interval": 5
            }
        }

        framework = await initialize_success_metrics_framework(config)

        # Test comprehensive status
        status = await framework.get_comprehensive_status()
        assert "framework_status" in status
        assert "success_metrics" in status
        assert "dashboard" in status
        assert "alerts" in status
        assert "collection" in status

        logger.info("   ✅ Comprehensive status retrieved")
        logger.info(f"   ✅ Framework health: {status['overall_health']}")

        # Test component status
        components = status['framework_status']['components']
        active_components = sum(1 for status in components.values() if status == "active")
        logger.info(f"   ✅ Active components: {active_components}/{len(components)}")

        # Test comprehensive report
        report = await framework.generate_comprehensive_report()
        assert "report_metadata" in report
        assert "executive_summary" in report
        assert "success_metrics_analysis" in report
        assert "goals_and_progress" in report

        logger.info("   ✅ Comprehensive report generated")

        # Test life's work assessment
        assessment = await framework.execute_lifes_work_assessment()
        assert "assessment_metadata" in assessment
        assert "overall_assessment" in assessment
        assert "success_probability" in assessment

        logger.info("   ✅ Life's work assessment completed")

        # Test success probabilities
        success_prob = assessment['success_probability']
        logger.info(f"   ✅ 100x goal probability: {success_prob['100x_goal_probability']:.1f}%")
        logger.info(f"   ✅ Democratization probability: {success_prob['democratization_goal_probability']:.1f}%")
        logger.info(f"   ✅ Overall success probability: {success_prob['overall_lifes_work_probability']:.1f}%")

        # Shutdown
        await framework.shutdown()
        logger.info("✅ Complete Framework Integration test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Complete Framework Integration test FAILED: {e}")
        return False


async def run_all_tests():
    """Run all validation tests"""
    logger.info("🚀 Starting Success Metrics Framework Validation")
    logger.info("=" * 60)

    tests = [
        ("Success Metrics Engine", test_success_metrics_engine),
        ("Automated Collection", test_automated_collection),
        ("Real-Time Dashboard", test_real_time_dashboard),
        ("Complete Framework Integration", test_framework_integration)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} Test...")
        logger.info("-" * 40)

        try:
            result = await test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")

    logger.info("-" * 60)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Success Metrics Framework is ready.")
        return True
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return False


def main():
    """Main test function"""
    try:
        print("🧪 AI-Scientist-v2 Success Metrics Framework Validation")
        print("=" * 60)
        print("Running comprehensive validation tests...")
        print("=" * 60)

        # Run all tests
        success = asyncio.run(run_all_tests())

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⏹️  Tests stopped by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)