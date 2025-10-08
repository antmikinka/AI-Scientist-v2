"""
Ethical Framework Agent Deployment Script

This script provides comprehensive deployment capabilities for the Ethical Framework Agent,
including installation, configuration, testing, and integration with the existing AI Scientist platform.

Usage:
    python deploy_ethical_framework.py install     # Install ethical framework
    python deploy_ethical_framework.py configure   # Configure ethical framework
    python deploy_ethical_framework.py test        # Run comprehensive tests
    python deploy_ethical_framework.py integrate   # Integrate with orchestrator
    python deploy_ethical_framework.py status      # Check deployment status
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_scientist.ethical.ethical_framework_agent import EthicalFrameworkAgent, get_ethical_framework
from ai_scientist.ethical.integration import EthicalIntegrationManager, EthicalOrchestratorWrapper
from ai_scientist.ethical.config import ConfigManager, get_config
from ai_scientist.orchestration.research_orchestrator_agent import ResearchOrchestratorAgent
from ai_scientist.security.security_manager import SecurityManager
from ai_scientist.core.logging_system import LoggingSystem


class EthicalFrameworkDeployer:
    """
    Comprehensive deployment manager for the Ethical Framework Agent
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.logger = self._setup_logging()
        self.deployment_log = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ethical_framework_deployment.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("EthicalFrameworkDeployer")

    def log_deployment_event(self, event_type: str, details: Dict[str, Any]):
        """Log deployment events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self.deployment_log.append(event)
        self.logger.info(f"Deployment Event: {event_type} - {details}")

    async def install_framework(self) -> Dict[str, Any]:
        """
        Install the Ethical Framework Agent and dependencies
        """
        try:
            self.log_deployment_event("installation_start", {"config_path": self.config_path})

            # Step 1: Create necessary directories
            directories = [
                "./logs/ethical_framework",
                "./audit/ethical_framework",
                "./config/ethical_framework",
                "./data/ethical_framework"
            ]

            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                self.log_deployment_event("directory_created", {"directory": directory})

            # Step 2: Create default configuration if not exists
            if not os.path.exists(self.config_path or "ethical_framework_config.yaml"):
                default_config_path = self.config_manager.create_default_config_file()
                self.log_deployment_event("default_config_created", {"path": default_config_path})

            # Step 3: Verify Python dependencies
            required_packages = [
                "numpy", "pandas", "scikit-learn", "torch", "transformers",
                "yaml", "asyncio", "aiofiles", "uvloop"
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                self.logger.warning(f"Missing required packages: {missing_packages}")
                self.log_deployment_event("missing_dependencies", {"packages": missing_packages})
                return {
                    "success": False,
                    "error": f"Missing required packages: {missing_packages}",
                    "install_command": f"pip install {' '.join(missing_packages)}"
                }

            # Step 4: Initialize configuration
            config = self.config_manager.get_config()
            self.log_deployment_event("configuration_initialized", {
                "environment": config.environment,
                "ethical_threshold": config.ethical_threshold
            })

            # Step 5: Test framework initialization
            try:
                framework = EthicalFrameworkAgent(self.config_manager._config_to_dict(config))
                await framework.initialize()
                await framework.shutdown()
                self.log_deployment_event("framework_test_successful", {})
            except Exception as e:
                self.log_deployment_event("framework_test_failed", {"error": str(e)})
                return {
                    "success": False,
                    "error": f"Framework initialization test failed: {e}"
                }

            self.log_deployment_event("installation_complete", {
                "status": "success",
                "config_summary": self.config_manager.get_config_summary()
            })

            return {
                "success": True,
                "message": "Ethical Framework Agent installed successfully",
                "config_path": self.config_path,
                "directories_created": directories,
                "dependencies_verified": len(missing_packages) == 0
            }

        except Exception as e:
            self.log_deployment_event("installation_failed", {"error": str(e)})
            return {
                "success": False,
                "error": f"Installation failed: {e}"
            }

    async def configure_framework(self, config_updates: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Configure the Ethical Framework Agent
        """
        try:
            self.log_deployment_event("configuration_start", {"updates": config_updates})

            config = self.config_manager.get_config()

            # Apply configuration updates
            if config_updates:
                self.config_manager.update_config(config_updates)
                self.log_deployment_event("configuration_updated", config_updates)

            # Validate configuration
            try:
                self.config_manager._validate_config()
                self.log_deployment_event("configuration_validated", {})
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Configuration validation failed: {e}"
                }

            # Test configuration with framework initialization
            config_dict = self.config_manager._config_to_dict(self.config_manager.get_config())
            try:
                framework = EthicalFrameworkAgent(config_dict)
                await framework.initialize()
                status = await framework.get_ethical_status()
                await framework.shutdown()

                self.log_deployment_event("configuration_test_successful", {
                    "framework_status": status
                })

            except Exception as e:
                self.log_deployment_event("configuration_test_failed", {"error": str(e)})
                return {
                    "success": False,
                    "error": f"Configuration test failed: {e}"
                }

            # Save configuration
            self.config_manager.save_config()
            self.log_deployment_event("configuration_saved", {"path": self.config_path})

            return {
                "success": True,
                "message": "Ethical Framework Agent configured successfully",
                "config_summary": self.config_manager.get_config_summary()
            }

        except Exception as e:
            self.log_deployment_event("configuration_failed", {"error": str(e)})
            return {
                "success": False,
                "error": f"Configuration failed: {e}"
            }

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive tests for the Ethical Framework Agent
        """
        try:
            self.log_deployment_event("testing_start", {})

            test_results = {
                "unit_tests": await self._run_unit_tests(),
                "integration_tests": await self._run_integration_tests(),
                "performance_tests": await self._run_performance_tests(),
                "security_tests": await self._run_security_tests(),
                "ethical_assessment_tests": await self._run_ethical_assessment_tests()
            }

            # Calculate overall success
            all_tests_passed = all(
                result["success"] for result in test_results.values()
            )

            test_summary = {
                "total_tests": sum(result.get("total_tests", 0) for result in test_results.values()),
                "passed_tests": sum(result.get("passed_tests", 0) for result in test_results.values()),
                "failed_tests": sum(result.get("failed_tests", 0) for result in test_results.values()),
                "overall_success": all_tests_passed
            }

            self.log_deployment_event("testing_complete", test_summary)

            return {
                "success": all_tests_passed,
                "test_results": test_results,
                "summary": test_summary
            }

        except Exception as e:
            self.log_deployment_event("testing_failed", {"error": str(e)})
            return {
                "success": False,
                "error": f"Testing failed: {e}"
            }

    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for ethical framework components"""
        try:
            self.logger.info("Running unit tests...")

            # Test configuration management
            config = self.config_manager.get_config()
            config_tests = [
                ("Config loading", bool(config)),
                ("Framework weights validation", sum([
                    config.framework_weights.utilitarian,
                    config.framework_weights.deontological,
                    config.framework_weights.virtue_ethics,
                    config.framework_weights.care_ethics,
                    config.framework_weights.principle_based,
                    config.framework_weights.precautionary
                ]) == 1.0),
                ("Threshold validation", 0 <= config.ethical_threshold <= 1)
            ]

            # Test framework initialization
            config_dict = self.config_manager._config_to_dict(config)
            framework = EthicalFrameworkAgent(config_dict)
            await framework.initialize()

            framework_tests = [
                ("Framework initialization", True),
                ("Status retrieval", bool(await framework.get_ethical_status())),
                ("Constraint system", len(framework.constraint_engine.constraints) > 0)
            ]

            await framework.shutdown()

            all_tests = config_tests + framework_tests
            passed = sum(1 for _, result in all_tests if result)

            return {
                "success": passed == len(all_tests),
                "total_tests": len(all_tests),
                "passed_tests": passed,
                "failed_tests": len(all_tests) - passed,
                "test_details": {name: result for name, result in all_tests}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Unit tests failed: {e}",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }

    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests with Research Orchestrator"""
        try:
            self.logger.info("Running integration tests...")

            # Initialize components
            config = self.config_manager.get_config()
            config_dict = self.config_manager._config_to_dict(config)

            ethical_framework = EthicalFrameworkAgent(config_dict)
            await ethical_framework.initialize()

            orchestrator = ResearchOrchestratorAgent(config_dict)
            await orchestrator.initialize()

            # Test integration manager
            from ai_scientist.ethical.integration import EthicalIntegrationManager
            integration_manager = EthicalIntegrationManager()
            await integration_manager.initialize(ethical_framework, orchestrator)

            integration_tests = [
                ("Integration manager initialization", True),
                ("Ethical framework integration", bool(integration_manager.ethical_framework)),
                ("Orchestrator integration", bool(integration_manager.orchestrator))
            ]

            # Test wrapper
            from ai_scientist.ethical.integration import EthicalOrchestratorWrapper
            wrapper = EthicalOrchestratorWrapper(orchestrator)
            await wrapper.initialize()

            wrapper_tests = [
                ("Wrapper initialization", True),
                ("Status check", bool(await wrapper.get_comprehensive_status()))
            ]

            # Cleanup
            await ethical_framework.shutdown()
            await orchestrator.shutdown()

            all_tests = integration_tests + wrapper_tests
            passed = sum(1 for _, result in all_tests if result)

            return {
                "success": passed == len(all_tests),
                "total_tests": len(all_tests),
                "passed_tests": passed,
                "failed_tests": len(all_tests) - passed,
                "test_details": {name: result for name, result in all_tests}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Integration tests failed: {e}",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }

    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests for ethical framework"""
        try:
            self.logger.info("Running performance tests...")

            import time

            # Test assessment performance
            config_dict = self.config_manager._config_to_dict(self.config_manager.get_config())
            framework = EthicalFrameworkAgent(config_dict)
            await framework.initialize()

            from ai_scientist.ethical.ethical_framework_agent import ResearchEthicsContext

            test_context = ResearchEthicsContext(
                research_domain="medical",
                methodologies=["experimental"],
                data_sources=["clinical_data"],
                human_involvement=True,
                environmental_impact=False,
                cultural_considerations=["western"],
                regulatory_requirements=["hipaa"],
                institutional_policies=["irb_approval"],
                stakeholder_groups=["patients", "researchers"],
                expected_outcomes=["treatment_improvement"],
                potential_risks ["data_privacy"],
                mitigation_strategies ["data_anonymization"]
            )

            # Measure assessment time
            start_time = time.time()
            assessment = await framework.assess_research_ethics(
                "test_request_001",
                "Clinical trial for new treatment with human participants",
                test_context
            )
            assessment_time = time.time() - start_time

            performance_tests = [
                ("Assessment time < 5 seconds", assessment_time < 5),
                ("Assessment completed", bool(assessment)),
                ("Ethical score generated", 0 <= assessment.overall_score <= 1),
                ("Compliance status determined", bool(assessment.compliance_status))
            ]

            await framework.shutdown()

            passed = sum(1 for _, result in performance_tests if result)

            return {
                "success": passed == len(performance_tests),
                "total_tests": len(performance_tests),
                "passed_tests": passed,
                "failed_tests": len(performance_tests) - passed,
                "performance_metrics": {
                    "assessment_time_seconds": assessment_time,
                    "ethical_score": assessment.overall_score,
                    "risk_level": assessment.risk_level.value
                },
                "test_details": {name: result for name, result in performance_tests}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Performance tests failed: {e}",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }

    async def _run_security_tests(self) -> Dict[str, Any]:
        """Run security tests for ethical framework"""
        try:
            self.logger.info("Running security tests...")

            # Test security manager
            security_manager = SecurityManager()

            security_tests = [
                ("Security manager initialization", bool(security_manager)),
                ("API key validation", await security_manager.validate_api_key("sk-test-key-123")),
                ("Authorization check", await security_manager.check_authorization("test_user", "test_resource", "read")),
                ("Security event logging", await security_manager.log_security_event("test_event", {"test": "data"}))
            ]

            # Test configuration security
            config = self.config_manager.get_config()
            config_security_tests = [
                ("Encryption enabled", config.security.encryption_enabled),
                ("Access control enabled", config.security.access_control_enabled),
                ("Secure data storage", config.security.secure_data_storage),
                ("Anonymization enabled", config.security.anonymization_enabled)
            ]

            all_tests = security_tests + config_security_tests
            passed = sum(1 for _, result in all_tests if result)

            return {
                "success": passed == len(all_tests),
                "total_tests": len(all_tests),
                "passed_tests": passed,
                "failed_tests": len(all_tests) - passed,
                "test_details": {name: result for name, result in all_tests}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Security tests failed: {e}",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }

    async def _run_ethical_assessment_tests(self) -> Dict[str, Any]:
        """Run ethical assessment functionality tests"""
        try:
            self.logger.info("Running ethical assessment tests...")

            config_dict = self.config_manager._config_to_dict(self.config_manager.get_config())
            framework = EthicalFrameworkAgent(config_dict)
            await framework.initialize()

            from ai_scientist.ethical.ethical_framework_agent import ResearchEthicsContext

            # Test various research scenarios
            test_cases = [
                {
                    "name": "Low risk research",
                    "content": "Mathematical analysis of existing datasets",
                    "context": ResearchEthicsContext(
                        research_domain="mathematics",
                        methodologies=["theoretical"],
                        data_sources=["public_data"],
                        human_involvement=False,
                        environmental_impact=False,
                        cultural_considerations=[],
                        regulatory_requirements=[],
                        institutional_policies=[],
                        stakeholder_groups=["researchers"],
                        expected_outcomes=["new_insights"],
                        potential_risks=[],
                        mitigation_strategies=[]
                    ),
                    "expected_risk_level": "low"
                },
                {
                    "name": "High risk research",
                    "content": "Clinical trial with human subjects testing new drug",
                    "context": ResearchEthicsContext(
                        research_domain="medical",
                        methodologies=["experimental"],
                        data_sources=["patient_data"],
                        human_involvement=True,
                        environmental_impact=False,
                        cultural_considerations=["western"],
                        regulatory_requirements=["fda", "hipaa"],
                        institutional_policies=["irb_approval"],
                        stakeholder_groups=["patients", "doctors", "researchers"],
                        expected_outcomes=["treatment_efficacy"],
                        potential_risks ["side_effects", "data_privacy"],
                        mitigation_strategies ["informed_consent", "data_anonymization"]
                    ),
                    "expected_risk_level": "high"
                }
            ]

            assessment_tests = []

            for i, test_case in enumerate(test_cases):
                try:
                    assessment = await framework.assess_research_ethics(
                        f"test_case_{i}",
                        test_case["content"],
                        test_case["context"]
                    )

                    risk_level_match = assessment.risk_level.value == test_case["expected_risk_level"]
                    ethical_score_valid = 0 <= assessment.overall_score <= 1
                    has_recommendations = len(assessment.recommendations) > 0

                    assessment_tests.extend([
                        (f"{test_case['name']} - Risk level correct", risk_level_match),
                        (f"{test_case['name']} - Ethical score valid", ethical_score_valid),
                        (f"{test_case['name']} - Has recommendations", has_recommendations)
                    ])

                except Exception as e:
                    assessment_tests.append((f"{test_case['name']} - Assessment failed", False))

            await framework.shutdown()

            passed = sum(1 for _, result in assessment_tests if result)

            return {
                "success": passed == len(assessment_tests),
                "total_tests": len(assessment_tests),
                "passed_tests": passed,
                "failed_tests": len(assessment_tests) - passed,
                "test_details": {name: result for name, result in assessment_tests}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Ethical assessment tests failed: {e}",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            }

    async def integrate_with_orchestrator(self) -> Dict[str, Any]:
        """
        Integrate Ethical Framework with Research Orchestrator
        """
        try:
            self.log_deployment_event("integration_start", {})

            config = self.config_manager.get_config()
            config_dict = self.config_manager._config_to_dict(config)

            # Initialize components
            ethical_framework = EthicalFrameworkAgent(config_dict)
            await ethical_framework.initialize()

            orchestrator = ResearchOrchestratorAgent(config_dict)
            await orchestrator.initialize()

            # Create integration manager
            integration_manager = EthicalIntegrationManager()
            await integration_manager.initialize(ethical_framework, orchestrator)

            # Create wrapper
            wrapper = EthicalOrchestratorWrapper(orchestrator)
            await wrapper.initialize()

            # Test integration with sample research request
            from ai_scientist.orchestration.research_orchestrator_agent import create_research_request

            test_request = await create_research_request(
                "Test medical research with ethical considerations",
                ethical_requirements={
                    "human_subjects": False,
                    "data_privacy": True,
                    "institutional_approval": True
                }
            )

            integration_result = await integration_manager.integrate_research_request(test_request)

            # Test ethical coordination
            ethical_results = await wrapper.coordinate_research_with_ethics(test_request)

            # Check integration status
            comprehensive_status = await wrapper.get_comprehensive_status()

            # Cleanup
            await ethical_framework.shutdown()
            await orchestrator.shutdown()

            integration_tests = [
                ("Integration manager created", bool(integration_manager)),
                ("Wrapper created", bool(wrapper)),
                ("Research request integration", integration_result["success"]),
                ("Ethical coordination", ethical_results.success),
                ("Comprehensive status", bool(comprehensive_status))
            ]

            passed = sum(1 for _, result in integration_tests if result)

            self.log_deployment_event("integration_complete", {
                "success": passed == len(integration_tests),
                "tests_passed": passed,
                "total_tests": len(integration_tests)
            })

            return {
                "success": passed == len(integration_tests),
                "message": "Ethical Framework integrated with Research Orchestrator",
                "integration_tests": {name: result for name, result in integration_tests},
                "ethical_results": {
                    "ethical_clearance": ethical_results.ethical_compliance.get("ethical_clearance", False),
                    "ethical_score": ethical_results.confidence_score,
                    "requires_oversight": ethical_results.ethical_compliance.get("requires_human_oversight", False)
                }
            }

        except Exception as e:
            self.log_deployment_event("integration_failed", {"error": str(e)})
            return {
                "success": False,
                "error": f"Integration failed: {e}"
            }

    async def check_deployment_status(self) -> Dict[str, Any]:
        """
        Check the current deployment status
        """
        try:
            self.log_deployment_event("status_check", {})

            status = {
                "installation_status": await self._check_installation_status(),
                "configuration_status": await self._check_configuration_status(),
                "integration_status": await self._check_integration_status(),
                "performance_metrics": await self._check_performance_metrics(),
                "deployment_history": self.deployment_log[-10:]  # Last 10 events
            }

            overall_health = all([
                status["installation_status"]["healthy"],
                status["configuration_status"]["healthy"],
                status["integration_status"]["healthy"]
            ])

            status["overall_health"] = overall_health

            return status

        except Exception as e:
            self.log_deployment_event("status_check_failed", {"error": str(e)})
            return {
                "overall_health": False,
                "error": f"Status check failed: {e}"
            }

    async def _check_installation_status(self) -> Dict[str, Any]:
        """Check installation status"""
        required_directories = [
            "./logs/ethical_framework",
            "./audit/ethical_framework",
            "./config/ethical_framework",
            "./data/ethical_framework"
        ]

        directories_exist = all(os.path.exists(dir) for dir in required_directories)
        config_exists = os.path.exists(self.config_path or "ethical_framework_config.yaml")

        return {
            "healthy": directories_exist and config_exists,
            "directories_exist": directories_exist,
            "config_exists": config_exists,
            "missing_directories": [dir for dir in required_directories if not os.path.exists(dir)]
        }

    async def _check_configuration_status(self) -> Dict[str, Any]:
        """Check configuration status"""
        try:
            config = self.config_manager.get_config()
            config_summary = self.config_manager.get_config_summary()

            return {
                "healthy": True,
                "config_loaded": True,
                "environment": config_summary["environment"],
                "ethical_threshold": config_summary["ethical_threshold"],
                "frameworks_enabled": config_summary["enabled_frameworks"]
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_integration_status(self) -> Dict[str, Any]:
        """Check integration status"""
        try:
            # Try to get framework instance
            framework = await get_ethical_framework()
            framework_status = await framework.get_ethical_status()

            return {
                "healthy": framework_status.get("status") == "active",
                "framework_active": framework_status.get("status") == "active",
                "total_assessments": framework_status.get("total_assessments", 0),
                "compliance_rate": framework_status.get("compliance_rate", 0)
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics"""
        try:
            import psutil
            import time

            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "system_healthy": cpu_percent < 80 and memory_percent < 80 and disk_percent < 80
            }
        except Exception as e:
            return {
                "system_healthy": False,
                "error": str(e)
            }

    async def generate_deployment_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive deployment report
        """
        try:
            status = await self.check_deployment_status()

            report = {
                "report_generated": datetime.now().isoformat(),
                "deployment_summary": {
                    "overall_health": status["overall_health"],
                    "installation_healthy": status["installation_status"]["healthy"],
                    "configuration_healthy": status["configuration_status"]["healthy"],
                    "integration_healthy": status["integration_status"]["healthy"],
                    "system_performance_healthy": status["performance_metrics"]["system_healthy"]
                },
                "detailed_status": status,
                "deployment_events": self.deployment_log,
                "recommendations": self._generate_recommendations(status)
            }

            # Save report
            report_path = f"ethical_framework_deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.log_deployment_event("report_generated", {"path": report_path})

            return {
                "success": True,
                "report": report,
                "report_path": report_path
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Report generation failed: {e}"
            }

    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate deployment recommendations based on status"""
        recommendations = []

        if not status["installation_status"]["healthy"]:
            recommendations.append("Complete installation by creating missing directories and configuration files")

        if not status["configuration_status"]["healthy"]:
            recommendations.append("Review and fix configuration issues")

        if not status["integration_status"]["healthy"]:
            recommendations.append("Check integration with Research Orchestrator")

        if not status["performance_metrics"]["system_healthy"]:
            recommendations.append("Address system performance issues (high CPU/memory usage)")

        if status["overall_health"]:
            recommendations.append("System is healthy - consider enabling advanced features")

        return recommendations


async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Ethical Framework Agent Deployment")
    parser.add_argument("command", choices=[
        "install", "configure", "test", "integrate", "status", "report"
    ], help="Deployment command to execute")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--config-updates", help="JSON string with configuration updates")
    parser.add_argument("--output", help="Output file path for reports")

    args = parser.parse_args()

    # Initialize deployer
    deployer = EthicalFrameworkDeployer(args.config)

    # Parse configuration updates if provided
    config_updates = None
    if args.config_updates:
        try:
            config_updates = json.loads(args.config_updates)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in config-updates")
            sys.exit(1)

    # Execute command
    result = None

    if args.command == "install":
        result = await deployer.install_framework()
    elif args.command == "configure":
        result = await deployer.configure_framework(config_updates)
    elif args.command == "test":
        result = await deployer.run_comprehensive_tests()
    elif args.command == "integrate":
        result = await deployer.integrate_with_orchestrator()
    elif args.command == "status":
        result = await deployer.check_deployment_status()
    elif args.command == "report":
        result = await deployer.generate_deployment_report()

    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    asyncio.run(main())