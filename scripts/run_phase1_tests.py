#!/usr/bin/env python3
"""
Phase 1 Comprehensive Test Execution Script

This script runs the complete Phase 1 testing strategy including:
- Critical fixes validation
- Integration testing
- Performance testing
- Security testing
- Quality gate validation

Usage:
    python scripts/run_phase1_tests.py [options]

Options:
    --critical-only     Run only critical tests
    --integration-only  Run only integration tests
    --performance-only  Run only performance tests
    --security-only     Run only security tests
    --full-suite        Run complete test suite (default)
    --report            Generate comprehensive report
    --ci-mode           Run in CI mode with strict quality gates
"""

import argparse
import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestRunner:
    """Comprehensive test runner for Phase 1"""

    def __init__(self, ci_mode: bool = False):
        self.ci_mode = ci_mode
        self.results = {
            'critical': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0},
            'integration': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0},
            'performance': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0},
            'security': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0},
            'overall': {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
        }
        self.start_time = time.time()
        self.report = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        self.report.append(f"[{timestamp}] {level}: {message}")

    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run command and capture results"""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300  # 5 minutes timeout
            )

            output = {
                'command': ' '.join(command),
                'description': description,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }

            if result.returncode == 0:
                self.log(f"✅ {description} completed successfully")
            else:
                self.log(f"❌ {description} failed with return code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr[:500]}...", "ERROR")

            return output

        except subprocess.TimeoutExpired:
            self.log(f"⏰ {description} timed out after 5 minutes", "ERROR")
            return {
                'command': ' '.join(command),
                'description': description,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Timeout exceeded',
                'success': False
            }
        except Exception as e:
            self.log(f"❌ {description} failed with exception: {e}", "ERROR")
            return {
                'command': ' '.join(command),
                'description': description,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output to extract test results"""
        results = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}

        for line in output.split('\n'):
            if 'passed' in line and 'failed' in line:
                # Parse line like: "8 passed, 1 skipped, 2 warnings in 3.71s"
                parts = line.split(',')[0]
                if 'passed' in parts:
                    results['passed'] = int(parts.split('passed')[0].strip())
            elif 'failed' in line:
                parts = line.split('failed')[0].strip()
                if parts:
                    results['failed'] = int(parts)
            elif 'skipped' in line:
                parts = line.split('skipped')[0].strip()
                if parts:
                    results['skipped'] = int(parts)
            elif 'error' in line.lower():
                results['errors'] += 1

        return results

    def run_critical_tests(self) -> bool:
        """Run critical Phase 1 tests"""
        self.log("🔥 Running Critical Tests for Phase 1")

        command = [
            'python', '-m', 'pytest',
            'tests/test_phase1_critical_fixes.py',
            '-v', '--tb=short', '-m', 'critical'
        ]

        result = self.run_command(command, "Critical Tests")

        if result['success']:
            results = self.parse_pytest_output(result['stdout'])
            self.results['critical'] = results
            self.results['overall']['passed'] += results['passed']
            self.results['overall']['failed'] += results['failed']
            self.results['overall']['skipped'] += results['skipped']
            self.results['overall']['errors'] += results['errors']

            self.log(f"Critical Tests Results: {results}")
            return results['failed'] == 0 and results['errors'] == 0
        else:
            self.results['critical']['failed'] = 1
            self.results['overall']['failed'] += 1
            return False

    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        self.log("🔗 Running Integration Tests")

        command = [
            'python', '-m', 'pytest',
            'tests/test_integration_framework.py',
            '-v', '--tb=short', '-m', 'integration'
        ]

        result = self.run_command(command, "Integration Tests")

        if result['success']:
            results = self.parse_pytest_output(result['stdout'])
            self.results['integration'] = results
            self.results['overall']['passed'] += results['passed']
            self.results['overall']['failed'] += results['failed']
            self.results['overall']['skipped'] += results['skipped']
            self.results['overall']['errors'] += results['errors']

            self.log(f"Integration Tests Results: {results}")
            return results['failed'] == 0 and results['errors'] == 0
        else:
            self.results['integration']['failed'] = 1
            self.results['overall']['failed'] += 1
            return False

    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        self.log("⚡ Running Performance Tests")

        command = [
            'python', '-m', 'pytest',
            'tests/test_performance_strategy.py',
            '-v', '--tb=short', '-m', 'performance'
        ]

        result = self.run_command(command, "Performance Tests")

        if result['success']:
            results = self.parse_pytest_output(result['stdout'])
            self.results['performance'] = results
            self.results['overall']['passed'] += results['passed']
            self.results['overall']['failed'] += results['failed']
            self.results['overall']['skipped'] += results['skipped']
            self.results['overall']['errors'] += results['errors']

            self.log(f"Performance Tests Results: {results}")
            return results['failed'] == 0 and results['errors'] == 0
        else:
            self.results['performance']['failed'] = 1
            self.results['overall']['failed'] += 1
            return False

    def run_security_tests(self) -> bool:
        """Run security tests"""
        self.log("🔒 Running Security Tests")

        command = [
            'python', '-m', 'pytest',
            'tests/test_security_requirements.py',
            '-v', '--tb=short', '-m', 'security'
        ]

        result = self.run_command(command, "Security Tests")

        if result['success']:
            results = self.parse_pytest_output(result['stdout'])
            self.results['security'] = results
            self.results['overall']['passed'] += results['passed']
            self.results['overall']['failed'] += results['failed']
            self.results['overall']['skipped'] += results['skipped']
            self.results['overall']['errors'] += results['errors']

            self.log(f"Security Tests Results: {results}")
            return results['failed'] == 0 and results['errors'] == 0
        else:
            self.results['security']['failed'] = 1
            self.results['overall']['failed'] += 1
            return False

    def run_full_suite(self) -> bool:
        """Run complete test suite"""
        self.log("🚀 Running Full Test Suite")

        # Run all test categories
        critical_success = self.run_critical_tests()
        integration_success = self.run_integration_tests()
        performance_success = self.run_performance_tests()
        security_success = self.run_security_tests()

        return all([critical_success, integration_success, performance_success, security_success])

    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PHASE 1 COMPREHENSIVE TESTING REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Duration: {time.time() - self.start_time:.2f} seconds")
        report_lines.append("")

        # Overall summary
        report_lines.append("📊 OVERALL SUMMARY")
        report_lines.append("-" * 40)
        overall = self.results['overall']
        total_tests = overall['passed'] + overall['failed'] + overall['skipped'] + overall['errors']
        success_rate = (overall['passed'] / total_tests * 100) if total_tests > 0 else 0

        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Passed: {overall['passed']} ({success_rate:.1f}%)")
        report_lines.append(f"Failed: {overall['failed']}")
        report_lines.append(f"Skipped: {overall['skipped']}")
        report_lines.append(f"Errors: {overall['errors']}")
        report_lines.append("")

        # Category breakdown
        report_lines.append("📈 CATEGORY BREAKDOWN")
        report_lines.append("-" * 40)

        for category in ['critical', 'integration', 'performance', 'security']:
            results = self.results[category]
            total = results['passed'] + results['failed'] + results['skipped'] + results['errors']
            if total > 0:
                success_rate = (results['passed'] / total * 100) if total > 0 else 0
                report_lines.append(f"{category.upper()}:")
                report_lines.append(f"  Total: {total}")
                report_lines.append(f"  Passed: {results['passed']} ({success_rate:.1f}%)")
                report_lines.append(f"  Failed: {results['failed']}")
                report_lines.append(f"  Skipped: {results['skipped']}")
                report_lines.append(f"  Errors: {results['errors']}")
                report_lines.append("")

        # Quality assessment
        report_lines.append("🎯 QUALITY ASSESSMENT")
        report_lines.append("-" * 40)

        if overall['failed'] == 0 and overall['errors'] == 0:
            report_lines.append("✅ ALL TESTS PASSED - Ready for deployment")
        elif self.results['critical']['failed'] == 0 and self.results['critical']['errors'] == 0:
            report_lines.append("⚠️  CRITICAL TESTS PASSED - Review non-critical failures")
        else:
            report_lines.append("❌ CRITICAL FAILURES - Fix required before deployment")

        report_lines.append("")

        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 40)

        if self.results['critical']['failed'] > 0:
            report_lines.append("🔴 Address critical test failures immediately")

        if self.results['security']['failed'] > 0:
            report_lines.append("🔴 Security failures require immediate attention")

        if self.results['performance']['failed'] > 0:
            report_lines.append("🟡 Performance issues should be reviewed")

        if self.results['integration']['failed'] > 0:
            report_lines.append("🟡 Integration issues may impact system stability")

        if overall['failed'] == 0:
            report_lines.append("🟢 System is ready for Phase 1 deployment")
            report_lines.append("🟢 Consider running additional load testing in production")
            report_lines.append("🟢 Monitor system performance and security metrics")

        report_lines.append("")
        report_lines.append("=" * 60)

        return '\n'.join(report_lines)

    def save_report(self, filename: str):
        """Save report to file"""
        report_content = self.generate_report()
        report_path = project_root / filename

        with open(report_path, 'w') as f:
            f.write(report_content)

        self.log(f"Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Test Execution Script")
    parser.add_argument('--critical-only', action='store_true', help='Run only critical tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--security-only', action='store_true', help='Run only security tests')
    parser.add_argument('--full-suite', action='store_true', help='Run complete test suite')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--ci-mode', action='store_true', help='Run in CI mode with strict quality gates')

    args = parser.parse_args()

    # Initialize test runner
    runner = TestRunner(ci_mode=args.ci_mode)

    # Run tests based on arguments
    success = False
    if args.critical_only:
        success = runner.run_critical_tests()
    elif args.integration_only:
        success = runner.run_integration_tests()
    elif args.performance_only:
        success = runner.run_performance_tests()
    elif args.security_only:
        success = runner.run_security_tests()
    else:
        # Default to full suite
        success = runner.run_full_suite()

    # Generate report if requested
    if args.report:
        report = runner.generate_report()
        print("\n" + report)
        runner.save_report("phase1_test_report.txt")

    # Exit with appropriate code
    if args.ci_mode:
        # In CI mode, exit with non-zero code if any tests failed
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    else:
        print(f"\nTest execution completed with {'success' if success else 'failures'}")
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()