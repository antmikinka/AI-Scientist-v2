# Advanced User Guide

## Overview

This comprehensive guide is designed for advanced users who want to leverage the full power of the AI-Scientist-v2 multi-agent system for complex research workflows. Learn advanced techniques, optimization strategies, and best practices for maximizing research productivity.

## Table of Contents

1. [Advanced Research Workflows](#advanced-research-workflows)
2. [Multi-Agent Orchestration](#multi-agent-orchestration)
3. [Custom Research Templates](#custom-research-templates)
4. [Data Integration Strategies](#data-integration-strategies)
5. [Performance Optimization](#performance-optimization)
6. [Advanced Configuration](#advanced-configuration)
7. [Collaboration Features](#collaboration-features)
8. [API Integration Examples](#api-integration-examples)
9. [Troubleshooting Advanced Issues](#troubleshooting-advanced-issues)
10. [Case Studies](#case-studies)

## Advanced Research Workflows

### 1. Complex Multi-Phase Research

#### Climate Change Impact Study Workflow

```python
# workflows/climate_research_workflow.py
from ai_scientist.research import ResearchOrchestrator
from ai_scientist.agents import ResearchAgent, DataAnalysisAgent, EthicalFrameworkAgent
import asyncio

class ClimateResearchWorkflow:
    """Advanced climate change impact research workflow"""

    def __init__(self):
        self.orchestrator = ResearchOrchestrator()
        self.agents = {}
        self.session_id = None

    async def initialize_research(self, config):
        """Initialize comprehensive climate research"""

        # Create research session
        self.session_id = await self.orchestrator.create_session({
            "title": "Global Climate Change Impact Assessment 2024-2050",
            "description": "Comprehensive analysis of climate change impacts across multiple domains",
            "domain": "climate_science",
            "coordination_mode": "parallel",
            "objectives": [
                "Analyze temperature trends and projections",
                "Assess impacts on biodiversity and ecosystems",
                "Evaluate economic consequences",
                "Model sea-level rise scenarios",
                "Develop mitigation strategies"
            ]
        })

        # Initialize specialized agents
        await self.setup_specialized_agents(config)

        return self.session_id

    async def setup_specialized_agents(self, config):
        """Setup domain-specific agents"""

        # Climate Modeling Agent
        climate_agent = ResearchAgent({
            "name": "Climate Modeling Specialist",
            "capabilities": ["climate_modeling", "data_assimilation", "scenario_analysis"],
            "model_preferences": ["CESM2", "GFDL-CM4", "IPSL-CM6A-LR"]
        })
        await climate_agent.initialize()
        self.agents["climate_modeler"] = climate_agent

        # Biodiversity Impact Agent
        biodiversity_agent = ResearchAgent({
            "name": "Biodiversity Impact Analyst",
            "capabilities": ["species_distribution_modeling", "ecosystem_analysis", "risk_assessment"],
            "expertise_domains": ["marine_biology", "terrestrial_ecology", "conservation_science"]
        })
        await biodiversity_agent.initialize()
        self.agents["biodiversity_analyst"] = biodiversity_agent

        # Economic Impact Agent
        economics_agent = ResearchAgent({
            "name": "Economic Impact Specialist",
            "capabilities": ["economic_modeling", "cost_benefit_analysis", "policy_analysis"],
            "models": ["CGE_models", "IAM_models", "risk_assessment_frameworks"]
        })
        await economics_agent.initialize()
        self.agents["economist"] = economics_agent

        # Data Analysis Agent
        data_agent = DataAnalysisAgent({
            "name": "Climate Data Analyst",
            "capabilities": ["statistical_analysis", "machine_learning", "visualization"],
            "tools": ["Python", "R", "MATLAB", "GIS"]
        })
        await data_agent.initialize()
        self.agents["data_analyst"] = data_agent

        # Ethical Oversight Agent
        ethical_agent = EthicalFrameworkAgent({
            "name": "Research Ethics Coordinator",
            "frameworks": ["utilitarian", "precautionary", "justice", "care_ethics"],
            "strictness": 0.9
        })
        await ethical_agent.initialize()
        self.agents["ethical_coordinator"] = ethical_agent

    async def execute_research_phases(self):
        """Execute multi-phase research workflow"""

        # Phase 1: Data Collection and Preparation
        await self.phase_1_data_collection()

        # Phase 2: Climate Modeling
        await self.phase_2_climate_modeling()

        # Phase 3: Impact Assessment
        await self.phase_3_impact_assessment()

        # Phase 4: Economic Analysis
        await self.phase_4_economic_analysis()

        # Phase 5: Mitigation Strategy Development
        await self.phase_5_mitigation_strategies()

        # Phase 6: Synthesis and Reporting
        await self.phase_6_synthesis()

    async def phase_1_data_collection(self):
        """Phase 1: Comprehensive data collection"""

        tasks = [
            {
                "agent": "data_analyst",
                "task": {
                    "type": "data_collection",
                    "datasets": [
                        "global_temperature_records",
                        "ghg_concentration_data",
                        "sea_level_measurements",
                        "biodiversity_survey_data",
                        "economic_indicators"
                    ],
                    "time_period": "1850-2024",
                    "quality_standards": "peer_reviewed"
                }
            },
            {
                "agent": "climate_modeler",
                "task": {
                    "type": "model_input_preparation",
                    "requirements": {
                        "spatial_resolution": "1x1 degree",
                        "temporal_resolution": "monthly",
                        "variables": ["temperature", "precipitation", "co2_concentration"]
                    }
                }
            }
        ]

        results = await self.orchestrator.execute_parallel_tasks(tasks)
        await self.validate_phase_results("data_collection", results)

    async def phase_2_climate_modeling(self):
        """Phase 2: Advanced climate modeling"""

        modeling_tasks = [
            {
                "agent": "climate_modeler",
                "task": {
                    "type": "ensemble_modeling",
                    "models": ["CESM2", "GFDL-CM4", "IPSL-CM6A-LR"],
                    "scenarios": ["SSP1-2.6", "SSP3-7.0", "SSP5-8.5"],
                    "time_horizon": "2024-2050",
                    "output_variables": [
                        "surface_temperature",
                        "precipitation_patterns",
                        "extreme_events_frequency",
                        "sea_level_pressure"
                    ]
                }
            },
            {
                "agent": "data_analyst",
                "task": {
                    "type": "model_validation",
                    "validation_period": "1980-2020",
                    "metrics": ["RMSE", "correlation_coefficient", "bias"]
                }
            }
        ]

        modeling_results = await self.orchestrator.execute_parallel_tasks(modeling_tasks)

        # Ethical review of modeling assumptions
        ethical_review = await self.agents["ethical_coordinator"].submit_task({
            "type": "ethical_assessment",
            "resource": modeling_results,
            "resource_type": "climate_models",
            "concerns": ["model_uncertainty", "regional_bias", "scenario_assumptions"]
        })

        await self.validate_phase_results("climate_modeling", modeling_results, ethical_review)

    async def phase_3_impact_assessment(self):
        """Phase 3: Multi-domain impact assessment"""

        impact_tasks = [
            {
                "agent": "biodiversity_analyst",
                "task": {
                    "type": "biodiversity_impact",
                    "assessments": [
                        "species_vulnerability_analysis",
                        "habitat_suitability_modeling",
                        "migration_pattern_assessment",
                        "extinction_risk_evaluation"
                    ],
                    "ecosystems": ["marine", "terrestrial", "freshwater", "arctic"]
                }
            },
            {
                "agent": "climate_modeler",
                "task": {
                    "type": "extreme_event_analysis",
                    "events": ["heatwaves", "droughts", "floods", "wildfires"],
                    "frequency_analysis": True,
                    "intensity_trends": True
                }
            }
        ]

        impact_results = await self.orchestrator.execute_parallel_tasks(impact_tasks)
        await self.validate_phase_results("impact_assessment", impact_results)

    async def phase_4_economic_analysis(self):
        """Phase 4: Economic consequence analysis"""

        economic_tasks = [
            {
                "agent": "economist",
                "task": {
                    "type": "economic_impact_modeling",
                    "sectors": ["agriculture", "tourism", "infrastructure", "health"],
                    "analysis_types": [
                        "direct_cost_assessment",
                        "indirect_impact_evaluation",
                        "adaptation_cost_benefit",
                        "mitigation_economic_analysis"
                    ],
                    "geographic_scope": "global",
                    "time_horizon": "2024-2050"
                }
            },
            {
                "agent": "data_analyst",
                "task": {
                    "type": "economic_data_integration",
                    "sources": ["World_Bank", "IMF", "national_statistics", "industry_reports"]
                }
            }
        ]

        economic_results = await self.orchestrator.execute_parallel_tasks(economic_tasks)
        await self.validate_phase_results("economic_analysis", economic_results)

    async def phase_5_mitigation_strategies(self):
        """Phase 5: Develop mitigation and adaptation strategies"""

        strategy_tasks = [
            {
                "agent": "climate_modeler",
                "task": {
                    "type": "mitigation_scenario_modeling",
                    "strategies": [
                        "carbon_capture_technologies",
                        "renewable_energy_transition",
                        "afforestation_programs",
                        "policy_interventions"
                    ],
                    "effectiveness_metrics": ["co2_reduction", "temperature_change", "cost_effectiveness"]
                }
            },
            {
                "agent": "economist",
                "task": {
                    "type": "policy_recommendations",
                    "focus_areas": ["carbon_pricing", "subsidy_reform", "international_cooperation"]
                }
            },
            {
                "agent": "ethical_coordinator",
                "task": {
                    "type": "ethical_evaluation",
                    "criteria": ["intergenerational_justice", "equitable_distribution", "precautionary_principle"]
                }
            }
        ]

        strategy_results = await self.orchestrator.execute_parallel_tasks(strategy_tasks)
        await self.validate_phase_results("mitigation_strategies", strategy_results)

    async def phase_6_synthesis(self):
        """Phase 6: Synthesize results and generate comprehensive report"""

        synthesis_task = {
            "agent": "data_analyst",
            "task": {
                "type": "comprehensive_synthesis",
                "components": [
                    "executive_summary",
                    "methodology_review",
                    "key_findings",
                    "uncertainty_analysis",
                    "policy_recommendations",
                    "future_research_needs"
                ],
                "format": "academic_paper",
                "visualization_requirements": [
                    "temperature_trend_plots",
                    "impact_distribution_maps",
                    "economic_cost_benefit_charts",
                    "scenario_comparison_tables"
                ]
            }
        }

        synthesis_results = await self.orchestrator.execute_task(synthesis_task)

        # Final ethical review
        final_ethical_review = await self.agents["ethical_coordinator"].submit_task({
            "type": "final_ethical_assessment",
            "resource": synthesis_results,
            "resource_type": "research_report",
            "compliance_check": True
        })

        return synthesis_results, final_ethical_review

    async def validate_phase_results(self, phase_name, results, ethical_review=None):
        """Validate phase results and ensure quality standards"""

        validation_results = {}

        for agent_id, result in results.items():
            # Quality metrics
            quality_score = await self.assess_result_quality(result)
            validation_results[agent_id] = {
                "quality_score": quality_score,
                "completeness": await self.check_completeness(result),
                "consistency": await self.check_consistency(result)
            }

        # Ethical compliance
        if ethical_review:
            validation_results["ethical_compliance"] = {
                "overall_score": ethical_review.get("overall_score", 0),
                "issues_identified": ethical_review.get("identified_issues", []),
                "approval_status": ethical_review.get("status", "pending")
            }

        # Store validation results
        await self.orchestrator.store_phase_validation(phase_name, validation_results)

        return validation_results

    async def assess_result_quality(self, result):
        """Assess quality of research results"""
        # Implementation for quality assessment
        return 0.85  # Placeholder

    async def check_completeness(self, result):
        """Check if results are complete"""
        # Implementation for completeness check
        return True  # Placeholder

    async def check_consistency(self, result):
        """Check internal consistency of results"""
        # Implementation for consistency check
        return True  # Placeholder

# Usage example
async def run_climate_research():
    workflow = ClimateResearchWorkflow()

    config = {
        "research_domain": "climate_science",
        "ethical_frameworks": ["utilitarian", "precautionary"],
        "quality_threshold": 0.8
    }

    session_id = await workflow.initialize_research(config)
    print(f"Research session created: {session_id}")

    # Execute full research workflow
    final_results, ethical_approval = await workflow.execute_research_phases()

    print("Research completed successfully!")
    print(f"Final ethical approval score: {ethical_approval['overall_score']}")

    return final_results
```

### 2. Real-time Research Monitoring

```python
# workflows/real_time_monitoring.py
from ai_scientist.monitoring import ResearchMonitor
from ai_scientist.websocket import WebSocketClient
import asyncio

class RealTimeResearchMonitor:
    """Real-time monitoring and adjustment of research workflows"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.monitor = ResearchMonitor()
        self.ws_client = WebSocketClient()
        self.alerts = []
        self.performance_metrics = {}

    async def start_monitoring(self):
        """Start real-time monitoring"""

        # Connect to WebSocket for real-time updates
        await self.ws_client.connect()

        # Subscribe to research events
        await self.ws_client.subscribe({
            "events": ["research_progress", "agent_status", "ethical_assessment"],
            "filters": {"session_id": self.session_id}
        })

        # Set up event handlers
        self.ws_client.on("research_progress", self.handle_progress_update)
        self.ws_client.on("agent_status", self.handle_agent_status)
        self.ws_client.on("ethical_assessment", self.handle_ethical_assessment)

        # Start monitoring loop
        asyncio.create_task(self.monitoring_loop())

    async def handle_progress_update(self, data):
        """Handle research progress updates"""
        progress = data.get("overall_progress", 0)
        current_phase = data.get("current_phase", "unknown")

        print(f"Research Progress: {progress:.1f}% - Phase: {current_phase}")

        # Check for progress issues
        if progress < 0.1 and current_phase != "initialization":
            await self.trigger_alert("LOW_PROGRESS", {
                "message": f"Low progress in phase {current_phase}",
                "progress": progress,
                "recommendation": "Investigate agent performance or task complexity"
            })

    async def handle_agent_status(self, data):
        """Handle agent status updates"""
        agent_id = data.get("agent_id")
        status = data.get("status")
        resource_usage = data.get("resource_usage", {})

        # Check for resource issues
        cpu_usage = resource_usage.get("cpu", 0)
        memory_usage = resource_usage.get("memory", 0)

        if cpu_usage > 0.9:
            await self.trigger_alert("HIGH_CPU_USAGE", {
                "agent_id": agent_id,
                "cpu_usage": cpu_usage,
                "recommendation": "Consider scaling resources or optimizing algorithms"
            })

        if memory_usage > 0.9:
            await self.trigger_alert("HIGH_MEMORY_USAGE", {
                "agent_id": agent_id,
                "memory_usage": memory_usage,
                "recommendation": "Consider memory optimization or increasing allocated memory"
            })

    async def handle_ethical_assessment(self, data):
        """Handle ethical assessment results"""
        overall_score = data.get("assessment", {}).get("overall_score", 0)
        status = data.get("assessment", {}).get("status", "unknown")

        if status == "non_compliant":
            await self.trigger_alert("ETHICAL_VIOLATION", {
                "assessment": data.get("assessment"),
                "recommendation": "Review and address ethical concerns before proceeding"
            })
        elif overall_score < 0.7:
            await self.trigger_alert("LOW_ETHICAL_SCORE", {
                "overall_score": overall_score,
                "recommendation": "Consider additional ethical safeguards"
            })

    async def trigger_alert(self, alert_type, details):
        """Trigger monitoring alert"""
        alert = {
            "type": alert_type,
            "timestamp": asyncio.get_event_loop().time(),
            "details": details
        }

        self.alerts.append(alert)
        print(f"ALERT: {alert_type} - {details['message']}")

        # Take automatic action if configured
        await self.handle_alert(alert)

    async def handle_alert(self, alert):
        """Handle monitoring alerts with automatic responses"""

        if alert["type"] == "LOW_PROGRESS":
            # Investigate and potentially reassign tasks
            await self.investigate_progress_issue()

        elif alert["type"] == "HIGH_CPU_USAGE":
            # Scale up resources or optimize
            await self.optimize_resource_usage("cpu", alert["details"]["agent_id"])

        elif alert["type"] == "ETHICAL_VIOLATION":
            # Pause research and require human review
            await self.pause_research_for_review()

    async def investigate_progress_issue(self):
        """Investigate and resolve progress issues"""
        print("Investigating progress issues...")

        # Get current agent statuses
        agent_statuses = await self.monitor.get_agent_statuses()

        # Identify bottlenecks
        bottlenecks = []
        for agent_id, status in agent_statuses.items():
            if status["status"] == "error":
                bottlenecks.append(agent_id)
            elif status["queue_size"] > 10:
                bottlenecks.append(agent_id)

        # Take corrective action
        for agent_id in bottlenecks:
            await self.resolve_agent_issue(agent_id)

    async def resolve_agent_issue(self, agent_id):
        """Resolve specific agent issues"""
        print(f"Resolving issues for agent: {agent_id}")

        # Get agent error details
        error_details = await self.monitor.get_agent_errors(agent_id)

        if error_details:
            # Restart agent if needed
            await self.monitor.restart_agent(agent_id)

        # Rebalance tasks if queue is full
        queue_size = await self.monitor.get_agent_queue_size(agent_id)
        if queue_size > 10:
            await self.rebalance_agent_tasks(agent_id)

    async def monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # Collect performance metrics
                metrics = await self.monitor.collect_metrics()
                self.performance_metrics = metrics

                # Check for anomalies
                anomalies = await self.detect_anomalies(metrics)
                for anomaly in anomalies:
                    await self.trigger_alert("ANOMALY_DETECTED", anomaly)

                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        report = {
            "session_id": self.session_id,
            "monitoring_period": "last_24_hours",
            "performance_summary": self.performance_metrics,
            "alerts_summary": {
                "total_alerts": len(self.alerts),
                "alert_types": self.categorize_alerts(),
                "resolution_status": self.check_alert_resolution()
            },
            "recommendations": await self.generate_monitoring_recommendations()
        }

        return report
```

## Multi-Agent Orchestration

### Advanced Coordination Patterns

```python
# orchestration/advanced_patterns.py
from ai_scientist.orchestration import CoordinationManager
from typing import List, Dict, Any
import asyncio

class AdvancedCoordinationManager(CoordinationManager):
    """Advanced multi-agent coordination with complex patterns"""

    def __init__(self):
        super().__init__()
        self.coordination_patterns = {
            "consensus": ConsensusPattern(),
            "competitive": CompetitivePattern(),
            "hierarchical": HierarchicalPattern(),
            "adaptive": AdaptivePattern()
        }

    async def execute_consensus_workflow(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with consensus-based coordination"""

        pattern = self.coordination_patterns["consensus"]
        return await pattern.execute(agents, task)

    async def execute_competitive_workflow(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with competitive coordination"""

        pattern = self.coordination_patterns["competitive"]
        return await pattern.execute(agents, task)

    async def execute_adaptive_workflow(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with adaptive coordination based on complexity"""

        # Analyze task complexity
        complexity_score = await self.analyze_task_complexity(task)

        if complexity_score > 0.8:
            pattern = self.coordination_patterns["hierarchical"]
        elif complexity_score > 0.5:
            pattern = self.coordination_patterns["consensus"]
        else:
            pattern = self.coordination_patterns["parallel"]

        return await pattern.execute(agents, task)

class ConsensusPattern:
    """Consensus-based coordination pattern"""

    async def execute(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with consensus mechanism"""

        initial_results = {}

        # Phase 1: Independent analysis
        for agent_id in agents:
            result = await self.submit_task_to_agent(agent_id, task)
            initial_results[agent_id] = result

        # Phase 2: Consensus building
        consensus_rounds = 0
        max_rounds = 3
        consensus_achieved = False

        while consensus_rounds < max_rounds and not consensus_achieved:
            consensus_achieved = await self.build_consensus(
                agents, initial_results, consensus_rounds
            )
            consensus_rounds += 1

        # Phase 3: Final synthesis
        if consensus_achieved:
            final_result = await self.synthesize_consensus_results(initial_results)
        else:
            final_result = await self.handle_consensus_failure(initial_results)

        return {
            "consensus_achieved": consensus_achieved,
            "consensus_rounds": consensus_rounds,
            "individual_results": initial_results,
            "final_result": final_result
        }

    async def build_consensus(self, agents: List[str], results: Dict, round_num: int) -> bool:
        """Build consensus among agents"""

        # Calculate agreement metrics
        agreement_scores = await self.calculate_agreement_scores(results)

        if agreement_scores["overall_agreement"] > 0.8:
            return True

        # If not enough agreement, facilitate discussion
        if round_num < 2:
            await self.facilitate_agent_discussion(agents, results, agreement_scores)

        return False

class CompetitivePattern:
    """Competitive coordination pattern"""

    async def execute(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with competitive approach"""

        # Set up competition parameters
        competition_config = {
            "time_limit": task.get("time_limit", 3600),  # 1 hour default
            "evaluation_criteria": task.get("criteria", ["accuracy", "efficiency", "innovation"]),
            "weights": task.get("weights", {"accuracy": 0.4, "efficiency": 0.3, "innovation": 0.3})
        }

        # Submit task to all agents simultaneously
        competition_results = {}

        for agent_id in agents:
            # Submit with competition metadata
            competitive_task = {
                **task,
                "competition": competition_config,
                "deadline": asyncio.get_event_loop().time() + competition_config["time_limit"]
            }

            result = await self.submit_competitive_task(agent_id, competitive_task)
            competition_results[agent_id] = result

        # Evaluate results
        evaluation = await self.evaluate_competition_results(
            competition_results, competition_config
        )

        # Select winner
        winner = evaluation["ranked_agents"][0]

        return {
            "competition_type": "competitive",
            "winner": winner,
            "evaluation": evaluation,
            "all_results": competition_results,
            "best_solution": competition_results[winner]["solution"]
        }

class AdaptivePattern:
    """Adaptive coordination pattern"""

    async def execute(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with adaptive coordination"""

        # Monitor task execution
        execution_monitor = ExecutionMonitor()

        # Start with parallel execution
        current_strategy = "parallel"
        strategy_history = []

        while not execution_monitor.is_complete():
            # Execute current strategy
            if current_strategy == "parallel":
                result = await self.execute_parallel_strategy(agents, task)
            elif current_strategy == "sequential":
                result = await self.execute_sequential_strategy(agents, task)
            elif current_strategy == "hierarchical":
                result = await self.execute_hierarchical_strategy(agents, task)

            # Monitor performance
            performance_metrics = await execution_monitor.get_performance_metrics()

            # Adapt strategy if needed
            new_strategy = await self.adapt_strategy(
                current_strategy, performance_metrics, strategy_history
            )

            if new_strategy != current_strategy:
                strategy_history.append({
                    "strategy": current_strategy,
                    "performance": performance_metrics,
                    "reason": "performance_optimization"
                })
                current_strategy = new_strategy
                print(f"Adapting strategy to: {current_strategy}")

            # Check completion
            if execution_monitor.is_complete():
                break

            await asyncio.sleep(10)  # Wait before next adaptation

        return {
            "adaptive_coordination": True,
            "strategy_history": strategy_history,
            "final_strategy": current_strategy,
            "result": result
        }

class HierarchicalPattern:
    """Hierarchical coordination pattern"""

    async def execute(self, agents: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with hierarchical coordination"""

        # Establish hierarchy
        hierarchy = await self.establish_hierarchy(agents, task)

        # Phase 1: Lead agent creates plan
        lead_agent = hierarchy["lead"]
        subtask_plan = await self.create_subtask_plan(lead_agent, task)

        # Phase 2: Delegate subtasks
        subtask_results = {}

        for subtask in subtask_plan["subtasks"]:
            assigned_agent = hierarchy["assignments"][subtask["id"]]
            result = await self.execute_subtask(assigned_agent, subtask)
            subtask_results[subtask["id"]] = result

        # Phase 3: Lead agent synthesizes results
        final_result = await self.synthesize_hierarchical_results(
            lead_agent, subtask_results, subtask_plan
        )

        return {
            "coordination_pattern": "hierarchical",
            "hierarchy": hierarchy,
            "subtask_results": subtask_results,
            "final_result": final_result
        }
```

## Custom Research Templates

### Research Template System

```python
# templates/research_templates.py
from ai_scientist.templates import ResearchTemplate
import json
from pathlib import Path

class CustomResearchTemplate(ResearchTemplate):
    """Base class for custom research templates"""

    def __init__(self, template_config: Dict[str, Any]):
        super().__init__(template_config)
        self.validate_template()

    def validate_template(self):
        """Validate template configuration"""
        required_fields = ["name", "domain", "phases", "agents"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Template missing required field: {field}")

class DrugDiscoveryTemplate(CustomResearchTemplate):
    """Template for drug discovery research"""

    def __init__(self):
        template_config = {
            "name": "AI-Assisted Drug Discovery",
            "domain": "pharmaceutical_research",
            "description": "Comprehensive drug discovery workflow using AI agents",
            "estimated_duration": "6-12 months",
            "phases": [
                {
                    "name": "Target Identification",
                    "duration": "4-6 weeks",
                    "agents": ["bioinformatics_specialist", "literature_reviewer"],
                    "tasks": [
                        "analyze_disease_mechanisms",
                        "identify_potential_targets",
                        "assess_druggability"
                    ],
                    "deliverables": ["target_shortlist", "validation_plan"]
                },
                {
                    "name": "Virtual Screening",
                    "duration": "6-8 weeks",
                    "agents": ["computational_chemist", "ml_specialist"],
                    "tasks": [
                        "prepare_compound_library",
                        "perform_molecular_docking",
                        "run_ml_predictions",
                        "filter_candidates"
                    ],
                    "deliverables": ["hit_compounds", "screening_report"]
                },
                {
                    "name": "Lead Optimization",
                    "duration": "8-12 weeks",
                    "agents": ["medicinal_chemist", "pharmacologist"],
                    "tasks": [
                        "structure_activity_analysis",
                        "admet_predictions",
                        "synthesize_analogs",
                        "biological_testing"
                    ],
                    "deliverables": ["lead_compounds", "optimization_report"]
                },
                {
                    "name": "Preclinical Testing",
                    "duration": "12-16 weeks",
                    "agents": ["toxicologist", "pharmacologist", "ethical_reviewer"],
                    "tasks": [
                        "in_vitro_testing",
                        "animal_studies",
                        "toxicity_assessment",
                        "efficacy_evaluation"
                    ],
                    "deliverables": ["preclinical_data", "safety_report", "go_no_go_recommendation"]
                }
            ],
            "agents": {
                "bioinformatics_specialist": {
                    "type": "research",
                    "capabilities": ["genomic_analysis", "pathway_analysis", "target_validation"],
                    "tools": ["BLAST", "Ensembl", "KEGG", "Reactome"]
                },
                "computational_chemist": {
                    "type": "research",
                    "capabilities": ["molecular_modeling", "docking", "md_simulation"],
                    "tools": ["AutoDock", "Schrödinger", "GROMACS"]
                },
                "ml_specialist": {
                    "type": "data_analysis",
                    "capabilities": ["deep_learning", "graph_neural_networks", "qsar_modeling"],
                    "tools": ["PyTorch", "TensorFlow", "RDKit"]
                },
                "medicinal_chemist": {
                    "type": "research",
                    "capabilities": ["synthetic_planning", "structure_design", "property_prediction"],
                    "tools": ["ChemDraw", "Reaxys", "SciFinder"]
                },
                "pharmacologist": {
                    "type": "research",
                    "capabilities": ["mechanism_of_action", "pharmacokinetics", "efficacy_testing"],
                    "tools": ["laboratory_equipment", "analytical_instruments"]
                },
                "toxicologist": {
                    "type": "research",
                    "capabilities": ["toxicity_assessment", "safety_evaluation", "risk_analysis"],
                    "tools": ["toxicology_databases", "testing_protocols"]
                },
                "ethical_reviewer": {
                    "type": "ethical_framework",
                    "capabilities": ["animal_ethics", "clinical_ethics", "regulatory_compliance"],
                    "frameworks": ["utilitarian", "deontological", "precautionary"]
                }
            },
            "ethical_considerations": {
                "animal_testing": "strict_regulatory_compliance",
                "human_subjects": "not_applicable_preclinical",
                "data_privacy": "protected_health_information",
                "intellectual_property": "patent_considerations"
            },
            "quality_metrics": {
                "target_validation_score": ">0.8",
                "hit_rate_threshold": ">2%",
                "lead_optimization_factor": ">10x",
                "safety_margin": ">100x_therapeutic_index"
            }
        }
        super().__init__(template_config)

class ClimateImpactTemplate(CustomResearchTemplate):
    """Template for climate impact assessment research"""

    def __init__(self):
        template_config = {
            "name": "Climate Change Impact Assessment",
            "domain": "environmental_science",
            "description": "Comprehensive assessment of climate change impacts across multiple domains",
            "estimated_duration": "3-6 months",
            "phases": [
                {
                    "name": "Baseline Assessment",
                    "duration": "4-6 weeks",
                    "agents": ["climate_scientist", "data_analyst"],
                    "tasks": [
                        "compile_historical_data",
                        "establish_baseline_conditions",
                        "identify_key_indicators"
                    ],
                    "deliverables": ["baseline_report", "indicator_dashboard"]
                },
                {
                    "name": "Climate Modeling",
                    "duration": "6-8 weeks",
                    "agents": ["climate_modeler", "statistician"],
                    "tasks": [
                        "select_climate_models",
                        "run_scenario_simulations",
                        "downscale_to_regional_level",
                        "validate_model_outputs"
                    ],
                    "deliverables": ["climate_projections", "uncertainty_analysis"]
                },
                {
                    "name": "Impact Analysis",
                    "duration": "6-10 weeks",
                    "agents": ["impact_specialist", "economist", "ecologist"],
                    "tasks": [
                        "assess_environmental_impacts",
                        "evaluate_economic_consequences",
                        "analyze_social_effects",
                        "identify_vulnerable_populations"
                    ],
                    "deliverables": ["impact_assessment", "vulnerability_maps", "economic_analysis"]
                },
                {
                    "name": "Adaptation Planning",
                    "duration": "4-6 weeks",
                    "agents": ["policy_analyst", "urban_planner", "stakeholder_coordinator"],
                    "tasks": [
                        "identify_adaptation_options",
                        "cost_benefit_analysis",
                        "stakeholder_consultation",
                        "implementation_roadmap"
                    ],
                    "deliverables": ["adaptation_strategy", "action_plan", "policy_recommendations"]
                }
            ],
            "agents": {
                "climate_scientist": {
                    "type": "research",
                    "capabilities": ["climate_analysis", "trend_detection", "attribution_studies"]
                },
                "climate_modeler": {
                    "type": "research",
                    "capabilities": ["gcm_modeling", "regional_downscaling", "scenario_analysis"]
                },
                "impact_specialist": {
                    "type": "research",
                    "capabilities": ["impact_assessment", "vulnerability_analysis", "risk_mapping"]
                },
                "ecologist": {
                    "type": "research",
                    "capabilities": ["ecosystem_analysis", "biodiversity_assessment", "species_modeling"]
                }
            }
        }
        super().__init__(template_config)

class TemplateManager:
    """Manager for research templates"""

    def __init__(self):
        self.templates = {}
        self.load_builtin_templates()

    def load_builtin_templates(self):
        """Load built-in templates"""
        self.templates["drug_discovery"] = DrugDiscoveryTemplate()
        self.templates["climate_impact"] = ClimateImpactTemplate()

    def register_template(self, template_name: str, template: CustomResearchTemplate):
        """Register a custom template"""
        self.templates[template_name] = template

    def get_template(self, template_name: str) -> CustomResearchTemplate:
        """Get template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        return self.templates[template_name]

    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates"""
        return [
            {
                "name": name,
                "domain": template.config["domain"],
                "description": template.config["description"],
                "duration": template.config["estimated_duration"]
            }
            for name, template in self.templates.items()
        ]

    def create_research_from_template(self, template_name: str, customizations: Dict = None):
        """Create research project from template"""
        template = self.get_template(template_name)

        # Apply customizations
        if customizations:
            template_config = template.config.copy()
            template_config.update(customizations)
            template = CustomResearchTemplate(template_config)

        return template

# Usage examples
async def use_drug_discovery_template():
    """Use drug discovery template for research"""

    template_manager = TemplateManager()

    # Get template
    template = template_manager.get_template("drug_discovery")

    # Customize for specific disease
    customizations = {
        "target_disease": "Alzheimer's_disease",
        "specific_focus": "amyloid_beta_targets",
        "timeline_acceleration": True
    }

    # Create customized research
    research = template_manager.create_research_from_template(
        "drug_discovery", customizations
    )

    # Execute research
    research_session_id = await research.execute()

    return research_session_id
```

This comprehensive Advanced User Guide provides detailed workflows, coordination patterns, and templates for complex research scenarios, enabling users to leverage the full power of the AI-Scientist-v2 multi-agent system for sophisticated scientific research.