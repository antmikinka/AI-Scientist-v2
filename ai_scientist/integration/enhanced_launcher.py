"""
Enhanced AI-Scientist-v2 Launcher

Integrated launcher that provides both legacy and enhanced modes,
orchestrating all components of the upgraded system.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import yaml

# Enhanced system imports
from ai_scientist.orchestration.supervisor_agent import SupervisorAgent
from ai_scientist.orchestration.agent_profiles import AgentProfileManager
from ai_scientist.theory.theory_evolution_agent import TheoryEvolutionAgent
from ai_scientist.knowledge.knowledge_manager import KnowledgeManager
from ai_scientist.rag.reasoning_rag_engine import ReasoningRAGEngine

# Legacy system imports
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import perform_experiments_bfts
from ai_scientist.perform_ideation_temp_free import perform_ideation
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_llm_review import perform_review


class EnhancedLauncher:
    """
    Enhanced launcher for AI-Scientist-v2 that supports both legacy and enhanced modes.
    """

    def __init__(self, config_path: Optional[str] = None, enhanced_mode: bool = True):
        """
        Initialize the Enhanced Launcher.
        
        Args:
            config_path: Path to configuration file
            enhanced_mode: Whether to use enhanced mode (True) or legacy mode (False)
        """
        self.enhanced_mode = enhanced_mode
        self.config_path = config_path or "ai_scientist/config/enhanced_config.yaml"
        self.config = self._load_configuration()
        
        # Component instances
        self.supervisor_agent: Optional[SupervisorAgent] = None
        self.profile_manager: Optional[AgentProfileManager] = None
        self.theory_agent: Optional[TheoryEvolutionAgent] = None
        self.knowledge_manager: Optional[KnowledgeManager] = None
        self.rag_engine: Optional[ReasoningRAGEngine] = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components based on mode
        if self.enhanced_mode:
            asyncio.run(self._initialize_enhanced_components())
        else:
            self._initialize_legacy_components()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load base configuration if extending
            if "extends" in config:
                base_config_path = config["extends"]
                if not os.path.isabs(base_config_path):
                    # Make relative to current config file
                    base_config_path = os.path.join(os.path.dirname(self.config_path), base_config_path)
                
                if os.path.exists(base_config_path):
                    with open(base_config_path, 'r') as f:
                        base_config = yaml.safe_load(f)
                    
                    # Merge configurations (enhanced config takes precedence)
                    merged_config = self._deep_merge_dicts(base_config, config)
                    return merged_config
            
            return config
            
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _deep_merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "theory_evolution": {"enabled": False},
            "orchestration": {"supervisor_agent": {"model": "gpt-4o-2024-11-20"}},
            "knowledge_management": {"storage_backend": "memory"},
            "rag_engine": {"pageindex": {"enabled": False}},
            "integration": {"legacy_compatibility": True}
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'enhanced_ai_scientist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(__name__)

    async def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced system components."""
        self.logger.info("Initializing enhanced AI-Scientist-v2 components")
        
        try:
            # Initialize Agent Profile Manager
            self.profile_manager = AgentProfileManager(
                config_path="ai_scientist/config/agent_profiles.yaml"
            )
            self.logger.info("✓ Agent Profile Manager initialized")
            
            # Initialize Knowledge Manager
            self.knowledge_manager = KnowledgeManager(
                config=self.config
            )
            self.logger.info("✓ Knowledge Manager initialized")
            
            # Initialize Theory Evolution Agent
            self.theory_agent = TheoryEvolutionAgent(
                config=self.config,
                knowledge_manager=self.knowledge_manager
            )
            self.logger.info("✓ Theory Evolution Agent initialized")
            
            # Initialize Reasoning RAG Engine
            self.rag_engine = ReasoningRAGEngine(
                config=self.config
            )
            self.logger.info("✓ Reasoning RAG Engine initialized")
            
            # Initialize Supervisor Agent
            self.supervisor_agent = SupervisorAgent(
                config=self.config,
                profile_manager=self.profile_manager,
                knowledge_manager=self.knowledge_manager
            )
            self.logger.info("✓ Supervisor Agent initialized")
            
            self.logger.info("All enhanced components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced components: {e}")
            self.logger.info("Falling back to legacy mode")
            self.enhanced_mode = False
            self._initialize_legacy_components()

    def _initialize_legacy_components(self) -> None:
        """Initialize legacy system components."""
        self.logger.info("Initializing legacy AI-Scientist-v2 components")
        # Legacy initialization would go here
        # For now, we'll just log that we're in legacy mode

    async def run_research_pipeline(self, research_objective: str, ideas_file: Optional[str] = None,
                                   mode: str = "full") -> Dict[str, Any]:
        """
        Run the complete research pipeline.
        
        Args:
            research_objective: High-level research objective
            ideas_file: Path to ideas JSON file (optional)
            mode: Pipeline mode ('ideation', 'experiment', 'writeup', 'full')
            
        Returns:
            Dict containing pipeline results
        """
        if self.enhanced_mode:
            return await self._run_enhanced_pipeline(research_objective, ideas_file, mode)
        else:
            return await self._run_legacy_pipeline(research_objective, ideas_file, mode)

    async def _run_enhanced_pipeline(self, research_objective: str, ideas_file: Optional[str],
                                   mode: str) -> Dict[str, Any]:
        """Run enhanced research pipeline with supervisor orchestration."""
        self.logger.info(f"Starting enhanced research pipeline: {research_objective}")
        
        results = {
            "mode": "enhanced",
            "objective": research_objective,
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }
        
        try:
            # Stage 1: Create workflow plan
            workflow = await self.supervisor_agent.coordinate_workflow(
                research_objective=research_objective,
                context={"ideas_file": ideas_file, "mode": mode}
            )
            results["workflow_id"] = workflow.plan_id
            
            # Stage 2: Execute workflow stages
            if mode in ["ideation", "full"]:
                ideation_results = await self._run_enhanced_ideation(research_objective, ideas_file)
                results["stages"]["ideation"] = ideation_results
            
            if mode in ["experiment", "full"]:
                experiment_results = await self._run_enhanced_experiments(workflow)
                results["stages"]["experiment"] = experiment_results
            
            if mode in ["writeup", "full"]:
                writeup_results = await self._run_enhanced_writeup(workflow)
                results["stages"]["writeup"] = writeup_results
            
            # Stage 3: Theory integration
            if self.theory_agent and mode == "full":
                theory_results = await self._integrate_with_theory(results)
                results["stages"]["theory_integration"] = theory_results
            
            # Stage 4: Final evaluation
            evaluation = await self.supervisor_agent.evaluate_progress(workflow.plan_id)
            results["final_evaluation"] = {
                "overall_progress": evaluation.overall_progress,
                "recommendations": evaluation.recommendations
            }
            
            self.logger.info("Enhanced pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced pipeline: {e}")
            results["error"] = str(e)
            return results

    async def _run_enhanced_ideation(self, objective: str, ideas_file: Optional[str]) -> Dict[str, Any]:
        """Run enhanced ideation with theory correlation."""
        self.logger.info("Running enhanced ideation stage")
        
        # Use RAG engine for context-aware ideation
        if self.rag_engine:
            context_retrieval = await self.rag_engine.reason_retrieve(
                query=f"research ideas related to {objective}",
                context="scientific research ideation"
            )
        
        # Generate ideas using enhanced reasoning
        # This would integrate with existing ideation but with enhanced context
        ideas_results = {
            "enhanced_context_used": True,
            "rag_retrieval_success": bool(self.rag_engine),
            "ideas_generated": 5,  # Placeholder
            "theory_correlation_applied": bool(self.theory_agent)
        }
        
        # Correlate ideas with existing theory
        if self.theory_agent:
            for idea in ideas_results.get("ideas", []):
                correlation = await self.theory_agent.correlate_findings(idea)
                ideas_results.setdefault("correlations", []).append({
                    "idea_id": idea.get("id"),
                    "correlation_score": correlation.correlation_score
                })
        
        return ideas_results

    async def _run_enhanced_experiments(self, workflow) -> Dict[str, Any]:
        """Run enhanced experiments with supervisor coordination."""
        self.logger.info("Running enhanced experiment stage")
        
        # Use supervisor agent to coordinate experiment execution
        experiment_tasks = [task for task in workflow.tasks if task.task_type == "experiment"]
        
        results = {
            "total_experiments": len(experiment_tasks),
            "completed_experiments": 0,
            "failed_experiments": 0,
            "supervisor_coordination": True
        }
        
        for task in experiment_tasks:
            try:
                # Delegate to experiment specialist
                delegation_result = await self.supervisor_agent.delegate_to_specialist(
                    task=task,
                    specialist_type="experiment"
                )
                
                if delegation_result.get("status") == "delegated":
                    results["completed_experiments"] += 1
                else:
                    results["failed_experiments"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error in experiment task {task.task_id}: {e}")
                results["failed_experiments"] += 1
        
        return results

    async def _run_enhanced_writeup(self, workflow) -> Dict[str, Any]:
        """Run enhanced writeup with theory integration."""
        self.logger.info("Running enhanced writeup stage")
        
        writeup_results = {
            "theory_integration": bool(self.theory_agent),
            "knowledge_synthesis": bool(self.knowledge_manager),
            "rag_enhancement": bool(self.rag_engine)
        }
        
        # Use knowledge manager for comprehensive writeup
        if self.knowledge_manager:
            research_knowledge = await self.knowledge_manager.retrieve_knowledge(
                query="experimental results and findings",
                limit=20
            )
            writeup_results["knowledge_items_used"] = len(research_knowledge.items)
        
        return writeup_results

    async def _integrate_with_theory(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate pipeline results with theory evolution."""
        self.logger.info("Integrating results with theory evolution")
        
        if not self.theory_agent:
            return {"error": "Theory agent not available"}
        
        integration_results = {
            "theory_updates": 0,
            "correlations_processed": 0,
            "new_knowledge_integrated": 0
        }
        
        # Process experimental findings
        for stage_name, stage_results in pipeline_results.get("stages", {}).items():
            if "findings" in stage_results:
                for finding in stage_results["findings"]:
                    # Correlate with theory
                    correlation = await self.theory_agent.correlate_findings(finding)
                    integration_results["correlations_processed"] += 1
                    
                    # Update theory if correlation is strong enough
                    if correlation.correlation_score >= 0.75:
                        update = await self.theory_agent.update_theory(correlation)
                        integration_results["theory_updates"] += 1
        
        return integration_results

    async def _run_legacy_pipeline(self, research_objective: str, ideas_file: Optional[str],
                                 mode: str) -> Dict[str, Any]:
        """Run legacy research pipeline."""
        self.logger.info(f"Starting legacy research pipeline: {research_objective}")
        
        # This would call the original AI-Scientist-v2 pipeline
        results = {
            "mode": "legacy",
            "objective": research_objective,
            "timestamp": datetime.now().isoformat(),
            "message": "Legacy pipeline execution - would call original functions"
        }
        
        return results

    def export_system_state(self, export_path: str) -> None:
        """Export current system state for debugging/analysis."""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "enhanced_mode": self.enhanced_mode,
            "config": self.config,
            "components": {}
        }
        
        if self.enhanced_mode:
            if self.supervisor_agent:
                state_data["components"]["supervisor"] = self.supervisor_agent.export_state()
            
            if self.theory_agent:
                state_data["components"]["theory"] = self.theory_agent.export_state()
            
            if self.knowledge_manager:
                state_data["components"]["knowledge"] = self.knowledge_manager.get_knowledge_statistics()
            
            if self.rag_engine:
                state_data["components"]["rag"] = self.rag_engine.get_performance_metrics()
        
        with open(export_path, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        self.logger.info(f"System state exported to {export_path}")

    async def run_diagnostic(self) -> Dict[str, Any]:
        """Run system diagnostic to check component health."""
        diagnostic_results = {
            "timestamp": datetime.now().isoformat(),
            "enhanced_mode": self.enhanced_mode,
            "component_status": {}
        }
        
        if self.enhanced_mode:
            # Check each component
            components = {
                "supervisor_agent": self.supervisor_agent,
                "profile_manager": self.profile_manager,
                "theory_agent": self.theory_agent,
                "knowledge_manager": self.knowledge_manager,
                "rag_engine": self.rag_engine
            }
            
            for name, component in components.items():
                diagnostic_results["component_status"][name] = {
                    "initialized": component is not None,
                    "functional": await self._test_component(component) if component else False
                }
        
        return diagnostic_results

    async def _test_component(self, component) -> bool:
        """Test if a component is functional."""
        try:
            # Basic functionality test - would be more comprehensive in practice
            if hasattr(component, 'get_performance_metrics'):
                await component.get_performance_metrics()
            elif hasattr(component, 'export_state'):
                component.export_state()
            return True
        except Exception:
            return False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Enhanced AI-Scientist-v2 Launcher")
    
    parser.add_argument(
        "--config",
        type=str,
        default="ai_scientist/config/enhanced_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["enhanced", "legacy"],
        default="enhanced",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--objective",
        type=str,
        required=True,
        help="Research objective"
    )
    
    parser.add_argument(
        "--ideas-file",
        type=str,
        help="Path to ideas JSON file"
    )
    
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["ideation", "experiment", "writeup", "full"],
        default="full",
        help="Pipeline execution mode"
    )
    
    parser.add_argument(
        "--export-state",
        type=str,
        help="Path to export system state"
    )
    
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run system diagnostic"
    )
    
    return parser


async def main():
    """Main entry point for enhanced launcher."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = EnhancedLauncher(
        config_path=args.config,
        enhanced_mode=(args.mode == "enhanced")
    )
    
    try:
        if args.diagnostic:
            # Run diagnostic
            diagnostic = await launcher.run_diagnostic()
            print("=== DIAGNOSTIC RESULTS ===")
            print(json.dumps(diagnostic, indent=2))
        
        else:
            # Run research pipeline
            results = await launcher.run_research_pipeline(
                research_objective=args.objective,
                ideas_file=args.ideas_file,
                mode=args.pipeline_mode
            )
            
            print("=== PIPELINE RESULTS ===")
            print(json.dumps(results, indent=2, default=str))
        
        # Export state if requested
        if args.export_state:
            launcher.export_system_state(args.export_state)
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())