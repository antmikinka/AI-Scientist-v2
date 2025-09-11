"""
Simple Enhanced AI-Scientist-v2 Launcher

Production-ready launcher that integrates OpenRouter with existing pipeline stages.
Provides enhanced features while maintaining compatibility with the original system.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import existing AI-Scientist functions
try:
    from ai_scientist.perform_ideation_temp_free import perform_ideation
    from ai_scientist.perform_writeup import perform_writeup  
    from ai_scientist.perform_llm_review import perform_review
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False

# Import OpenRouter integration
from ai_scientist.openrouter import (
    OpenRouterConfig, get_global_client, RAGSystem
)

logger = logging.getLogger(__name__)

class SimpleEnhancedLauncher:
    """
    Simple enhanced launcher that integrates OpenRouter with existing AI-Scientist pipeline.
    Focuses on practical enhancements without complex orchestration.
    """
    
    def __init__(self, config: OpenRouterConfig, rag_system: Optional[RAGSystem] = None):
        """
        Initialize the enhanced launcher.
        
        Args:
            config: OpenRouter configuration
            rag_system: Optional RAG system for document retrieval
        """
        self.config = config
        self.rag_system = rag_system
        self.client = get_global_client()
        self.logger = logging.getLogger(__name__)
        
        # Results tracking
        self.results = {
            "start_time": datetime.now().isoformat(),
            "config_used": {
                "enhanced_pipeline": config.use_enhanced_pipeline,
                "rag_enabled": rag_system is not None,
                "models": {stage: cfg.model for stage, cfg in config.stage_configs.items()}
            },
            "stages": {}
        }
    
    async def run_complete_pipeline(self, idea_name: str, base_dir: str = "./results") -> Dict[str, Any]:
        """
        Run the complete AI-Scientist pipeline with OpenRouter integration.
        
        Args:
            idea_name: Name of the research idea to pursue
            base_dir: Base directory for results
            
        Returns:
            Complete pipeline results
        """
        self.logger.info(f"Starting enhanced pipeline for idea: {idea_name}")
        
        # Create results directory
        results_dir = Path(base_dir) / idea_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Enhanced Ideation
            if self.config.use_enhanced_pipeline:
                ideation_results = await self.run_enhanced_ideation(idea_name, results_dir)
                self.results["stages"]["ideation"] = ideation_results
            else:
                self.logger.info("Skipping enhanced ideation (disabled in config)")
            
            # Stage 2: Experiments (would integrate with existing experiment system)
            experiment_results = await self.run_experiment_stage(idea_name, results_dir)
            self.results["stages"]["experiments"] = experiment_results
            
            # Stage 3: Enhanced Analysis
            analysis_results = await self.run_analysis_stage(idea_name, results_dir)
            self.results["stages"]["analysis"] = analysis_results
            
            # Stage 4: Enhanced Writeup
            writeup_results = await self.run_writeup_stage(idea_name, results_dir)
            self.results["stages"]["writeup"] = writeup_results
            
            # Stage 5: Enhanced Review
            review_results = await self.run_review_stage(idea_name, results_dir)
            self.results["stages"]["review"] = review_results
            
            # Final summary
            self.results["end_time"] = datetime.now().isoformat()
            self.results["success"] = True
            self.results["summary"] = self._generate_pipeline_summary()
            
            # Save results
            results_file = results_dir / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline completed successfully. Results saved to {results_file}")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.results["error"] = str(e)
            self.results["success"] = False
            return self.results
    
    async def run_enhanced_ideation(self, idea_name: str, results_dir: Path) -> Dict[str, Any]:
        """
        Run enhanced ideation with RAG context and improved prompting.
        
        Args:
            idea_name: Research idea name
            results_dir: Results directory
            
        Returns:
            Ideation results
        """
        self.logger.info("Running enhanced ideation stage")
        
        stage_config = self.config.stage_configs.get("ideation")
        if not stage_config:
            raise ValueError("No ideation stage configuration found")
        
        # Get RAG context if available
        rag_context = ""
        if self.rag_system:
            try:
                context_results = self.rag_system.search(
                    f"research ideas and methodologies related to {idea_name}",
                    max_results=5
                )
                if context_results:
                    rag_context = "\n\n=== RELEVANT RESEARCH CONTEXT ===\n"
                    for i, (content, score, metadata) in enumerate(context_results, 1):
                        source = metadata.get('title', 'Unknown')
                        rag_context += f"\n{i}. {source} (relevance: {score:.2f})\n{content[:500]}...\n"
                    rag_context += "\n=== END CONTEXT ===\n"
                    
            except Exception as e:
                self.logger.warning(f"RAG context retrieval failed: {e}")
        
        # Enhanced ideation prompt
        enhanced_prompt = f"""
        You are an expert research scientist tasked with developing a comprehensive research proposal.
        
        Research Topic: {idea_name}
        
        {rag_context}
        
        Please provide a detailed research proposal that includes:
        1. Clear problem statement and research objectives
        2. Novel methodology and experimental approach  
        3. Expected outcomes and significance
        4. Implementation plan with specific steps
        5. Potential challenges and mitigation strategies
        
        Format your response as a structured research proposal that could guide implementation.
        """
        
        try:
            # Use OpenRouter for ideation
            messages = [
                {"role": "system", "content": "You are a world-class research scientist with expertise in developing innovative research proposals."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            response, _ = await self.client.get_response(
                messages=messages,
                model=stage_config.model,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
                cache_strategy=stage_config.caching,
                stage="ideation"  # Add stage for cost tracking
            )
            
            # Save ideation results
            ideation_file = results_dir / "ideation_enhanced.md" 
            with open(ideation_file, 'w') as f:
                f.write(f"# Enhanced Ideation Results for {idea_name}\n\n")
                f.write(f"**Model Used:** {stage_config.model}\n")
                f.write(f"**RAG Context:** {'Yes' if rag_context else 'No'}\n\n")
                f.write(response)
            
            return {
                "status": "completed",
                "model_used": stage_config.model,
                "rag_context_used": bool(rag_context),
                "output_file": str(ideation_file),
                "response_length": len(response)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced ideation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_experiment_stage(self, idea_name: str, results_dir: Path) -> Dict[str, Any]:
        """
        Run experiment stage with OpenRouter integration.
        
        Args:
            idea_name: Research idea name
            results_dir: Results directory
            
        Returns:
            Experiment results
        """
        self.logger.info("Running experiment stage")
        
        stage_config = self.config.stage_configs.get("experiment")
        if not stage_config:
            raise ValueError("No experiment stage configuration found")
        
        try:
            # For now, this is a placeholder that would integrate with existing experiment system
            # In a full implementation, this would call existing experiment functions with OpenRouter
            
            # Enhanced experiment planning
            planning_prompt = f"""
            Based on the research idea "{idea_name}", create a detailed experimental plan.
            
            Please provide:
            1. Specific experimental design and methodology
            2. Data collection procedures  
            3. Analysis methods and statistical approaches
            4. Success metrics and evaluation criteria
            5. Timeline and resource requirements
            
            Focus on reproducible and scientifically rigorous approaches.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert experimental scientist specializing in rigorous research design."},
                {"role": "user", "content": planning_prompt}
            ]
            
            response, _ = await self.client.get_response(
                messages=messages,
                model=stage_config.model,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
                stage="experiment"  # Add stage for cost tracking
            )
            
            # Save experiment plan
            experiment_file = results_dir / "experiment_plan.md"
            with open(experiment_file, 'w') as f:
                f.write(f"# Experiment Plan for {idea_name}\n\n")
                f.write(f"**Model Used:** {stage_config.model}\n\n")
                f.write(response)
            
            return {
                "status": "completed",
                "model_used": stage_config.model,
                "output_file": str(experiment_file),
                "note": "Full experiment execution would integrate with existing systems"
            }
            
        except Exception as e:
            self.logger.error(f"Experiment stage failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_analysis_stage(self, idea_name: str, results_dir: Path) -> Dict[str, Any]:
        """
        Run analysis stage with enhanced reasoning.
        
        Args:
            idea_name: Research idea name
            results_dir: Results directory
            
        Returns:
            Analysis results
        """
        self.logger.info("Running analysis stage")
        
        stage_config = self.config.stage_configs.get("analysis")
        if not stage_config:
            raise ValueError("No analysis stage configuration found")
        
        try:
            # Enhanced analysis with reasoning models (O1/O3)
            analysis_prompt = f"""
            Perform a comprehensive analysis of the research proposal and experimental plan for "{idea_name}".
            
            Please analyze:
            1. Scientific validity and rigor of the proposed methodology
            2. Potential limitations and confounding variables
            3. Expected statistical power and effect sizes
            4. Reproducibility and generalizability
            5. Novel contributions to the field
            6. Risk assessment and mitigation strategies
            
            Provide detailed reasoning for each assessment.
            """
            
            messages = [
                {"role": "system", "content": "You are a senior research analyst with expertise in scientific methodology and statistical analysis."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Use reasoning configuration for O1/O3 models if applicable
            kwargs = {}
            if stage_config.reasoning_config and ("o1" in stage_config.model.lower() or "o3" in stage_config.model.lower()):
                kwargs["reasoning_config"] = stage_config.reasoning_config
            
            response, _ = await self.client.get_response(
                messages=messages,
                model=stage_config.model,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
                **kwargs
            )
            
            # Save analysis results
            analysis_file = results_dir / "analysis_results.md"
            with open(analysis_file, 'w') as f:
                f.write(f"# Analysis Results for {idea_name}\n\n")
                f.write(f"**Model Used:** {stage_config.model}\n")
                f.write(f"**Reasoning Enhanced:** {'Yes' if kwargs.get('reasoning_config') else 'No'}\n\n")
                f.write(response)
            
            return {
                "status": "completed",
                "model_used": stage_config.model,
                "reasoning_enhanced": bool(kwargs.get('reasoning_config')),
                "output_file": str(analysis_file)
            }
            
        except Exception as e:
            self.logger.error(f"Analysis stage failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_writeup_stage(self, idea_name: str, results_dir: Path) -> Dict[str, Any]:
        """
        Run writeup stage with enhanced formatting and citations.
        
        Args:
            idea_name: Research idea name
            results_dir: Results directory
            
        Returns:
            Writeup results
        """
        self.logger.info("Running writeup stage")
        
        stage_config = self.config.stage_configs.get("writeup") 
        if not stage_config:
            raise ValueError("No writeup stage configuration found")
        
        try:
            # Enhanced writeup with academic formatting
            writeup_prompt = f"""
            Write a comprehensive research paper for the study on "{idea_name}".
            
            Structure the paper with:
            1. Abstract (250 words max)
            2. Introduction with literature review
            3. Methods and methodology
            4. Results and findings 
            5. Discussion and implications
            6. Conclusions and future work
            7. References (in academic format)
            
            Use formal academic language and ensure scientific rigor throughout.
            Include placeholder citations where appropriate.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert academic writer specializing in scientific publications. Write clear, rigorous, and well-structured research papers."},
                {"role": "user", "content": writeup_prompt}
            ]
            
            response, _ = await self.client.get_response(
                messages=messages,
                model=stage_config.model,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
                stage="writeup"  # Add stage for cost tracking
            )
            
            # Save writeup
            writeup_file = results_dir / "research_paper.md"
            with open(writeup_file, 'w') as f:
                f.write(response)
            
            # Also save as LaTeX if tools are configured
            if "latex_formatting" in stage_config.tools:
                latex_file = results_dir / "research_paper.tex"
                latex_content = self._convert_to_latex(response)
                with open(latex_file, 'w') as f:
                    f.write(latex_content)
            
            return {
                "status": "completed",
                "model_used": stage_config.model,
                "output_file": str(writeup_file),
                "latex_generated": "latex_formatting" in stage_config.tools
            }
            
        except Exception as e:
            self.logger.error(f"Writeup stage failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def run_review_stage(self, idea_name: str, results_dir: Path) -> Dict[str, Any]:
        """
        Run review stage with comprehensive evaluation.
        
        Args:
            idea_name: Research idea name
            results_dir: Results directory
            
        Returns:
            Review results
        """
        self.logger.info("Running review stage")
        
        stage_config = self.config.stage_configs.get("review")
        if not stage_config:
            raise ValueError("No review stage configuration found")
        
        try:
            # Read the research paper for review
            paper_file = results_dir / "research_paper.md"
            if paper_file.exists():
                with open(paper_file, 'r') as f:
                    paper_content = f.read()
            else:
                paper_content = f"Research paper for {idea_name} (content not available)"
            
            # Enhanced review with detailed evaluation
            review_prompt = f"""
            You are a peer reviewer for a scientific journal. Review this research paper thoroughly.
            
            PAPER TO REVIEW:
            {paper_content[:4000]}  # Truncate to fit in context
            
            Provide a comprehensive review that includes:
            1. Overall assessment and recommendation (Accept/Minor Revision/Major Revision/Reject)
            2. Strengths of the work
            3. Major concerns and weaknesses
            4. Minor issues and suggestions
            5. Technical soundness evaluation
            6. Novelty and significance assessment
            7. Writing quality and clarity
            8. Specific recommendations for improvement
            
            Be constructive but critical, following standard peer review practices.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert peer reviewer with extensive experience in scientific publishing. Provide thorough, fair, and constructive reviews."},
                {"role": "user", "content": review_prompt}
            ]
            
            response, _ = await self.client.get_response(
                messages=messages,
                model=stage_config.model,
                temperature=stage_config.temperature,
                max_tokens=stage_config.max_tokens,
                stage="review"  # Add stage for cost tracking
            )
            
            # Save review results
            review_file = results_dir / "peer_review.md"
            with open(review_file, 'w') as f:
                f.write(f"# Peer Review for {idea_name}\n\n")
                f.write(f"**Reviewer Model:** {stage_config.model}\n\n")
                f.write(response)
            
            return {
                "status": "completed", 
                "model_used": stage_config.model,
                "output_file": str(review_file),
                "paper_reviewed": str(paper_file)
            }
            
        except Exception as e:
            self.logger.error(f"Review stage failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _convert_to_latex(self, markdown_content: str) -> str:
        """
        Simple conversion from markdown to LaTeX (placeholder implementation).
        
        Args:
            markdown_content: Markdown content to convert
            
        Returns:
            LaTeX formatted content
        """
        # This is a very basic conversion - a full implementation would use proper tools
        latex_content = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath,amsfonts,amssymb}
\\usepackage{graphicx}
\\usepackage[margin=1in]{geometry}

\\title{Research Paper}
\\author{AI-Scientist-v2}
\\date{\\today}

\\begin{document}
\\maketitle

"""
        # Convert headers
        content = markdown_content.replace("# ", "\\section{").replace("\n", "}\n")
        content = content.replace("## ", "\\subsection{").replace("}\n", "}\n")
        content = content.replace("### ", "\\subsubsection{").replace("}\n", "}\n")
        
        latex_content += content + "\n\n\\end{document}"
        return latex_content
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of pipeline execution.
        
        Returns:
            Pipeline summary
        """
        summary = {
            "total_stages": len(self.results.get("stages", {})),
            "successful_stages": 0,
            "failed_stages": 0,
            "models_used": set(),
            "features_used": []
        }
        
        # Analyze stage results
        for stage_name, stage_result in self.results.get("stages", {}).items():
            if stage_result.get("status") == "completed":
                summary["successful_stages"] += 1
            else:
                summary["failed_stages"] += 1
            
            if "model_used" in stage_result:
                summary["models_used"].add(stage_result["model_used"])
            
            # Track feature usage
            if stage_result.get("rag_context_used"):
                summary["features_used"].append("RAG Context")
            if stage_result.get("reasoning_enhanced"):
                summary["features_used"].append("Reasoning Enhancement")
            if stage_result.get("latex_generated"):
                summary["features_used"].append("LaTeX Generation")
        
        summary["models_used"] = list(summary["models_used"])
        summary["features_used"] = list(set(summary["features_used"]))
        
        return summary

async def run_enhanced_pipeline(config: OpenRouterConfig, rag_system: Optional[RAGSystem], 
                               idea_name: str, results_dir: str = "./results") -> Dict[str, Any]:
    """
    Convenience function to run the enhanced pipeline.
    
    Args:
        config: OpenRouter configuration
        rag_system: Optional RAG system
        idea_name: Research idea name
        results_dir: Results directory
        
    Returns:
        Pipeline results
    """
    launcher = SimpleEnhancedLauncher(config, rag_system)
    return await launcher.run_complete_pipeline(idea_name, results_dir)