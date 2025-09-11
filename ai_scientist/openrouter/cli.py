"""
Interactive CLI Configuration System

User-friendly command-line interface for configuring OpenRouter integration
with AI-Scientist-v2. Supports model selection, pipeline configuration,
and advanced feature setup.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import argparse

# Rich imports for enhanced CLI (optional)
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .config import (
    OpenRouterConfig, StageConfig, RAGConfig, PipelineStage,
    create_default_config, load_config, save_config, validate_config,
    create_config_from_template, get_available_models
)
from .client import initialize_openrouter, get_global_client

logger = logging.getLogger(__name__)

class CLIInterface:
    """Command-line interface for OpenRouter configuration"""
    
    def __init__(self):
        """Initialize CLI interface"""
        if RICH_AVAILABLE:
            self.console = Console()
            self.use_rich = True
        else:
            self.console = None
            self.use_rich = False
        
        self.config: Optional[OpenRouterConfig] = None

    def print(self, message: str, style: Optional[str] = None):
        """Print message with optional styling"""
        if self.use_rich and style:
            self.console.print(message, style=style)
        elif self.use_rich:
            self.console.print(message)
        else:
            print(message)

    def input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if self.use_rich:
            return Prompt.ask(prompt, default=default)
        else:
            display_prompt = f"{prompt}"
            if default:
                display_prompt += f" [{default}]"
            display_prompt += ": "
            
            response = input(display_prompt).strip()
            return response if response else (default or "")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Get user confirmation"""
        if self.use_rich:
            return Confirm.ask(prompt, default=default)
        else:
            default_str = "Y/n" if default else "y/N"
            response = input(f"{prompt} [{default_str}]: ").strip().lower()
            
            if not response:
                return default
            return response.startswith('y')

    def int_input(self, prompt: str, default: Optional[int] = None, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """Get integer input with validation"""
        if self.use_rich:
            return IntPrompt.ask(prompt, default=default)
        else:
            while True:
                try:
                    display_prompt = f"{prompt}"
                    if default is not None:
                        display_prompt += f" [{default}]"
                    display_prompt += ": "
                    
                    response = input(display_prompt).strip()
                    if not response and default is not None:
                        value = default
                    else:
                        value = int(response)
                    
                    if min_val is not None and value < min_val:
                        self.print(f"Value must be at least {min_val}")
                        continue
                    if max_val is not None and value > max_val:
                        self.print(f"Value must be at most {max_val}")
                        continue
                    
                    return value
                except ValueError:
                    self.print("Please enter a valid integer")

    def float_input(self, prompt: str, default: Optional[float] = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Get float input with validation"""
        if self.use_rich:
            return FloatPrompt.ask(prompt, default=default)
        else:
            while True:
                try:
                    display_prompt = f"{prompt}"
                    if default is not None:
                        display_prompt += f" [{default}]"
                    display_prompt += ": "
                    
                    response = input(display_prompt).strip()
                    if not response and default is not None:
                        value = default
                    else:
                        value = float(response)
                    
                    if min_val is not None and value < min_val:
                        self.print(f"Value must be at least {min_val}")
                        continue
                    if max_val is not None and value > max_val:
                        self.print(f"Value must be at most {max_val}")
                        continue
                    
                    return value
                except ValueError:
                    self.print("Please enter a valid number")

    def select_from_list(self, prompt: str, options: List[str], default_index: int = 0) -> Tuple[int, str]:
        """Select from a list of options"""
        self.print(f"\n{prompt}")
        for i, option in enumerate(options):
            marker = ">" if i == default_index else " "
            self.print(f"{marker} {i + 1}. {option}")
        
        while True:
            try:
                choice = self.int_input("Select option", default=default_index + 1, min_val=1, max_val=len(options))
                return choice - 1, options[choice - 1]
            except (ValueError, IndexError):
                self.print("Invalid selection")

    def show_table(self, title: str, data: List[Dict[str, str]]):
        """Display tabular data"""
        if self.use_rich:
            table = Table(title=title)
            if data:
                # Add columns
                for key in data[0].keys():
                    table.add_column(key.replace('_', ' ').title())
                
                # Add rows
                for row in data:
                    table.add_row(*row.values())
                
                self.console.print(table)
        else:
            self.print(f"\n{title}")
            self.print("=" * len(title))
            for item in data:
                for key, value in item.items():
                    self.print(f"{key}: {value}")
                self.print()

    async def run_setup_wizard(self) -> OpenRouterConfig:
        """Run the interactive setup wizard"""
        
        self.print("=" * 60, "bold cyan" if self.use_rich else None)
        self.print("    AI-Scientist-v2 OpenRouter Configuration Wizard", "bold cyan" if self.use_rich else None)
        self.print("=" * 60, "bold cyan" if self.use_rich else None)
        
        # Check for existing configuration
        config_path = os.path.expanduser("~/.ai_scientist/openrouter_config.yaml")
        if Path(config_path).exists():
            if self.confirm("Existing configuration found. Load it?"):
                self.config = load_config(config_path)
            else:
                self.config = create_default_config()
        else:
            self.config = create_default_config()
        
        # Step 1: API Key Setup
        await self._setup_api_key()
        
        # Step 2: Pipeline Selection
        await self._setup_pipeline_selection()
        
        # Step 3: Model Configuration
        await self._setup_models()
        
        # Step 4: Advanced Features
        await self._setup_advanced_features()
        
        # Step 5: RAG Configuration
        await self._setup_rag_system()
        
        # Step 6: Validation and Save
        await self._validate_and_save()
        
        return self.config

    async def _setup_api_key(self):
        """Setup OpenRouter API key"""
        self.print("\n📋 Step 1: API Key Configuration", "bold yellow" if self.use_rich else None)
        
        current_key = os.getenv("OPENROUTER_API_KEY")
        if current_key:
            self.print(f"✓ Found API key in environment: {current_key[:12]}...")
            if not self.confirm("Use this API key?"):
                current_key = None
        
        if not current_key:
            api_key = self.input("Enter your OpenRouter API key")
            if api_key:
                self.config.api_key = api_key
                self.print("💡 Consider setting OPENROUTER_API_KEY environment variable for security")
        
        # Test API key
        if self.confirm("Test API connection?"):
            await self._test_api_connection()

    async def _test_api_connection(self):
        """Test OpenRouter API connection"""
        try:
            if self.use_rich:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                ) as progress:
                    progress.add_task(description="Testing API connection...", total=None)
                    
                    client = initialize_openrouter(self.config.api_key)
                    models = await client.get_available_models()
            else:
                self.print("Testing API connection...")
                client = initialize_openrouter(self.config.api_key)
                models = await client.get_available_models()
            
            self.print(f"✅ Success! Found {len(models)} available models", "bold green" if self.use_rich else None)
            
        except Exception as e:
            self.print(f"❌ Connection failed: {e}", "bold red" if self.use_rich else None)
            if not self.confirm("Continue anyway?"):
                sys.exit(1)

    async def _setup_pipeline_selection(self):
        """Setup pipeline selection preferences"""
        self.print("\n🔧 Step 2: Pipeline Configuration", "bold yellow" if self.use_rich else None)
        
        pipeline_options = [
            "Enhanced Pipeline (Recommended) - Full feature set with theory evolution",
            "Original Pipeline - Use existing AI-Scientist-v2 workflow", 
            "Hybrid - Let me choose per execution"
        ]
        
        choice, _ = self.select_from_list("Select default pipeline mode:", pipeline_options, 0)
        
        if choice == 0:
            self.config.use_enhanced_pipeline = True
            self.config.use_original_pipeline = False
        elif choice == 1:
            self.config.use_enhanced_pipeline = False
            self.config.use_original_pipeline = True
        else:
            self.config.use_enhanced_pipeline = True
            self.config.use_original_pipeline = True

    async def _setup_models(self):
        """Setup model configurations for each stage"""
        self.print("\n🤖 Step 3: Model Configuration", "bold yellow" if self.use_rich else None)
        
        # Option to use template
        if self.confirm("Use a configuration template?"):
            await self._select_template()
        
        # Get available models
        try:
            client = get_global_client()
            available_models = await client.get_available_models()
            model_names = [model["id"] for model in available_models]
        except Exception as e:
            self.print(f"Warning: Could not fetch models ({e}). Using predefined list.")
            model_names = [
                "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1-preview", "openai/o1-mini",
                "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku", "anthropic/claude-3-opus",
                "google/gemini-2.0-flash", "google/gemini-1.5-pro", 
                "deepseek/deepseek-v3", "x-ai/grok-2", "meta-llama/llama-3.1-405b"
            ]
        
        # Configure each stage
        if self.confirm("Configure models for each pipeline stage?"):
            for stage_name in ["ideation", "experiment", "analysis", "writeup", "review"]:
                await self._configure_stage(stage_name, model_names)

    async def _select_template(self):
        """Select configuration template"""
        templates = ["research", "cost_optimized", "high_quality", "experimental"]
        template_descriptions = [
            "Balanced configuration for research work",
            "Cost-optimized models with caching enabled",
            "High-quality models for best results",
            "Experimental models for cutting-edge features"
        ]
        
        self.print("\nAvailable templates:")
        for i, (template, desc) in enumerate(zip(templates, template_descriptions)):
            self.print(f"  {i + 1}. {template}: {desc}")
        
        choice, template = self.select_from_list("Select template:", templates, 0)
        
        template_config = create_config_from_template(template)
        self.config.stage_configs = template_config.stage_configs
        
        self.print(f"✅ Applied {template} template", "bold green" if self.use_rich else None)

    async def _configure_stage(self, stage_name: str, available_models: List[str]):
        """Configure a specific pipeline stage"""
        self.print(f"\n⚙️  Configuring {stage_name} stage", "bold blue" if self.use_rich else None)
        
        current_config = self.config.stage_configs.get(stage_name, StageConfig())
        
        # Model selection with enhanced interface
        if self.confirm(f"Configure model for {stage_name}? (current: {current_config.model})"):
            self.print(f"Current model: {current_config.model}")
            
            # Show recommended models for this stage
            recommended = self._get_recommended_models(stage_name)
            if recommended:
                self.print("\nRecommended models for this stage:")
                for i, model in enumerate(recommended[:10], 1):
                    marker = ">" if model == current_config.model else " "
                    self.print(f"{marker} {i}. {model}")
                
                # Quick selection option
                if self.confirm("Choose from recommended models?"):
                    choice, selected_model = self.select_from_list("Select recommended model:", recommended[:10])
                    current_config.model = selected_model
                else:
                    model = self.input("Enter custom model name", current_config.model)
                    if model in available_models or self.confirm(f"Model {model} not found in available models. Use anyway?"):
                        current_config.model = model
            else:
                model = self.input("Enter model name", current_config.model)
                if model in available_models or self.confirm(f"Model {model} not found in available models. Use anyway?"):
                    current_config.model = model
        
        # Advanced configuration
        if self.confirm("Configure advanced settings?"):
            # Temperature
            temp = self.float_input("Temperature (0.0-2.0)", current_config.temperature, 0.0, 2.0)
            current_config.temperature = temp
            
            # Max tokens
            tokens = self.int_input("Max tokens", current_config.max_tokens, 1, 128000)
            current_config.max_tokens = tokens
            
            # Caching strategy
            cache_options = ["auto", "none", "ephemeral", "explicit"]
            cache_descriptions = [
                "Automatic caching (recommended for most models)",
                "No caching",
                "Ephemeral caching (for Anthropic/Gemini)",
                "Explicit cache control"
            ]
            
            self.print("\nCaching strategies:")
            for i, (option, desc) in enumerate(zip(cache_options, cache_descriptions)):
                marker = ">" if option == current_config.caching else " "
                self.print(f"{marker} {i + 1}. {option}: {desc}")
            
            choice, cache_strategy = self.select_from_list("Select caching strategy:", cache_options)
            current_config.caching = cache_strategy
            
            # Fallback models
            if self.confirm("Configure fallback models?"):
                fallback_models = []
                while len(fallback_models) < 3:
                    fallback = self.input(f"Fallback model {len(fallback_models) + 1} (or press Enter to skip)", "")
                    if not fallback:
                        break
                    fallback_models.append(fallback)
                
                if fallback_models:
                    current_config.fallback_models = fallback_models
        
        # Tool configuration for applicable stages
        if stage_name in ["ideation", "experiment", "writeup", "review"]:
            if self.confirm("Configure tools for this stage?"):
                available_tools = self._get_available_tools(stage_name)
                if available_tools:
                    self.print(f"Available tools for {stage_name}:")
                    selected_tools = []
                    
                    for tool in available_tools:
                        if self.confirm(f"Enable {tool}?", tool in current_config.tools):
                            selected_tools.append(tool)
                    
                    current_config.tools = selected_tools
        
        self.config.stage_configs[stage_name] = current_config

    def _get_recommended_models(self, stage_name: str) -> List[str]:
        """Get recommended models for a stage"""
        recommendations = {
            "ideation": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "x-ai/grok-2", "google/gemini-2.0-flash", "deepseek/deepseek-v3"],
            "experiment": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-v3", "openai/gpt-4o-mini", "google/gemini-1.5-pro"],
            "analysis": ["openai/o1-preview", "openai/o1", "google/gemini-2.0-flash", "anthropic/claude-3.5-sonnet", "openai/o1-mini"],
            "writeup": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro", "deepseek/deepseek-v3"],
            "review": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "openai/o1-preview", "google/gemini-2.0-flash", "anthropic/claude-3-opus"]
        }
        return recommendations.get(stage_name, [])
    
    def _get_available_tools(self, stage_name: str) -> List[str]:
        """Get available tools for a stage"""
        tools_by_stage = {
            "ideation": [
                "semantic_scholar", "arxiv_search", "literature_review", 
                "research_trends", "domain_analysis", "novelty_check"
            ],
            "experiment": [
                "code_execution", "data_analysis", "statistical_tests",
                "visualization", "experiment_design", "parameter_tuning"
            ],
            "writeup": [
                "latex_formatting", "citation_search", "grammar_check",
                "scientific_writing", "figure_generation", "reference_manager"
            ],
            "review": [
                "plagiarism_check", "quality_assessment", "methodology_review",
                "statistical_validation", "reproducibility_check", "ethics_review"
            ]
        }
        return tools_by_stage.get(stage_name, [])

    async def _setup_advanced_features(self):
        """Setup advanced OpenRouter features"""
        self.print("\n🚀 Step 4: Advanced Features", "bold yellow" if self.use_rich else None)
        
        # Streaming responses
        if self.confirm("Enable streaming responses?", self.config.enable_streaming):
            self.config.enable_streaming = True
            self.print("  ✓ Streaming will provide real-time response generation")
        
        # Parallel processing
        if self.confirm("Enable parallel processing?", self.config.enable_parallel_processing):
            self.config.enable_parallel_processing = True
            max_concurrent = self.int_input("Max concurrent requests", self.config.max_concurrent_requests, 1, 20)
            self.config.max_concurrent_requests = max_concurrent
            self.print(f"  ✓ Will process up to {max_concurrent} requests simultaneously")
        
        # Cost optimization and budgets
        if self.confirm("Enable cost optimization features?", self.config.enable_cost_optimization):
            self.config.enable_cost_optimization = True
            
            # Budget controls
            if self.confirm("Set budget limits?"):
                if self.confirm("Set maximum cost per request?"):
                    max_cost = self.float_input("Max cost per request ($)", min_val=0.01)
                    self.config.max_cost_per_request = max_cost
                
                if self.confirm("Enable budget alerts?", self.config.budget_alerts):
                    self.config.budget_alerts = True
            
            self.print("  ✓ Cost optimization and monitoring enabled")
        
        # Retry and error handling
        if self.confirm("Configure retry behavior?"):
            retry_attempts = self.int_input("Number of retry attempts", self.config.retry_attempts, 1, 10)
            self.config.retry_attempts = retry_attempts
            self.print(f"  ✓ Will retry failed requests up to {retry_attempts} times")
        
        # Global prompt customization
        if self.confirm("Configure global prompt enhancements?"):
            use_system_prompts = self.confirm("Use enhanced system prompts for better results?")
            if use_system_prompts:
                # This could be expanded to configure custom system prompts per stage
                self.print("  ✓ Enhanced system prompts will be used")
        
        # Provider preferences
        if self.confirm("Configure provider preferences?"):
            providers = ["openai", "anthropic", "google", "deepseek", "x-ai", "meta-llama"]
            self.print("Available providers:")
            for i, provider in enumerate(providers, 1):
                self.print(f"  {i}. {provider}")
            
            preferred_providers = []
            while len(preferred_providers) < 3:
                choice = self.input(f"Preferred provider {len(preferred_providers) + 1} (or press Enter to skip)", "")
                if not choice:
                    break
                if choice in providers:
                    preferred_providers.append(choice)
                elif choice.isdigit() and 1 <= int(choice) <= len(providers):
                    preferred_providers.append(providers[int(choice) - 1])
            
            if preferred_providers:
                self.print(f"  ✓ Provider preference order: {' → '.join(preferred_providers)}")

    async def _setup_rag_system(self):
        """Setup RAG document ingestion system"""
        self.print("\n📚 Step 5: RAG Document System", "bold yellow" if self.use_rich else None)
        
        if self.confirm("Enable RAG document ingestion?", self.config.rag_config.enabled):
            self.config.rag_config.enabled = True
            
            # Document ingestion settings
            if self.confirm("Configure document settings?"):
                chunk_size = self.int_input("Chunk size", self.config.rag_config.chunk_size, 100, 2000)
                self.config.rag_config.chunk_size = chunk_size
                
                chunk_overlap = self.int_input("Chunk overlap", self.config.rag_config.chunk_overlap, 0, chunk_size // 2)
                self.config.rag_config.chunk_overlap = chunk_overlap
                
                threshold = self.float_input("Similarity threshold", self.config.rag_config.similarity_threshold, 0.0, 1.0)
                self.config.rag_config.similarity_threshold = threshold
            
            # Auto-ingestion
            if self.confirm("Enable automatic document discovery?", self.config.rag_config.auto_ingest):
                self.config.rag_config.auto_ingest = True

    async def _validate_and_save(self):
        """Validate configuration and save"""
        self.print("\n✅ Step 6: Validation and Save", "bold yellow" if self.use_rich else None)
        
        # Validate configuration
        errors = validate_config(self.config)
        if errors:
            self.print("❌ Configuration validation errors:", "bold red" if self.use_rich else None)
            for error in errors:
                self.print(f"  • {error}")
            
            if not self.confirm("Continue with invalid configuration?"):
                return
        else:
            self.print("✅ Configuration is valid!", "bold green" if self.use_rich else None)
        
        # Show summary
        self._show_configuration_summary()
        
        # Save configuration
        if self.confirm("Save configuration?"):
            config_path = self.input("Configuration file path", "~/.ai_scientist/openrouter_config.yaml")
            config_path = os.path.expanduser(config_path)
            
            save_config(self.config, config_path)
            self.print(f"✅ Configuration saved to {config_path}", "bold green" if self.use_rich else None)
            
            # Set environment variable suggestion
            if self.config.api_key and not os.getenv("OPENROUTER_API_KEY"):
                self.print("\n💡 Recommendation: Set environment variable for security:")
                self.print(f"export OPENROUTER_API_KEY='{self.config.api_key}'")

    def _show_configuration_summary(self):
        """Show configuration summary"""
        self.print("\n📋 Configuration Summary", "bold cyan" if self.use_rich else None)
        
        summary_data = []
        
        # Pipeline settings
        pipeline_mode = "Enhanced" if self.config.use_enhanced_pipeline else "Original"
        summary_data.append({"Setting": "Pipeline Mode", "Value": pipeline_mode})
        
        # Model configurations
        for stage_name, stage_config in self.config.stage_configs.items():
            summary_data.append({
                "Setting": f"{stage_name.title()} Model", 
                "Value": f"{stage_config.model} (temp: {stage_config.temperature})"
            })
        
        # Features
        features = []
        if self.config.enable_streaming:
            features.append("Streaming")
        if self.config.enable_parallel_processing:
            features.append(f"Parallel ({self.config.max_concurrent_requests})")
        if self.config.enable_cost_optimization:
            features.append("Cost Optimization")
        if self.config.rag_config.enabled:
            features.append("RAG System")
        
        summary_data.append({"Setting": "Enabled Features", "Value": ", ".join(features) or "None"})
        
        self.show_table("Configuration Summary", summary_data)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="OpenRouter Configuration CLI")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("--validate", type=str, help="Validate configuration file")
    parser.add_argument("--template", type=str, help="Create config from template")
    parser.add_argument("--models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    cli = CLIInterface()
    
    try:
        if args.setup:
            config = asyncio.run(cli.run_setup_wizard())
            cli.print("✅ Setup completed successfully!", "bold green" if cli.use_rich else None)
        
        elif args.validate:
            config = load_config(args.validate)
            errors = validate_config(config)
            if errors:
                cli.print("❌ Configuration errors:", "bold red" if cli.use_rich else None)
                for error in errors:
                    cli.print(f"  • {error}")
                sys.exit(1)
            else:
                cli.print("✅ Configuration is valid", "bold green" if cli.use_rich else None)
        
        elif args.template:
            try:
                config = create_config_from_template(args.template)
                save_config(config, f"{args.template}_config.yaml")
                cli.print(f"✅ Created {args.template} configuration", "bold green" if cli.use_rich else None)
            except ValueError as e:
                cli.print(f"❌ {e}", "bold red" if cli.use_rich else None)
                sys.exit(1)
        
        elif args.models:
            async def list_models():
                try:
                    client = get_global_client()
                    models = await client.get_available_models()
                    
                    model_data = []
                    for model in models[:20]:  # Show first 20
                        model_data.append({
                            "Model": model["id"],
                            "Provider": model.get("owned_by", "Unknown"),
                            "Context": str(model.get("context_length", "Unknown")),
                            "Tools": "Yes" if "tools" in model.get("supported_parameters", []) else "No"
                        })
                    
                    cli.show_table(f"Available Models (showing {len(model_data)} of {len(models)})", model_data)
                    
                except Exception as e:
                    cli.print(f"❌ Failed to fetch models: {e}", "bold red" if cli.use_rich else None)
            
            asyncio.run(list_models())
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        cli.print("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        cli.print(f"\n❌ Error: {e}", "bold red" if cli.use_rich else None)
        sys.exit(1)

if __name__ == "__main__":
    main()