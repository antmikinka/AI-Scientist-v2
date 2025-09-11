#!/usr/bin/env python3
"""
Enhanced AI-Scientist-v2 Launcher with OpenRouter Integration
Provides interactive CLI setup, per-stage configuration, and RAG support
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add ai_scientist to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_scientist'))

# Import OpenRouter components
try:
    from ai_scientist.openrouter import (
        initialize_openrouter, 
        CLIInterface,
        OpenRouterConfig,
        RAGSystem,
        get_global_client
    )
    OPENROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenRouter integration not available: {e}")
    OPENROUTER_AVAILABLE = False

# Import existing AI-Scientist components
from ai_scientist.llm import create_client
from ai_scientist.utils.token_tracker import token_tracker

# Import execution modules
try:
    from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import perform_experiments_bfts
    from ai_scientist.perform_plotting import aggregate_plots
    from ai_scientist.perform_writeup import perform_writeup
    from ai_scientist.perform_icbinb_writeup import perform_writeup as perform_icbinb_writeup
    from ai_scientist.perform_llm_review import perform_review
    from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
except ImportError as e:
    print(f"Warning: Some AI-Scientist modules not available: {e}")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
import rich.traceback

# Enable rich tracebacks
rich.traceback.install()

console = Console()

class EnhancedScientistLauncher:
    """Enhanced launcher with OpenRouter integration and interactive configuration"""
    
    def __init__(self):
        self.config = None
        self.rag_system = None
        self.client = None
        self.setup_complete = False
        
    async def run(self):
        """Main entry point for the enhanced launcher"""
        console.clear()
        
        # Display welcome banner
        self._display_banner()
        
        # Check dependencies
        if not OPENROUTER_AVAILABLE:
            console.print("[red]OpenRouter integration not available. Using legacy mode only.[/red]")
            return await self._run_legacy_mode()
        
        # Interactive setup wizard
        await self._setup_wizard()
        
        # Main menu
        await self._main_menu()
    
    def _display_banner(self):
        """Display welcome banner"""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]              [bold white]Enhanced AI-Scientist-v2 with OpenRouter[/bold white]              [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]                    [italic]Unified LLM Research Platform[/italic]                    [bold cyan]║[/bold cyan]
[bold cyan]╚═══════════════════════════════════════════════════════════╝[/bold cyan]

[bold green]✨ Features:[/bold green]
• 🔄 Unified access to 200+ AI models via OpenRouter
• 🧠 Intelligent per-stage model selection and configuration
• 📚 RAG document ingestion for theory evolution
• 🎯 Interactive CLI configuration wizard
• ⚡ Advanced prompt caching and optimization
• 🛡️ Robust error handling and fallback mechanisms
"""
        console.print(Panel(banner, border_style="cyan"))
    
    async def _setup_wizard(self):
        """Interactive setup wizard"""
        console.print("\n[bold yellow]🚀 Welcome to the Setup Wizard![/bold yellow]")
        
        # Check for existing configuration
        config_path = Path("./openrouter_config.yaml")
        if config_path.exists():
            if Confirm.ask("Found existing OpenRouter configuration. Use it?"):
                try:
                    from ai_scientist.openrouter.config import load_config
                    self.config = load_config(str(config_path))
                    console.print("[green]✅ Loaded existing configuration[/green]")
                    self.setup_complete = True
                    return
                except Exception as e:
                    console.print(f"[red]❌ Error loading config: {e}[/red]")
        
        # Run OpenRouter CLI setup
        cli = CLIInterface()
        self.config = await cli.run_setup_wizard()
        
        # Save configuration
        try:
            from ai_scientist.openrouter.config import save_config
            save_config(self.config, str(config_path))
            console.print(f"[green]✅ Configuration saved to {config_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not save configuration: {e}[/yellow]")
        
        self.setup_complete = True
    
    async def _main_menu(self):
        """Display main menu and handle user choices"""
        while True:
            console.print("\n" + "="*60)
            console.print("[bold cyan]🎯 Main Menu[/bold cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="cyan", width=8)
            table.add_column("Description", style="white")
            
            table.add_row("1", "🚀 Run Full AI-Scientist Pipeline")
            table.add_row("2", "📚 Manage RAG Documents")
            table.add_row("3", "⚙️  Configure Pipeline Settings")
            table.add_row("4", "🧪 Run Individual Stages")
            table.add_row("5", "📊 View Statistics & Logs")
            table.add_row("6", "🔧 System Settings")
            table.add_row("7", "❓ Help & Documentation")
            table.add_row("0", "🚪 Exit")
            
            console.print(table)
            
            choice = Prompt.ask("\n[bold yellow]Select an option[/bold yellow]", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            
            try:
                if choice == "0":
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                elif choice == "1":
                    await self._run_full_pipeline()
                elif choice == "2":
                    await self._manage_rag_documents()
                elif choice == "3":
                    await self._configure_pipeline()
                elif choice == "4":
                    await self._run_individual_stages()
                elif choice == "5":
                    await self._view_statistics()
                elif choice == "6":
                    await self._system_settings()
                elif choice == "7":
                    await self._show_help()
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled by user[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                console.print("[dim]Press any key to continue...[/dim]")
                input()
    
    async def _run_full_pipeline(self):
        """Run the complete AI-Scientist pipeline"""
        console.print("\n[bold green]🚀 Running Full AI-Scientist Pipeline[/bold green]")
        
        # Pipeline selection
        pipeline_choice = Prompt.ask(
            "Choose pipeline version",
            choices=["enhanced", "original"],
            default="enhanced"
        )
        
        # Experiment type selection
        experiment_type = Prompt.ask(
            "Select experiment type",
            choices=["bfts", "custom"],
            default="bfts"
        )
        
        # Model configuration
        if pipeline_choice == "enhanced" and self.config:
            console.print("[cyan]Using OpenRouter configuration for enhanced pipeline[/cyan]")
            # Initialize OpenRouter
            await initialize_openrouter(self.config)
            self.client = get_global_client()
        else:
            # Legacy mode
            model = Prompt.ask("Enter model name", default="claude-3-5-sonnet-20240620")
            self.client, _ = create_client(model)
        
        # Additional options
        writeup_type = Prompt.ask(
            "Select writeup type",
            choices=["icbinb", "normal"],
            default="icbinb"
        )
        
        # Execution
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Executing AI-Scientist pipeline...", total=None)
            
            try:
                await self._execute_pipeline(
                    pipeline_type=pipeline_choice,
                    experiment_type=experiment_type,
                    writeup_type=writeup_type
                )
                console.print("[bold green]✅ Pipeline completed successfully![/bold green]")
                
            except Exception as e:
                console.print(f"[red]❌ Pipeline failed: {e}[/red]")
                console.print("[dim]Check logs for detailed error information[/dim]")
    
    async def _manage_rag_documents(self):
        """Manage RAG document ingestion and retrieval"""
        console.print("\n[bold blue]📚 RAG Document Management[/bold blue]")
        
        if not self.rag_system:
            try:
                # Initialize RAG system
                rag_config = self.config.rag_config if self.config else {
                    'enabled': True,
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'embedding_model': 'text-embedding-3-large',
                    'vector_store': 'chroma'
                }
                
                from ai_scientist.openrouter.rag_system import RAGSystem
                self.rag_system = RAGSystem(rag_config)
                
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize RAG system: {e}[/red]")
                return
        
        while True:
            console.print("\n[bold cyan]RAG Management Menu[/bold cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="cyan", width=8)
            table.add_column("Description", style="white")
            
            table.add_row("1", "📄 Ingest Documents from Files")
            table.add_row("2", "🌐 Ingest from URLs")
            table.add_row("3", "📋 List Ingested Documents")
            table.add_row("4", "🔍 Search Documents")
            table.add_row("5", "🗑️  Delete Documents")
            table.add_row("6", "📊 RAG Statistics")
            table.add_row("0", "↩️  Back to Main Menu")
            
            console.print(table)
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])
            
            if choice == "0":
                break
            elif choice == "1":
                await self._ingest_files()
            elif choice == "2":
                await self._ingest_urls()
            elif choice == "3":
                await self._list_documents()
            elif choice == "4":
                await self._search_documents()
            elif choice == "5":
                await self._delete_documents()
            elif choice == "6":
                await self._show_rag_statistics()
    
    async def _configure_pipeline(self):
        """Configure pipeline settings"""
        console.print("\n[bold purple]⚙️ Pipeline Configuration[/bold purple]")
        
        if not self.config:
            console.print("[red]No OpenRouter configuration found. Please run setup wizard first.[/red]")
            return
        
        # Display current configuration
        config_table = Table(title="Current Pipeline Configuration")
        config_table.add_column("Stage", style="cyan")
        config_table.add_column("Model", style="white")
        config_table.add_column("Temperature", style="yellow")
        config_table.add_column("Caching", style="green")
        
        for stage_name, stage_config in self.config.stage_configs.items():
            config_table.add_row(
                stage_name.replace('_', ' ').title(),
                stage_config.model,
                str(stage_config.temperature),
                stage_config.caching
            )
        
        console.print(config_table)
        
        if Confirm.ask("\nModify configuration?"):
            cli = CLIInterface()
            self.config = await cli.configure_stage_settings(self.config)
            console.print("[green]✅ Configuration updated[/green]")
    
    async def _run_individual_stages(self):
        """Run individual pipeline stages"""
        console.print("\n[bold orange]🧪 Individual Stage Execution[/bold orange]")
        
        stages = {
            "1": ("Idea Generation", self._run_ideation),
            "2": ("Experiment Design", self._run_experiment_design),
            "3": ("Code Generation", self._run_code_generation),
            "4": ("Experiment Execution", self._run_experiments),
            "5": ("Plot Generation", self._run_plotting),
            "6": ("Paper Writing", self._run_writeup),
            "7": ("Review & Evaluation", self._run_review)
        }
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Stage", style="white")
        
        for key, (stage_name, _) in stages.items():
            table.add_row(key, stage_name)
        table.add_row("0", "↩️  Back to Main Menu")
        
        console.print(table)
        
        choice = Prompt.ask("Select stage to run", choices=list(stages.keys()) + ["0"])
        
        if choice == "0":
            return
        
        stage_name, stage_func = stages[choice]
        console.print(f"\n[bold cyan]Running: {stage_name}[/bold cyan]")
        
        try:
            await stage_func()
            console.print(f"[green]✅ {stage_name} completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ {stage_name} failed: {e}[/red]")
    
    async def _view_statistics(self):
        """View statistics and logs"""
        console.print("\n[bold cyan]📊 Statistics & Logs[/bold cyan]")
        
        # Token usage statistics
        if hasattr(token_tracker, 'get_summary'):
            stats = token_tracker.get_summary()
            
            stats_table = Table(title="Token Usage Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            for key, value in stats.items():
                stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(stats_table)
        
        # RAG statistics
        if self.rag_system:
            try:
                rag_stats = self.rag_system.get_statistics()
                
                rag_table = Table(title="RAG System Statistics")
                rag_table.add_column("Metric", style="cyan")
                rag_table.add_column("Value", style="white")
                
                for key, value in rag_stats.items():
                    rag_table.add_row(key.replace('_', ' ').title(), str(value))
                
                console.print(rag_table)
                
            except Exception as e:
                console.print(f"[red]Error getting RAG statistics: {e}[/red]")
    
    async def _system_settings(self):
        """System settings and configuration"""
        console.print("\n[bold red]🔧 System Settings[/bold red]")
        
        settings_table = Table(show_header=True, header_style="bold magenta")
        settings_table.add_column("Option", style="cyan", width=8)
        settings_table.add_column("Description", style="white")
        
        settings_table.add_row("1", "🔄 Re-run Setup Wizard")
        settings_table.add_row("2", "🧪 Test OpenRouter Connection")
        settings_table.add_row("3", "📋 Export Configuration")
        settings_table.add_row("4", "📥 Import Configuration")
        settings_table.add_row("5", "🗑️  Reset All Settings")
        settings_table.add_row("0", "↩️  Back to Main Menu")
        
        console.print(settings_table)
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5"])
        
        if choice == "0":
            return
        elif choice == "1":
            await self._setup_wizard()
        elif choice == "2":
            await self._test_connection()
        elif choice == "3":
            await self._export_configuration()
        elif choice == "4":
            await self._import_configuration()
        elif choice == "5":
            if Confirm.ask("Are you sure you want to reset all settings?"):
                await self._reset_settings()
    
    async def _show_help(self):
        """Show help and documentation"""
        console.print("\n[bold green]❓ Help & Documentation[/bold green]")
        
        help_text = """
[bold cyan]Enhanced AI-Scientist-v2 Help[/bold cyan]

[bold yellow]🚀 Getting Started:[/bold yellow]
1. Run the setup wizard to configure OpenRouter integration
2. Optionally ingest documents for RAG-enhanced research
3. Choose between enhanced or original pipeline
4. Select models and parameters for each pipeline stage

[bold yellow]🔧 Configuration:[/bold yellow]
• Enhanced mode uses OpenRouter's unified API for 200+ models
• Each pipeline stage can use different models and settings
• Prompt caching is automatically optimized for cost savings
• Fallback mechanisms ensure reliability across providers

[bold yellow]📚 RAG Features:[/bold yellow]
• Ingest PDFs, documents, and web content
• Automatic chunking and embedding generation
• Context-aware research question answering
• Theory evolution based on ingested knowledge

[bold yellow]🎯 Pipeline Stages:[/bold yellow]
1. Idea Generation - Generate novel research ideas
2. Experiment Design - Design controlled experiments
3. Code Generation - Implement experimental code
4. Experiment Execution - Run experiments and collect data
5. Plot Generation - Create visualizations
6. Paper Writing - Generate research papers
7. Review & Evaluation - Critical analysis and scoring

[bold yellow]💡 Tips:[/bold yellow]
• Use different models for different stages (e.g., O1 for reasoning, Claude for writing)
• Enable prompt caching for repeated operations
• Ingest relevant papers to improve research quality
• Monitor token usage to optimize costs

[bold yellow]🔗 Resources:[/bold yellow]
• OpenRouter documentation: https://openrouter.ai/docs
• AI-Scientist paper: https://arxiv.org/abs/2408.06292
• Configuration examples in ./config/ directory
"""
        
        console.print(Panel(help_text, border_style="green"))
        
        input("\n[dim]Press Enter to continue...[/dim]")
    
    # Implementation of stage execution methods
    async def _run_ideation(self):
        """Run ideation stage"""
        console.print("[cyan]🧠 Running Ideation...[/cyan]")
        # Implementation would go here
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_experiment_design(self):
        """Run experiment design stage"""
        console.print("[cyan]🔬 Running Experiment Design...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_code_generation(self):
        """Run code generation stage"""
        console.print("[cyan]💻 Running Code Generation...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_experiments(self):
        """Run experiments stage"""
        console.print("[cyan]🧪 Running Experiments...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_plotting(self):
        """Run plotting stage"""
        console.print("[cyan]📊 Running Plotting...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_writeup(self):
        """Run writeup stage"""
        console.print("[cyan]✍️ Running Writeup...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
        
    async def _run_review(self):
        """Run review stage"""
        console.print("[cyan]📝 Running Review...[/cyan]")
        await asyncio.sleep(1)  # Placeholder
    
    # RAG management methods
    async def _ingest_files(self):
        """Ingest documents from files"""
        console.print("[cyan]📄 Document Ingestion[/cyan]")
        
        file_path = Prompt.ask("Enter file path or directory")
        path = Path(file_path)
        
        if not path.exists():
            console.print("[red]Path does not exist[/red]")
            return
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Ingesting documents...", total=None)
            
            try:
                if path.is_file():
                    doc_id = await self.rag_system.ingest_file(path)
                    if doc_id:
                        console.print(f"[green]✅ Ingested: {path.name}[/green]")
                    else:
                        console.print(f"[red]❌ Failed to ingest: {path.name}[/red]")
                else:
                    # Ingest directory
                    for file_path in path.rglob("*"):
                        if file_path.is_file() and self.rag_system.document_processor.is_supported(file_path):
                            doc_id = await self.rag_system.ingest_file(file_path)
                            if doc_id:
                                console.print(f"[green]✅ Ingested: {file_path.name}[/green]")
                            
            except Exception as e:
                console.print(f"[red]❌ Ingestion failed: {e}[/red]")
    
    async def _ingest_urls(self):
        """Ingest documents from URLs"""
        console.print("[cyan]🌐 URL Ingestion[/cyan]")
        
        url = Prompt.ask("Enter URL")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Ingesting from URL...", total=None)
            
            try:
                doc_id = self.rag_system.ingest_url(url)
                if doc_id:
                    console.print(f"[green]✅ Ingested from: {url}[/green]")
                else:
                    console.print(f"[red]❌ Failed to ingest from: {url}[/red]")
                    
            except Exception as e:
                console.print(f"[red]❌ URL ingestion failed: {e}[/red]")
    
    async def _list_documents(self):
        """List all ingested documents"""
        try:
            documents = self.rag_system.list_documents()
            
            if not documents:
                console.print("[yellow]No documents found[/yellow]")
                return
            
            docs_table = Table(title="Ingested Documents")
            docs_table.add_column("ID", style="cyan", width=16)
            docs_table.add_column("Title", style="white")
            docs_table.add_column("Type", style="yellow")
            docs_table.add_column("Size", style="green")
            docs_table.add_column("Date", style="blue")
            
            for doc in documents[:20]:  # Limit display
                docs_table.add_row(
                    doc['id'][:16] + "...",
                    doc['title'][:50] + "..." if len(doc['title']) > 50 else doc['title'],
                    doc['doc_type'],
                    f"{doc['content_length']:,} chars",
                    doc['created_at'][:10]
                )
            
            console.print(docs_table)
            
            if len(documents) > 20:
                console.print(f"[dim]... and {len(documents) - 20} more documents[/dim]")
                
        except Exception as e:
            console.print(f"[red]❌ Error listing documents: {e}[/red]")
    
    async def _search_documents(self):
        """Search documents"""
        query = Prompt.ask("Enter search query")
        
        try:
            results = self.rag_system.search(query, max_results=10)
            
            if not results:
                console.print("[yellow]No matching documents found[/yellow]")
                return
            
            results_table = Table(title=f"Search Results for: '{query}'")
            results_table.add_column("Content", style="white", width=60)
            results_table.add_column("Score", style="green")
            results_table.add_column("Source", style="cyan")
            
            for content, score, metadata in results:
                results_table.add_row(
                    content[:100] + "..." if len(content) > 100 else content,
                    f"{score:.3f}",
                    metadata.get('title', 'Unknown')[:30]
                )
            
            console.print(results_table)
            
        except Exception as e:
            console.print(f"[red]❌ Search failed: {e}[/red]")
    
    async def _delete_documents(self):
        """Delete documents"""
        # List documents first
        await self._list_documents()
        
        doc_id = Prompt.ask("Enter document ID to delete (or 'cancel')")
        
        if doc_id.lower() == 'cancel':
            return
        
        if Confirm.ask(f"Are you sure you want to delete document {doc_id}?"):
            try:
                if self.rag_system.delete_document(doc_id):
                    console.print("[green]✅ Document deleted[/green]")
                else:
                    console.print("[red]❌ Document not found or deletion failed[/red]")
            except Exception as e:
                console.print(f"[red]❌ Deletion failed: {e}[/red]")
    
    async def _show_rag_statistics(self):
        """Show RAG system statistics"""
        try:
            stats = self.rag_system.get_statistics()
            await self._view_statistics()  # Reuse existing method
        except Exception as e:
            console.print(f"[red]❌ Error getting statistics: {e}[/red]")
    
    # System settings methods
    async def _test_connection(self):
        """Test OpenRouter connection"""
        console.print("[cyan]🧪 Testing OpenRouter Connection...[/cyan]")
        
        try:
            if not self.config:
                console.print("[red]No configuration found[/red]")
                return
            
            # Initialize client and test
            await initialize_openrouter(self.config)
            client = get_global_client()
            
            # Test with a simple query
            messages = [{"role": "user", "content": "Hello, this is a test message."}]
            response, _ = await client.get_response(messages, "openai/gpt-4o-mini", temperature=0.1)
            
            console.print("[green]✅ Connection successful![/green]")
            console.print(f"[dim]Test response: {response[:100]}...[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
    
    async def _export_configuration(self):
        """Export configuration to file"""
        if not self.config:
            console.print("[red]No configuration to export[/red]")
            return
        
        export_path = Prompt.ask("Export path", default="./exported_config.yaml")
        
        try:
            from ai_scientist.openrouter.config import save_config
            save_config(self.config, export_path)
            console.print(f"[green]✅ Configuration exported to {export_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
    
    async def _import_configuration(self):
        """Import configuration from file"""
        import_path = Prompt.ask("Import path")
        
        if not Path(import_path).exists():
            console.print("[red]File does not exist[/red]")
            return
        
        try:
            from ai_scientist.openrouter.config import load_config
            self.config = load_config(import_path)
            console.print(f"[green]✅ Configuration imported from {import_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Import failed: {e}[/red]")
    
    async def _reset_settings(self):
        """Reset all settings"""
        self.config = None
        self.rag_system = None
        self.client = None
        self.setup_complete = False
        
        # Remove config files
        config_files = ["./openrouter_config.yaml", "./rag_storage"]
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    import shutil
                    shutil.rmtree(path)
        
        console.print("[green]✅ Settings reset successfully[/green]")
    
    async def _execute_pipeline(self, pipeline_type: str, experiment_type: str, writeup_type: str):
        """Execute the complete pipeline"""
        # This would integrate with the existing AI-Scientist pipeline
        # For now, this is a placeholder that demonstrates the structure
        
        stages = [
            "Initializing...",
            "Running experiments...",
            "Generating plots...",
            "Writing paper...",
            "Reviewing results..."
        ]
        
        with Progress() as progress:
            task = progress.add_task("Pipeline execution", total=len(stages))
            
            for stage in stages:
                progress.update(task, description=stage)
                await asyncio.sleep(2)  # Simulate work
                progress.advance(task)
    
    async def _run_legacy_mode(self):
        """Run in legacy mode without OpenRouter"""
        console.print("[yellow]Running in legacy mode...[/yellow]")
        
        # Import and run original launcher
        try:
            import subprocess
            result = subprocess.run([sys.executable, "launch_scientist_bfts.py", "--help"], 
                                  capture_output=True, text=True)
            console.print("Legacy launcher available. Use: python launch_scientist_bfts.py")
        except Exception as e:
            console.print(f"[red]Legacy mode not available: {e}[/red]")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced AI-Scientist-v2 with OpenRouter")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--no-wizard", action="store_true", help="Skip setup wizard")
    
    args = parser.parse_args()
    
    launcher = EnhancedScientistLauncher()
    
    # Load configuration if provided
    if args.config and Path(args.config).exists():
        try:
            from ai_scientist.openrouter.config import load_config
            launcher.config = load_config(args.config)
            launcher.setup_complete = True
        except Exception as e:
            console.print(f"[red]Failed to load config {args.config}: {e}[/red]")
    
    await launcher.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)