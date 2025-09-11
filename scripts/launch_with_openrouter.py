#!/usr/bin/env python3
"""
Enhanced AI-Scientist-v2 Launcher with OpenRouter Integration
Provides interactive setup and advanced features for AI research automation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the ai_scientist directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_scientist'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_openrouter_integration():
    """Initialize OpenRouter integration"""
    try:
        from ai_scientist.openrouter import (
            initialize_openrouter,
            load_config,
            save_config,
            create_default_config
        )
        
        logger.info("OpenRouter integration available")
        
        # Check for existing configuration
        config_path = os.path.expanduser("~/.ai_scientist/openrouter_config.yaml")
        existing_config = None
        
        try:
            existing_config = load_config(config_path)
            logger.info("Found existing OpenRouter configuration")
            
            # Initialize with existing config
            initialize_openrouter(existing_config.api_key)
            return existing_config
        except:
            logger.info("No existing configuration found")
            
            # Check for API key in environment
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                logger.info("Found OpenRouter API key in environment")
                default_config = create_default_config()
                default_config.api_key = api_key
                
                # Initialize with default config
                initialize_openrouter(default_config.api_key)
                
                # Save the configuration
                save_config(default_config, config_path)
                return default_config
            else:
                logger.warning("No OpenRouter API key found. Use --configure to run setup wizard.")
                return None
                
    except ImportError as e:
        logger.error(f"OpenRouter integration not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Error setting up OpenRouter integration: {e}")
        return None

def run_configuration_wizard():
    """Run the interactive configuration wizard"""
    try:
        from ai_scientist.openrouter.cli import CLIInterface
        import asyncio
        
        print("\n🚀 Starting AI-Scientist-v2 OpenRouter Configuration Wizard...")
        
        cli = CLIInterface()
        config = asyncio.run(cli.run_setup_wizard())
        
        if config:
            print("\n✅ Configuration completed successfully!")
            return config
        else:
            print("\n❌ Configuration was not completed.")
            return None
            
    except ImportError:
        print("❌ OpenRouter integration not available. Please install required dependencies.")
        return None
    except KeyboardInterrupt:
        print("\n\n⏹️  Configuration cancelled by user.")
        return None
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        return None

def setup_rag_system(config):
    """Initialize RAG system if enabled"""
    if not config or not config.rag_config.enabled:
        return None
    
    try:
        from ai_scientist.openrouter import RAGSystem
        
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(config.rag_config)
        
        # Check for documents to ingest
        docs_dir = Path("./documents")
        if docs_dir.exists():
            logger.info(f"Found documents directory: {docs_dir}")
            
            # Auto-ingest documents
            supported_extensions = ['.pdf', '.txt', '.md', '.docx']
            documents_found = []
            
            for ext in supported_extensions:
                documents_found.extend(list(docs_dir.glob(f"**/*{ext}")))
            
            if documents_found:
                print(f"\n📚 Found {len(documents_found)} documents to ingest:")
                for doc in documents_found[:5]:  # Show first 5
                    print(f"  • {doc.name}")
                if len(documents_found) > 5:
                    print(f"  • ... and {len(documents_found) - 5} more")
                
                if input("\nIngest these documents? (y/N): ").strip().lower().startswith('y'):
                    ingested_count = 0
                    for doc_path in documents_found:
                        try:
                            # Run async ingest_file in sync context
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            doc_id = loop.run_until_complete(rag_system.ingest_file(doc_path))
                            if doc_id:
                                ingested_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to ingest {doc_path}: {e}")
                    
                    print(f"✅ Successfully ingested {ingested_count}/{len(documents_found)} documents")
        
        return rag_system
        
    except ImportError:
        logger.warning("RAG system dependencies not available")
        return None
    except Exception as e:
        logger.error(f"Error setting up RAG system: {e}")
        return None

def launch_enhanced_pipeline(config, rag_system=None):
    """Launch the enhanced AI-Scientist pipeline"""
    try:
        # Import simple enhanced launcher
        from ai_scientist.integration.simple_enhanced_launcher import run_enhanced_pipeline
        import asyncio
        
        logger.info("Starting enhanced AI-Scientist pipeline...")
        
        # For demo purposes, let's run with a sample idea
        idea_name = "machine_learning_optimization"
        
        # Run async pipeline in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            run_enhanced_pipeline(config, rag_system, idea_name)
        )
        
        print("\n=== ENHANCED PIPELINE RESULTS ===")
        print(f"Success: {results.get('success', False)}")
        print(f"Stages completed: {len(results.get('stages', {}))}")
        
        for stage, result in results.get('stages', {}).items():
            print(f"  {stage}: {result.get('status', 'unknown')}")
        
        return results.get('success', False)
        
    except ImportError as e:
        logger.warning(f"Enhanced pipeline not available: {e}, falling back to original...")
        return launch_original_pipeline()
    except Exception as e:
        logger.error(f"Error launching enhanced pipeline: {e}")
        return launch_original_pipeline()

def launch_original_pipeline():
    """Launch the original AI-Scientist pipeline"""
    try:
        # Import and run the original launcher
        import launch_scientist_bfts
        
        logger.info("Starting original AI-Scientist pipeline...")
        return launch_scientist_bfts.main()
        
    except Exception as e:
        logger.error(f"Error launching original pipeline: {e}")
        return False

def display_welcome_banner():
    """Display welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🚀 AI-Scientist-v2 with OpenRouter Integration              ║
║                                                                            ║
║  Advanced AI Research Automation with Unified LLM Access                  ║
║  • 200+ Models from Multiple Providers                                    ║
║  • Intelligent Cost Optimization                                          ║
║  • RAG Document Integration                                               ║
║  • Per-Stage Model Configuration                                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AI-Scientist-v2 with OpenRouter Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Run interactive configuration wizard"
    )
    
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original AI-Scientist pipeline (bypass OpenRouter)"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG system"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display welcome banner
    display_welcome_banner()
    
    # Handle configuration wizard
    if args.configure:
        config = run_configuration_wizard()
        if not config:
            sys.exit(1)
        print("\n🎉 Configuration completed! You can now launch AI-Scientist-v2.")
        print("Run this script again without --configure to start the system.")
        sys.exit(0)
    
    # Handle original pipeline mode
    if args.use_original:
        print("🔄 Using original AI-Scientist pipeline (OpenRouter disabled)")
        os.environ["USE_OPENROUTER"] = "false"
        success = launch_original_pipeline()
        sys.exit(0 if success else 1)
    
    # Set up OpenRouter integration
    print("🔧 Setting up OpenRouter integration...")
    config = setup_openrouter_integration()
    
    if not config:
        print("\n⚠️  OpenRouter integration not available.")
        print("Options:")
        print("  1. Run with --configure to set up OpenRouter")
        print("  2. Run with --use-original to use the original pipeline")
        print("  3. Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    # Enable OpenRouter
    os.environ["USE_OPENROUTER"] = "true"
    
    # Set up RAG system
    rag_system = None
    if not args.no_rag and config.rag_config.enabled:
        print("\n📚 Setting up RAG system...")
        rag_system = setup_rag_system(config)
    
    # Launch pipeline
    print(f"\n🚀 Launching AI-Scientist-v2...")
    print(f"   Pipeline: {'Enhanced' if config.use_enhanced_pipeline else 'Original'}")
    print(f"   RAG System: {'Enabled' if rag_system else 'Disabled'}")
    print()
    
    try:
        if config.use_enhanced_pipeline:
            success = launch_enhanced_pipeline(config, rag_system)
        else:
            success = launch_original_pipeline()
        
        if success:
            print("\n🎉 AI-Scientist-v2 completed successfully!")
        else:
            print("\n❌ AI-Scientist-v2 encountered errors.")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()