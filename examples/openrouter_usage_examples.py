#!/usr/bin/env python3
"""
OpenRouter Integration Usage Examples

This script demonstrates various ways to use the OpenRouter integration
with AI-Scientist-v2, including basic completions, advanced features,
cost tracking, and pipeline integration.

Run with:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python examples/openrouter_usage_examples.py
"""

import asyncio
import json
import os
import logging
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_basic_completion():
    """Example 1: Basic text completion"""
    print("\n" + "="*60)
    print("Example 1: Basic Text Completion")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import initialize_openrouter
        
        # Initialize client
        client = initialize_openrouter()
        
        # Simple completion
        response, history = await client.get_response(
            messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
            model="anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"Model: anthropic/claude-3.5-sonnet")
        print(f"Response: {response}")
        print(f"Message history length: {len(history)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_model_comparison():
    """Example 2: Compare responses from different models"""
    print("\n" + "="*60)
    print("Example 2: Model Comparison")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import get_global_client
        
        client = get_global_client()
        
        question = "What are the key principles of machine learning?"
        models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "google/gemini-2.0-flash"
        ]
        
        for model in models:
            try:
                response, _ = await client.get_response(
                    messages=[{"role": "user", "content": question}],
                    model=model,
                    temperature=0.5,
                    max_tokens=150
                )
                
                print(f"\n🤖 {model}:")
                print(f"{response[:200]}..." if len(response) > 200 else response)
                
            except Exception as e:
                print(f"❌ {model} failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_batch_responses():
    """Example 3: Batch responses for ensembling"""
    print("\n" + "="*60)
    print("Example 3: Batch Responses for Ensembling")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import get_global_client
        
        client = get_global_client()
        
        # Get multiple responses for ensembling
        responses, histories = await client.get_batch_responses(
            messages=[{"role": "user", "content": "Design a simple machine learning experiment."}],
            model="openai/gpt-4o-mini",
            n_responses=3,
            temperature=0.8,
            max_tokens=150
        )
        
        print(f"Generated {len(responses)} responses:")
        for i, response in enumerate(responses, 1):
            print(f"\n📝 Response {i}:")
            print(f"{response[:150]}..." if len(response) > 150 else response)
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_tool_calling():
    """Example 4: Function/Tool calling"""
    print("\n" + "="*60)
    print("Example 4: Tool Calling")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import get_global_client, ToolDefinition
        
        client = get_global_client()
        
        # Create tool definitions
        tools = client.create_tool_definitions()
        
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Use tools
        final_response, history = await client.call_function(
            messages=[{"role": "user", "content": "Calculate 15 * 23 + 47 and tell me the result."}],
            model="openai/gpt-4o-mini",
            tools=tools,
            tool_choice="auto",
            max_iterations=3
        )
        
        print(f"Final response: {final_response}")
        print(f"Conversation length: {len(history)} messages")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_cost_tracking():
    """Example 5: Cost tracking and budget management"""
    print("\n" + "="*60)
    print("Example 5: Cost Tracking")
    print("="*60)
    
    try:
        from ai_scientist.openrouter.cost_tracker import get_global_cost_tracker
        from ai_scientist.openrouter import get_global_client
        
        # Get cost tracker
        tracker = get_global_cost_tracker()
        
        # Set a test budget
        tracker.set_budget("test_budget", 1.0, "total")  # $1 total budget
        
        client = get_global_client()
        
        # Make some API calls to generate costs
        for i in range(3):
            await client.get_response(
                messages=[{"role": "user", "content": f"Count to {i+3}"}],
                model="openai/gpt-4o-mini",
                max_tokens=50,
                stage="example"  # For cost tracking
            )
        
        # Get usage statistics
        usage = tracker.get_current_usage("all")
        print(f"Total cost: ${usage['total_cost']:.6f}")
        print(f"Total tokens: {usage['total_tokens']:,}")
        print(f"Total requests: {usage['total_requests']}")
        
        # Get cost optimization suggestions
        suggestions = tracker.get_optimization_suggestions()
        if suggestions:
            print("\n💡 Cost optimization suggestions:")
            for suggestion in suggestions[:2]:  # Show first 2
                print(f"   • {suggestion.description}")
                print(f"     Potential savings: ${suggestion.potential_savings:.4f}")
        else:
            print("\n💡 No optimization suggestions (not enough data)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_rag_system():
    """Example 6: RAG document system"""
    print("\n" + "="*60)
    print("Example 6: RAG Document System")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import RAGSystem
        from ai_scientist.openrouter.config import RAGConfig
        
        # Create temporary directory for RAG storage
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize RAG system
            rag_config = RAGConfig(
                enabled=True,
                chunk_size=500,
                chunk_overlap=100
            )
            rag_system = RAGSystem(rag_config, temp_dir)
            
            # Create a test document
            test_doc = Path(temp_dir) / "test_document.txt"
            test_content = """
            Artificial Intelligence (AI) is a transformative technology that enables machines to learn, reason, and make decisions.
            
            Key AI concepts include:
            - Machine Learning: Algorithms that improve automatically through experience
            - Deep Learning: Neural networks with multiple layers that can learn complex patterns
            - Natural Language Processing: AI's ability to understand and generate human language
            - Computer Vision: AI systems that can interpret and understand visual information
            
            Applications of AI include autonomous vehicles, medical diagnosis, financial trading, and recommendation systems.
            The field continues to evolve rapidly with new breakthroughs in areas like large language models and generative AI.
            """
            
            with open(test_doc, 'w') as f:
                f.write(test_content)
            
            # Ingest the document
            doc_id = await rag_system.ingest_file(test_doc)
            print(f"Ingested document: {doc_id}")
            
            # Search for relevant content
            results = rag_system.search("machine learning algorithms", max_results=3)
            print(f"\nSearch results ({len(results)} found):")
            
            for i, (content, score, metadata) in enumerate(results, 1):
                print(f"\n🔍 Result {i} (score: {score:.3f}):")
                print(f"   {content[:150]}..." if len(content) > 150 else content)
            
            # Generate context for a query
            context = rag_system.get_context_for_query("deep learning", max_context_length=1000)
            if context:
                print(f"\n📚 Generated context length: {len(context)} characters")
                print(f"Context preview: {context[:200]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_configuration():
    """Example 7: Configuration management"""
    print("\n" + "="*60)
    print("Example 7: Configuration Management")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import (
            create_default_config, 
            create_config_from_template,
            validate_config
        )
        
        # Create default configuration
        default_config = create_default_config()
        print(f"Default config created with {len(default_config.stage_configs)} stages")
        
        # Validate configuration
        errors = validate_config(default_config)
        if errors:
            print(f"❌ Validation errors: {errors}")
        else:
            print("✅ Configuration is valid")
        
        # Create configuration from template
        templates = ["research", "cost_optimized", "high_quality"]
        for template in templates:
            try:
                config = create_config_from_template(template)
                print(f"✅ Created {template} template configuration")
                
                # Show model selections for this template
                models = {stage: cfg.model for stage, cfg in config.stage_configs.items()}
                print(f"   Models: {models}")
                
            except Exception as e:
                print(f"❌ Failed to create {template} template: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_enhanced_pipeline():
    """Example 8: Enhanced pipeline execution"""
    print("\n" + "="*60)
    print("Example 8: Enhanced Pipeline (Demo)")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import create_default_config
        from ai_scientist.integration.simple_enhanced_launcher import SimpleEnhancedLauncher
        
        # Create lightweight configuration for demo
        config = create_default_config()
        
        # Use fast, inexpensive models for demo
        for stage_config in config.stage_configs.values():
            stage_config.model = "openai/gpt-4o-mini"  # Fast and cheap
            stage_config.max_tokens = 100  # Limit tokens for demo
        
        # Initialize launcher (without RAG for simplicity)
        launcher = SimpleEnhancedLauncher(config, rag_system=None)
        
        # Create temporary results directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run just the ideation stage as a demo
            results_dir = Path(temp_dir)
            result = await launcher.run_enhanced_ideation("ai_safety_research", results_dir)
            
            print(f"Pipeline stage result: {result['status']}")
            print(f"Model used: {result.get('model_used', 'unknown')}")
            
            if result['status'] == 'completed':
                output_file = result.get('output_file')
                if output_file and Path(output_file).exists():
                    with open(output_file, 'r') as f:
                        content = f.read()
                    print(f"Generated content length: {len(content)} characters")
                    print(f"Content preview: {content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_streaming():
    """Example 9: Streaming responses"""
    print("\n" + "="*60)
    print("Example 9: Streaming Responses")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import get_global_client
        
        client = get_global_client()
        
        print("🌊 Streaming response from Claude:")
        print("-" * 40)
        
        full_response = ""
        async for chunk in client.stream_response(
            messages=[{"role": "user", "content": "Write a haiku about artificial intelligence."}],
            model="anthropic/claude-3-haiku",
            max_tokens=100
        ):
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    print(content, end="", flush=True)
                    full_response += content
        
        print("\n" + "-" * 40)
        print(f"Complete response length: {len(full_response)} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_fallback_models():
    """Example 10: Model fallback demonstration"""
    print("\n" + "="*60)
    print("Example 10: Model Fallback")
    print("="*60)
    
    try:
        from ai_scientist.openrouter import get_global_client
        
        client = get_global_client()
        
        # Try with an intentionally failing primary model and working fallbacks
        response, history = await client.get_response_with_fallback(
            messages=[{"role": "user", "content": "What is machine learning?"}],
            primary_model="invalid/model-that-does-not-exist",
            fallback_models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
            max_tokens=100
        )
        
        print("✅ Fallback system worked!")
        print(f"Response: {response[:150]}..." if len(response) > 150 else response)
        
    except Exception as e:
        print(f"❌ Error (expected if no fallbacks work): {e}")

async def main():
    """Run all examples"""
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        print("   export OPENROUTER_API_KEY='sk-or-v1-...'")
        return
    
    print("🚀 OpenRouter Integration Examples")
    print(f"API Key: {'✓ Found' if os.getenv('OPENROUTER_API_KEY') else '❌ Not Found'}")
    
    examples = [
        example_basic_completion,
        example_model_comparison,
        example_batch_responses,
        example_tool_calling,
        example_cost_tracking,
        example_rag_system,
        example_configuration,
        example_enhanced_pipeline,
        example_streaming,
        example_fallback_models
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            await example_func()
        except KeyboardInterrupt:
            print(f"\n⏹️ Examples interrupted at #{i}")
            break
        except Exception as e:
            print(f"\n💥 Example #{i} failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay between examples
        await asyncio.sleep(0.5)
    
    print("\n" + "="*60)
    print("🎉 Examples completed!")
    print("="*60)
    
    # Final cost summary if cost tracking is available
    try:
        from ai_scientist.openrouter.cost_tracker import get_global_cost_tracker
        tracker = get_global_cost_tracker()
        usage = tracker.get_current_usage("all")
        print(f"\n💰 Total cost for examples: ${usage['total_cost']:.6f}")
        print(f"📊 Total tokens used: {usage['total_tokens']:,}")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(main())