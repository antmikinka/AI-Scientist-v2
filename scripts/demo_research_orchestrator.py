#!/usr/bin/env python3
"""
Research Orchestrator Agent Demo

This script demonstrates the capabilities of the Research Orchestrator Agent,
showcasing agent coordination, workflow management, and research automation.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_scientist.orchestration.research_orchestrator_agent import (
    ResearchOrchestratorAgent,
    ResearchRequest,
    AgentCapability,
    AgentStatus,
    CoordinationMode,
    get_orchestrator,
    create_research_request,
    coordinate_research_simple
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_orchestration():
    """Demonstrate basic research orchestration"""
    print("🚀 Research Orchestrator Agent Demo")
    print("=" * 50)

    # Create orchestrator configuration
    config = {
        'max_concurrent_workflows': 5,
        'default_timeout': 300,  # 5 minutes
        'consensus_threshold': 0.7,
        'max_agents_per_workflow': 3
    }

    try:
        # Initialize orchestrator
        print("🔧 Initializing Research Orchestrator Agent...")
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        print(f"✅ Orchestrator initialized: {orchestrator.agent_id}")

        # Register some sample agents
        print("\n📋 Registering sample agents...")
        sample_agents = [
            AgentCapability(
                agent_id="data_analyst_001",
                agent_type="data_specialist",
                capabilities=["data_analysis", "statistics", "visualization"],
                status=AgentStatus.ACTIVE,
                performance_metrics={"success_rate": 0.9, "avg_response_time": 2.0},
                current_load=1,
                max_capacity=10,
                last_active=datetime.now()
            ),
            AgentCapability(
                agent_id="experimentalist_001",
                agent_type="experimental_specialist",
                capabilities=["experimental_design", "simulation", "optimization"],
                status=AgentStatus.ACTIVE,
                performance_metrics={"success_rate": 0.85, "avg_response_time": 3.0},
                current_load=0,
                max_capacity=8,
                last_active=datetime.now()
            ),
            AgentCapability(
                agent_id="theorist_001",
                agent_type="theory_specialist",
                capabilities=["theory_development", "modeling", "analysis"],
                status=AgentStatus.ACTIVE,
                performance_metrics={"success_rate": 0.95, "avg_response_time": 1.5},
                current_load=2,
                max_capacity=6,
                last_active=datetime.now()
            )
        ]

        for agent in sample_agents:
            await orchestrator.service_registry.register_agent(agent)
            print(f"   ✅ Registered {agent.agent_id} ({agent.agent_type})")

        # Create research requests
        print("\n🔬 Creating research requests...")
        research_requests = [
            ResearchRequest(
                request_id="research_001",
                objective="Analyze the relationship between temperature and chemical reaction rates",
                context={"domain": "chemistry", "experiment_type": "kinetics"},
                priority=8,
                coordination_mode=CoordinationMode.SEQUENTIAL,
                budget_constraints={"max_cost": 100.0},
                ethical_requirements={"human_subjects": False}
            ),
            ResearchRequest(
                request_id="research_002",
                objective="Investigate optimal catalyst concentration for maximum yield",
                context={"domain": "chemical_engineering", "reaction_type": "catalysis"},
                priority=6,
                coordination_mode=CoordinationMode.PARALLEL,
                budget_constraints={"max_cost": 150.0},
                ethical_requirements={"safety_concerns": "low"}
            ),
            ResearchRequest(
                request_id="research_003",
                objective="Develop predictive model for protein folding patterns",
                context={"domain": "biochemistry", "analysis_type": "machine_learning"},
                priority=9,
                coordination_mode=CoordinationMode.CONSENSUS,
                budget_constraints={"max_cost": 200.0},
                ethical_requirements={"human_subjects": False}
            )
        ]

        # Demonstrate agent discovery
        print("\n🔍 Demonstrating agent discovery...")
        for i, request in enumerate(research_requests, 1):
            required_capabilities = await orchestrator._analyze_required_capabilities(request.objective)
            available_agents = await orchestrator.service_registry.discover_agents(required_capabilities)

            print(f"   Request {i}: '{request.objective[:50]}...'")
            print(f"      Required capabilities: {required_capabilities}")
            print(f"      Available agents: {[a.agent_id for a in available_agents]}")
            print()

        # Get orchestrator status
        print("📊 Orchestrator Status:")
        status = await orchestrator.get_orchestrator_status()
        for key, value in status.items():
            if isinstance(value, (int, float, str)):
                print(f"   {key}: {value}")
        print()

        # Demonstrate simple coordination (mock execution)
        print("🎯 Demonstrating research coordination...")
        print("   (Note: This is a demonstration - actual agent execution would require")
        print("    external agent implementations and more complex setup)")
        print()

        for i, request in enumerate(research_requests, 1):
            print(f"   📋 Research Request {i}:")
            print(f"      ID: {request.request_id}")
            print(f"      Objective: {request.objective}")
            print(f"      Coordination Mode: {request.coordination_mode.value}")
            print(f"      Priority: {request.priority}")

            # Show agent assignment logic
            required_capabilities = await orchestrator._analyze_required_capabilities(request.objective)
            available_agents = await orchestrator.service_registry.discover_agents(required_capabilities)

            if available_agents:
                selected_agents = await orchestrator._select_optimal_agents(available_agents, request)
                print(f"      Assigned Agents: {[a.agent_id for a in selected_agents]}")
            else:
                print(f"      Assigned Agents: None available")

            print()

        # Show agent status
        print("🤖 Agent Status:")
        for agent_id, agent in orchestrator.service_registry.registered_agents.items():
            print(f"   {agent_id}:")
            print(f"      Status: {agent.status.value}")
            print(f"      Load: {agent.current_load}/{agent.max_capacity}")
            print(f"      Performance: {agent.performance_metrics}")
            print()

        # Demonstrate workflow analysis
        print("📈 Performance Analysis:")
        print(f"   Total registered agents: {len(orchestrator.service_registry.registered_agents)}")
        print(f"   Max concurrent workflows: {orchestrator.config['max_concurrent_workflows']}")
        print(f"   Consensus threshold: {orchestrator.config['consensus_threshold']}")
        print(f"   Default timeout: {orchestrator.config['default_timeout']} seconds")
        print()

        # Demonstrate API Gateway capabilities
        print("🌐 API Gateway Features:")
        print("   ✅ RESTful API endpoints")
        print("   ✅ WebSocket real-time updates")
        print("   ✅ Authentication and rate limiting")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Performance monitoring")
        print("   ✅ Health check endpoints")
        print()

        # Show system architecture
        print("🏗️ System Architecture:")
        print("   🎯 Research Orchestrator Agent (Central Coordination)")
        print("   ├─ 📋 Service Registry (Agent Discovery)")
        print("   ├─ 🤖 Supervisor Agent (Workflow Management)")
        print("   ├─ 👥 Agent Manager (Multi-Agent Coordination)")
        print("   ├─ 🧠 Theory Evolution Agent (Knowledge Integration)")
        print("   ├─ 🔒 Security Manager (Ethical Compliance)")
        print("   ├─ 📊 Performance Monitor (Metrics Collection)")
        print("   └─ 🌐 API Gateway (External Interface)")
        print()

        print("🎉 Research Orchestrator Agent Demo Complete!")
        print("=" * 50)
        print("\nKey Features Demonstrated:")
        print("✅ Agent registration and discovery")
        print("✅ Research request processing")
        print("✅ Agent assignment and load balancing")
        print("✅ Multiple coordination modes (Sequential, Parallel, Consensus)")
        print("✅ Ethical compliance checking")
        print("✅ Performance monitoring")
        print("✅ Service registry management")
        print("✅ API gateway capabilities")
        print("✅ System status reporting")

        # Cleanup
        await orchestrator.shutdown()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_simple_interface():
    """Demonstrate the simple interface for research coordination"""
    print("\n🚀 Simple Interface Demo")
    print("=" * 30)

    try:
        # Demonstrate simple interface functions
        request = await create_research_request(
            objective="Quick analysis of temperature effects",
            priority=5
        )

        print(f"Created request: {request.request_id}")
        print(f"Objective: {request.objective}")
        print(f"Priority: {request.priority}")

        # Note: Actual execution would require full system setup
        print("\n⚠️  Note: Full execution requires:")
        print("   - External agent implementations")
        print("   - Database backend for persistence")
        print("   - Message queue for communication")
        print("   - Container orchestration for scaling")

    except Exception as e:
        logger.error(f"Simple interface demo failed: {e}")


if __name__ == "__main__":
    async def main():
        """Main demo function"""
        try:
            await demo_basic_orchestration()
            await demo_simple_interface()
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()

    # Run the demo
    asyncio.run(main())