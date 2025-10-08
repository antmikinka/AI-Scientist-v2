# Research Orchestrator Guide

## 🎯 Introduction

The Research Orchestrator Agent is the central coordination system for the AI-Scientist-v2 multi-agent ecosystem. It provides intelligent agent discovery, workflow coordination, ethical compliance integration, and performance optimization to enable autonomous scientific research at scale.

## 🏗️ Architecture Overview

```
🎯 Research Orchestrator Agent
├─ 📋 Service Registry (Agent Discovery & Management)
├─ 🤖 Supervisor Agent (Workflow Management)
├─ 👥 Agent Manager (Multi-Agent Coordination)
├─ 🧠 Theory Evolution Agent (Knowledge Integration)
├─ 🛡️ Ethical Framework Agent (Governance & Compliance)
├─ 📊 Performance Monitor (Metrics & Optimization)
└─ 🌐 API Gateway (External Interface)
```

## 🚀 Getting Started

### Initialization
```python
import asyncio
from ai_scientist.orchestration.research_orchestrator_agent import get_orchestrator

async def initialize_orchestrator():
    # Create orchestrator configuration
    config = {
        'max_concurrent_workflows': 10,
        'default_timeout': 3600,
        'consensus_threshold': 0.7,
        'max_agents_per_workflow': 5,
        'ethical_integration_enabled': True
    }

    # Initialize the orchestrator
    orchestrator = await get_orchestrator(config)

    print(f"✅ Orchestrator initialized: {orchestrator.agent_id}")
    return orchestrator

# Initialize
orchestrator = asyncio.run(initialize_orchestrator())
```

### Basic Usage
```python
from ai_scientist.orchestration.research_orchestrator_agent import create_research_request, CoordinationMode

# Create a research request
request = await create_research_request(
    objective="Investigate novel catalyst materials for hydrogen production",
    context={
        "domain": "materials_science",
        "research_type": "catalyst_discovery"
    },
    priority=8,
    coordination_mode=CoordinationMode.CONSENSUS,
    ethical_requirements={
        "human_subjects": False,
        "environmental_impact": "low"
    }
)

# Execute research
results = await orchestrator.coordinate_research(request)
```

## 🤖 Agent Management

### Agent Registration
```python
from ai_scientist.orchestration.research_orchestrator_agent import AgentCapability, AgentStatus

# Create a specialized agent
chemistry_agent = AgentCapability(
    agent_id="catalyst_specialist_001",
    agent_type="domain_specialist",
    capabilities=["catalyst_design", "materials_modeling", "reaction_optimization"],
    status=AgentStatus.ACTIVE,
    performance_metrics={
        "success_rate": 0.92,
        "avg_response_time": 2.5,
        "total_tasks": 150
    },
    current_load=0,
    max_capacity=8,
    last_activity=datetime.now()
)

# Register the agent
success = await orchestrator.service_registry.register_agent(chemistry_agent)
print(f"Agent registered: {success}")
```

### Agent Discovery
```python
# Discover agents for specific capabilities
required_capabilities = ["catalyst_design", "materials_modeling"]

available_agents = await orchestrator.service_registry.discover_agents(
    required_capabilities,
    max_agents=3
)

print(f"Found {len(available_agents)} agents:")
for agent in available_agents:
    print(f"  - {agent.agent_id}: {agent.capabilities}")
    print(f"    Performance: {agent.performance_metrics}")
    print(f"    Load: {agent.current_load}/{agent.max_capacity}")
```

### Agent Status Monitoring
```python
# Get orchestrator status
status = await orchestrator.get_orchestrator_status()

print("📊 System Status:")
print(f"  Active Workflows: {status['active_workflows']}")
print(f"  Success Rate: {status['success_rate']:.2f}")
print(f"  Registered Agents: {status['registered_agents']}")

# Monitor individual agents
for agent_id, agent in orchestrator.service_registry.registered_agents.items():
    print(f"\n🤖 Agent: {agent_id}")
    print(f"  Status: {agent.status.value}")
    print(f"  Current Load: {agent.current_load}/{agent.max_capacity}")
    print(f"  Availability Score: {agent.availability_score:.2f}")
```

## 🔄 Coordination Modes

### Sequential Mode
Agents work in a predetermined sequence.

```python
sequential_request = await create_research_request(
    objective="Develop new drug candidate for malaria treatment",
    coordination_mode=CoordinationMode.SEQUENTIAL,
    context={
        "sequence": [
            "target_identification",
            "compound_screening",
            "lead_optimization",
            "preclinical_testing"
        ]
    }
)

results = await orchestrator.coordinate_research(sequential_request)
```

### Parallel Mode
Multiple agents work simultaneously.

```python
parallel_request = await create_research_request(
    objective="Comprehensive climate change impact analysis",
    coordination_mode=CoordinationMode.PARALLEL,
    context={
        "parallel_tasks": [
            "temperature_data_analysis",
            "sea_level_rise_modeling",
            "ecosystem_impact_assessment",
            "economic_impact_projection"
        ]
    }
)

results = await orchestrator.coordinate_research(parallel_request)
```

### Consensus Mode
Multiple agents work independently and build consensus.

```python
consensus_request = await create_research_request(
    objective="Determine optimal neural network architecture",
    coordination_mode=CoordinationMode.CONSENSUS,
    context={
        "consensus_requirements": {
            "min_agents": 3,
            "confidence_threshold": 0.7,
            "voting_mechanism": "weighted"
        }
    }
)

results = await orchestrator.coordinate_research(consensus_request)
```

### Competitive Mode
Agents compete to find the best solution.

```python
competitive_request = await create_research_request(
    objective="Optimize protein folding prediction algorithm",
    coordination_mode=CoordinationMode.COMPETITIVE,
    context={
        "competition_criteria": {
            "accuracy_weight": 0.6,
            "speed_weight": 0.2,
            "efficiency_weight": 0.2
        }
    }
)

results = await orchestrator.coordinate_research(competitive_request)
```

## 🛡️ Ethical Integration

### Ethical Compliance Checking
```python
# Create research with ethical considerations
ethical_request = await create_research_request(
    objective="Genetic study of disease susceptibility in diverse populations",
    ethical_requirements={
        "human_subjects": True,
        "genetic_data": True,
        "privacy_protection": "required",
        "informed_consent": "required",
        "data_anonymization": "required"
    }
)

results = await orchestrator.coordinate_research(ethical_request)

# Check ethical compliance
if results.ethical_compliance["approved"]:
    print("✅ Research ethically approved")
    print(f"Risk Level: {results.ethical_compliance['risk_level']}")
    print(f"Compliance Score: {results.ethical_compliance['compliance_score']:.2f}")
else:
    print("❌ Research blocked due to ethical concerns")
    print(f"Reason: {results.ethical_compliance.get('reason', 'Unknown')}")
```

### Ethical Framework Configuration
```python
# Configure ethical framework weights
ethical_config = {
    'framework_weights': {
        'utilitarian': 0.2,        # Maximize overall benefit
        'deontological': 0.2,     # Follow ethical duties
        'virtue': 0.15,           # Moral character focus
        'care': 0.15,             # Relationship ethics
        'principle_based': 0.2,   # Belmont principles
        'precautionary': 0.1      # Risk-averse approach
    },
    'human_oversight_level': 'high',
    'enable_cross_cultural_considerations': True
}

# Update orchestrator configuration
orchestrator.config.update(ethical_config)
```

## 📊 Performance Monitoring

### Real-time Metrics
```python
# Get current performance metrics
metrics = await orchestrator.performance_monitor.get_current_metrics()

print("📈 Performance Metrics:")
print(f"  CPU Usage: {metrics['system_metrics']['cpu_usage']:.1f}%")
print(f"  Memory Usage: {metrics['system_metrics']['memory_usage']:.1f}%")
print(f"  Active Workflows: {metrics['workflow_metrics']['active_workflows']}")
print(f"  Throughput: {metrics['performance_metrics']['throughput']:.1f}/min")
print(f"  Error Rate: {metrics['performance_metrics']['error_rate']:.2%}")
```

### Historical Performance
```python
# Get historical metrics (last hour)
historical_metrics = await orchestrator.performance_monitor.get_historical_metrics(
    duration_minutes=60
)

print("📊 Historical Performance:")
for metric in historical_metrics[-5:]:  # Last 5 data points
    timestamp = datetime.fromtimestamp(metric['timestamp'])
    print(f"  {timestamp}: {metric['throughput']:.1f} workflows/min")
```

### Agent Performance Reports
```python
# Get agent performance summary
agent_report = await orchestrator.performance_monitor.get_agent_performance_report()

print("🤖 Agent Performance Summary:")
print(f"Top Performers: {agent_report['top_performers']}")
print(f"Underperformers: {agent_report['underperformers']}")

for agent_id, summary in agent_report['agent_summary'].items():
    print(f"\n{agent_id}:")
    print(f"  Performance Score: {summary['performance_score']:.2f}")
    print(f"  Success Rate: {summary['success_rate']:.2f}")
    print(f"  Avg Response Time: {summary['average_response_time']:.2f}s")
```

## 🌐 API Gateway Integration

### Starting the API Server
```python
from ai_scientist.orchestration.api_gateway import APIGateway
import uvicorn

# Create API gateway
gateway = APIGateway({
    "orchestrator": {
        'max_concurrent_workflows': 10,
        'default_timeout': 3600
    },
    "security": {
        "enable_auth": True
    }
})

# Start server (in production)
uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
```

### Making API Requests
```python
import requests
import json

# Research request via API
api_url = "http://localhost:8000/research"
headers = {
    "Authorization": "Bearer sk-your-api-key",
    "Content-Type": "application/json"
}

data = {
    "objective": "Develop quantum algorithm for optimization",
    "context": {"domain": "quantum_computing"},
    "priority": 9,
    "coordination_mode": "consensus",
    "ethical_requirements": {"human_subjects": False}
}

response = requests.post(api_url, json=data, headers=headers)
results = response.json()

print("🔬 API Results:", json.dumps(results, indent=2))
```

### WebSocket Integration
```python
import websockets
import json
import asyncio

async def listen_to_updates():
    uri = "ws://localhost:8000/ws"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"type": "subscribe"}))

        async for message in websocket:
            data = json.loads(message)
            print(f"📡 Real-time Update: {data}")

# Listen for real-time updates
asyncio.run(listen_to_updates())
```

## 🔧 Advanced Configuration

### Workflow Configuration
```python
advanced_config = {
    # Workflow settings
    'max_concurrent_workflows': 20,
    'default_timeout': 7200,  # 2 hours

    # Coordination settings
    'coordination_modes': {
        'sequential': {
            'enable_dependencies': True,
            'max_retries': 3
        },
        'parallel': {
            'max_parallel_agents': 10,
            'load_balancing': 'intelligent'
        },
        'consensus': {
            'threshold': 0.7,
            'voting_mechanism': 'weighted',
            'min_participants': 3
        },
        'competitive': {
            'scoring_method': 'multi_objective',
            'selection_criteria': 'highest_score'
        }
    },

    # Performance settings
    'performance_monitoring': {
        'enable_real_time_metrics': True,
        'metrics_retention_hours': 24,
        'enable_alerts': True,
        'alert_thresholds': {
            'error_rate': 0.1,
            'response_time': 300,
            'memory_usage': 0.8
        }
    },

    # Ethical settings
    'ethical_integration': {
        'enabled': True,
        'framework_weights': {
            'utilitarian': 0.2,
            'deontological': 0.2,
            'virtue': 0.15,
            'care': 0.15,
            'principle_based': 0.2,
            'precautionary': 0.1
        },
        'human_oversight': {
            'required_for_high_risk': True,
            'approval_threshold': 0.8
        }
    }
}

orchestrator = await get_orchestrator(advanced_config)
```

### Agent-Specific Configuration
```python
# Configure agent behavior
agent_config = {
    'agent_behavior': {
        'load_balancing_strategy': 'least_loaded',
        'performance_weighting': {
            'success_rate': 0.4,
            'response_time': 0.3,
            'reliability': 0.3
        },
        'selection_criteria': {
            'min_performance_score': 0.7,
            'max_load_factor': 0.8
        }
    },

    'discovery': {
        'capability_matching': 'fuzzy',
        'similarity_threshold': 0.7,
        'consider_agent_load': True
    }
}
```

## 🚨 Error Handling and Troubleshooting

### Common Issues

#### Agent Not Found
```python
# Check if agents are registered
if len(orchestrator.service_registry.registered_agents) == 0:
    print("⚠️  No agents registered. Register agents first:")

    # Register default agents
    await register_default_agents(orchestrator)
```

#### Ethical Compliance Failure
```python
if not results.ethical_compliance["approved"]:
    print(f"❌ Ethical compliance failed:")
    print(f"  Risk Level: {results.ethical_compliance['risk_level']}")
    print(f"  Risk Factors: {results.ethical_compliance['risk_factors']}")

    # Suggest modifications
    suggestions = generate_ethical_suggestions(results.ethical_compliance)
    print(f"💡 Suggestions: {suggestions}")
```

#### Performance Issues
```python
# Check system performance
status = await orchestrator.get_orchestrator_status()

if status['average_workflow_time'] > 300:  # > 5 minutes
    print("⚠️  Slow workflow performance detected")

    # Optimize configuration
    await optimize_performance(orchestrator)
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger('ai_scientist.orchestration').setLevel(logging.DEBUG)

# Run with debug information
results = await orchestrator.coordinate_research(request)
```

## 📈 Best Practices

### 1. Agent Management
- **Register agents with clear capabilities** - Use specific, descriptive capability names
- **Monitor agent performance** - Regularly review success rates and response times
- **Balance agent loads** - Distribute workload evenly across capable agents
- **Retire underperforming agents** - Remove or retrain agents with low performance scores

### 2. Coordination Mode Selection
- **Sequential** for quality-critical, linear processes
- **Parallel** for time-sensitive, independent tasks
- **Consensus** for high-stakes decisions requiring validation
- **Competitive** for optimization and innovation tasks

### 3. Ethical Configuration
- **Use appropriate framework weights** for your research domain
- **Enable human oversight** for sensitive or high-risk research
- **Regular review ethical decisions** and update frameworks accordingly
- **Document ethical considerations** for all research projects

### 4. Performance Optimization
- **Monitor system metrics** regularly
- **Set appropriate timeouts** for different research types
- **Use caching** for repeated similar requests
- **Scale resources** based on demand patterns

## 🎯 Use Cases

### 1. Drug Discovery Research
```python
# Multi-stage drug discovery pipeline
drug_request = await create_research_request(
    objective="Discover novel inhibitors for target protein X",
    coordination_mode=CoordinationMode.SEQUENTIAL,
    context={
        "pipeline_stages": [
            "target_validation",
            "compound_screening",
            "lead_optimization",
            "preclinical_testing"
        ],
        "domain": "pharmaceutical_research"
    },
    ethical_requirements={
        "human_subjects": False,
        "animal_testing": "minimize",
        "environmental_impact": "consider"
    }
)
```

### 2. Climate Science Analysis
```python
# Large-scale climate data analysis
climate_request = await create_research_request(
    objective="Analyze global temperature patterns and predict future trends",
    coordination_mode=CoordinationMode.PARALLEL,
    context={
        "data_sources": [
            "satellite_data",
            "ground_stations",
            "ocean_buoys",
            "ice_cores"
        ],
        "analysis_types": [
            "statistical_analysis",
            "machine_learning_modeling",
            "climate_simulation"
        ]
    },
    ethical_requirements={
        "human_subjects": False,
        "environmental_impact": "high",
        "data_privacy": "protect"
    }
)
```

### 3. Materials Science Discovery
```python
# Novel materials discovery
materials_request = await create_research_request(
    objective="Discover new superconducting materials",
    coordination_mode=CoordinationMode.CONSENSUS,
    context={
        "search_space": "chemical_compounds",
        "properties_target": {
            "critical_temperature": "optimize",
            "stability": "ensure",
            "cost_effectiveness": "consider"
        }
    },
    ethical_requirements={
        "human_subjects": False,
        "environmental_impact": "low",
        "commercial_applications": "consider"
    }
)
```

## 📚 Additional Resources

- [Multi-Agent System Overview](MULTI_AGENT_OVERVIEW.md)
- [Agent Development Guide](AGENT_DEVELOPMENT.md)
- [Ethical Framework Guide](ETHICAL_FRAMEWORK.md)
- [API Integration Guide](API_DEVELOPMENT.md)
- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION.md)

---

**🎉 Master the Research Orchestrator to unlock the full potential of multi-agent scientific research!**