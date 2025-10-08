# Quick Start Guide - Multi-Agent AI Scientist

## 🚀 Get Started in 10 Minutes

This guide will help you get up and running with the AI-Scientist-v2 multi-agent system quickly and easily.

### Prerequisites
- Python 3.11 or higher
- OpenRouter API key
- Git

### Installation

#### 1. Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-username/AI-Scientist-v2.git
cd AI-Scientist-v2

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_openrouter.txt
```

#### 2. Configure API Key
```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

#### 3. Launch the System
```bash
# Launch the multi-agent system
python scripts/launch_enhanced_scientist.py
```

## 🎯 Your First Multi-Agent Research Project

### Step 1: Initialize the Research Orchestrator
```python
import asyncio
from ai_scientist.orchestration.research_orchestrator_agent import get_orchestrator

async def main():
    # Initialize the orchestrator
    orchestrator = await get_orchestrator({
        'max_concurrent_workflows': 5,
        'consensus_threshold': 0.7,
        'ethical_integration_enabled': True
    })

    print("✅ Multi-Agent System Initialized!")
    return orchestrator

# Run it
orchestrator = asyncio.run(main())
```

### Step 2: Register Specialized Agents
```python
from ai_scientist.orchestration.research_orchestrator_agent import AgentCapability, AgentStatus

# Register chemistry specialist
chemistry_agent = AgentCapability(
    agent_id="chemistry_expert_001",
    agent_type="domain_specialist",
    capabilities=["molecular_modeling", "reaction_optimization", "catalysis"],
    status=AgentStatus.ACTIVE,
    performance_metrics={"success_rate": 0.92, "avg_response_time": 2.5},
    current_load=0,
    max_capacity=8
)

# Register data analysis specialist
data_agent = AgentCapability(
    agent_id="data_analyst_001",
    agent_type="task_specialist",
    capabilities=["statistical_analysis", "visualization", "pattern_recognition"],
    status=AgentStatus.ACTIVE,
    performance_metrics={"success_rate": 0.88, "avg_response_time": 1.8},
    current_load=0,
    max_capacity=10
)

# Register agents with the system
await orchestrator.service_registry.register_agent(chemistry_agent)
await orchestrator.service_registry.register_agent(data_agent)

print("🤖 Specialized Agents Registered!")
```

### Step 3: Create Your First Research Request
```python
from ai_scientist.orchestration.research_orchestrator_agent import create_research_request, CoordinationMode

# Create a research request
research_request = await create_research_request(
    objective="Analyze the efficiency of platinum vs. palladium catalysts in hydrogen fuel cells",
    context={
        "domain": "electrochemistry",
        "experiment_type": "catalytic_efficiency",
        "materials": ["platinum", "palladium", "hydrogen", "fuel_cell"]
    },
    priority=8,
    coordination_mode=CoordinationMode.CONSENSUS,
    ethical_requirements={
        "human_subjects": False,
        "environmental_impact": "low",
        "safety_concerns": "medium"
    }
)

print("📋 Research Request Created!")
```

### Step 4: Execute Multi-Agent Research
```python
# Execute the research with multi-agent coordination
results = await orchestrator.coordinate_research(research_request)

# Display results
print(f"🔬 Research Results:")
print(f"   Success: {results.success}")
print(f"   Confidence: {results.confidence_score:.2f}")
print(f"   Execution Time: {results.execution_time:.2f} seconds")
print(f"   Agents Used: {list(results.agent_contributions.values())}")

if results.success:
    print("   📊 Key Findings:")
    for key, value in results.results.items():
        if isinstance(value, str) and len(value) < 100:
            print(f"      {key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"      {key}: {value}")

    if results.recommendations:
        print("   💡 Recommendations:")
        for rec in results.recommendations:
            print(f"      - {rec}")
```

## 🌐 Using the API Gateway

### Start the API Server
```bash
# Start the API server
python -m ai_scientist.orchestration.api_gateway
```

### Make Research Requests via API
```python
import requests
import json

# API endpoint
url = "http://localhost:8000/research"

# Research request data
data = {
    "objective": "Optimize neural network architecture for image classification",
    "context": {"domain": "machine_learning", "task": "computer_vision"},
    "priority": 7,
    "coordination_mode": "parallel",
    "ethical_requirements": {"human_subjects": False}
}

# Headers with authentication
headers = {
    "Authorization": "Bearer sk-your-api-key-here",
    "Content-Type": "application/json"
}

# Make the request
response = requests.post(url, json=data, headers=headers)
results = response.json()

print("🔬 API Research Results:", json.dumps(results, indent=2))
```

## 🎮 Demo and Examples

### Run the Multi-Agent Demo
```bash
# Run the comprehensive demo
python scripts/demo_research_orchestrator.py
```

This demo will show:
- Agent registration and discovery
- Multi-coordination modes
- Ethical compliance checking
- Performance monitoring
- API gateway functionality

## 🛡️ Ethical Compliance in Action

The system automatically handles ethical compliance:

```python
# Request with ethical considerations
ethical_request = await create_research_request(
    objective="Study genetic markers for disease susceptibility",
    context={"domain": "genetics", "study_type": "population_genetics"},
    ethical_requirements={
        "human_subjects": True,
        "genetic_data": True,
        "privacy_concerns": "high",
        "consent_required": True
    }
)

results = await orchestrator.coordinate_research(ethical_request)

# Check ethical compliance
if results.ethical_compliance["approved"]:
    print("✅ Ethical compliance verified")
    print(f"   Risk Level: {results.ethical_compliance['risk_level']}")
    print(f"   Compliance Score: {results.ethical_compliance['compliance_score']:.2f}")
else:
    print("❌ Ethical compliance failed")
    print(f"   Reason: {results.ethical_compliance.get('reason', 'Unknown')}")
```

## 🔧 Configuration Options

### Basic Configuration
```python
config = {
    # Workflow settings
    'max_concurrent_workflows': 10,
    'default_timeout': 3600,

    # Coordination settings
    'consensus_threshold': 0.7,
    'max_agents_per_workflow': 5,

    # Ethical settings
    'ethical_integration_enabled': True
}
```

## 🎉 You're Ready!

**🚀 Congratulations! You're now ready to revolutionize scientific research with multi-agent AI!**

**Next Step**: Run `python scripts/demo_research_orchestrator.py` to see the full system in action!

## 📚 Additional Resources

- [Multi-Agent System Overview](MULTI_AGENT_OVERVIEW.md)
- [Research Orchestrator Guide](RESEARCH_ORCHESTRATOR.md)
- [Agent Development Guide](AGENT_DEVELOPMENT.md)
- [Ethical Framework Guide](ETHICAL_FRAMEWORK.md)