# Agent Development Guide

## Overview

This comprehensive guide provides everything needed to develop, integrate, and deploy specialized AI agents within the AI-Scientist-v2 multi-agent ecosystem. Learn how to create agents for specific research domains, implement custom capabilities, and ensure ethical compliance.

## Table of Contents

1. [Agent Architecture](#agent-architecture)
2. [Development Environment Setup](#development-environment-setup)
3. [Agent Types and Templates](#agent-types-and-templates)
4. [Core Agent Interface](#core-agent-interface)
5. [Agent Lifecycle Management](#agent-lifecycle-management)
6. [Capability Development](#capability-development)
7. [Ethical Framework Integration](#ethical-framework-integration)
8. [Testing and Validation](#testing-and-validation)
9. [Deployment and Registration](#deployment-and-registration)
10. [Best Practices](#best-practices)
11. [Advanced Topics](#advanced-topics)

## Agent Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Instance                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Core      │  │ Capabilities│  │   Ethical Layer     │  │
│  │   Engine    │  │   Manager   │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Communication│  │  State      │  │   Performance       │  │
│  │   Manager   │  │  Manager    │  │   Monitor           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                External Interfaces                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   REST API  │  │  WebSocket  │  │   Message Queue     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Agent States

- **INITIALIZING**: Agent is starting up and loading configuration
- **IDLE**: Agent is ready to accept tasks
- **BUSY**: Agent is actively processing a task
- **PAUSED**: Agent is temporarily paused
- **ERROR**: Agent has encountered an error
- **OFFLINE**: Agent is not connected

## Development Environment Setup

### Prerequisites

```bash
# Python 3.11+ required
python --version

# Install development dependencies
pip install -r requirements_dev.txt

# Install agent development tools
pip install ai-scientist-agent-tools
```

### Development Tools

```bash
# Create agent project
ai-scientist agent create --name "MyResearchAgent" --type "research"

# Validate agent configuration
ai-scientist agent validate --path ./my_research_agent

# Test agent locally
ai-scientist agent test --path ./my_research_agent

# Generate agent documentation
ai-scientist agent docs --path ./my_research_agent
```

### IDE Configuration

VS Code settings (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

## Agent Types and Templates

### Research Agent

```python
# templates/research_agent.py
from ai_scientist.agents.base import BaseAgent
from ai_scientist.agents.capabilities import ResearchCapability
from typing import Dict, Any, List
import asyncio

class ResearchAgent(BaseAgent):
    """Specialized agent for research tasks"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_type = "research"
        self.capabilities = [
            ResearchCapability(
                name="literature_review",
                description="Conduct comprehensive literature reviews"
            ),
            ResearchCapability(
                name="hypothesis_generation",
                description="Generate novel research hypotheses"
            ),
            ResearchCapability(
                name="experiment_design",
                description="Design scientific experiments"
            )
        ]

    async def initialize(self):
        """Initialize agent-specific resources"""
        await super().initialize()

        # Load domain knowledge
        await self.load_domain_knowledge()

        # Initialize research models
        await self.initialize_research_models()

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process research task"""
        task_type = task.get("type")

        if task_type == "literature_review":
            return await self.conduct_literature_review(task)
        elif task_type == "hypothesis_generation":
            return await self.generate_hypotheses(task)
        elif task_type == "experiment_design":
            return await self.design_experiment(task)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    async def conduct_literature_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct literature review"""
        query = task.get("query")
        databases = task.get("databases", ["pubmed", "arxiv"])
        max_papers = task.get("max_papers", 100)

        # Search for relevant papers
        papers = await self.search_literature(query, databases, max_papers)

        # Analyze and synthesize findings
        synthesis = await self.synthesize_literature(papers)

        # Generate summary and recommendations
        summary = await self.generate_summary(synthesis)

        return {
            "papers": papers,
            "synthesis": synthesis,
            "summary": summary,
            "recommendations": await self.generate_recommendations(synthesis)
        }

    async def generate_hypotheses(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research hypotheses"""
        domain = task.get("domain")
        context = task.get("context", {})
        constraints = task.get("constraints", {})

        # Analyze existing knowledge
        knowledge_analysis = await self.analyze_domain_knowledge(domain)

        # Generate initial hypotheses
        initial_hypotheses = await self.generate_initial_hypotheses(
            knowledge_analysis, context
        )

        # Evaluate and rank hypotheses
        evaluated_hypotheses = await self.evaluate_hypotheses(
            initial_hypotheses, constraints
        )

        return {
            "hypotheses": evaluated_hypotheses,
            "rationale": await self.generate_hypothesis_rationale(evaluated_hypotheses),
            "testability_scores": await self.assess_testability(evaluated_hypotheses)
        }

    async def design_experiment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Design scientific experiment"""
        hypothesis = task.get("hypothesis")
        domain = task.get("domain")
        resources = task.get("resources", {})

        # Design experimental methodology
        methodology = await self.design_methodology(hypothesis, domain, resources)

        # Define variables and controls
        variables = await self.define_variables(methodology)

        # Determine sample size and power
        power_analysis = await self.conduct_power_analysis(methodology, variables)

        # Create experimental protocol
        protocol = await self.create_experimental_protocol(methodology, variables)

        return {
            "methodology": methodology,
            "variables": variables,
            "power_analysis": power_analysis,
            "protocol": protocol,
            "ethical_considerations": await self.assess_ethical_considerations(protocol),
            "timeline": await self.estimate_timeline(protocol)
        }
```

### Data Analysis Agent

```python
# templates/data_analysis_agent.py
from ai_scientist.agents.base import BaseAgent
from ai_scientist.agents.capabilities import AnalysisCapability
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import asyncio

class DataAnalysisAgent(BaseAgent):
    """Specialized agent for data analysis tasks"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_type = "data_analysis"
        self.capabilities = [
            AnalysisCapability(
                name="statistical_analysis",
                description="Perform comprehensive statistical analysis"
            ),
            AnalysisCapability(
                name="machine_learning",
                description="Apply machine learning algorithms"
            ),
            AnalysisCapability(
                name="data_visualization",
                description="Create insightful data visualizations"
            ),
            AnalysisCapability(
                name="data_preprocessing",
                description="Clean and preprocess datasets"
            )
        ]

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process data analysis task"""
        task_type = task.get("type")

        if task_type == "statistical_analysis":
            return await self.perform_statistical_analysis(task)
        elif task_type == "machine_learning":
            return await self.apply_machine_learning(task)
        elif task_type == "visualization":
            return await self.create_visualizations(task)
        elif task_type == "preprocessing":
            return await self.preprocess_data(task)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    async def perform_statistical_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        data_path = task.get("data_path")
        analysis_type = task.get("analysis_type")
        variables = task.get("variables", [])

        # Load and validate data
        data = await self.load_data(data_path)
        validation_result = await self.validate_data(data)

        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.errors}")

        # Perform descriptive statistics
        descriptive_stats = await self.compute_descriptive_statistics(data, variables)

        # Perform inferential statistics
        inferential_stats = await self.perform_inferential_statistics(
            data, analysis_type, variables
        )

        # Check assumptions
        assumption_tests = await self.test_assumptions(data, analysis_type)

        # Generate interpretation
        interpretation = await self.interpret_results(
            descriptive_stats, inferential_stats, assumption_tests
        )

        return {
            "descriptive_statistics": descriptive_stats,
            "inferential_statistics": inferential_stats,
            "assumption_tests": assumption_tests,
            "interpretation": interpretation,
            "visualizations": await self.create_statistical_plots(data, variables)
        }

    async def apply_machine_learning(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply machine learning algorithms"""
        data_path = task.get("data_path")
        target_variable = task.get("target_variable")
        algorithms = task.get("algorithms", ["random_forest", "svm", "neural_network"])
        task_type = task.get("task_type", "classification")

        # Load and prepare data
        data = await self.load_data(data_path)
        X, y = await self.prepare_data_for_ml(data, target_variable, task_type)

        # Split data
        X_train, X_test, y_train, y_test = await self.split_data(X, y)

        results = {}

        for algorithm in algorithms:
            # Train model
            model = await self.train_model(X_train, y_train, algorithm, task_type)

            # Evaluate model
            evaluation = await self.evaluate_model(model, X_test, y_test, task_type)

            # Feature importance
            importance = await self.compute_feature_importance(model, X_train.columns)

            results[algorithm] = {
                "model": model,
                "evaluation": evaluation,
                "feature_importance": importance,
                "predictions": await self.make_predictions(model, X_test)
            }

        # Compare models
        comparison = await self.compare_models(results)

        # Select best model
        best_model = await self.select_best_model(results, comparison)

        return {
            "results": results,
            "comparison": comparison,
            "best_model": best_model,
            "recommendations": await self.generate_ml_recommendations(results, comparison)
        }
```

### Ethical Framework Agent

```python
# templates/ethical_framework_agent.py
from ai_scientist.agents.base import BaseAgent
from ai_scientist.agents.capabilities import EthicalCapability
from typing import Dict, Any, List
import asyncio

class EthicalFrameworkAgent(BaseAgent):
    """Specialized agent for ethical assessment and oversight"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent_type = "ethical_framework"
        self.capabilities = [
            EthicalCapability(
                name="ethical_assessment",
                description="Assess research ethical compliance"
            ),
            EthicalCapability(
                name="bias_detection",
                description="Detect and analyze potential biases"
            ),
            EthicalCapability(
                name="risk_assessment",
                description="Assess potential risks and harms"
            ),
            EthicalCapability(
                name="compliance_check",
                description="Check regulatory compliance"
            )
        ]

        # Initialize ethical frameworks
        self.frameworks = {
            "utilitarian": UtilitarianFramework(),
            "deontological": DeontologicalFramework(),
            "virtue": VirtueFramework(),
            "care": CareEthicsFramework(),
            "justice": JusticeFramework(),
            "precautionary": PrecautionaryFramework()
        }

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process ethical assessment task"""
        task_type = task.get("type")

        if task_type == "ethical_assessment":
            return await self.conduct_ethical_assessment(task)
        elif task_type == "bias_detection":
            return await self.detect_biases(task)
        elif task_type == "risk_assessment":
            return await self.assess_risks(task)
        elif task_type == "compliance_check":
            return await self.check_compliance(task)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    async def conduct_ethical_assessment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive ethical assessment"""
        resource = task.get("resource")
        resource_type = task.get("resource_type")  # experiment, agent, session
        frameworks = task.get("frameworks", list(self.frameworks.keys()))

        assessment_results = {}

        for framework_name in frameworks:
            framework = self.frameworks[framework_name]

            # Assess using specific framework
            framework_result = await framework.assess(resource, resource_type)

            assessment_results[framework_name] = {
                "score": framework_result.score,
                "weight": framework_result.weight,
                "assessment": framework_result.assessment,
                "issues": framework_result.issues,
                "recommendations": framework_result.recommendations
            }

        # Calculate overall score
        overall_score = await self.calculate_overall_score(assessment_results)

        # Identify critical issues
        critical_issues = await self.identify_critical_issues(assessment_results)

        # Generate recommendations
        recommendations = await self.generate_overall_recommendations(
            assessment_results, critical_issues
        )

        # Determine if human review is required
        human_review_required = await self.assess_human_review_need(
            overall_score, critical_issues
        )

        return {
            "overall_score": overall_score,
            "framework_results": assessment_results,
            "critical_issues": critical_issues,
            "recommendations": recommendations,
            "human_review_required": human_review_required,
            "compliance_status": await self.determine_compliance_status(overall_score),
            "assessment_metadata": {
                "assessed_by": self.agent_id,
                "assessment_version": "2.1",
                "frameworks_used": frameworks,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
```

## Core Agent Interface

### Base Agent Class

```python
# ai_scientist/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
from enum import Enum

class AgentStatus(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"

class BaseAgent(ABC):
    """Base class for all AI agents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", self.generate_agent_id())
        self.name = config.get("name", "Unnamed Agent")
        self.agent_type = config.get("agent_type", "generic")
        self.capabilities = []
        self.status = AgentStatus.INITIALIZING
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "error_count": 0,
            "last_activity": None
        }

        # State management
        self.state = {}
        self.current_task = None
        self.task_queue = asyncio.Queue()

        # Communication
        self.event_handlers = {}
        self.message_handlers = {}

    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources"""
        pass

    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities"""
        pass

    async def start(self):
        """Start the agent"""
        try:
            await self.initialize()
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.name} started successfully")

            # Start task processing loop
            asyncio.create_task(self.task_processing_loop())

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to start agent: {e}")
            raise

    async def stop(self):
        """Stop the agent"""
        self.status = AgentStatus.OFFLINE

        # Complete current task if possible
        if self.current_task:
            await self.handle_task_cancellation()

        self.logger.info(f"Agent {self.name} stopped")

    async def task_processing_loop(self):
        """Main task processing loop"""
        while self.status != AgentStatus.OFFLINE:
            try:
                # Wait for task
                task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )

                # Process task
                await self.execute_task(task)

            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
                self.metrics["error_count"] += 1

    async def execute_task(self, task: Dict[str, Any]):
        """Execute a single task"""
        self.status = AgentStatus.BUSY
        self.current_task = task

        start_time = asyncio.get_event_loop().time()

        try:
            # Log task start
            self.logger.info(f"Executing task: {task.get('type', 'unknown')}")

            # Process task
            result = await self.process_task(task)

            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self.update_metrics(True, execution_time)

            # Emit completion event
            await self.emit_event("task_completed", {
                "task_id": task.get("id"),
                "result": result,
                "execution_time": execution_time
            })

            return result

        except Exception as e:
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            self.update_metrics(False, execution_time)

            # Log error
            self.logger.error(f"Task execution failed: {e}")

            # Emit error event
            await self.emit_event("task_failed", {
                "task_id": task.get("id"),
                "error": str(e),
                "execution_time": execution_time
            })

            # Re-raise if critical
            if self.is_critical_error(e):
                raise

        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None
            self.metrics["last_activity"] = datetime.utcnow()

    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for processing"""
        task_id = task.get("id", self.generate_task_id())
        task["id"] = task_id
        task["submitted_at"] = datetime.utcnow().isoformat()

        await self.task_queue.put(task)

        self.logger.info(f"Task submitted: {task_id}")
        return task_id

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.value,
            "capabilities": self.get_capabilities(),
            "current_task": self.current_task.get("id") if self.current_task else None,
            "queue_size": self.task_queue.qsize(),
            "metrics": self.metrics,
            "last_activity": self.metrics["last_activity"].isoformat() if self.metrics["last_activity"] else None
        }

    def update_metrics(self, success: bool, execution_time: float):
        """Update agent performance metrics"""
        self.metrics["tasks_completed"] += 1

        # Update success rate
        total_tasks = self.metrics["tasks_completed"]
        if success:
            successful_tasks = total_tasks - self.metrics["error_count"]
        else:
            self.metrics["error_count"] += 1
            successful_tasks = total_tasks - self.metrics["error_count"]

        self.metrics["success_rate"] = successful_tasks / total_tasks if total_tasks > 0 else 0

        # Update average response time
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        )

    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")

    def on_event(self, event_type: str, handler):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def generate_agent_id(self) -> str:
        """Generate unique agent ID"""
        import uuid
        return f"agent_{uuid.uuid4().hex[:12]}"

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return f"task_{uuid.uuid4().hex[:12]}"

    def is_critical_error(self, error: Exception) -> bool:
        """Determine if error is critical"""
        critical_errors = [
            "AuthenticationError",
            "PermissionError",
            "ResourceNotFoundError"
        ]
        return any(err in str(type(error)) for err in critical_errors)
```

### Capability System

```python
# ai_scientist/agents/capabilities.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import asyncio

class Capability(ABC):
    """Base class for agent capabilities"""

    def __init__(self, name: str, description: str, **kwargs):
        self.name = name
        self.description = description
        self.parameters = kwargs.get("parameters", {})
        self.dependencies = kwargs.get("dependencies", [])
        self.version = kwargs.get("version", "1.0")

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the capability"""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate capability parameters"""
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get capability metadata"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "version": self.version
        }

class ResearchCapability(Capability):
    """Research-specific capability"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.domain = kwargs.get("domain", "general")
        self.methodology = kwargs.get("methodology", "mixed")

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research capability"""
        # Implementation specific to research capabilities
        pass

class AnalysisCapability(Capability):
    """Data analysis capability"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_type = kwargs.get("analysis_type", "statistical")
        self.supported_formats = kwargs.get("supported_formats", ["csv", "json", "parquet"])

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis capability"""
        # Implementation specific to analysis capabilities
        pass

class EthicalCapability(Capability):
    """Ethical assessment capability"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frameworks = kwargs.get("frameworks", ["utilitarian", "deontological"])
        self.strictness = kwargs.get("strictness", 0.8)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ethical capability"""
        # Implementation specific to ethical capabilities
        pass
```

## Agent Lifecycle Management

### Registration and Discovery

```python
# ai_scientist/agents/registry.py
from typing import Dict, List, Optional
import asyncio
import logging

class AgentRegistry:
    """Registry for managing agent instances"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_metadata: Dict[str, Dict] = {}
        self.logger = logging.getLogger("agent_registry")

    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent"""
        try:
            # Validate agent
            if not await self.validate_agent(agent):
                return False

            # Register agent
            self.agents[agent.agent_id] = agent
            self.agent_metadata[agent.agent_id] = {
                "name": agent.name,
                "type": agent.agent_type,
                "capabilities": agent.get_capabilities(),
                "registered_at": asyncio.get_event_loop().time()
            }

            self.logger.info(f"Agent registered: {agent.name} ({agent.agent_id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await agent.stop()
            del self.agents[agent_id]
            del self.agent_metadata[agent_id]

            self.logger.info(f"Agent unregistered: {agent_id}")
            return True
        return False

    def discover_agents(self, agent_type: Optional[str] = None,
                       capabilities: Optional[List[str]] = None) -> List[str]:
        """Discover agents based on criteria"""
        matching_agents = []

        for agent_id, metadata in self.agent_metadata.items():
            # Filter by type
            if agent_type and metadata["type"] != agent_type:
                continue

            # Filter by capabilities
            if capabilities:
                agent_caps = set(metadata["capabilities"])
                required_caps = set(capabilities)
                if not required_caps.issubset(agent_caps):
                    continue

            matching_agents.append(agent_id)

        return matching_agents

    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    async def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get agent status"""
        agent = self.agents.get(agent_id)
        if agent:
            return await agent.get_status()
        return None

    def list_agents(self) -> List[Dict]:
        """List all registered agents"""
        return [
            {
                "agent_id": agent_id,
                **metadata
            }
            for agent_id, metadata in self.agent_metadata.items()
        ]

    async def validate_agent(self, agent: BaseAgent) -> bool:
        """Validate agent before registration"""
        # Check required methods
        required_methods = ["initialize", "process_task", "get_capabilities"]
        for method in required_methods:
            if not hasattr(agent, method):
                self.logger.error(f"Agent missing required method: {method}")
                return False

        # Check capabilities
        capabilities = agent.get_capabilities()
        if not capabilities:
            self.logger.error("Agent must have at least one capability")
            return False

        return True
```

## Capability Development

### Creating Custom Capabilities

```python
# examples/custom_capabilities.py
from ai_scientist.agents.capabilities import Capability
from typing import Dict, Any
import asyncio

class ClimateModelingCapability(Capability):
    """Custom capability for climate modeling"""

    def __init__(self, **kwargs):
        super().__init__(
            name="climate_modeling",
            description="Perform climate modeling and simulation",
            parameters={
                "model_type": "string",
                "time_period": "string",
                "spatial_resolution": "string",
                "emissions_scenario": "string"
            },
            dependencies=["numpy", "xarray", "climate_models"],
            **kwargs
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute climate modeling"""
        model_type = context.get("model_type", "general_circulation")
        time_period = context.get("time_period", "2020-2100")
        spatial_resolution = context.get("spatial_resolution", "2.5deg")
        emissions_scenario = context.get("emissions_scenario", "RCP8.5")

        # Load climate data
        climate_data = await self.load_climate_data(context)

        # Initialize model
        model = await self.initialize_model(model_type, spatial_resolution)

        # Run simulation
        simulation_results = await self.run_simulation(
            model, climate_data, time_period, emissions_scenario
        )

        # Analyze results
        analysis = await self.analyze_results(simulation_results)

        # Generate visualizations
        visualizations = await self.create_visualizations(simulation_results)

        return {
            "model_type": model_type,
            "simulation_results": simulation_results,
            "analysis": analysis,
            "visualizations": visualizations,
            "metadata": {
                "time_period": time_period,
                "spatial_resolution": spatial_resolution,
                "emissions_scenario": emissions_scenario
            }
        }

    async def load_climate_data(self, context: Dict[str, Any]):
        """Load climate data for modeling"""
        # Implementation for loading climate data
        pass

    async def initialize_model(self, model_type: str, resolution: str):
        """Initialize climate model"""
        # Implementation for model initialization
        pass

    async def run_simulation(self, model, data, time_period: str, scenario: str):
        """Run climate simulation"""
        # Implementation for running simulation
        pass

    async def analyze_results(self, results):
        """Analyze simulation results"""
        # Implementation for result analysis
        pass

    async def create_visualizations(self, results):
        """Create visualizations of results"""
        # Implementation for creating visualizations
        pass

class DrugDiscoveryCapability(Capability):
    """Custom capability for drug discovery"""

    def __init__(self, **kwargs):
        super().__init__(
            name="drug_discovery",
            description="AI-assisted drug discovery and analysis",
            parameters={
                "target_protein": "string",
                "compound_library": "string",
                "screening_method": "string"
            },
            dependencies=["rdkit", "cheminformatics", "ml_models"],
            **kwargs
        )

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute drug discovery process"""
        target_protein = context.get("target_protein")
        compound_library = context.get("compound_library")
        screening_method = context.get("screening_method", "virtual_screening")

        # Load target protein structure
        protein_structure = await self.load_protein_structure(target_protein)

        # Prepare compound library
        compounds = await self.prepare_compound_library(compound_library)

        # Perform virtual screening
        screening_results = await self.perform_virtual_screening(
            protein_structure, compounds, screening_method
        )

        # Analyze hits
        hit_analysis = await self.analyze_hits(screening_results)

        # Predict ADMET properties
        admet_predictions = await self.predict_admet_properties(hit_analysis)

        return {
            "target_protein": target_protein,
            "screening_results": screening_results,
            "hit_analysis": hit_analysis,
            "admet_predictions": admet_predictions,
            "recommendations": await self.generate_recommendations(hit_analysis)
        }
```

## Testing and Validation

### Agent Testing Framework

```python
# tests/test_agents.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from ai_scientist.agents.base import BaseAgent
from ai_scientist.agents.registry import AgentRegistry

class TestAgent(BaseAgent):
    """Test agent for unit testing"""

    def __init__(self, config):
        super().__init__(config)
        self.agent_type = "test"
        self.test_results = []

    async def initialize(self):
        """Initialize test agent"""
        self.test_results.append("initialized")

    async def process_task(self, task):
        """Process test task"""
        task_type = task.get("type", "test")

        if task_type == "success_task":
            return {"status": "success", "result": "test_result"}
        elif task_type == "error_task":
            raise ValueError("Test error")
        else:
            return {"status": "unknown_task"}

    def get_capabilities(self):
        """Get test agent capabilities"""
        return ["test_capability"]

class TestAgentFramework:
    """Test suite for agent framework"""

    @pytest.fixture
    def agent_config(self):
        """Test agent configuration"""
        return {
            "name": "Test Agent",
            "agent_type": "test"
        }

    @pytest.fixture
    def test_agent(self, agent_config):
        """Create test agent"""
        return TestAgent(agent_config)

    @pytest.mark.asyncio
    async def test_agent_initialization(self, test_agent):
        """Test agent initialization"""
        await test_agent.initialize()
        assert "initialized" in test_agent.test_results
        assert test_agent.status.value == "idle"

    @pytest.mark.asyncio
    async def test_successful_task_execution(self, test_agent):
        """Test successful task execution"""
        await test_agent.initialize()

        task = {
            "type": "success_task",
            "id": "test_task_1"
        }

        result = await test_agent.execute_task(task)

        assert result["status"] == "success"
        assert result["result"] == "test_result"
        assert test_agent.metrics["tasks_completed"] == 1
        assert test_agent.metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_failed_task_execution(self, test_agent):
        """Test failed task execution"""
        await test_agent.initialize()

        task = {
            "type": "error_task",
            "id": "test_task_2"
        }

        with pytest.raises(ValueError):
            await test_agent.execute_task(task)

        assert test_agent.metrics["error_count"] == 1
        assert test_agent.metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_agent_registry(self, test_agent):
        """Test agent registry functionality"""
        registry = AgentRegistry()

        # Register agent
        success = await registry.register_agent(test_agent)
        assert success
        assert test_agent.agent_id in registry.agents

        # Discover agents
        discovered = registry.discover_agents(agent_type="test")
        assert test_agent.agent_id in discovered

        # Get agent status
        status = await registry.get_agent_status(test_agent.agent_id)
        assert status is not None
        assert status["name"] == "Test Agent"

        # Unregister agent
        success = await registry.unregister_agent(test_agent.agent_id)
        assert success
        assert test_agent.agent_id not in registry.agents

# Integration tests
class TestAgentIntegration:
    """Integration tests for agent system"""

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test coordination between multiple agents"""
        # Create research agent
        research_config = {"name": "Research Agent", "agent_type": "research"}
        research_agent = TestAgent(research_config)

        # Create ethical agent
        ethical_config = {"name": "Ethical Agent", "agent_type": "ethical"}
        ethical_agent = TestAgent(ethical_config)

        # Initialize agents
        await research_agent.initialize()
        await ethical_agent.initialize()

        # Create registry
        registry = AgentRegistry()
        await registry.register_agent(research_agent)
        await registry.register_agent(ethical_agent)

        # Test task coordination
        research_task = {
            "type": "success_task",
            "id": "coordination_test"
        }

        # Submit task to research agent
        task_id = await research_agent.submit_task(research_task)
        assert task_id is not None

        # Wait for completion
        await asyncio.sleep(0.1)

        # Verify task completion
        assert research_agent.metrics["tasks_completed"] == 1
```

### Performance Testing

```python
# tests/test_agent_performance.py
import pytest
import asyncio
import time
from ai_scientist.agents.base import BaseAgent

class PerformanceTestAgent(BaseAgent):
    """Agent for performance testing"""

    def __init__(self, config):
        super().__init__(config)
        self.agent_type = "performance_test"
        self.processing_delay = config.get("processing_delay", 0.1)

    async def initialize(self):
        pass

    async def process_task(self, task):
        # Simulate processing time
        await asyncio.sleep(self.processing_delay)
        return {"status": "completed", "task_id": task["id"]}

    def get_capabilities(self):
        return ["performance_test"]

class TestAgentPerformance:
    """Performance tests for agents"""

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self):
        """Test concurrent task processing"""
        config = {"name": "Performance Agent", "processing_delay": 0.01}
        agent = PerformanceTestAgent(config)
        await agent.initialize()

        # Submit multiple tasks concurrently
        tasks = []
        for i in range(10):
            task = {"type": "test", "id": f"task_{i}"}
            tasks.append(agent.submit_task(task))

        # Wait for all tasks to complete
        start_time = time.time()
        await asyncio.sleep(0.5)  # Allow time for processing

        # Check results
        assert agent.metrics["tasks_completed"] == 10
        assert agent.metrics["success_rate"] == 1.0

        processing_time = time.time() - start_time
        assert processing_time < 0.2  # Should be much less than sequential processing

    @pytest.mark.asyncio
    async def test_agent_memory_usage(self):
        """Test agent memory usage"""
        import psutil
        import os

        config = {"name": "Memory Test Agent"}
        agent = PerformanceTestAgent(config)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        await agent.initialize()

        # Process many tasks
        for i in range(100):
            task = {"type": "test", "id": f"task_{i}", "data": "x" * 1000}
            await agent.submit_task(task)

        await asyncio.sleep(2)  # Wait for processing

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
```

## Deployment and Registration

### Agent Deployment Script

```python
# scripts/deploy_agent.py
import asyncio
import argparse
import json
from pathlib import Path
from ai_scientist.agents.registry import AgentRegistry
from ai_scientist.agents.research_agent import ResearchAgent
from ai_scientist.agents.data_analysis_agent import DataAnalysisAgent
from ai_scientist.agents.ethical_framework_agent import EthicalFrameworkAgent

AGENT_CLASSES = {
    "research": ResearchAgent,
    "data_analysis": DataAnalysisAgent,
    "ethical_framework": EthicalFrameworkAgent
}

async def deploy_agent(config_file: str, registry_url: str = None):
    """Deploy an agent from configuration file"""

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create agent instance
    agent_type = config.get("agent_type")
    if agent_type not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_class = AGENT_CLASSES[agent_type]
    agent = agent_class(config)

    # Start agent
    await agent.start()

    # Register with registry if provided
    if registry_url:
        registry = AgentRegistry(registry_url)
        success = await registry.register_agent(agent)
        if not success:
            raise RuntimeError("Failed to register agent with registry")

    print(f"Agent deployed successfully: {agent.name} ({agent.agent_id})")

    # Keep agent running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
        await agent.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy AI Scientist Agent")
    parser.add_argument("config", help="Agent configuration file")
    parser.add_argument("--registry", help="Registry URL")
    args = parser.parse_args()

    asyncio.run(deploy_agent(args.config, args.registry))
```

### Agent Configuration Templates

```json
// configs/research_agent_template.json
{
  "agent_id": "auto_generated",
  "name": "Climate Research Agent",
  "agent_type": "research",
  "description": "Specialized agent for climate change research",
  "capabilities": [
    {
      "name": "climate_modeling",
      "parameters": {
        "default_model": "CESM2",
        "supported_scenarios": ["RCP4.5", "RCP8.5", "SSP1-2.6", "SSP5-8.5"]
      }
    },
    {
      "name": "data_analysis",
      "parameters": {
        "supported_formats": ["netcdf", "csv", "hdf5"],
        "max_file_size": "10GB"
      }
    }
  ],
  "resources": {
    "memory": "8GB",
    "cpu": 4,
    "gpu": false,
    "storage": "100GB"
  },
  "ethical_framework": {
    "enabled": true,
    "frameworks": ["utilitarian", "precautionary"],
    "strictness": 0.8,
    "auto_review": true
  },
  "communication": {
    "api_port": 8082,
    "websocket_port": 8083,
    "message_queue": "redis://localhost:6379/1"
  },
  "monitoring": {
    "metrics_enabled": true,
    "log_level": "INFO",
    "health_check_interval": 30
  }
}
```

This comprehensive Agent Development Guide provides everything needed to create, test, and deploy specialized AI agents within the AI-Scientist-v2 ecosystem, with detailed examples, testing frameworks, and best practices for production deployment.