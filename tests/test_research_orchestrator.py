"""
Comprehensive tests for Research Orchestrator Agent

This test suite validates the Research Orchestrator Agent's functionality including:
- Agent coordination and workflow management
- Service registry and agent discovery
- Research request processing
- Consensus building mechanisms
- API gateway functionality
- Performance and scalability
- Error handling and resilience
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the orchestrator components
from ai_scientist.orchestration.research_orchestrator_agent import (
    ResearchOrchestratorAgent,
    ResearchRequest,
    ResearchResults,
    ServiceRegistry,
    AgentCapability,
    AgentStatus,
    CoordinationMode,
    get_orchestrator,
    create_research_request,
    coordinate_research_simple
)

from ai_scientist.orchestration.api_gateway import (
    APIGateway,
    ResearchRequestModel,
    ResearchResponseModel,
    OrchestratorStatusModel,
    ConnectionManager
)


class TestServiceRegistry:
    """Test service registry functionality"""

    @pytest.fixture
    def service_registry(self):
        """Create service registry instance"""
        return ServiceRegistry()

    @pytest.fixture
    def sample_agent(self):
        """Create sample agent capability"""
        return AgentCapability(
            agent_id="test_agent_1",
            agent_type="research_specialist",
            capabilities=["data_analysis", "experimental_design"],
            status=AgentStatus.ACTIVE,
            performance_metrics={"success_rate": 0.9, "avg_response_time": 1.0},
            current_load=0,
            max_capacity=10,
            last_active=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_register_agent(self, service_registry, sample_agent):
        """Test agent registration"""
        result = await service_registry.register_agent(sample_agent)

        assert result is True
        assert sample_agent.agent_id in service_registry.registered_agents
        assert "data_analysis" in service_registry.agent_capabilities
        assert "experimental_design" in service_registry.agent_capabilities

    @pytest.mark.asyncio
    async def test_discover_agents(self, service_registry, sample_agent):
        """Test agent discovery"""
        # Register agent
        await service_registry.register_agent(sample_agent)

        # Discover with matching capabilities
        agents = await service_registry.discover_agents(["data_analysis"])

        assert len(agents) == 1
        assert agents[0].agent_id == sample_agent.agent_id

        # Discover with non-matching capabilities
        agents = await service_registry.discover_agents["nonexistent_capability"]

        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_agent_load_management(self, service_registry, sample_agent):
        """Test agent load management"""
        await service_registry.register_agent(sample_agent)

        # Increase load
        sample_agent.current_load = 8  # Near capacity

        # Should still be discoverable
        agents = await service_registry.discover_agents(["data_analysis"])
        assert len(agents) == 1

        # Max out capacity
        sample_agent.current_load = 10  # At capacity

        # Should not be discoverable
        agents = await service_registry.discover_agents(["data_analysis"])
        assert len(agents) == 0


class TestResearchOrchestratorAgent:
    """Test Research Orchestrator Agent functionality"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'max_concurrent_workflows': 5,
            'default_timeout': 300,  # 5 minutes for testing
            'consensus_threshold': 0.6,
            'max_agents_per_workflow': 3
        }

    @pytest.fixture
    async def orchestrator(self, config):
        """Create orchestrator instance"""
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()

    @pytest.fixture
    def sample_request(self):
        """Create sample research request"""
        return ResearchRequest(
            request_id="test_req_001",
            objective="Analyze the relationship between temperature and reaction rates",
            context={"experiment_type": "chemical_kinetics"},
            priority=5,
            coordination_mode=CoordinationMode.SEQUENTIAL,
            budget_constraints={"max_cost": 100.0},
            ethical_requirements={"human_subjects": False}
        )

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, config):
        """Test orchestrator initialization"""
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        assert orchestrator.agent_id is not None
        assert orchestrator.service_registry is not None
        assert orchestrator.supervisor is not None

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_research_coordination_sequential(self, orchestrator, sample_request):
        """Test sequential research coordination"""
        # Mock the supervisor to return successful results
        mock_workflow_plan = Mock()
        mock_workflow_plan.tasks = []
        mock_workflow_plan.created_at = datetime.now()

        with patch.object(orchestrator.supervisor, 'coordinate_workflow', return_value=mock_workflow_plan):
            with patch.object(orchestrator, '_discover_and_assign_agents', return_value=["agent_1"]):
                with patch.object(orchestrator, '_execute_workflow', return_value={
                    "success": True,
                    "results": {"analysis": "Temperature positively correlates with reaction rate"},
                    "agent_contributions": {"task_1": "agent_1"},
                    "confidence": 0.85
                }):
                    with patch.object(orchestrator.theory_agent, 'correlate_findings', return_value={
                        "theory_match": "Arrhenius equation",
                        "confidence": 0.9
                    }):
                        results = await orchestrator.coordinate_research(sample_request)

                        assert results.success is True
                        assert results.request_id == sample_request.request_id
                        assert "theoretical_insights" in results.results
                        assert results.confidence_score > 0.8

    @pytest.mark.asyncio
    async def test_agent_discovery_and_assignment(self, orchestrator, sample_request):
        """Test agent discovery and assignment logic"""
        # Register mock agents
        agent1 = AgentCapability(
            agent_id="agent_1",
            agent_type="data_analyst",
            capabilities=["data_analysis", "statistics"],
            status=AgentStatus.ACTIVE,
            performance_metrics={"success_rate": 0.9},
            current_load=2,
            max_capacity=10,
            last_active=datetime.now()
        )

        agent2 = AgentCapability(
            agent_id="agent_2",
            agent_type="experimentalist",
            capabilities=["experimental_design", "lab_simulation"],
            status=AgentStatus.ACTIVE,
            performance_metrics={"success_rate": 0.8},
            current_load=1,
            max_capacity=8,
            last_active=datetime.now()
        )

        await orchestrator.service_registry.register_agent(agent1)
        await orchestrator.service_registry.register_agent(agent2)

        assigned_agents = await orchestrator._discover_and_assign_agents(sample_request)

        assert len(assigned_agents) > 0
        assert "agent_1" in assigned_agents or "agent_2" in assigned_agents

    @pytest.mark.asyncio
    async def test_ethical_compliance_check(self, orchestrator, sample_request):
        """Test ethical compliance checking"""
        with patch.object(orchestrator.security_manager, 'check_research_compliance', return_value={
            "approved": True,
            "risk_level": "low",
            "compliance_score": 0.95
        }):
            result = await orchestrator._ethical_compliance_check(sample_request)

            assert result["approved"] is True
            assert result["risk_level"] == "low"

    @pytest.mark.asyncio
    async def test_ethical_compliance_failure(self, orchestrator, sample_request):
        """Test ethical compliance failure"""
        with patch.object(orchestrator.security_manager, 'check_research_compliance', return_value={
            "approved": False,
            "reason": "Human subjects research without proper authorization",
            "risk_level": "high"
        }):
            result = await orchestrator._ethical_compliance_check(sample_request)

            assert result["approved"] is False
            assert "risk_level" in result

    @pytest.mark.asyncio
    async def test_workflow_execution_parallel(self, orchestrator, sample_request):
        """Test parallel workflow execution"""
        # Modify request for parallel execution
        sample_request.coordination_mode = CoordinationMode.PARALLEL

        mock_workflow_plan = Mock()
        mock_task = Mock()
        mock_task.task_id = "task_1"
        mock_workflow_plan.tasks = [mock_task]
        mock_workflow_plan.created_at = datetime.now()

        with patch.object(orchestrator, 'agent_assignments', {sample_request.request_id: ["agent_1", "agent_2"]}):
            with patch.object(orchestrator.supervisor, 'delegate_to_specialist', return_value={
                "success": True,
                "result": "Parallel analysis completed",
                "confidence": 0.8
            }):
                results = await orchestrator._execute_parallel_workflow(mock_workflow_plan, sample_request)

                assert results["success"] is True
                assert "results" in results

    @pytest.mark.asyncio
    async def test_consensus_building(self, orchestrator):
        """Test consensus building mechanism"""
        agent_results = {
            "agent_1": {
                "success": True,
                "results": {"temperature_coefficient": 0.8},
                "confidence": 0.85
            },
            "agent_2": {
                "success": True,
                "results": {"temperature_coefficient": 0.9},
                "confidence": 0.75
            },
            "agent_3": {
                "success": True,
                "results": {"temperature_coefficient": 0.85},
                "confidence": 0.8
            }
        }

        consensus = await orchestrator._build_consensus(agent_results, None)

        assert consensus["success"] is True
        assert consensus["consensus_met"] is True
        assert "temperature_coefficient" in consensus["results"]

    @pytest.mark.asyncio
    async def test_consensus_failure(self, orchestrator):
        """Test consensus building failure"""
        agent_results = {
            "agent_1": {
                "success": True,
                "results": {"hypothesis": "A"},
                "confidence": 0.4
            },
            "agent_2": {
                "success": True,
                "results": {"hypothesis": "B"},
                "confidence": 0.3
            }
        }

        consensus = await orchestrator._build_consensus(agent_results, None)

        assert consensus["success"] is False
        assert consensus["consensus_met"] is False
        assert "threshold not met" in consensus["error"]

    @pytest.mark.asyncio
    async def test_orchestrator_status(self, orchestrator):
        """Test orchestrator status reporting"""
        status = await orchestrator.get_orchestrator_status()

        assert "orchestrator_id" in status
        assert "status" in status
        assert "active_workflows" in status
        assert "registered_agents" in status
        assert "success_rate" in status


class TestAPIGateway:
    """Test API Gateway functionality"""

    @pytest.fixture
    def api_config(self):
        """API configuration"""
        return {
            "orchestrator": {
                'max_concurrent_workflows': 5,
                'default_timeout': 300
            },
            "security": {
                "enable_auth": False  # Disable for testing
            },
            "cors_origins": ["*"]
        }

    @pytest.fixture
    async def api_gateway(self, api_config):
        """Create API gateway instance"""
        return APIGateway(api_config)

    @pytest.fixture
    def sample_request_data(self):
        """Sample API request data"""
        return {
            "objective": "Analyze protein folding patterns",
            "context": {"domain": "biochemistry"},
            "priority": 8,
            "coordination_mode": "sequential",
            "ethical_requirements": {"human_subjects": False}
        }

    @pytest.mark.asyncio
    async def test_api_gateway_initialization(self, api_config):
        """Test API gateway initialization"""
        gateway = APIGateway(api_config)
        await gateway._startup()

        assert gateway.orchestrator is not None
        assert gateway.security_manager is not None

        await gateway._shutdown()

    @pytest.mark.asyncio
    async def test_health_check(self, api_gateway):
        """Test health check endpoint"""
        health = await api_gateway.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "version" in health

    def test_request_model_validation(self):
        """Test request model validation"""
        valid_data = {
            "objective": "Test objective",
            "priority": 5,
            "coordination_mode": "sequential"
        }

        request = ResearchRequestModel(**valid_data)
        assert request.objective == "Test objective"
        assert request.priority == 5

    def test_invalid_request_model(self):
        """Test invalid request model"""
        invalid_data = {
            "objective": "",  # Empty objective should fail
            "priority": 15  # Priority too high
        }

        with pytest.raises(Exception):
            ResearchRequestModel(**invalid_data)


class TestConnectionManager:
    """Test WebSocket connection manager"""

    @pytest.fixture
    def connection_manager(self):
        """Create connection manager"""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_connection_management(self, connection_manager):
        """Test WebSocket connection management"""
        # Mock WebSocket
        mock_websocket = Mock()

        # Test connection
        await connection_manager.connect(mock_websocket)
        assert mock_websocket in connection_manager.active_connections

        # Test disconnection
        connection_manager.disconnect(mock_websocket)
        assert mock_websocket not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast(self, connection_manager):
        """Test message broadcasting"""
        # Mock multiple WebSockets
        mock_ws1 = Mock()
        mock_ws2 = Mock()
        mock_ws3 = Mock()

        await connection_manager.connect(mock_ws1)
        await connection_manager.connect(mock_ws2)
        await connection_manager.connect(mock_ws3)

        test_message = "Test broadcast message"
        await connection_manager.broadcast(test_message)

        # Verify all connections received the message
        mock_ws1.send_text.assert_called_once_with(test_message)
        mock_ws2.send_text.assert_called_once_with(test_message)
        mock_ws3.send_text.assert_called_once_with(test_message)


class TestPerformanceAndScalability:
    """Test performance and scalability aspects"""

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self):
        """Test concurrent workflow execution"""
        config = {'max_concurrent_workflows': 10, 'default_timeout': 60}
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            # Create multiple research requests
            requests = []
            for i in range(5):
                request = ResearchRequest(
                    request_id=f"concurrent_req_{i}",
                    objective=f"Test objective {i}",
                    coordination_mode=CoordinationMode.SEQUENTIAL
                )
                requests.append(request)

            # Mock execution to avoid actual agent calls
            with patch.object(orchestrator, '_discover_and_assign_agents', return_value=[f"agent_{i}"]):
                with patch.object(orchestrator, '_execute_workflow', return_value={
                    "success": True,
                    "results": {"test": f"result_{i}"},
                    "confidence": 0.8
                }):
                    with patch.object(orchestrator.theory_agent, 'correlate_findings', return_value={}):
                        # Execute workflows concurrently
                        tasks = [orchestrator.coordinate_research(req) for req in requests]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Verify all completed successfully
                        successful_results = [r for r in results if not isinstance(r, Exception)]
                        assert len(successful_results) == len(requests)

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_memory_usage_cleanup(self):
        """Test memory cleanup mechanisms"""
        config = {'max_concurrent_workflows': 3, 'default_timeout': 1}  # Short timeout
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            # Create and complete some workflows
            for i in range(3):
                request = ResearchRequest(
                    request_id=f"memory_test_{i}",
                    objective=f"Test objective {i}",
                    coordination_mode=CoordinationMode.SEQUENTIAL
                )

                # Add to active workflows (simulating execution)
                mock_workflow = Mock()
                mock_workflow.created_at = datetime.now() - timedelta(seconds=2)  # Past timeout
                orchestrator.active_workflows[request.request_id] = mock_workflow
                orchestrator.agent_assignments[request.request_id] = [f"agent_{i}"]

            # Trigger timeout check
            await orchestrator._check_workflow_timeouts()

            # Verify cleanup occurred
            assert len(orchestrator.active_workflows) == 0
            assert len(orchestrator.agent_assignments) == 0

        finally:
            await orchestrator.shutdown()


class TestErrorHandling:
    """Test error handling and resilience"""

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self):
        """Test handling of agent failures"""
        config = {'max_concurrent_workflows': 3, 'default_timeout': 60}
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            request = ResearchRequest(
                request_id="failure_test",
                objective="Test with agent failure",
                coordination_mode=CoordinationMode.SEQUENTIAL
            )

            # Mock agent discovery to return agents
            with patch.object(orchestrator, '_discover_and_assign_agents', return_value=["failing_agent"]):
                # Mock workflow execution to fail
                with patch.object(orchestrator, '_execute_workflow', return_value={
                    "success": False,
                    "error": "Agent execution failed"
                }):
                    results = await orchestrator.coordinate_research(request)

                    assert results.success is False
                    assert "error" in results.results

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling"""
        config = {'max_concurrent_workflows': 3, 'default_timeout': 0.1}  # Very short timeout
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            request = ResearchRequest(
                request_id="timeout_test",
                objective="Test timeout",
                coordination_mode=CoordinationMode.SEQUENTIAL
            )

            # Add a workflow that will timeout
            mock_workflow = Mock()
            mock_workflow.created_at = datetime.now()
            orchestrator.active_workflows[request.request_id] = mock_workflow

            # Wait for timeout
            await asyncio.sleep(0.2)

            # Check if workflow was cleaned up
            await orchestrator._check_workflow_timeouts()

            assert request.request_id not in orchestrator.active_workflows

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_exception_isolation(self):
        """Test that exceptions in one workflow don't affect others"""
        config = {'max_concurrent_workflows': 3, 'default_timeout': 60}
        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            # Create two requests - one that will fail, one that should succeed
            failing_request = ResearchRequest(
                request_id="failing_req",
                objective="This will fail",
                coordination_mode=CoordinationMode.SEQUENTIAL
            )

            successful_request = ResearchRequest(
                request_id="successful_req",
                objective="This should succeed",
                coordination_mode=CoordinationMode.SEQUENTIAL
            )

            # Mock the failing request to raise an exception
            with patch.object(orchestrator, '_discover_and_assign_agents', side_effect=Exception("Test exception")):
                # Execute failing request (should handle exception gracefully)
                failing_results = await orchestrator.coordinate_research(failing_request)
                assert failing_results.success is False

            # Mock successful execution for the second request
            with patch.object(orchestrator, '_discover_and_assign_agents', return_value=["good_agent"]):
                with patch.object(orchestrator, '_execute_workflow', return_value={
                    "success": True,
                    "results": {"test": "success"},
                    "confidence": 0.9
                }):
                    with patch.object(orchestrator.theory_agent, 'correlate_findings', return_value={}):
                        successful_results = await orchestrator.coordinate_research(successful_request)
                        assert successful_results.success is True

        finally:
            await orchestrator.shutdown()


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self):
        """Test complete end-to-end research workflow"""
        config = {
            'max_concurrent_workflows': 5,
            'default_timeout': 60,
            'consensus_threshold': 0.7
        }

        orchestrator = ResearchOrchestratorAgent(config)
        await orchestrator.initialize()

        try:
            # Register some test agents
            test_agent = AgentCapability(
                agent_id="integration_test_agent",
                agent_type="general_researcher",
                capabilities=["data_analysis", "theory_development"],
                status=AgentStatus.ACTIVE,
                performance_metrics={"success_rate": 0.9},
                current_load=0,
                max_capacity=5,
                last_active=datetime.now()
            )

            await orchestrator.service_registry.register_agent(test_agent)

            # Create research request
            request = ResearchRequest(
                request_id="integration_test",
                objective="Investigate the effects of catalyst concentration on reaction rate",
                context={"domain": "chemical_engineering"},
                priority=7,
                coordination_mode=CoordinationMode.SEQUENTIAL,
                ethical_requirements={"safety_concerns": "low"}
            )

            # Mock supervisor workflow creation
            mock_workflow = Mock()
            mock_workflow.tasks = []
            mock_workflow.created_at = datetime.now()

            with patch.object(orchestrator.supervisor, 'coordinate_workflow', return_value=mock_workflow):
                with patch.object(orchestrator.supervisor, 'delegate_to_specialist', return_value={
                    "success": True,
                    "result": "Catalyst concentration shows logarithmic relationship with reaction rate",
                    "confidence": 0.85
                }):
                    with patch.object(orchestrator.theory_agent, 'correlate_findings', return_value={
                        "matches": ["Michaelis-Menten kinetics"],
                        "confidence": 0.8
                    }):
                        results = await orchestrator.coordinate_research(request)

                        # Verify complete workflow
                        assert results.success is True
                        assert results.request_id == request.request_id
                        assert results.ethical_compliance["approved"] is True
                        assert "theoretical_insights" in results.results
                        assert results.execution_time > 0

                        # Verify agent was used
                        assert len(results.agent_contributions) > 0

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_api_integration(self):
        """Test API integration with orchestrator"""
        api_config = {
            "orchestrator": {
                'max_concurrent_workflows': 3,
                'default_timeout': 60
            },
            "security": {
                "enable_auth": False
            }
        }

        gateway = APIGateway(api_config)
        await gateway._startup()

        try:
            # Test health check
            health = await gateway.health_check()
            assert health["status"] == "healthy"

            # Test orchestrator status
            status = await gateway.get_orchestrator_status()
            assert "orchestrator_id" in status
            assert status["status"] == "active"

        finally:
            await gateway._shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])