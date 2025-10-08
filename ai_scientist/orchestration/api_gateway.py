"""
API Gateway for Research Orchestrator Agent

This module provides RESTful API interfaces for the Research Orchestrator Agent,
enabling external systems to interact with the multi-agent research ecosystem.
Includes authentication, rate limiting, real-time status updates, and comprehensive
error handling.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .research_orchestrator_agent import (
    ResearchOrchestratorAgent,
    ResearchRequest,
    ResearchResults,
    get_orchestrator,
    create_research_request,
    coordinate_research_simple
)
from ..security.security_manager import SecurityManager


# Pydantic models for API requests/responses
class ResearchRequestModel(BaseModel):
    objective: str = Field(..., description="Research objective to accomplish")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(default=1, ge=1, le=10, description="Request priority (1-10)")
    deadline: Optional[datetime] = Field(None, description="Optional deadline")
    coordination_mode: str = Field(default="sequential", description="Coordination strategy")
    required_agents: List[str] = Field(default_factory=list, description="Specific agents required")
    budget_constraints: Dict[str, Any] = Field(default_factory=dict, description="Budget limitations")
    ethical_requirements: Dict[str, Any] = Field(default_factory=dict, description="Ethical constraints")

class ResearchResponseModel(BaseModel):
    request_id: str
    success: bool
    results: Dict[str, Any]
    execution_time: float
    agent_contributions: Dict[str, Any]
    confidence_score: float
    ethical_compliance: Dict[str, Any]
    recommendations: List[str]
    next_steps: List[str]

class OrchestratorStatusModel(BaseModel):
    orchestrator_id: str
    status: str
    active_workflows: int
    total_workflows_completed: int
    registered_agents: int
    average_workflow_time: float
    success_rate: float
    agent_assignments: Dict[str, List[str]]
    performance_metrics: Dict[str, Any]

class AgentCapabilityModel(BaseModel):
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: str
    performance_metrics: Dict[str, float]
    current_load: int
    max_capacity: int
    availability_score: float

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    requests_count: int = 0
    window_start: datetime = None
    max_requests: int = 100
    window_seconds: int = 3600  # 1 hour

    def __post_init__(self):
        if self.window_start is None:
            self.window_start = datetime.now()

    def can_make_request(self) -> bool:
        """Check if request can be made within rate limit"""
        now = datetime.now()
        if (now - self.window_start).total_seconds() > self.window_seconds:
            # Reset window
            self.requests_count = 0
            self.window_start = now
            return True

        return self.requests_count < self.max_requests

    def record_request(self):
        """Record a request"""
        self.requests_count += 1


class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed, remove it
                self.active_connections.remove(connection)


class APIGateway:
    """API Gateway for Research Orchestrator Agent"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.orchestrator: Optional[ResearchOrchestratorAgent] = None
        self.security_manager: Optional[SecurityManager] = None
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.connection_manager = ConnectionManager()
        self.api_key_validity: Dict[str, datetime] = {}

        # FastAPI app
        self.app = FastAPI(
            title="Research Orchestrator API",
            description="RESTful API for multi-agent research coordination",
            version="1.0.0",
            lifespan=self.lifespan
        )

        # Security
        self.security = HTTPBearer()

        # Setup middleware
        self._setup_middleware()

        # Setup routes
        self._setup_routes()

        self.logger = logging.getLogger(f"{__name__}.APIGateway")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifespan context manager for FastAPI app"""
        # Startup
        await self._startup()
        yield
        # Shutdown
        await self._shutdown()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""
        self.app.post("/research", response_model=ResearchResponseModel)(self.create_research_request)
        self.app.get("/research/{request_id}", response_model=ResearchResponseModel)(self.get_research_results)
        self.app.get("/status", response_model=OrchestratorStatusModel)(self.get_orchestrator_status)
        self.app.get("/agents", response_model=List[AgentCapabilityModel])(self.get_available_agents)
        self.app.post("/agents/register", response_model=dict)(self.register_agent)
        self.app.get("/health")(self.health_check)
        self.app.websocket("/ws")(self.websocket_endpoint)
        self.app.get("/metrics")(self.get_metrics)

    async def _startup(self):
        """Startup initialization"""
        try:
            # Initialize orchestrator
            self.orchestrator = await get_orchestrator(self.config.get("orchestrator", {}))

            # Initialize security manager
            self.security_manager = SecurityManager(self.config.get("security", {}))

            self.logger.info("API Gateway started successfully")

        except Exception as e:
            self.logger.error(f"API Gateway startup failed: {e}")
            raise

    async def _shutdown(self):
        """Shutdown cleanup"""
        try:
            if self.orchestrator:
                await self.orchestrator.shutdown()

            self.logger.info("API Gateway shutdown successfully")

        except Exception as e:
            self.logger.error(f"API Gateway shutdown failed: {e}")

    async def _authenticate(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Authenticate API requests"""
        try:
            api_key = credentials.credentials

            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")

            # Check rate limiting
            if not await self._check_rate_limit(api_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            return api_key

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")

    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        try:
            # Simple API key validation (in production, use proper authentication)
            if api_key.startswith("sk-") and len(api_key) > 20:
                # Mark as valid for 1 hour
                self.api_key_validity[api_key] = datetime.now() + timedelta(hours=1)
                return True

            # Check if already validated
            if api_key in self.api_key_validity:
                if datetime.now() < self.api_key_validity[api_key]:
                    return True
                else:
                    # Expired
                    del self.api_key_validity[api_key]

            return False

        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

    async def _check_rate_limit(self, api_key: str) -> bool:
        """Check rate limiting for API key"""
        try:
            if api_key not in self.rate_limits:
                self.rate_limits[api_key] = RateLimitInfo()

            return self.rate_limits[api_key].can_make_request()

        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return False

    async def create_research_request(
        self,
        request: ResearchRequestModel,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(_authenticate)
    ) -> ResearchResponseModel:
        """Create and execute a research request"""
        try:
            # Record rate limit usage
            if api_key in self.rate_limits:
                self.rate_limits[api_key].record_request()

            # Convert to internal request format
            research_request = ResearchRequest(
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                objective=request.objective,
                context=request.context,
                priority=request.priority,
                deadline=request.deadline,
                coordination_mode=getattr(request.coordination_mode, "SEQUENTIAL"),
                required_agents=request.required_agents,
                budget_constraints=request.budget_constraints,
                ethical_requirements=request.ethical_requirements
            )

            # Broadcast start of research
            await self.connection_manager.broadcast(json.dumps({
                "event": "research_started",
                "request_id": research_request.request_id,
                "objective": request.objective,
                "timestamp": datetime.now().isoformat()
            }))

            # Execute research
            results = await self.orchestrator.coordinate_research(research_request)

            # Broadcast completion
            await self.connection_manager.broadcast(json.dumps({
                "event": "research_completed",
                "request_id": research_request.request_id,
                "success": results.success,
                "execution_time": results.execution_time,
                "timestamp": datetime.now().isoformat()
            }))

            # Convert to response model
            return ResearchResponseModel(
                request_id=results.request_id,
                success=results.success,
                results=results.results,
                execution_time=results.execution_time,
                agent_contributions=results.agent_contributions,
                confidence_score=results.confidence_score,
                ethical_compliance=results.ethical_compliance,
                recommendations=results.recommendations,
                next_steps=results.next_steps
            )

        except Exception as e:
            self.logger.error(f"Research request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_research_results(
        self,
        request_id: str,
        api_key: str = Depends(_authenticate)
    ) -> ResearchResponseModel:
        """Get results for a specific research request"""
        try:
            # For now, return error since we don't store request history
            # In a production system, this would query a database
            raise HTTPException(status_code=404, detail="Research request not found")

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get research results: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_orchestrator_status(
        self,
        api_key: str = Depends(_authenticate)
    ) -> OrchestratorStatusModel:
        """Get orchestrator status"""
        try:
            status = await self.orchestrator.get_orchestrator_status()

            return OrchestratorStatusModel(**status)

        except Exception as e:
            self.logger.error(f"Failed to get orchestrator status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_available_agents(
        self,
        api_key: str = Depends(_authenticate)
    ) -> List[AgentCapabilityModel]:
        """Get list of available agents"""
        try:
            agents = []
            for agent_id, agent in self.orchestrator.service_registry.registered_agents.items():
                agent_model = AgentCapabilityModel(
                    agent_id=agent.agent_id,
                    agent_type=agent.agent_type,
                    capabilities=agent.capabilities,
                    status=agent.status.value,
                    performance_metrics=agent.performance_metrics,
                    current_load=agent.current_load,
                    max_capacity=agent.max_capacity,
                    availability_score=agent.availability_score
                )
                agents.append(agent_model)

            return agents

        except Exception as e:
            self.logger.error(f"Failed to get available agents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def register_agent(
        self,
        agent_data: Dict[str, Any],
        api_key: str = Depends(_authenticate)
    ) -> Dict[str, str]:
        """Register a new agent (for internal use)"""
        try:
            # This would typically be called by agents themselves
            # For now, return success message
            return {"message": "Agent registration endpoint (internal use)"}

        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "orchestrator_ready": self.orchestrator is not None,
                "security_ready": self.security_manager is not None,
                "version": "1.0.0"
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        try:
            await self.connection_manager.connect(websocket)

            # Send initial status
            status = await self.orchestrator.get_orchestrator_status()
            await websocket.send_text(json.dumps({
                "event": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }))

            # Keep connection alive and listen for messages
            while True:
                try:
                    # Wait for messages (client can send ping or subscribe to events)
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except WebSocketDisconnect:
                    break

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.connection_manager.disconnect(websocket)

    async def get_metrics(self, api_key: str = Depends(_authenticate)) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            # Get orchestrator metrics
            status = await self.orchestrator.get_orchestrator_status()

            # Calculate additional metrics
            metrics = {
                "system_metrics": status["performance_metrics"],
                "api_metrics": {
                    "active_api_keys": len(self.api_key_validity),
                    "rate_limited_keys": len([k for k, v in self.rate_limits.items() if not v.can_make_request()]),
                    "websocket_connections": len(self.connection_manager.active_connections)
                },
                "timestamp": datetime.now().isoformat()
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# FastAPI app instance (for direct use)
api_gateway = APIGateway()
app = api_gateway.app

# Convenience functions for standalone deployment
async def start_api_server(config: Dict[str, Any] = None, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    import uvicorn

    gateway = APIGateway(config)

    await uvicorn.run(
        gateway.app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    # Example usage
    asyncio.run(start_api_server())