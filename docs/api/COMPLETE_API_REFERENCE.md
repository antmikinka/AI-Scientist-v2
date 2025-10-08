# Complete API Reference

## Overview

This document provides a comprehensive reference for the AI-Scientist-v2 REST API and WebSocket interfaces. The API enables programmatic access to all multi-agent system capabilities including research orchestration, agent management, ethical oversight, and data processing.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Base URL and Endpoints](#base-url-and-endpoints)
4. [Common Response Formats](#common-response-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [REST API Endpoints](#rest-api-endpoints)
   - [Authentication Endpoints](#authentication-endpoints)
   - [Agent Management](#agent-management)
   - [Research Orchestration](#research-orchestration)
   - [Experiments](#experiments)
   - [Data Management](#data-management)
   - [Ethical Framework](#ethical-framework)
   - [Monitoring](#monitoring)
8. [WebSocket API](#websocket-api)
9. [API Versioning](#api-versioning)
10. [SDK Examples](#sdk-examples)

## API Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   REST API  │  │   WebSocket │  │   GraphQL (Future)  │  │
│  │             │  │     API     │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Authentication                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │     JWT     │  │   OAuth 2   │  │    API Keys         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Multi-Agent System                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Research    │  │ Ethical     │  │ Specialized         │  │
│  │ Orchestrator│  │ Framework   │  │ Agents              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Supported Operations

- **Agent Management**: Register, configure, and monitor AI agents
- **Research Orchestration**: Create and manage research workflows
- **Experiment Management**: Design, execute, and analyze experiments
- **Data Processing**: Upload, process, and analyze research data
- **Ethical Oversight**: Real-time ethical compliance monitoring
- **System Monitoring**: Performance metrics and health status

## Authentication

### JWT Authentication

```http
Authorization: Bearer <jwt_token>
```

#### JWT Token Structure

```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_id",
    "exp": 1640995200,
    "iat": 1640908800,
    "permissions": ["read", "write", "admin"],
    "session_id": "session_uuid"
  }
}
```

#### Login Endpoint

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
      "id": "user_uuid",
      "username": "user@example.com",
      "permissions": ["read", "write"]
    }
  }
}
```

### API Key Authentication

```http
X-API-Key: your_api_key_here
```

#### API Key Generation

```http
POST /api/v1/auth/api-keys
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "My Application Key",
  "permissions": ["agents:read", "experiments:write"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

## Base URL and Endpoints

### Base URLs

- **Development**: `http://localhost:8080/api/v1`
- **Staging**: `https://staging.ai-scientist.com/api/v1`
- **Production**: `https://api.ai-scientist.com/api/v1`

### WebSocket URL

- **Development**: `ws://localhost:8080/ws`
- **Staging**: `wss://staging.ai-scientist.com/ws`
- **Production**: `wss://api.ai-scientist.com/ws`

## Common Response Formats

### Success Response

```json
{
  "status": "success",
  "data": {
    // Response data here
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_uuid",
    "version": "v1"
  }
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "agent_type",
      "message": "Invalid agent type specified"
    }
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_uuid",
    "version": "v1"
  }
}
```

### Paginated Response

```json
{
  "status": "success",
  "data": [
    // Array of items
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_uuid",
    "version": "v1"
  }
}
```

## Error Handling

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `AGENT_ERROR` | 422 | Agent execution failed |
| `ETHICAL_VIOLATION` | 451 | Ethical compliance issue |

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "errors": [
        {
          "field": "agent_name",
          "message": "Agent name is required",
          "code": "REQUIRED_FIELD"
        },
        {
          "field": "agent_type",
          "message": "Invalid agent type",
          "code": "INVALID_VALUE"
        }
      ]
    }
  },
  "meta": {
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_uuid",
    "version": "v1"
  }
}
```

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### Rate Limits by Endpoint

| Endpoint Type | Rate Limit | Window |
|---------------|------------|--------|
| Authentication | 10 requests | 1 minute |
| Agent Management | 100 requests | 1 minute |
| Research Operations | 50 requests | 1 minute |
| Data Upload | 20 requests | 1 minute |
| Monitoring | 1000 requests | 1 minute |

## REST API Endpoints

### Authentication Endpoints

#### Login

```http
POST /api/v1/auth/login
```

**Request Body:**
```json
{
  "username": "string",
  "password": "string",
  "remember_me": false
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "access_token": "string",
    "refresh_token": "string",
    "token_type": "bearer",
    "expires_in": 3600,
    "user": {
      "id": "string",
      "username": "string",
      "email": "string",
      "permissions": ["string"]
    }
  }
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

#### Logout

```http
POST /api/v1/auth/logout
Authorization: Bearer <jwt_token>
```

#### Get User Profile

```http
GET /api/v1/auth/profile
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "string",
    "username": "string",
    "email": "string",
    "permissions": ["string"],
    "created_at": "2024-01-01T12:00:00Z",
    "last_login": "2024-01-01T12:00:00Z"
  }
}
```

### Agent Management

#### List Agents

```http
GET /api/v1/agents?page=1&per_page=20&status=active&type=research
Authorization: Bearer <jwt_token>
```

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (active, inactive, error)
- `type` (string): Filter by agent type
- `search` (string): Search term

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "agent_uuid",
      "name": "Research Agent Alpha",
      "type": "research",
      "status": "active",
      "capabilities": ["text_generation", "analysis"],
      "created_at": "2024-01-01T12:00:00Z",
      "last_active": "2024-01-01T12:30:00Z",
      "performance": {
        "tasks_completed": 150,
        "success_rate": 0.95,
        "avg_response_time": 2.5
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 50,
    "total_pages": 3
  }
}
```

#### Register Agent

```http
POST /api/v1/agents
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "string",
  "type": "research|ethical|analysis|data_processing",
  "description": "string",
  "capabilities": ["string"],
  "configuration": {
    "model": "string",
    "parameters": {},
    "resources": {
      "memory": 2048,
      "cpu": 2
    }
  },
  "ethical_framework": {
    "enabled": true,
    "frameworks": ["utilitarian", "deontological"],
    "strictness": 0.8
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "agent_uuid",
    "name": "string",
    "type": "string",
    "status": "pending",
    "created_at": "2024-01-01T12:00:00Z",
    "api_key": "agent_api_key"
  }
}
```

#### Get Agent Details

```http
GET /api/v1/agents/{agent_id}
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "agent_uuid",
    "name": "string",
    "type": "string",
    "status": "active",
    "description": "string",
    "capabilities": ["string"],
    "configuration": {},
    "ethical_framework": {},
    "performance": {
      "tasks_completed": 150,
      "success_rate": 0.95,
      "avg_response_time": 2.5,
      "last_error": null
    },
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  }
}
```

#### Update Agent

```http
PUT /api/v1/agents/{agent_id}
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

#### Delete Agent

```http
DELETE /api/v1/agents/{agent_id}
Authorization: Bearer <jwt_token>
```

#### Execute Agent Task

```http
POST /api/v1/agents/{agent_id}/execute
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "task": "string",
  "parameters": {},
  "context": {
    "session_id": "string",
    "priority": "normal|high|urgent",
    "timeout": 300
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "task_id": "task_uuid",
    "status": "queued",
    "estimated_duration": 120,
    "queue_position": 2
  }
}
```

#### Get Agent Status

```http
GET /api/v1/agents/{agent_id}/status
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "agent_id": "agent_uuid",
    "status": "active|busy|idle|error|offline",
    "current_task": {
      "id": "task_uuid",
      "type": "string",
      "progress": 0.65,
      "started_at": "2024-01-01T12:00:00Z",
      "estimated_completion": "2024-01-01T12:05:00Z"
    },
    "resource_usage": {
      "cpu": 0.45,
      "memory": 0.67,
      "network": 0.12
    },
    "health_metrics": {
      "response_time": 1.2,
      "error_rate": 0.01,
      "uptime": 0.99
    }
  }
}
```

### Research Orchestration

#### Create Research Session

```http
POST /api/v1/research/sessions
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "title": "string",
  "description": "string",
  "domain": "string",
  "objectives": ["string"],
  "agents": [
    {
      "id": "agent_uuid",
      "role": "string"
    }
  ],
  "coordination_mode": "sequential|parallel|consensus|competitive",
  "ethical_requirements": {
    "human_review_required": true,
    "data_privacy": "high",
    "bias_detection": true
  },
  "timeline": {
    "start_date": "2024-01-01T12:00:00Z",
    "duration_days": 30,
    "milestones": [
      {
        "name": "string",
        "due_date": "2024-01-15T12:00:00Z"
      }
    ]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "session_uuid",
    "title": "string",
    "status": "initialized",
    "created_at": "2024-01-01T12:00:00Z",
    "agents": [
      {
        "id": "agent_uuid",
        "name": "string",
        "role": "string"
      }
    ]
  }
}
```

#### Get Research Session

```http
GET /api/v1/research/sessions/{session_id}
Authorization: Bearer <jwt_token>
```

#### List Research Sessions

```http
GET /api/v1/research/sessions?page=1&per_page=20&status=active
Authorization: Bearer <jwt_token>
```

#### Start Research Session

```http
POST /api/v1/research/sessions/{session_id}/start
Authorization: Bearer <jwt_token>
```

#### Pause Research Session

```http
POST /api/v1/research/sessions/{session_id}/pause
Authorization: Bearer <jwt_token>
```

#### Stop Research Session

```http
POST /api/v1/research/sessions/{session_id}/stop
Authorization: Bearer <jwt_token>
```

#### Get Research Progress

```http
GET /api/v1/research/sessions/{session_id}/progress
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "session_id": "session_uuid",
    "overall_progress": 0.45,
    "phase": "ideation",
    "current_activities": [
      {
        "agent_id": "agent_uuid",
        "activity": "Generating research hypotheses",
        "progress": 0.8
      }
    ],
    "milestones": [
      {
        "name": "Ideation Complete",
        "completed": true,
        "completed_at": "2024-01-01T12:30:00Z"
      },
      {
        "name": "Experiment Design",
        "completed": false,
        "progress": 0.3
      }
    ],
    "timeline": {
      "started_at": "2024-01-01T12:00:00Z",
      "estimated_completion": "2024-01-01T18:00:00Z",
      "time_elapsed": 1800,
      "time_remaining": 21600
    }
  }
}
```

### Experiments

#### Create Experiment

```http
POST /api/v1/experiments
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "session_uuid",
  "title": "string",
  "hypothesis": "string",
  "methodology": "string",
  "data_requirements": {
    "datasets": ["string"],
    "sample_size": 1000,
    "features": ["string"]
  },
  "agents": [
    {
      "id": "agent_uuid",
      "role": "data_analyst|experiment_runner"
    }
  ],
  "parameters": {},
  "ethical_review": {
    "auto_review": true,
    "human_review": false,
    "risk_level": "low"
  }
}
```

#### Execute Experiment

```http
POST /api/v1/experiments/{experiment_id}/execute
Authorization: Bearer <jwt_token>
```

#### Get Experiment Results

```http
GET /api/v1/experiments/{experiment_id}/results
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "experiment_id": "experiment_uuid",
    "status": "completed",
    "results": {
      "summary": "string",
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90
      },
      "visualizations": [
        {
          "type": "scatter_plot",
          "url": "https://.../plot.png",
          "description": "string"
        }
      ],
      "data": {
        "raw_data_url": "https://.../data.csv",
        "processed_data_url": "https://.../processed.json"
      },
      "statistical_analysis": {
        "p_value": 0.001,
        "confidence_interval": [0.92, 0.98],
        "effect_size": 0.75
      }
    },
    "execution_time": 1200,
    "completed_at": "2024-01-01T12:20:00Z"
  }
}
```

### Data Management

#### Upload Data

```http
POST /api/v1/data/upload
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: Binary file data
- `metadata`: JSON metadata string
- `session_id`: Research session ID (optional)

**Response:**
```json
{
  "status": "success",
  "data": {
    "data_id": "data_uuid",
    "filename": "dataset.csv",
    "size": 1048576,
    "type": "csv",
    "checksum": "sha256_hash",
    "uploaded_at": "2024-01-01T12:00:00Z",
    "processed": false
  }
}
```

#### Get Data Details

```http
GET /api/v1/data/{data_id}
Authorization: Bearer <jwt_token>
```

#### Process Data

```http
POST /api/v1/data/{data_id}/process
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "processing_type": "cleaning|normalization|feature_extraction|analysis",
  "parameters": {},
  "output_format": "csv|json|parquet"
}
```

#### List Data

```http
GET /api/v1/data?page=1&per_page=20&type=csv&session_id=session_uuid
Authorization: Bearer <jwt_token>
```

### Ethical Framework

#### Get Ethical Assessment

```http
GET /api/v1/ethical/assessment/{resource_id}
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "resource_id": "resource_uuid",
    "resource_type": "experiment|agent|research_session",
    "assessment": {
      "overall_score": 0.85,
      "status": "compliant",
      "frameworks": {
        "utilitarian": {
          "score": 0.9,
          "assessment": "Maximizes overall benefit"
        },
        "deontological": {
          "score": 0.8,
          "assessment": "Follows ethical principles"
        },
        "care_ethics": {
          "score": 0.85,
          "assessment": "Considers impact on stakeholders"
        }
      },
      "identified_issues": [],
      "recommendations": [
        "Consider including diverse perspectives",
        "Validate findings with external review"
      ],
      "human_review_required": false,
      "assessed_at": "2024-01-01T12:00:00Z"
    }
  }
}
```

#### Request Ethical Review

```http
POST /api/v1/ethical/review
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "resource_id": "string",
  "resource_type": "experiment|agent|research_session",
  "priority": "normal|high|urgent",
  "specific_concerns": ["string"],
  "auto_approve": false
}
```

#### Get Ethical Guidelines

```http
GET /api/v1/ethical/guidelines?domain=research&framework=all
Authorization: Bearer <jwt_token>
```

### Monitoring

#### Get System Health

```http
GET /api/v1/monitoring/health
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "overall_status": "healthy",
    "services": {
      "api_gateway": {
        "status": "healthy",
        "response_time": 45,
        "uptime": 0.999
      },
      "database": {
        "status": "healthy",
        "response_time": 12,
        "connections": 15
      },
      "redis": {
        "status": "healthy",
        "response_time": 2,
        "memory_usage": 0.45
      },
      "agents": {
        "status": "healthy",
        "active_agents": 8,
        "total_tasks": 150,
        "success_rate": 0.95
      }
    },
    "metrics": {
      "requests_per_minute": 250,
      "error_rate": 0.01,
      "average_response_time": 120
    }
  }
}
```

#### Get Performance Metrics

```http
GET /api/v1/monitoring/metrics?start_time=2024-01-01T00:00:00Z&end_time=2024-01-01T23:59:59Z&metric=cpu,memory,requests
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "time_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-01T23:59:59Z"
    },
    "metrics": {
      "cpu": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 0.65
        }
      ],
      "memory": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 0.78
        }
      ],
      "requests_per_minute": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 250
        }
      ]
    }
  }
}
```

#### Get Agent Performance

```http
GET /api/v1/monitoring/agents/{agent_id}/performance?period=24h
Authorization: Bearer <jwt_token>
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('wss://api.ai-scientist.com/ws');
ws.onopen = function() {
    // Authenticate
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'jwt_token_here'
    }));
};
```

### Message Format

```json
{
  "type": "string",
  "id": "message_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {}
}
```

### WebSocket Message Types

#### Authentication

```json
{
  "type": "auth",
  "data": {
    "token": "jwt_token_here"
  }
}
```

**Response:**
```json
{
  "type": "auth_response",
  "data": {
    "status": "success|error",
    "session_id": "session_uuid"
  }
}
```

#### Subscribe to Events

```json
{
  "type": "subscribe",
  "data": {
    "events": ["agent_status", "research_progress", "experiment_results"],
    "filters": {
      "session_id": "session_uuid",
      "agent_id": "agent_uuid"
    }
  }
}
```

#### Agent Status Updates

```json
{
  "type": "agent_status",
  "data": {
    "agent_id": "agent_uuid",
    "status": "busy",
    "current_task": "task_uuid",
    "progress": 0.65,
    "resource_usage": {
      "cpu": 0.45,
      "memory": 0.67
    }
  }
}
```

#### Research Progress Updates

```json
{
  "type": "research_progress",
  "data": {
    "session_id": "session_uuid",
    "overall_progress": 0.45,
    "current_phase": "experimentation",
    "milestone_reached": {
      "name": "Ideation Complete",
      "reached_at": "2024-01-01T12:30:00Z"
    }
  }
}
```

#### Experiment Results

```json
{
  "type": "experiment_results",
  "data": {
    "experiment_id": "experiment_uuid",
    "status": "completed",
    "results": {
      "summary": "string",
      "metrics": {},
      "visualizations": []
    }
  }
}
```

#### Ethical Assessment Updates

```json
{
  "type": "ethical_assessment",
  "data": {
    "resource_id": "resource_uuid",
    "assessment_score": 0.85,
    "issues_identified": [],
    "human_review_required": false
  }
}
```

## API Versioning

### Version Strategy

The API uses semantic versioning with URL-based versioning:

- `/api/v1/` - Current stable version
- `/api/v2/` - Next version (when available)
- `/api/latest/` - Always points to latest stable version

### Version Headers

```http
API-Version: v1
Supported-Versions: v1, v2
Deprecated-Versions: v0
```

### Version Deprecation

- Versions are supported for at least 12 months
- 6-month deprecation notice before removal
- Migration guides provided for major updates

## SDK Examples

### Python SDK

```python
from ai_scientist_sdk import AIScientistClient

# Initialize client
client = AIScientistClient(
    base_url="https://api.ai-scientist.com",
    api_key="your_api_key_here"
)

# Create research session
session = client.research.create_session(
    title="Climate Change Impact Study",
    description="Analyzing the impact of climate change on biodiversity",
    domain="environmental_science",
    objectives=["Analyze temperature trends", "Assess species impact"],
    coordination_mode="parallel"
)

# Register agents
research_agent = client.agents.register(
    name="Climate Research Agent",
    type="research",
    capabilities=["data_analysis", "modeling"]
)

ethical_agent = client.agents.register(
    name="Ethical Review Agent",
    type="ethical",
    capabilities=["ethical_analysis", "bias_detection"]
)

# Add agents to session
client.research.add_agents(
    session_id=session.id,
    agents=[
        {"id": research_agent.id, "role": "primary_researcher"},
        {"id": ethical_agent.id, "role": "ethical_reviewer"}
    ]
)

# Start research
client.research.start_session(session.id)

# Monitor progress with WebSocket
def on_progress(data):
    print(f"Progress: {data['overall_progress']:.1%}")
    print(f"Current phase: {data['current_phase']}")

client.websocket.subscribe(
    events=["research_progress"],
    filters={"session_id": session.id},
    callback=on_progress
)

# Wait for completion
import time
while True:
    progress = client.research.get_progress(session.id)
    if progress.overall_progress >= 1.0:
        break
    time.sleep(10)

# Get results
results = client.research.get_results(session.id)
print(f"Research completed: {results.summary}")
```

### JavaScript SDK

```javascript
import { AIScientistClient } from '@ai-scientist/sdk';

// Initialize client
const client = new AIScientistClient({
  baseURL: 'https://api.ai-scientist.com',
  apiKey: 'your_api_key_here'
});

// Create research session
const session = await client.research.createSession({
  title: 'Drug Discovery Project',
  description: 'AI-assisted drug discovery for rare diseases',
  domain: 'pharmaceutical_research',
  objectives: ['Identify potential compounds', 'Analyze efficacy'],
  coordinationMode: 'sequential'
});

// Start research with WebSocket monitoring
const ws = client.websocket.connect();

ws.on('research_progress', (data) => {
  console.log(`Progress: ${(data.overallProgress * 100).toFixed(1)}%`);
  console.log(`Phase: ${data.currentPhase}`);
});

await client.research.startSession(session.id);

// Handle experiment results
ws.on('experiment_results', async (data) => {
  console.log('Experiment completed:', data.results);

  // Download visualizations
  for (const viz of data.results.visualizations) {
    const response = await fetch(viz.url);
    const blob = await response.blob();
    // Handle visualization download
  }
});
```

### CLI Examples

```bash
# Authenticate
ai-scientist auth login --username user@example.com

# Create research session
ai-scientist research create \
  --title "Market Analysis Study" \
  --domain "economics" \
  --coordination-mode "parallel"

# List agents
ai-scientist agents list --status active

# Execute experiment
ai-scientist experiments run \
  --session-id session_uuid \
  --hypothesis "AI models improve market predictions" \
  --data-file market_data.csv

# Monitor progress
ai-scientist monitor --session-id session_uuid --follow

# Get results
ai-scientist results download --session-id session_uuid --format pdf
```

This comprehensive API reference provides complete documentation for integrating with the AI-Scientist-v2 multi-agent system, including all endpoints, authentication methods, WebSocket connectivity, and SDK examples for various programming languages.