# WebSocket Integration Guide

## Overview

This guide provides comprehensive documentation for integrating with the AI-Scientist-v2 WebSocket API. WebSocket connections enable real-time monitoring of research progress, agent status updates, experiment results, and system events.

## Table of Contents

1. [WebSocket Overview](#websocket-overview)
2. [Connection Management](#connection-management)
3. [Authentication](#authentication)
4. [Message Protocol](#message-protocol)
5. [Event Types](#event-types)
6. [Subscription Management](#subscription-management)
7. [Error Handling](#error-handling)
8. [Integration Examples](#integration-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## WebSocket Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   WebSocket Server                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Client 1  │  │   Client 2  │  │   Client N          │  │
│  │Connection   │  │Connection   │  │   Connection        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Event Broadcasting                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Agent Events│  │ Research    │  │ System Events       │  │
│  │             │  │ Progress    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Authentication                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   JWT       │  │   API Keys  │  │   Session Tokens    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Features

- **Real-time Updates**: Live monitoring of research progress and agent activities
- **Event Filtering**: Subscribe to specific events and resources
- **Multi-client Support**: Handle multiple concurrent connections
- **Automatic Reconnection**: Built-in reconnection logic with exponential backoff
- **Message Acknowledgment**: Reliable message delivery with acknowledgment support
- **Rate Limiting**: Client-side rate limiting to prevent overwhelming the server

## Connection Management

### WebSocket URLs

- **Development**: `ws://localhost:8080/ws`
- **Staging**: `wss://staging.ai-scientist.com/ws`
- **Production**: `wss://api.ai-scientist.com/ws`

### Connection Lifecycle

1. **Establish Connection**: WebSocket handshake
2. **Authenticate**: Send authentication message
3. **Subscribe**: Subscribe to desired events
4. **Receive Events**: Handle incoming real-time events
5. **Unsubscribe**: Clean up subscriptions
6. **Close Connection**: Graceful shutdown

### Connection States

```javascript
const ConnectionState = {
  CONNECTING: 0,    // Connection in progress
  OPEN: 1,          // Connection established and authenticated
  CLOSING: 2,       // Connection closing
  CLOSED: 3,        // Connection closed
  RECONNECTING: 4,  // Attempting to reconnect
  ERROR: 5          // Connection error
};
```

## Authentication

### JWT Authentication

```javascript
const ws = new WebSocket('wss://api.ai-scientist.com/ws');

ws.onopen = function() {
  // Send authentication message
  ws.send(JSON.stringify({
    type: 'auth',
    data: {
      token: 'your_jwt_token_here'
    }
  }));
};
```

### API Key Authentication

```javascript
ws.send(JSON.stringify({
  type: 'auth',
  data: {
    api_key: 'your_api_key_here'
  }
}));
```

### Authentication Response

**Success Response:**
```json
{
  "type": "auth_response",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "status": "success",
    "session_id": "session_uuid",
    "user_id": "user_uuid",
    "permissions": ["read", "write", "admin"],
    "expires_at": "2024-01-01T13:00:00Z"
  }
}
```

**Error Response:**
```json
{
  "type": "auth_response",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "status": "error",
    "error": {
      "code": "INVALID_TOKEN",
      "message": "Authentication failed"
    }
  }
}
```

### Token Refresh

```javascript
// Handle token expiration
ws.onmessage = function(event) {
  const message = JSON.parse(event.data);

  if (message.type === 'token_refresh_required') {
    // Refresh token and re-authenticate
    refreshToken().then(newToken => {
      ws.send(JSON.stringify({
        type: 'auth',
        data: { token: newToken }
      }));
    });
  }
};
```

## Message Protocol

### Message Structure

All WebSocket messages follow this structure:

```json
{
  "type": "string",
  "id": "message_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {},
  "meta": {
    "version": "1.0",
    "source": "agent_service",
    "correlation_id": "correlation_uuid"
  }
}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `auth` | Client → Server | Authentication request |
| `auth_response` | Server → Client | Authentication response |
| `subscribe` | Client → Server | Subscribe to events |
| `unsubscribe` | Client → Server | Unsubscribe from events |
| `subscription_response` | Server → Client | Subscription confirmation |
| `ping` | Client → Server | Keep-alive ping |
| `pong` | Server → Client | Keep-alive pong |
| `agent_status` | Server → Client | Agent status update |
| `research_progress` | Server → Client | Research progress update |
| `experiment_results` | Server → Client | Experiment results |
| `ethical_assessment` | Server → Client | Ethical assessment |
| `system_event` | Server → Client | System event notification |
| `error` | Server → Client | Error message |

### Message Acknowledgment

```javascript
// Request acknowledgment
ws.send(JSON.stringify({
  type: 'subscribe',
  id: generateMessageId(),
  data: {
    events: ['agent_status'],
    require_ack: true
  }
}));

// Handle acknowledgment
ws.onmessage = function(event) {
  const message = JSON.parse(event.data);

  if (message.type === 'ack') {
    console.log(`Message ${message.data.original_id} acknowledged`);
  }
};
```

## Event Types

### Agent Status Events

```json
{
  "type": "agent_status",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "agent_id": "agent_uuid",
    "name": "Research Agent Alpha",
    "status": "busy|idle|error|offline",
    "current_task": {
      "id": "task_uuid",
      "type": "data_analysis",
      "progress": 0.65,
      "started_at": "2024-01-01T11:45:00Z",
      "estimated_completion": "2024-01-01T12:15:00Z"
    },
    "performance": {
      "cpu_usage": 0.45,
      "memory_usage": 0.67,
      "response_time": 1.2,
      "success_rate": 0.95
    },
    "health": {
      "last_heartbeat": "2024-01-01T12:00:00Z",
      "uptime": 86400,
      "error_count": 2
    }
  }
}
```

### Research Progress Events

```json
{
  "type": "research_progress",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "session_id": "session_uuid",
    "title": "Climate Change Impact Study",
    "overall_progress": 0.45,
    "current_phase": "experimentation",
    "phase_progress": 0.30,
    "activities": [
      {
        "agent_id": "agent_uuid",
        "activity": "Running climate simulations",
        "progress": 0.80,
        "status": "in_progress"
      }
    ],
    "milestones": [
      {
        "name": "Data Collection Complete",
        "status": "completed",
        "completed_at": "2024-01-01T11:30:00Z"
      },
      {
        "name": "Experiment Design",
        "status": "in_progress",
        "progress": 0.60
      }
    ],
    "timeline": {
      "started_at": "2024-01-01T10:00:00Z",
      "estimated_completion": "2024-01-01T18:00:00Z",
      "time_elapsed": 7200,
      "time_remaining": 21600
    }
  }
}
```

### Experiment Results Events

```json
{
  "type": "experiment_results",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "experiment_id": "experiment_uuid",
    "session_id": "session_uuid",
    "title": "Temperature Trend Analysis",
    "status": "completed|failed|partial",
    "results": {
      "summary": "Analysis shows significant temperature increase over the past decade",
      "key_findings": [
        "Average temperature increased by 1.2°C",
        "Correlation with CO2 levels: r = 0.87",
        "Statistical significance: p < 0.001"
      ],
      "metrics": {
        "accuracy": 0.94,
        "confidence_interval": [0.89, 0.99],
        "sample_size": 10000,
        "effect_size": 0.75
      },
      "visualizations": [
        {
          "type": "time_series",
          "title": "Temperature Over Time",
          "url": "https://api.ai-scientist.com/viz/plot_123.png",
          "description": "Temperature trends from 2010-2024"
        },
        {
          "type": "scatter_plot",
          "title": "Temperature vs CO2 Levels",
          "url": "https://api.ai-scientist.com/viz/plot_124.png",
          "description": "Correlation between temperature and CO2"
        }
      ],
      "data": {
        "raw_data_url": "https://api.ai-scientist.com/data/exp_123_raw.csv",
        "processed_data_url": "https://api.ai-scientist.com/data/exp_123_processed.json",
        "metadata": {
          "columns": ["date", "temperature", "co2_level"],
          "rows": 3650,
          "file_size": "2.3MB"
        }
      }
    },
    "execution": {
      "duration": 1200,
      "started_at": "2024-01-01T11:40:00Z",
      "completed_at": "2024-01-01T12:00:00Z",
      "agent_id": "agent_uuid",
      "computing_resources": {
        "cpu_time": 1800,
        "memory_peak": "4.5GB",
        "gpu_usage": "30%"
      }
    },
    "quality_metrics": {
      "data_quality_score": 0.92,
      "methodology_score": 0.88,
      "reproducibility_score": 0.95
    }
  }
}
```

### Ethical Assessment Events

```json
{
  "type": "ethical_assessment",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "resource_id": "resource_uuid",
    "resource_type": "experiment|agent|research_session",
    "assessment": {
      "overall_score": 0.85,
      "status": "compliant|review_required|non_compliant",
      "confidence": 0.92,
      "frameworks": {
        "utilitarian": {
          "score": 0.90,
          "weight": 0.25,
          "assessment": "Maximizes overall benefit for society"
        },
        "deontological": {
          "score": 0.80,
          "weight": 0.20,
          "assessment": "Follows established ethical principles"
        },
        "virtue_ethics": {
          "score": 0.85,
          "weight": 0.15,
          "assessment": "Demonstrates scientific integrity"
        },
        "care_ethics": {
          "score": 0.88,
          "weight": 0.20,
          "assessment": "Considers impact on affected stakeholders"
        },
        "justice_ethics": {
          "score": 0.82,
          "weight": 0.10,
          "assessment": "Ensures fair distribution of benefits"
        },
        "precautionary": {
          "score": 0.78,
          "weight": 0.10,
          "assessment": "Addresses potential risks appropriately"
        }
      },
      "identified_issues": [
        {
          "severity": "low|medium|high",
          "category": "bias|privacy|safety|transparency",
          "description": "Potential sampling bias in dataset selection",
          "recommendation": "Include more diverse data sources",
          "auto_fixable": true
        }
      ],
      "recommendations": [
        "Include diverse demographic representation in analysis",
        "Validate findings with independent peer review",
        "Document all data preprocessing steps"
      ],
      "human_review_required": false,
      "auto_approved": true,
      "review_deadline": "2024-01-08T12:00:00Z"
    },
    "metadata": {
      "assessed_by": "ethical_framework_agent",
      "assessment_version": "2.1",
      "assessment_duration": 45,
      "assessed_at": "2024-01-01T12:00:00Z"
    }
  }
}
```

### System Events

```json
{
  "type": "system_event",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "event_type": "maintenance|scale|error|update",
    "severity": "info|warning|error|critical",
    "title": "System Maintenance Scheduled",
    "description": "Scheduled maintenance will affect agent services",
    "affected_services": ["agent_service", "research_orchestrator"],
    "scheduled_time": "2024-01-02T02:00:00Z",
    "estimated_duration": 1800,
    "actions_required": [
      "Save all active research sessions",
      "Export critical experiment results"
    ],
    "alternatives": {
      "continue_offline": true,
      "use_backup_agents": false
    }
  }
}
```

## Subscription Management

### Subscribe to Events

```javascript
// Subscribe to multiple event types
ws.send(JSON.stringify({
  type: 'subscribe',
  id: generateMessageId(),
  data: {
    events: [
      'agent_status',
      'research_progress',
      'experiment_results',
      'ethical_assessment'
    ],
    filters: {
      session_id: 'session_uuid',
      agent_ids: ['agent_uuid_1', 'agent_uuid_2'],
      severity: ['warning', 'error']
    },
    options: {
      include_history: false,
      batch_size: 10,
      throttle_ms: 1000
    }
  }
}));
```

### Subscription Response

```json
{
  "type": "subscription_response",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "status": "success|error",
    "subscription_id": "sub_uuid",
    "subscribed_events": ["agent_status", "research_progress"],
    "active_filters": {
      session_id: "session_uuid"
    },
    "estimated_events_per_minute": 15,
    "retention_hours": 24
  }
}
```

### Unsubscribe from Events

```javascript
// Unsubscribe from specific events
ws.send(JSON.stringify({
  type: 'unsubscribe',
  id: generateMessageId(),
  data: {
    subscription_id: 'sub_uuid',
    events: ['agent_status']
  }
}));

// Unsubscribe from all events
ws.send(JSON.stringify({
  type: 'unsubscribe',
  id: generateMessageId(),
  data: {
    subscription_id: 'sub_uuid',
    all: true
  }
}));
```

### Advanced Filtering

```javascript
// Complex subscription with filters
ws.send(JSON.stringify({
  type: 'subscribe',
  id: generateMessageId(),
  data: {
    events: ['experiment_results'],
    filters: {
      session_id: 'session_uuid',
      status: ['completed', 'failed'],
      date_range: {
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-31T23:59:59Z'
      },
      custom_filters: {
        confidence_threshold: 0.8,
        include_visualizations: true
      }
    },
    options: {
      batch_events: true,
      batch_size: 5,
      batch_interval_ms: 2000
    }
  }
}));
```

## Error Handling

### Error Message Format

```json
{
  "type": "error",
  "id": "msg_uuid",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": {
    "code": "SUBSCRIPTION_FAILED",
    "message": "Failed to subscribe to requested events",
    "details": {
      "invalid_events": ["invalid_event_type"],
      "permission_denied": ["system_events"]
    },
    "retry_after": 60,
    "correlation_id": "correlation_uuid"
  }
}
```

### Common Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `AUTHENTICATION_FAILED` | Invalid credentials | Re-authenticate |
| `PERMISSION_DENIED` | Insufficient permissions | Request proper permissions |
| `SUBSCRIPTION_FAILED` | Invalid subscription request | Fix subscription parameters |
| `RATE_LIMITED` | Too many requests | Implement backoff |
| `CONNECTION_LIMIT` | Too many connections | Close unused connections |
| `INVALID_MESSAGE` | Malformed message | Fix message format |
| `SERVICE_UNAVAILABLE` | Temporary service issue | Retry with exponential backoff |

### Error Handling Best Practices

```javascript
class WebSocketClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1000;
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.authenticate();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.handleError(error);
    };

    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      this.handleClose(event);
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(JSON.parse(event.data));
    };
  }

  handleError(error) {
    // Log error for debugging
    console.error('WebSocket error:', error);

    // Implement retry logic for recoverable errors
    if (this.isRecoverableError(error)) {
      this.scheduleReconnect();
    }
  }

  handleClose(event) {
    if (event.code !== 1000) {
      // Abnormal closure, attempt to reconnect
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
      console.log(`Reconnecting in ${delay}ms...`);

      setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnect attempts reached');
      this.onError(new Error('Failed to reconnect after maximum attempts'));
    }
  }

  isRecoverableError(error) {
    const recoverableCodes = [1006, 1011]; // Abnormal closure, server error
    return recoverableCodes.includes(error.code);
  }
}
```

## Integration Examples

### JavaScript/TypeScript Client

```typescript
interface WebSocketMessage {
  type: string;
  id: string;
  timestamp: string;
  data: any;
  meta?: {
    version: string;
    source: string;
    correlation_id: string;
  };
}

interface SubscriptionOptions {
  events: string[];
  filters?: Record<string, any>;
  options?: {
    includeHistory?: boolean;
    batchSize?: number;
    throttleMs?: number;
  };
}

class AIScientistWebSocket {
  private ws: WebSocket | null = null;
  private subscriptions: Map<string, SubscriptionOptions> = new Map();
  private messageHandlers: Map<string, Function[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private isConnecting = false;

  constructor(private url: string, private authToken: string) {}

  async connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.authenticate();
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
        reject(error);
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        this.isConnecting = false;
        this.handleReconnect();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(JSON.parse(event.data));
      };
    });
  }

  private authenticate(): void {
    this.send({
      type: 'auth',
      data: { token: this.authToken }
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('Received message:', message.type, message.data);

    // Handle authentication response
    if (message.type === 'auth_response') {
      if (message.data.status === 'success') {
        console.log('Authentication successful');
        this.resubscribeAll();
      } else {
        console.error('Authentication failed:', message.data.error);
      }
      return;
    }

    // Handle subscription responses
    if (message.type === 'subscription_response') {
      console.log('Subscription confirmed:', message.data);
      return;
    }

    // Handle errors
    if (message.type === 'error') {
      console.error('WebSocket error:', message.data);
      return;
    }

    // Route to appropriate handlers
    const handlers = this.messageHandlers.get(message.type) || [];
    handlers.forEach(handler => {
      try {
        handler(message.data);
      } catch (error) {
        console.error('Error in message handler:', error);
      }
    });
  }

  async subscribe(options: SubscriptionOptions): Promise<string> {
    const messageId = this.generateId();
    const subscription = {
      ...options,
      id: messageId
    };

    this.subscriptions.set(messageId, subscription);

    await this.send({
      type: 'subscribe',
      id: messageId,
      data: options
    });

    return messageId;
  }

  unsubscribe(subscriptionId: string): void {
    this.subscriptions.delete(subscriptionId);

    this.send({
      type: 'unsubscribe',
      id: this.generateId(),
      data: {
        subscription_id: subscriptionId,
        all: true
      }
    });
  }

  on(eventType: string, handler: Function): void {
    const handlers = this.messageHandlers.get(eventType) || [];
    handlers.push(handler);
    this.messageHandlers.set(eventType, handlers);
  }

  off(eventType: string, handler?: Function): void {
    if (!handler) {
      this.messageHandlers.delete(eventType);
      return;
    }

    const handlers = this.messageHandlers.get(eventType) || [];
    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
      this.messageHandlers.set(eventType, handlers);
    }
  }

  private async resubscribeAll(): Promise<void> {
    for (const [id, subscription] of this.subscriptions) {
      await this.send({
        type: 'subscribe',
        id: id,
        data: subscription
      });
    }
  }

  private async send(message: Partial<WebSocketMessage>): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }

    const fullMessage: WebSocketMessage = {
      type: message.type!,
      id: message.id || this.generateId(),
      timestamp: new Date().toISOString(),
      data: message.data
    };

    this.ws.send(JSON.stringify(fullMessage));
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Reconnecting in ${delay}ms...`);

      setTimeout(() => {
        this.reconnectAttempts++;
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnect attempts reached');
    }
  }

  private generateId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.subscriptions.clear();
    this.messageHandlers.clear();
  }
}

// Usage example
const client = new AIScientistWebSocket(
  'wss://api.ai-scientist.com/ws',
  'your_jwt_token_here'
);

await client.connect();

// Subscribe to research progress
const subscriptionId = await client.subscribe({
  events: ['research_progress', 'experiment_results'],
  filters: {
    session_id: 'session_uuid'
  }
});

// Handle events
client.on('research_progress', (data) => {
  console.log(`Research progress: ${(data.overall_progress * 100).toFixed(1)}%`);
  updateProgressBar(data.overall_progress);
});

client.on('experiment_results', (data) => {
  console.log('Experiment completed:', data.results.summary);
  displayResults(data.results);
});
```

### Python Client

```python
import asyncio
import json
import websockets
from typing import Dict, List, Callable, Any
import logging
from datetime import datetime

class AIScientistWebSocket:
    def __init__(self, url: str, auth_token: str):
        self.url = url
        self.auth_token = auth_token
        self.websocket = None
        self.subscriptions: Dict[str, Dict] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.is_connected = False
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Connect to WebSocket server"""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.websocket = await websockets.connect(self.url)
                self.is_connected = True
                self.reconnect_attempts = 0
                self.logger.info("WebSocket connected")

                # Start message handler
                asyncio.create_task(self.message_handler())

                # Authenticate
                await self.authenticate()

                # Resubscribe to previous subscriptions
                await self.resubscribe_all()

                return True

            except Exception as e:
                self.reconnect_attempts += 1
                delay = min(2 ** self.reconnect_attempts, 30)
                self.logger.error(f"Connection failed (attempt {self.reconnect_attempts}): {e}")
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        raise Exception("Failed to connect after maximum attempts")

    async def authenticate(self):
        """Send authentication message"""
        await self.send_message({
            'type': 'auth',
            'data': {'token': self.auth_token}
        })

    async def message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            self.logger.warning("WebSocket connection closed")
            await self.handle_reconnect()
        except Exception as e:
            self.logger.error(f"Error in message handler: {e}")
            self.is_connected = False

    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming message"""
        message_type = message.get('type')

        # Handle authentication response
        if message_type == 'auth_response':
            if message['data']['status'] == 'success':
                self.logger.info("Authentication successful")
            else:
                self.logger.error(f"Authentication failed: {message['data']}")
            return

        # Handle subscription responses
        if message_type == 'subscription_response':
            self.logger.info(f"Subscription confirmed: {message['data']}")
            return

        # Handle errors
        if message_type == 'error':
            self.logger.error(f"WebSocket error: {message['data']}")
            return

        # Route to appropriate handlers
        handlers = self.message_handlers.get(message_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message['data'])
                else:
                    handler(message['data'])
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")

    async def subscribe(self, events: List[str], filters: Dict[str, Any] = None) -> str:
        """Subscribe to events"""
        subscription_id = f"sub_{datetime.now().timestamp()}"

        subscription_data = {
            'events': events,
            'filters': filters or {}
        }

        self.subscriptions[subscription_id] = subscription_data

        await self.send_message({
            'type': 'subscribe',
            'data': subscription_data
        })

        return subscription_id

    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from events"""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]

            await self.send_message({
                'type': 'unsubscribe',
                'data': {
                    'subscription_id': subscription_id,
                    'all': True
                }
            })

    def on(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable = None):
        """Unregister event handler"""
        if event_type in self.message_handlers:
            if handler:
                self.message_handlers[event_type].remove(handler)
            else:
                del self.message_handlers[event_type]

    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket server"""
        if not self.is_connected or not self.websocket:
            raise Exception("WebSocket is not connected")

        full_message = {
            'type': message['type'],
            'id': f"msg_{datetime.now().timestamp()}",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'data': message['data']
        }

        await self.websocket.send(json.dumps(full_message))

    async def resubscribe_all(self):
        """Resubscribe to all previous subscriptions"""
        for subscription_id, subscription_data in self.subscriptions.items():
            await self.send_message({
                'type': 'subscribe',
                'data': subscription_data
            })

    async def handle_reconnect(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(2 ** self.reconnect_attempts, 30)
            self.logger.info(f"Reconnecting in {delay} seconds...")
            await asyncio.sleep(delay)
            await self.connect()
        else:
            self.logger.error("Max reconnect attempts reached")

    async def close(self):
        """Close WebSocket connection"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
        self.subscriptions.clear()
        self.message_handlers.clear()

# Usage example
async def main():
    client = AIScientistWebSocket(
        'wss://api.ai-scientist.com/ws',
        'your_jwt_token_here'
    )

    # Connect to WebSocket
    await client.connect()

    # Subscribe to events
    subscription_id = await client.subscribe(
        events=['research_progress', 'experiment_results'],
        filters={'session_id': 'session_uuid'}
    )

    # Define event handlers
    def on_research_progress(data):
        progress = data['overall_progress'] * 100
        print(f"Research progress: {progress:.1f}%")

    def on_experiment_results(data):
        print(f"Experiment completed: {data['results']['summary']}")

    # Register handlers
    client.on('research_progress', on_research_progress)
    client.on('experiment_results', on_experiment_results)

    # Keep connection alive
    try:
        while client.is_connected:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Connection Management

1. **Implement Exponential Backoff**: Use exponential backoff for reconnection attempts
2. **Limit Concurrent Connections**: Reuse connections when possible
3. **Handle Authentication Expiration**: Refresh tokens before they expire
4. **Graceful Shutdown**: Properly close connections when done

### Performance Optimization

1. **Batch Events**: Use event batching for high-frequency updates
2. **Filter Events**: Subscribe only to necessary events
3. **Debounce Handlers**: Debounce rapid successive events
4. **Use Web Workers**: Offload heavy processing to web workers

### Error Handling

1. **Comprehensive Error Logging**: Log all errors with context
2. **Retry Logic**: Implement retry logic for recoverable errors
3. **Fallback Mechanisms**: Provide fallbacks when WebSocket is unavailable
4. **User Notification**: Notify users of connection issues

### Security

1. **Use Secure Connections**: Always use WSS in production
2. **Validate Messages**: Validate incoming message structure
3. **Rate Limiting**: Implement client-side rate limiting
4. **Token Security**: Securely store and manage authentication tokens

## Troubleshooting

### Common Issues

#### Connection Fails

**Symptoms**: WebSocket connection fails to establish
**Solutions**:
- Check network connectivity
- Verify WebSocket URL is correct
- Ensure authentication token is valid
- Check firewall settings

#### Authentication Fails

**Symptoms**: Authentication response shows error
**Solutions**:
- Verify JWT token is not expired
- Check token has required permissions
- Ensure token is sent in correct format
- Verify API key is valid (if using API key auth)

#### Missing Events

**Symptoms**: Expected events are not received
**Solutions**:
- Verify subscription was successful
- Check event filters are correct
- Ensure events match subscription criteria
- Check server logs for event generation

#### Frequent Disconnections

**Symptoms**: WebSocket connection drops frequently
**Solutions**:
- Check network stability
- Implement proper reconnection logic
- Verify server-side configuration
- Check for rate limiting

#### Performance Issues

**Symptoms**: High latency or memory usage
**Solutions**:
- Implement event batching
- Use efficient message handlers
- Optimize filtering criteria
- Monitor resource usage

### Debug Tools

```javascript
// Enable debug logging
const client = new AIScientistWebSocket(url, token, {
  debug: true,
  logLevel: 'debug'
});

// Monitor connection state
client.on('connection_state_change', (state) => {
  console.log('Connection state:', state);
});

// Monitor message flow
client.on('message_sent', (message) => {
  console.log('Sent:', message);
});

client.on('message_received', (message) => {
  console.log('Received:', message);
});
```

### Health Monitoring

```python
# Health check for WebSocket connection
async def health_check(client):
    if not client.is_connected:
        return False

    try:
        # Send ping message
        await client.send_message({'type': 'ping'})
        return True
    except Exception:
        return False

# Periodic health monitoring
async def monitor_connection(client):
    while True:
        if not await health_check(client):
            print("Connection unhealthy, attempting reconnect...")
            await client.connect()

        await asyncio.sleep(30)
```

This comprehensive WebSocket integration guide provides everything needed to implement real-time monitoring and event handling for the AI-Scientist-v2 system, with detailed examples in multiple programming languages and best practices for production use.