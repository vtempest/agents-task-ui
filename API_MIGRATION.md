# Next.js API Migration Guide

This document describes the Next.js API implementation that mirrors the Python FastAPI backend functionality.

## Architecture Overview

### Migration Strategy

The migration follows a **hybrid architecture** approach:

1. **Next.js API Layer** - Handles HTTP requests, authentication, and database operations
2. **Python Background Worker** - Maintains agent execution engine (Dramatiq + LiteLLM)
3. **Shared Infrastructure** - Supabase (PostgreSQL) and Redis for data and queuing

### What's Migrated

✅ **API Routes (Next.js)**
- Agent CRUD operations
- Thread management
- Message handling
- Agent version control
- Project management
- Health checks
- Authentication & authorization

### What Remains in Python

🐍 **Background Processing**
- Agent execution engine (AgentPress)
- LLM integration (Claude, OpenAI via LiteLLM)
- Tool execution
- Streaming response handling
- Background job processing (Dramatiq)

## API Endpoints

### Base URL
- **Development**: `http://localhost:3000/api`
- **Production**: `https://kortix.com/api`

### Authentication

All endpoints (except `/health` and public endpoints) require JWT authentication:

```bash
Authorization: Bearer <supabase-jwt-token>
```

## Key Endpoints

### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/agents` | List all agents (paginated) |
| POST | `/agents` | Create a new agent |
| GET | `/agents/{agentId}` | Get specific agent |
| PUT | `/agents/{agentId}` | Update agent configuration |
| DELETE | `/agents/{agentId}` | Delete agent |
| GET | `/agents/{agentId}/versions` | List agent versions |
| POST | `/agents/{agentId}/versions` | Create new version |

### Threads

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/threads` | List all threads (paginated) |
| POST | `/threads` | Create a new thread |
| GET | `/threads/{threadId}` | Get specific thread |
| PUT | `/threads/{threadId}` | Update thread |
| DELETE | `/threads/{threadId}` | Delete thread |
| GET | `/threads/{threadId}/messages` | List messages in thread |
| POST | `/threads/{threadId}/messages` | Create message |

### Agent Runs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/agent-runs` | List agent runs |
| POST | `/agent-runs` | Start agent execution |
| GET | `/agent-runs/{runId}` | Get run status |

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/projects/{projectId}` | Get project (supports public access) |
| PUT | `/projects/{projectId}` | Update project |
| DELETE | `/projects/{projectId}` | Delete project |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

## Code Structure

```
frontend/
├── src/
│   ├── app/
│   │   └── api/                    # Next.js API routes
│   │       ├── agents/             # Agent endpoints
│   │       │   ├── route.ts        # GET /agents, POST /agents
│   │       │   └── [agentId]/
│   │       │       ├── route.ts    # GET/PUT/DELETE /agents/{id}
│   │       │       └── versions/
│   │       │           └── route.ts # Agent versions
│   │       ├── threads/            # Thread endpoints
│   │       │   ├── route.ts
│   │       │   └── [threadId]/
│   │       │       ├── route.ts
│   │       │       └── messages/
│   │       │           └── route.ts
│   │       ├── agent-runs/         # Agent run endpoints
│   │       │   ├── route.ts
│   │       │   └── [runId]/
│   │       │       └── route.ts
│   │       ├── projects/           # Project endpoints
│   │       │   └── [projectId]/
│   │       │       └── route.ts
│   │       └── health/
│   │           └── route.ts
│   └── lib/
│       └── api/                    # Shared utilities
│           ├── auth.ts             # Authentication helpers
│           ├── db.ts               # Supabase client
│           └── types.ts            # TypeScript types
└── openapi.yaml                    # OpenAPI 3.1 specification
```

## Usage Examples

### Create an Agent

```typescript
const response = await fetch('/api/agents', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify({
    name: 'Customer Support Agent',
    description: 'Handles customer inquiries',
    system_prompt: 'You are a helpful customer support agent...',
    model: 'claude-3-5-sonnet-20241022',
    icon_name: 'user-headset',
    agentpress_tools: {
      'web-search': { enabled: true },
      'file-operations': { enabled: true }
    }
  })
});

const { agent } = await response.json();
```

### List Threads

```typescript
const response = await fetch('/api/threads?page=1&limit=50', {
  headers: {
    'Authorization': `Bearer ${token}`,
  },
});

const { threads, pagination } = await response.json();
```

### Start Agent Run

```typescript
const response = await fetch('/api/agent-runs', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify({
    thread_id: 'thread-uuid',
    agent_id: 'agent-uuid',
    message: 'Hello, can you help me?'
  })
});

const { run } = await response.json();
// run.status === 'queued'
```

## Environment Variables

Required environment variables in `.env.local`:

```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Optional: Redis (for future agent run coordination)
REDIS_URL=redis://localhost:6379
```

## Features

### ✅ Implemented

- **Authentication**: JWT-based auth via Supabase
- **Authorization**: Row-level security checks
- **Pagination**: Configurable page size (max 1000)
- **CRUD Operations**: Full create, read, update, delete for agents, threads, messages
- **Version Control**: Agent version management
- **Public Access**: Support for public projects/threads
- **Suna Agent Protection**: Respects Suna default agent restrictions
- **Error Handling**: Comprehensive error responses
- **TypeScript**: Full type safety

### 🚧 TODO: Python Worker Integration

For full agent execution, the Python background worker is still required:

1. **Agent Execution**: Complex multi-step agent workflows
2. **Streaming**: SSE-based streaming responses
3. **Tool Execution**: File operations, web search, code execution
4. **LLM Integration**: Claude/OpenAI API calls via LiteLLM
5. **Queue Management**: Dramatiq task processing

### Integration Points

The Next.js API and Python worker communicate via:

- **Supabase**: Shared database for state
- **Redis**: Job queue and response streaming
- **Agent Runs Table**: Status tracking

```typescript
// Next.js creates the run
POST /api/agent-runs
→ Creates record in agent_runs table with status='queued'
→ Publishes job to Redis queue

// Python worker processes
Dramatiq picks up job from Redis
→ Executes agent (LLM + tools)
→ Updates agent_runs.status to 'running'
→ Streams responses to Redis
→ Updates status to 'completed'

// Frontend polls or subscribes
GET /api/agent-runs/{runId}
→ Returns current status
```

## Migration Benefits

1. **Performance**: Next.js Edge runtime for low latency
2. **Type Safety**: Full TypeScript support
3. **Simplified Deployment**: Single Next.js deployment
4. **Developer Experience**: Hot reload, better debugging
5. **Cost**: Reduced backend infrastructure needs
6. **Scalability**: Vercel Edge Network distribution

## Testing

```bash
# Start Next.js dev server
cd frontend
npm run dev

# Test health check
curl http://localhost:3000/api/health

# Test with authentication
export TOKEN="your-supabase-jwt"
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:3000/api/agents
```

## API Documentation

Full OpenAPI 3.1 specification available at:
- **File**: `openapi.yaml`
- **Swagger UI**: Coming soon

## Comparison: Python vs Next.js

| Feature | Python (FastAPI) | Next.js API |
|---------|------------------|-------------|
| Language | Python 3.11+ | TypeScript |
| Runtime | Uvicorn/ASGI | Node.js/Edge |
| Auth | Custom JWT | Supabase Auth |
| ORM | None (direct SQL) | Supabase Client |
| Validation | Pydantic | Zod (optional) |
| Deployment | Docker/EC2 | Vercel/Next.js |
| Hot Reload | ✅ | ✅ |
| Type Safety | ✅ | ✅ |
| Background Jobs | Dramatiq | ❌ (use Python) |
| Streaming | SSE | ❌ (use Python) |

## Contributing

When adding new endpoints:

1. Create route file in `src/app/api/[endpoint]/route.ts`
2. Add authentication with `verifyAndGetUserId()`
3. Define types in `src/lib/api/types.ts`
4. Update `openapi.yaml` with endpoint documentation
5. Add integration tests

## Support

For issues or questions:
- GitHub Issues: https://github.com/vtempest/agents-task-ui/issues
- Documentation: See README.md
