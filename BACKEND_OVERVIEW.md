# Backend Architecture - Python FastAPI

This document provides a comprehensive overview of the Python backend architecture for the Kortix AI Agent Platform.

## 📁 Directory Structure

```
backend/
├── api.py                          # Main FastAPI application entry point
├── run_agent_background.py         # Dramatiq background worker for agent execution
├── start.py                        # Startup/management script
├── core/                           # Core application modules
│   ├── admin/                      # Admin APIs
│   │   ├── admin_api.py           # User/system admin endpoints
│   │   ├── billing_admin_api.py   # Billing administration
│   │   └── notification_admin_api.py # Notification management
│   │
│   ├── agentpress/                # Agent execution engine
│   │   ├── agent.py               # Core agent class
│   │   ├── context_manager.py     # Context/memory management
│   │   ├── error_processor.py     # Error handling
│   │   ├── native_tool_parser.py  # Native tool integration
│   │   ├── prompt_caching.py      # Prompt cache optimization
│   │   ├── response_processor.py  # LLM response handling
│   │   ├── thread_manager.py      # Thread lifecycle management
│   │   ├── tool.py                # Base tool class
│   │   ├── tool_registry.py       # Tool discovery and loading
│   │   └── xml_tool_parser.py     # XML-based tool parsing
│   │
│   ├── ai_models/                 # AI model management
│   │   ├── ai_models.py           # Model definitions
│   │   ├── manager.py             # Model lifecycle
│   │   └── registry.py            # Available models registry
│   │
│   ├── api_models/                # Pydantic request/response models
│   │   ├── agents.py              # Agent-related models
│   │   ├── threads.py             # Thread/message models
│   │   ├── common.py              # Shared models (pagination, etc.)
│   │   └── imports.py             # Common imports
│   │
│   ├── billing/                   # Complete billing system (100+ files)
│   │   ├── api.py                 # Billing API endpoints
│   │   ├── core/                  # Core billing logic
│   │   ├── credits/               # Credit management
│   │   │   ├── calculator.py      # Usage calculation
│   │   │   ├── integration.py     # Billing integration
│   │   │   └── manager.py         # Credit operations
│   │   ├── domain/                # Domain entities
│   │   │   └── entities/
│   │   │       ├── credit_account.py
│   │   │       └── subscription.py
│   │   ├── endpoints/             # API endpoints
│   │   │   ├── account_state.py   # Account billing state
│   │   │   ├── admin.py           # Admin operations
│   │   │   ├── core.py            # Core endpoints
│   │   │   ├── payments.py        # Payment processing
│   │   │   ├── subscriptions.py   # Subscription management
│   │   │   ├── trial.py           # Free trial handling
│   │   │   └── webhooks.py        # Payment webhooks
│   │   ├── external/              # External payment integrations
│   │   │   ├── revenuecat/        # RevenueCat (mobile IAP)
│   │   │   └── stripe/            # Stripe payment processing
│   │   └── payments/              # Payment processing
│   │
│   ├── composio_integration/      # Composio 150+ integrations
│   ├── credentials/               # Secure credential storage
│   ├── google/                    # Google Workspace integrations
│   ├── knowledge_base/            # Knowledge base system
│   ├── mcp_module/                # Model Context Protocol
│   ├── notifications/             # Notification system
│   ├── referrals/                 # Referral program
│   ├── sandbox/                   # Code execution sandbox
│   ├── services/                  # Core services
│   ├── setup/                     # Initial setup
│   ├── templates/                 # Agent template marketplace
│   ├── tools/                     # Agent tools (20+ tools)
│   ├── triggers/                  # Automation triggers
│   ├── utils/                     # Utility functions
│   ├── versioning/                # Agent versioning
│   ├── account_deletion.py        # Account deletion logic
│   ├── accounts_api.py            # Account management
│   ├── agent_crud.py              # Agent CRUD operations
│   ├── agent_runs.py              # Agent run lifecycle
│   ├── agent_service.py           # Agent business logic
│   ├── auth.py                    # Authentication
│   ├── threads.py                 # Thread management
│   └── [... see full structure below]
│
├── supabase/migrations/           # Database migrations (60+)
└── tests/                         # Test suite
```

## 🏗️ Core Components

### 1. Main Application (`api.py`)
- FastAPI with CORS and rate limiting
- Health check and metrics endpoints
- Request logging with structlog
- Memory watchdog monitoring
- Lifespan management

### 2. Agent Execution Engine (`agentpress/`)
- Multi-provider LLM support (Claude, GPT-4)
- Streaming responses via SSE
- Tool calling with parallel execution
- Context management and memory
- Error recovery

### 3. Background Worker (`run_agent_background.py`)
- Dramatiq task queue
- Distributed locking via Redis
- Agent run orchestration
- Automatic retries

### 4. Database Layer (`services/supabase.py`)
- Async Supabase client
- Connection pooling
- Row Level Security

### 5. Billing System (`billing/`)
- Credit-based consumption
- Stripe subscriptions
- RevenueCat mobile IAP
- Usage analytics

### 6. Tools System (`tools/`)
20+ extensible tools including:
- Browser automation
- Code execution (E2B, Replit, Daytona)
- Document processing
- File operations
- Google Suite integration
- Image editing
- Web search and research

## 📊 API Endpoints

### Core APIs
- `/agents` - Agent CRUD operations
- `/agent-runs` - Agent execution
- `/threads` - Thread management
- `/versions` - Version control

### Feature APIs
- `/billing` - Billing & subscriptions
- `/knowledge-base` - Knowledge base
- `/templates` - Template marketplace
- `/triggers` - Automation
- `/mcp` - MCP integrations
- `/credentials` - Secure credentials
- `/notifications` - Notifications
- `/composio` - Composio integrations

### Admin APIs
- `/admin` - User management
- `/admin/billing` - Billing admin
- `/admin/notifications` - Notification admin

### System APIs
- `/health` - Health check
- `/metrics/queue` - Queue metrics
- `/api-keys` - API key management

## 🔐 Authentication & Authorization

### JWT Authentication
```python
verify_and_get_user_id_from_jwt(request)
```

### Role-Based Access Control
- `user` - Standard user access
- `admin` - Admin operations
- `superadmin` - Full system access

## 🚀 Running the Backend

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python api.py

# Run background worker
python run_agent_background.py
```

### Production
```bash
# FastAPI with multiple workers
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Background worker (separate process)
python run_agent_background.py
```

## 📈 Key Dependencies

```python
fastapi==0.115.12          # Web framework
dramatiq==1.18.0           # Task queue
supabase==2.17.0           # Database
litellm>=1.77.5            # Multi-provider LLM
stripe==11.6.0             # Payments
anthropic>=0.69.0          # Claude
openai>=1.99.5             # GPT-4
```

## 📖 Related Documentation

- [API_MIGRATION.md](API_MIGRATION.md) - Next.js API migration
- [openapi.yaml](openapi.yaml) - OpenAPI specification
- [README.md](README.md) - Main project README
