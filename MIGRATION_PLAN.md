# Supabase to SQLite/Turso Migration Plan

## Overview
This document outlines the migration strategy from Supabase to SQLite (local) + Turso (cloud) with custom authentication system.

**Migration Scope:**
- ✅ Web application (Next.js frontend + FastAPI backend)
- ✅ Mobile application (React Native)
- ✅ Authentication system (Priority 1)
- ✅ Database (PostgreSQL → SQLite/Turso)
- ✅ Realtime (Supabase Realtime → Server-Sent Events)
- ✅ File Storage (Supabase Storage → Local filesystem)
- ✅ Multi-tenancy (Basejump → Custom SQLite schema)

**Estimated Timeline:** 50-100 hours

---

## Phase 1: Authentication Migration (PRIORITY)

### Current Architecture Analysis

#### Backend (Python/FastAPI)
- **Location:** `backend/core/auth.py`, `backend/core/utils/auth_utils.py`
- **JWT Verification:** Uses Supabase JWT secret with HS256 algorithm
- **Token Flow:**
  - Bearer token in Authorization header
  - JWT payload contains `sub` (user_id)
  - Token verified with `SUPABASE_JWT_SECRET`
- **API Key Support:** Format `pk_xxx:sk_xxx` for programmatic access
- **Role System:** user, admin, super_admin (stored in `user_roles` table)
- **Dependencies:**
  - Supabase client for DB queries
  - Redis for caching account lookups

#### Frontend (Next.js)
- **Location:** `frontend/src/lib/supabase/client.ts`, `frontend/src/lib/supabase/server.ts`
- **Auth Client:** Uses `@supabase/ssr` package
- **Token Management:** Automatic via Supabase client
- **API Integration:** `frontend/src/lib/api-client.ts` attaches Bearer token
- **Middleware:** `frontend/src/middleware.ts` validates auth + billing checks

#### Mobile (React Native)
- **Location:** `apps/mobile/lib/utils/auth-types.ts`
- **Auth Types:** User, Session, SignInCredentials, SignUpCredentials
- **OAuth Support:** Google, GitHub, Apple

#### Basejump Multi-Tenant Schema
- **Location:** `backend/supabase/migrations/20240414161947_basejump-accounts.sql`
- **Tables:**
  - `basejump.accounts` - Organizations/workspaces
    - `id` (uuid, primary key)
    - `primary_owner_user_id` (references auth.users)
    - `name`, `slug`
    - `personal_account` (boolean)
    - `public_metadata`, `private_metadata` (jsonb)
  - `basejump.account_user` - Account memberships
    - `user_id`, `account_id` (composite key)
    - `account_role` (enum: 'owner', 'member')
- **Functions:**
  - `run_new_user_setup()` - Creates personal account on signup
  - `add_current_user_to_new_account()` - Adds user to new accounts
  - `has_role_on_account()` - Permission checking
  - `get_accounts_with_role()` - List user's accounts

### New Architecture Design

#### Database Setup

**Local Development:** SQLite file
```bash
database/local.db
```

**Production:** Turso (SQLite in the cloud)
```bash
# Connection string format
libsql://[database-name]-[org-name].turso.io
```

#### Auth Schema (SQLite)

```sql
-- Users table
CREATE TABLE users (
  id TEXT PRIMARY KEY,  -- UUID stored as TEXT
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,  -- Argon2 hash
  full_name TEXT,
  email_verified INTEGER DEFAULT 0,  -- SQLite boolean (0/1)
  created_at INTEGER NOT NULL,  -- Unix timestamp
  updated_at INTEGER NOT NULL,
  metadata TEXT  -- JSON stored as TEXT
);

-- Sessions table
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,  -- UUID
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  access_token TEXT UNIQUE NOT NULL,
  refresh_token TEXT UNIQUE NOT NULL,
  expires_at INTEGER NOT NULL,  -- Unix timestamp
  created_at INTEGER NOT NULL,
  user_agent TEXT,
  ip_address TEXT
);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_access_token ON sessions(access_token);

-- Accounts table (basejump equivalent)
CREATE TABLE accounts (
  id TEXT PRIMARY KEY,  -- UUID
  primary_owner_user_id TEXT NOT NULL REFERENCES users(id),
  name TEXT,
  slug TEXT UNIQUE,
  personal_account INTEGER NOT NULL DEFAULT 0,  -- SQLite boolean
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  public_metadata TEXT,  -- JSON
  private_metadata TEXT  -- JSON
);
CREATE INDEX idx_accounts_owner ON accounts(primary_owner_user_id);
CREATE INDEX idx_accounts_slug ON accounts(slug);

-- Account users table (basejump equivalent)
CREATE TABLE account_users (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  account_role TEXT NOT NULL CHECK(account_role IN ('owner', 'member')),
  created_at INTEGER NOT NULL,
  PRIMARY KEY (user_id, account_id)
);
CREATE INDEX idx_account_users_user ON account_users(user_id);
CREATE INDEX idx_account_users_account ON account_users(account_id);

-- User roles table
CREATE TABLE user_roles (
  user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK(role IN ('user', 'admin', 'super_admin'))
);

-- API keys table
CREATE TABLE api_keys (
  id TEXT PRIMARY KEY,  -- UUID
  account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  public_key TEXT UNIQUE NOT NULL,  -- pk_xxx
  secret_key_hash TEXT NOT NULL,  -- Hashed sk_xxx
  name TEXT,
  created_at INTEGER NOT NULL,
  last_used_at INTEGER
);
CREATE INDEX idx_api_keys_public ON api_keys(public_key);
CREATE INDEX idx_api_keys_account ON api_keys(account_id);
```

#### JWT Implementation

**Token Structure:**
```json
{
  "sub": "user_id",
  "email": "user@example.com",
  "iat": 1234567890,
  "exp": 1234571490,
  "type": "access"
}
```

**Libraries:**
- Python: `PyJWT`
- Secret: Environment variable `JWT_SECRET` (generate 256-bit random key)

**Token Lifetimes:**
- Access token: 15 minutes
- Refresh token: 30 days

#### Password Hashing
- Library: `argon2-cffi` (Python), `argon2` (JavaScript)
- Algorithm: Argon2id (winner of Password Hashing Competition)
- Parameters: Default secure settings

---

## Phase 2: Implementation Steps

### Step 1: Database Setup

#### 1.1 Install Dependencies
```bash
# Backend
cd backend
pip install libsql-client argon2-cffi PyJWT

# Frontend
cd frontend
npm install argon2-browser @libsql/client
```

#### 1.2 Create Migration Files
```bash
backend/database/migrations/
  001_create_users.sql
  002_create_sessions.sql
  003_create_accounts.sql
  004_create_account_users.sql
  005_create_user_roles.sql
  006_create_api_keys.sql
```

#### 1.3 Setup Turso
```bash
# Install Turso CLI
curl -sSfL https://get.tur.so/install.sh | bash

# Create database
turso db create agents-task-ui-prod

# Get connection URL
turso db show agents-task-ui-prod

# Create auth token
turso db tokens create agents-task-ui-prod
```

#### 1.4 Database Connection Layer
```python
# backend/core/database.py
from libsql_client import create_client
import os

class DatabaseConnection:
    def __init__(self):
        self.is_local = os.getenv("ENV") == "local"

        if self.is_local:
            self.client = create_client("file:database/local.db")
        else:
            self.client = create_client(
                url=os.getenv("TURSO_DATABASE_URL"),
                auth_token=os.getenv("TURSO_AUTH_TOKEN")
            )

    async def execute(self, query: str, params: list = None):
        return await self.client.execute(query, params or [])

    async def query(self, query: str, params: list = None):
        result = await self.client.execute(query, params or [])
        return result.rows
```

### Step 2: Backend Auth Implementation

#### 2.1 Password Hashing Utilities
```python
# backend/core/utils/password.py
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher()

def hash_password(password: str) -> str:
    return ph.hash(password)

def verify_password(password: str, hash: str) -> bool:
    try:
        ph.verify(hash, password)
        return True
    except VerifyMismatchError:
        return False
```

#### 2.2 JWT Utilities
```python
# backend/core/utils/jwt_utils.py
import jwt
from datetime import datetime, timedelta
from core.utils.config import config

def generate_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=15),
        "type": "access"
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm="HS256")

def generate_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=30),
        "type": "refresh"
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm="HS256")

def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, config.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### 2.3 Auth Endpoints
```python
# backend/api/routes/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import time

router = APIRouter(prefix="/auth", tags=["auth"])

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str = None

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    user: dict

@router.post("/signup", response_model=TokenResponse)
async def signup(req: SignupRequest, db: DatabaseConnection = Depends(get_db)):
    # Check if user exists
    existing = await db.query(
        "SELECT id FROM users WHERE email = ?",
        [req.email]
    )
    if existing:
        raise HTTPException(400, "Email already registered")

    # Create user
    user_id = str(uuid.uuid4())
    password_hash = hash_password(req.password)
    now = int(time.time())

    await db.execute(
        "INSERT INTO users (id, email, password_hash, full_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        [user_id, req.email, password_hash, req.full_name, now, now]
    )

    # Create personal account (basejump behavior)
    account_id = user_id  # Use same ID for personal account
    await db.execute(
        "INSERT INTO accounts (id, primary_owner_user_id, name, personal_account, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
        [account_id, user_id, req.full_name or req.email.split('@')[0], now, now]
    )

    # Add user to account
    await db.execute(
        "INSERT INTO account_users (user_id, account_id, account_role, created_at) VALUES (?, ?, 'owner', ?)",
        [user_id, account_id, now]
    )

    # Create default role
    await db.execute(
        "INSERT INTO user_roles (user_id, role) VALUES (?, 'user')",
        [user_id]
    )

    # Generate tokens
    access_token = generate_access_token(user_id, req.email)
    refresh_token = generate_refresh_token(user_id)

    # Store session
    session_id = str(uuid.uuid4())
    expires_at = int(time.time()) + (30 * 24 * 60 * 60)  # 30 days
    await db.execute(
        "INSERT INTO sessions (id, user_id, access_token, refresh_token, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [session_id, user_id, access_token, refresh_token, expires_at, now]
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {
            "id": user_id,
            "email": req.email,
            "full_name": req.full_name
        }
    }

@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: DatabaseConnection = Depends(get_db)):
    # Find user
    users = await db.query(
        "SELECT id, email, password_hash, full_name FROM users WHERE email = ?",
        [req.email]
    )
    if not users:
        raise HTTPException(401, "Invalid credentials")

    user = users[0]

    # Verify password
    if not verify_password(req.password, user['password_hash']):
        raise HTTPException(401, "Invalid credentials")

    # Generate tokens (same as signup)
    # ...

@router.post("/refresh")
async def refresh(refresh_token: str, db: DatabaseConnection = Depends(get_db)):
    # Verify refresh token
    payload = verify_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid token type")

    user_id = payload["sub"]

    # Check session exists
    sessions = await db.query(
        "SELECT id FROM sessions WHERE user_id = ? AND refresh_token = ?",
        [user_id, refresh_token]
    )
    if not sessions:
        raise HTTPException(401, "Session not found")

    # Get user
    users = await db.query(
        "SELECT email FROM users WHERE id = ?",
        [user_id]
    )
    if not users:
        raise HTTPException(401, "User not found")

    # Generate new access token
    access_token = generate_access_token(user_id, users[0]['email'])

    # Update session
    await db.execute(
        "UPDATE sessions SET access_token = ? WHERE user_id = ? AND refresh_token = ?",
        [access_token, user_id, refresh_token]
    )

    return {"access_token": access_token}

@router.post("/logout")
async def logout(
    request: Request,
    db: DatabaseConnection = Depends(get_db)
):
    # Get token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing token")

    token = auth_header.split(" ")[1]

    # Delete session
    await db.execute(
        "DELETE FROM sessions WHERE access_token = ?",
        [token]
    )

    return {"message": "Logged out"}
```

#### 2.4 Update Auth Middleware
```python
# backend/core/auth.py (replace existing)
from fastapi import HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.utils.jwt_utils import verify_token
from core.database import DatabaseConnection

security = HTTPBearer()

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: DatabaseConnection = Depends(get_db)
) -> dict:
    token = credentials.credentials

    # Verify JWT
    payload = verify_token(token)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(401, "Invalid token")

    # Check session exists
    sessions = await db.query(
        "SELECT id FROM sessions WHERE access_token = ?",
        [token]
    )
    if not sessions:
        raise HTTPException(401, "Session expired")

    return {
        "user_id": user_id,
        "token": token
    }
```

### Step 3: Frontend Auth Implementation

#### 3.1 Create Auth Context
```typescript
// frontend/src/contexts/auth-context.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';

interface User {
  id: string;
  email: string;
  full_name?: string;
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

interface AuthContextType extends AuthState {
  signup: (email: string, password: string, fullName?: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    accessToken: null,
    refreshToken: null,
    isLoading: true,
    isAuthenticated: false,
  });

  // Load tokens from localStorage on mount
  useEffect(() => {
    const accessToken = localStorage.getItem('access_token');
    const refreshToken = localStorage.getItem('refresh_token');
    const userStr = localStorage.getItem('user');

    if (accessToken && refreshToken && userStr) {
      setState({
        user: JSON.parse(userStr),
        accessToken,
        refreshToken,
        isLoading: false,
        isAuthenticated: true,
      });
    } else {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  // Auto-refresh token before expiry (15 min - 1 min = 14 min)
  useEffect(() => {
    if (!state.refreshToken) return;

    const interval = setInterval(() => {
      refreshSession();
    }, 14 * 60 * 1000);  // 14 minutes

    return () => clearInterval(interval);
  }, [state.refreshToken]);

  const signup = async (email: string, password: string, fullName?: string) => {
    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name: fullName }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    const data = await response.json();

    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    localStorage.setItem('user', JSON.stringify(data.user));

    setState({
      user: data.user,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      isLoading: false,
      isAuthenticated: true,
    });
  };

  const login = async (email: string, password: string) => {
    const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();

    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    localStorage.setItem('user', JSON.stringify(data.user));

    setState({
      user: data.user,
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      isLoading: false,
      isAuthenticated: true,
    });
  };

  const logout = async () => {
    if (state.accessToken) {
      await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${state.accessToken}` },
      });
    }

    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');

    setState({
      user: null,
      accessToken: null,
      refreshToken: null,
      isLoading: false,
      isAuthenticated: false,
    });
  };

  const refreshSession = async () => {
    if (!state.refreshToken) return;

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: state.refreshToken }),
      });

      if (!response.ok) {
        await logout();
        return;
      }

      const data = await response.json();

      localStorage.setItem('access_token', data.access_token);

      setState(prev => ({
        ...prev,
        accessToken: data.access_token,
      }));
    } catch (error) {
      await logout();
    }
  };

  return (
    <AuthContext.Provider value={{ ...state, signup, login, logout, refreshSession }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```

#### 3.2 Update API Client
```typescript
// frontend/src/lib/api-client.ts (replace supabase auth)
async function makeRequest<T = any>(
  url: string,
  options: RequestInit & ApiClientOptions = {}
): Promise<ApiResponse<T>> {
  // ... existing code ...

  // Replace Supabase auth with custom auth
  const accessToken = localStorage.getItem('access_token');

  const headers: Record<string, string> = {};

  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }

  Object.assign(headers, fetchOptions.headers as Record<string, string>);

  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  // ... rest of existing code ...
}
```

#### 3.3 Update Middleware
```typescript
// frontend/src/middleware.ts (replace Supabase auth)
export async function middleware(request: NextRequest) {
  // ... existing code ...

  // Replace Supabase auth with custom auth check
  const accessToken = request.cookies.get('access_token')?.value;

  if (!accessToken) {
    // Redirect to login
    const url = request.nextUrl.clone();
    url.pathname = '/auth';
    url.searchParams.set('redirect', pathname);
    return NextResponse.redirect(url);
  }

  // Optionally verify token with backend
  // Or just trust it (verification happens on backend anyway)

  return supabaseResponse;
}
```

### Step 4: Mobile App Auth

#### 4.1 Create Auth Service
```typescript
// apps/mobile/lib/services/auth-service.ts
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL || '';

export class AuthService {
  static async signup(email: string, password: string, fullName?: string) {
    const response = await fetch(`${API_URL}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name: fullName }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Signup failed');
    }

    const data = await response.json();

    await AsyncStorage.setItem('access_token', data.access_token);
    await AsyncStorage.setItem('refresh_token', data.refresh_token);
    await AsyncStorage.setItem('user', JSON.stringify(data.user));

    return data;
  }

  static async login(email: string, password: string) {
    const response = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();

    await AsyncStorage.setItem('access_token', data.access_token);
    await AsyncStorage.setItem('refresh_token', data.refresh_token);
    await AsyncStorage.setItem('user', JSON.stringify(data.user));

    return data;
  }

  static async logout() {
    const accessToken = await AsyncStorage.getItem('access_token');

    if (accessToken) {
      await fetch(`${API_URL}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${accessToken}` },
      });
    }

    await AsyncStorage.removeItem('access_token');
    await AsyncStorage.removeItem('refresh_token');
    await AsyncStorage.removeItem('user');
  }

  static async getAccessToken(): Promise<string | null> {
    return await AsyncStorage.getItem('access_token');
  }

  static async refreshToken() {
    const refreshToken = await AsyncStorage.getItem('refresh_token');

    if (!refreshToken) {
      throw new Error('No refresh token');
    }

    const response = await fetch(`${API_URL}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      await this.logout();
      throw new Error('Refresh failed');
    }

    const data = await response.json();
    await AsyncStorage.setItem('access_token', data.access_token);

    return data.access_token;
  }
}
```

---

## Phase 3: Database Migration

### Step 1: Migrate Core Tables
- All 180+ Supabase migrations need conversion
- PostgreSQL → SQLite syntax differences
- UUID → TEXT conversion
- Timestamp → INTEGER (Unix timestamps)
- JSONB → TEXT

### Step 2: Data Migration Script
```python
# scripts/migrate_data.py
# Connect to Supabase (source) and SQLite (dest)
# Migrate users, accounts, agents, threads, messages, etc.
```

---

## Phase 4: Realtime Migration (SSE)

### Backend SSE Implementation
```python
# backend/api/routes/realtime.py
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

@router.get("/realtime/presence/{thread_id}")
async def presence_stream(thread_id: str, user_id: str = Depends(get_current_user)):
    async def event_generator():
        while True:
            # Fetch presence data
            # yield event
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())
```

### Frontend SSE Client
```typescript
// frontend/src/hooks/use-realtime.ts
import { useEffect, useState } from 'react';

export function useRealtimePresence(threadId: string) {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    const accessToken = localStorage.getItem('access_token');
    const eventSource = new EventSource(
      `${API_URL}/realtime/presence/${threadId}?token=${accessToken}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setUsers(data.users);
    };

    return () => eventSource.close();
  }, [threadId]);

  return users;
}
```

---

## Phase 5: File Storage Migration

### Local Filesystem Storage
```python
# backend/core/storage.py
import os
import uuid
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def upload_file(file: UploadFile, account_id: str) -> str:
    # Create account directory
    account_dir = UPLOAD_DIR / account_id
    account_dir.mkdir(exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    filepath = account_dir / f"{file_id}{ext}"

    # Save file
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    return file_id

async def get_file(file_id: str, account_id: str) -> Path:
    filepath = UPLOAD_DIR / account_id / f"{file_id}.*"
    # Find file
    # Return path
```

---

## Testing Checklist

### Authentication
- [ ] User signup creates user + personal account
- [ ] Login returns valid tokens
- [ ] Token refresh works
- [ ] Logout invalidates session
- [ ] Protected routes require auth
- [ ] Role-based access control works
- [ ] API key authentication works
- [ ] Mobile auth works

### Multi-Tenancy
- [ ] Personal accounts created automatically
- [ ] Team accounts can be created
- [ ] Users can be added to accounts
- [ ] Role permissions work (owner vs member)
- [ ] Account isolation works

### Database
- [ ] Local SQLite works
- [ ] Turso connection works
- [ ] All queries converted from PostgreSQL
- [ ] Indexes perform well

### Realtime
- [ ] SSE connections work
- [ ] Presence tracking works
- [ ] Thread updates stream correctly

### File Storage
- [ ] Files upload to local filesystem
- [ ] File retrieval works
- [ ] Account isolation for files

---

## Environment Variables

```bash
# Backend (.env)
JWT_SECRET=<generate-256-bit-random-key>
ENV=local  # or production
TURSO_DATABASE_URL=libsql://....turso.io
TURSO_AUTH_TOKEN=<turso-token>

# Frontend (.env.local)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_ENV_MODE=local  # or production

# Mobile (.env)
EXPO_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Rollback Plan
1. Keep Supabase instance running during migration
2. Implement feature flag to switch between old/new auth
3. Gradual rollout: 10% → 50% → 100%
4. Database backups before major migrations

---

## Next Steps
1. ✅ Complete this migration plan
2. ⏳ Begin implementation (Step 1: Database Setup)
3. Continue with auth endpoints
4. Frontend integration
5. Mobile integration
6. Testing
7. Production deployment
