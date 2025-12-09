/**
 * Type definitions for API requests and responses
 */

// ============================================================================
// Common Types
// ============================================================================

export interface PaginationParams {
  page?: number;
  limit?: number;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  pages: number;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

// ============================================================================
// Agent Types
// ============================================================================

export interface Agent {
  agent_id: string;
  account_id: string;
  name: string;
  description?: string;
  system_prompt?: string;
  model?: string;
  icon_name?: string;
  icon_color?: string;
  icon_background?: string;
  agentpress_tools?: Record<string, any>;
  configured_mcps?: any[];
  custom_mcps?: any[];
  current_version_id?: string;
  version_count?: number;
  metadata?: Record<string, any>;
  is_public?: boolean;
  created_at: string;
  updated_at: string;
}

export interface AgentCreateRequest {
  name: string;
  description?: string;
  system_prompt?: string;
  model?: string;
  icon_name?: string;
  icon_color?: string;
  icon_background?: string;
  agentpress_tools?: Record<string, any>;
  configured_mcps?: any[];
  custom_mcps?: any[];
  metadata?: Record<string, any>;
  is_public?: boolean;
}

export interface AgentUpdateRequest extends Partial<AgentCreateRequest> {
  // All fields from create are optional in update
}

export interface AgentResponse {
  agent: Agent;
}

export interface AgentsResponse {
  agents: Agent[];
  pagination: PaginationInfo;
}

// ============================================================================
// Thread Types
// ============================================================================

export interface Thread {
  thread_id: string;
  account_id: string;
  project_id?: string;
  metadata?: Record<string, any>;
  is_public?: boolean;
  created_at: string;
  updated_at: string;
  project?: Project;
}

export interface ThreadCreateRequest {
  project_id?: string;
  metadata?: Record<string, any>;
  is_public?: boolean;
}

export interface ThreadsResponse {
  threads: Thread[];
  pagination: PaginationInfo;
}

// ============================================================================
// Message Types
// ============================================================================

export interface Message {
  message_id: string;
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface MessageCreateRequest {
  thread_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, any>;
}

export interface MessagesResponse {
  messages: Message[];
  pagination?: PaginationInfo;
}

// ============================================================================
// Project Types
// ============================================================================

export interface Project {
  project_id: string;
  account_id: string;
  name: string;
  description?: string;
  icon_name?: string;
  sandbox?: Record<string, any>;
  is_public?: boolean;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreateRequest {
  name: string;
  description?: string;
  icon_name?: string;
  sandbox?: Record<string, any>;
  is_public?: boolean;
}

// ============================================================================
// Agent Run Types
// ============================================================================

export interface AgentRun {
  run_id: string;
  thread_id: string;
  agent_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  error?: string;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

export interface AgentRunStartRequest {
  thread_id: string;
  agent_id: string;
  message?: string;
  metadata?: Record<string, any>;
}

export interface AgentRunResponse {
  run: AgentRun;
}

// ============================================================================
// Agent Version Types
// ============================================================================

export interface AgentVersion {
  version_id: string;
  agent_id: string;
  version_number: number;
  version_name: string;
  system_prompt: string;
  model?: string;
  agentpress_tools?: Record<string, any>;
  configured_mcps?: any[];
  custom_mcps?: any[];
  config?: Record<string, any>;
  is_active: boolean;
  created_by: string;
  created_at: string;
}

export interface AgentVersionCreateRequest {
  version_name?: string;
  system_prompt?: string;
  model?: string;
  agentpress_tools?: Record<string, any>;
  configured_mcps?: any[];
  custom_mcps?: any[];
}

export interface AgentVersionsResponse {
  versions: AgentVersion[];
}

// ============================================================================
// Health Check Types
// ============================================================================

export interface HealthCheckResponse {
  status: 'ok' | 'error';
  timestamp: string;
  instance_id?: string;
}
