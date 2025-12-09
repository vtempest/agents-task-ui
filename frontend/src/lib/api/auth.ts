/**
 * Authentication utilities for Next.js API routes
 * Handles JWT verification and user authorization
 */

import { NextRequest } from 'next/server';
import { supabaseAdmin } from './db';

export interface AuthenticatedUser {
  userId: string;
  email?: string;
}

/**
 * Extract and verify JWT token from request headers
 * @throws Error if token is invalid or missing
 */
export async function verifyAndGetUserId(request: NextRequest): Promise<string> {
  const authHeader = request.headers.get('authorization');

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    throw new Error('Missing or invalid authorization header');
  }

  const token = authHeader.substring(7);

  try {
    // Verify JWT token with Supabase
    const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);

    if (error || !user) {
      throw new Error('Invalid token');
    }

    return user.id;
  } catch (error) {
    throw new Error('Authentication failed');
  }
}

/**
 * Optional user ID extraction (for public endpoints that support both auth and anon)
 */
export async function getOptionalUserId(request: NextRequest): Promise<string | null> {
  try {
    return await verifyAndGetUserId(request);
  } catch {
    return null;
  }
}

/**
 * Verify user has access to a specific thread
 */
export async function verifyThreadAccess(
  threadId: string,
  userId: string
): Promise<boolean> {
  const { data, error } = await supabaseAdmin
    .from('threads')
    .select('thread_id, account_id, is_public')
    .eq('thread_id', threadId)
    .single();

  if (error || !data) {
    return false;
  }

  // Allow access if user owns thread or thread is public
  return data.account_id === userId || data.is_public === true;
}

/**
 * Verify user has access to a specific agent
 */
export async function verifyAgentAccess(
  agentId: string,
  userId: string
): Promise<boolean> {
  const { data, error } = await supabaseAdmin
    .from('agents')
    .select('agent_id, account_id')
    .eq('agent_id', agentId)
    .single();

  if (error || !data) {
    return false;
  }

  return data.account_id === userId;
}

/**
 * Verify user has access to a specific project
 */
export async function verifyProjectAccess(
  projectId: string,
  userId: string
): Promise<boolean> {
  const { data, error } = await supabaseAdmin
    .from('projects')
    .select('project_id, account_id, is_public')
    .eq('project_id', projectId)
    .single();

  if (error || !data) {
    return false;
  }

  return data.account_id === userId || data.is_public === true;
}
