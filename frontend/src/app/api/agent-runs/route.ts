/**
 * Agent Runs API - List and Start agent runs
 * GET /api/agent-runs - List agent runs
 * POST /api/agent-runs/start - Start a new agent run
 *
 * Note: This is a simplified version. For production, you'd need to:
 * 1. Integrate with the Python background worker via Redis
 * 2. Handle streaming responses
 * 3. Implement proper queue management
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyThreadAccess, verifyAgentAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { AgentRunStartRequest, AgentRunResponse } from '@/lib/api/types';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const searchParams = request.nextUrl.searchParams;

    // Optional filters
    const threadId = searchParams.get('thread_id');
    const agentId = searchParams.get('agent_id');
    const status = searchParams.get('status');

    // Pagination
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 500);
    const offset = (page - 1) * limit;

    // Build query
    let query = supabaseAdmin
      .from('agent_runs')
      .select(`
        *,
        threads!inner(account_id)
      `, { count: 'exact' })
      .eq('threads.account_id', userId)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    // Apply filters
    if (threadId) {
      query = query.eq('thread_id', threadId);
    }
    if (agentId) {
      query = query.eq('agent_id', agentId);
    }
    if (status) {
      query = query.eq('status', status);
    }

    const { data: runs, error, count } = await query;

    if (error) {
      console.error('Error fetching agent runs:', error);
      return NextResponse.json(
        { error: 'Failed to fetch agent runs' },
        { status: 500 }
      );
    }

    const totalPages = count ? Math.ceil(count / limit) : 0;

    return NextResponse.json({
      runs: runs || [],
      pagination: {
        page,
        limit,
        total: count || 0,
        pages: totalPages,
      },
    });
  } catch (error) {
    console.error('Agent runs fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body: AgentRunStartRequest = await request.json();

    // Validate required fields
    if (!body.thread_id || !body.agent_id) {
      return NextResponse.json(
        { error: 'thread_id and agent_id are required' },
        { status: 400 }
      );
    }

    // Verify thread access
    const hasThreadAccess = await verifyThreadAccess(body.thread_id, userId);
    if (!hasThreadAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Verify agent access
    const hasAgentAccess = await verifyAgentAccess(body.agent_id, userId);
    if (!hasAgentAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Create user message if provided
    if (body.message) {
      await supabaseAdmin
        .from('messages')
        .insert({
          thread_id: body.thread_id,
          role: 'user',
          content: body.message,
          metadata: {},
        });
    }

    // Create agent run
    const runData = {
      thread_id: body.thread_id,
      agent_id: body.agent_id,
      status: 'queued',
      metadata: body.metadata || {},
    };

    const { data: run, error } = await supabaseAdmin
      .from('agent_runs')
      .insert(runData)
      .select()
      .single();

    if (error) {
      console.error('Error creating agent run:', error);
      return NextResponse.json(
        { error: 'Failed to create agent run' },
        { status: 500 }
      );
    }

    // TODO: Queue the run for processing by the Python background worker
    // This would typically involve:
    // 1. Publishing to Redis queue
    // 2. Python worker picks up the job via Dramatiq
    // 3. Python worker processes the agent run
    // 4. Results are streamed back via Redis and SSE

    const response: AgentRunResponse = { run };
    return NextResponse.json(response, { status: 201 });
  } catch (error) {
    console.error('Agent run creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
