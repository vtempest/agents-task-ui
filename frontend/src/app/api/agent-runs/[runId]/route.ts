/**
 * Agent Run API - Get and Stop specific agent run
 * GET /api/agent-runs/[runId] - Get agent run status
 * POST /api/agent-runs/[runId]/stop - Stop a running agent
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyThreadAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ runId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { runId } = await context.params;

    // Fetch agent run with thread info
    const { data: run, error } = await supabaseAdmin
      .from('agent_runs')
      .select(`
        *,
        threads!inner(account_id, thread_id, project_id, metadata)
      `)
      .eq('run_id', runId)
      .single();

    if (error || !run) {
      return NextResponse.json(
        { error: 'Agent run not found' },
        { status: 404 }
      );
    }

    // Verify user owns the thread
    if (run.threads.account_id !== userId) {
      return NextResponse.json(
        { error: 'Access denied' },
        { status: 403 }
      );
    }

    return NextResponse.json({ run });
  } catch (error) {
    console.error('Agent run fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
