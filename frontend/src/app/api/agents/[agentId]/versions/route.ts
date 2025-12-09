/**
 * Agent Versions API - List and Create versions
 * GET /api/agents/[agentId]/versions - List all versions
 * POST /api/agents/[agentId]/versions - Create new version
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyAgentAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { AgentVersionCreateRequest, AgentVersionsResponse } from '@/lib/api/types';

interface RouteContext {
  params: Promise<{ agentId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { agentId } = await context.params;

    // Verify access
    const hasAccess = await verifyAgentAccess(agentId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Fetch versions
    const { data: versions, error } = await supabaseAdmin
      .from('agent_versions')
      .select('*')
      .eq('agent_id', agentId)
      .order('version_number', { ascending: false });

    if (error) {
      console.error('Error fetching agent versions:', error);
      return NextResponse.json(
        { error: 'Failed to fetch agent versions' },
        { status: 500 }
      );
    }

    const response: AgentVersionsResponse = {
      versions: versions || [],
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Agent versions fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function POST(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { agentId } = await context.params;
    const body: AgentVersionCreateRequest = await request.json();

    // Verify access
    const hasAccess = await verifyAgentAccess(agentId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Get current agent data
    const { data: agent, error: agentError } = await supabaseAdmin
      .from('agents')
      .select('*')
      .eq('agent_id', agentId)
      .single();

    if (agentError || !agent) {
      return NextResponse.json(
        { error: 'Agent not found' },
        { status: 404 }
      );
    }

    // Get current max version number
    const { data: maxVersionData } = await supabaseAdmin
      .from('agent_versions')
      .select('version_number')
      .eq('agent_id', agentId)
      .order('version_number', { ascending: false })
      .limit(1)
      .single();

    const nextVersionNumber = (maxVersionData?.version_number || 0) + 1;

    // Prepare version data
    const versionData = {
      agent_id: agentId,
      version_number: nextVersionNumber,
      version_name: body.version_name || `v${nextVersionNumber}`,
      system_prompt: body.system_prompt || agent.system_prompt || '',
      model: body.model || agent.model,
      agentpress_tools: body.agentpress_tools || agent.agentpress_tools || {},
      configured_mcps: body.configured_mcps || agent.configured_mcps || [],
      custom_mcps: body.custom_mcps || agent.custom_mcps || [],
      is_active: false, // New versions start as inactive
      created_by: userId,
    };

    // Insert version
    const { data: version, error } = await supabaseAdmin
      .from('agent_versions')
      .insert(versionData)
      .select()
      .single();

    if (error) {
      console.error('Error creating agent version:', error);
      return NextResponse.json(
        { error: 'Failed to create agent version' },
        { status: 500 }
      );
    }

    // Update agent's version count
    await supabaseAdmin
      .from('agents')
      .update({
        version_count: nextVersionNumber,
      })
      .eq('agent_id', agentId);

    return NextResponse.json({ version }, { status: 201 });
  } catch (error) {
    console.error('Agent version creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
