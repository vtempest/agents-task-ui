/**
 * Agent API - Get, Update, Delete
 * GET /api/agents/[agentId] - Get specific agent
 * PUT /api/agents/[agentId] - Update agent
 * DELETE /api/agents/[agentId] - Delete agent
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyAgentAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { AgentUpdateRequest, AgentResponse } from '@/lib/api/types';

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

    // Fetch agent
    const { data: agent, error } = await supabaseAdmin
      .from('agents')
      .select('*')
      .eq('agent_id', agentId)
      .single();

    if (error || !agent) {
      return NextResponse.json(
        { error: 'Agent not found' },
        { status: 404 }
      );
    }

    const response: AgentResponse = { agent };
    return NextResponse.json(response);
  } catch (error) {
    console.error('Agent fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { agentId } = await context.params;
    const body: AgentUpdateRequest = await request.json();

    // Verify access
    const hasAccess = await verifyAgentAccess(agentId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Get existing agent to check for restrictions
    const { data: existingAgent, error: fetchError } = await supabaseAdmin
      .from('agents')
      .select('*')
      .eq('agent_id', agentId)
      .single();

    if (fetchError || !existingAgent) {
      return NextResponse.json(
        { error: 'Agent not found' },
        { status: 404 }
      );
    }

    // Check for Suna default agent restrictions
    const metadata = existingAgent.metadata || {};
    const isSunaAgent = metadata.is_suna_default === true;
    const restrictions = metadata.restrictions || {};

    if (isSunaAgent) {
      // Validate restricted fields
      if (body.name !== undefined && body.name !== existingAgent.name && restrictions.name_editable === false) {
        return NextResponse.json(
          { error: "Suna's name cannot be modified" },
          { status: 403 }
        );
      }
      if (body.system_prompt !== undefined && restrictions.system_prompt_editable === false) {
        return NextResponse.json(
          { error: "Suna's system prompt cannot be modified" },
          { status: 403 }
        );
      }
      if (body.agentpress_tools !== undefined && restrictions.tools_editable === false) {
        return NextResponse.json(
          { error: "Suna's default tools cannot be modified" },
          { status: 403 }
        );
      }
      if ((body.configured_mcps !== undefined || body.custom_mcps !== undefined) &&
          restrictions.mcps_editable === false) {
        return NextResponse.json(
          { error: "Suna's integrations cannot be modified" },
          { status: 403 }
        );
      }
    }

    // Prepare update data (only include provided fields)
    const updateData: any = {};
    if (body.name !== undefined) updateData.name = body.name;
    if (body.description !== undefined) updateData.description = body.description;
    if (body.system_prompt !== undefined) updateData.system_prompt = body.system_prompt;
    if (body.model !== undefined) updateData.model = body.model;
    if (body.icon_name !== undefined) updateData.icon_name = body.icon_name;
    if (body.icon_color !== undefined) updateData.icon_color = body.icon_color;
    if (body.icon_background !== undefined) updateData.icon_background = body.icon_background;
    if (body.agentpress_tools !== undefined) updateData.agentpress_tools = body.agentpress_tools;
    if (body.configured_mcps !== undefined) updateData.configured_mcps = body.configured_mcps;
    if (body.custom_mcps !== undefined) updateData.custom_mcps = body.custom_mcps;
    if (body.metadata !== undefined) updateData.metadata = { ...existingAgent.metadata, ...body.metadata };
    if (body.is_public !== undefined) updateData.is_public = body.is_public;

    updateData.updated_at = new Date().toISOString();

    // Update agent
    const { data: agent, error } = await supabaseAdmin
      .from('agents')
      .update(updateData)
      .eq('agent_id', agentId)
      .select()
      .single();

    if (error) {
      console.error('Error updating agent:', error);
      return NextResponse.json(
        { error: 'Failed to update agent' },
        { status: 500 }
      );
    }

    const response: AgentResponse = { agent };
    return NextResponse.json(response);
  } catch (error) {
    console.error('Agent update error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function DELETE(
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

    // Delete agent (cascades will handle related records)
    const { error } = await supabaseAdmin
      .from('agents')
      .delete()
      .eq('agent_id', agentId);

    if (error) {
      console.error('Error deleting agent:', error);
      return NextResponse.json(
        { error: 'Failed to delete agent' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'Agent deleted successfully' });
  } catch (error) {
    console.error('Agent deletion error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
