/**
 * Agents API - List and Create
 * GET /api/agents - List all agents for authenticated user
 * POST /api/agents - Create a new agent
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { AgentCreateRequest, AgentsResponse, AgentResponse } from '@/lib/api/types';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const searchParams = request.nextUrl.searchParams;

    // Pagination parameters
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);
    const offset = (page - 1) * limit;

    // Optional filters
    const search = searchParams.get('search');
    const isPublic = searchParams.get('is_public');

    // Build query
    let query = supabaseAdmin
      .from('agents')
      .select('*', { count: 'exact' })
      .eq('account_id', userId)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    // Apply filters
    if (search) {
      query = query.or(`name.ilike.%${search}%,description.ilike.%${search}%`);
    }
    if (isPublic !== null) {
      query = query.eq('is_public', isPublic === 'true');
    }

    const { data: agents, error, count } = await query;

    if (error) {
      console.error('Error fetching agents:', error);
      return NextResponse.json(
        { error: 'Failed to fetch agents' },
        { status: 500 }
      );
    }

    const totalPages = count ? Math.ceil(count / limit) : 0;

    const response: AgentsResponse = {
      agents: agents || [],
      pagination: {
        page,
        limit,
        total: count || 0,
        pages: totalPages,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Agent list error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body: AgentCreateRequest = await request.json();

    // Validate required fields
    if (!body.name) {
      return NextResponse.json(
        { error: 'Agent name is required' },
        { status: 400 }
      );
    }

    // Prepare agent data
    const agentData = {
      account_id: userId,
      name: body.name,
      description: body.description || '',
      system_prompt: body.system_prompt || '',
      model: body.model || 'claude-3-5-sonnet-20241022',
      icon_name: body.icon_name,
      icon_color: body.icon_color,
      icon_background: body.icon_background,
      agentpress_tools: body.agentpress_tools || {},
      configured_mcps: body.configured_mcps || [],
      custom_mcps: body.custom_mcps || [],
      metadata: body.metadata || {},
      is_public: body.is_public || false,
      version_count: 0,
    };

    // Insert agent
    const { data: agent, error } = await supabaseAdmin
      .from('agents')
      .insert(agentData)
      .select()
      .single();

    if (error) {
      console.error('Error creating agent:', error);
      return NextResponse.json(
        { error: 'Failed to create agent' },
        { status: 500 }
      );
    }

    const response: AgentResponse = { agent };
    return NextResponse.json(response, { status: 201 });
  } catch (error) {
    console.error('Agent creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
