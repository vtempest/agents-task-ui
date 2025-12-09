/**
 * Templates API - Marketplace
 * GET /api/templates - List public templates
 * POST /api/templates - Create template from agent
 * GET /api/templates/{templateId} - Get template details
 * POST /api/templates/{templateId}/install - Install template
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, getOptionalUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;

    // Optional authentication (for public templates)
    const userId = await getOptionalUserId(request);

    // Pagination
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 100);
    const offset = (page - 1) * limit;

    // Filters
    const category = searchParams.get('category');
    const tag = searchParams.get('tag');
    const search = searchParams.get('search');

    // Build query for public templates
    let query = supabaseAdmin
      .from('agent_templates')
      .select('*, profiles(name)', { count: 'exact' })
      .eq('is_public', true)
      .order('download_count', { ascending: false })
      .range(offset, offset + limit - 1);

    // Apply filters
    if (category) {
      query = query.contains('categories', [category]);
    }
    if (tag) {
      query = query.contains('tags', [tag]);
    }
    if (search) {
      query = query.or(`name.ilike.%${search}%,metadata->description.ilike.%${search}%`);
    }

    const { data: templates, error, count } = await query;

    if (error) {
      console.error('Error fetching templates:', error);
      return NextResponse.json(
        { error: 'Failed to fetch templates' },
        { status: 500 }
      );
    }

    const totalPages = count ? Math.ceil(count / limit) : 0;

    return NextResponse.json({
      templates: templates || [],
      pagination: {
        page,
        limit,
        total: count || 0,
        pages: totalPages,
      },
    });
  } catch (error) {
    console.error('Templates fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch templates' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body = await request.json();

    // Validate required fields
    if (!body.agent_id) {
      return NextResponse.json(
        { error: 'agent_id is required' },
        { status: 400 }
      );
    }

    // Verify agent ownership
    const { data: agent, error: agentError } = await supabaseAdmin
      .from('agents')
      .select('*')
      .eq('agent_id', body.agent_id)
      .eq('account_id', userId)
      .single();

    if (agentError || !agent) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Create template from agent
    const templateData = {
      creator_id: userId,
      name: agent.name,
      system_prompt: agent.system_prompt,
      model: agent.model,
      agentpress_tools: agent.agentpress_tools,
      configured_mcps: agent.configured_mcps || [],
      custom_mcps: agent.custom_mcps || [],
      icon_name: agent.icon_name,
      icon_color: agent.icon_color,
      icon_background: agent.icon_background,
      tags: body.tags || [],
      categories: body.categories || [],
      is_public: body.make_public || false,
      download_count: 0,
      metadata: {
        description: agent.description,
        source_agent_id: body.agent_id,
        ...body.metadata,
      },
    };

    const { data: template, error } = await supabaseAdmin
      .from('agent_templates')
      .insert(templateData)
      .select()
      .single();

    if (error) {
      console.error('Error creating template:', error);
      return NextResponse.json(
        { error: 'Failed to create template' },
        { status: 500 }
      );
    }

    return NextResponse.json({ template }, { status: 201 });
  } catch (error) {
    console.error('Template creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
