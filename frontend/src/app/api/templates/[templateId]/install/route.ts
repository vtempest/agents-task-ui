/**
 * Template Installation API
 * POST /api/templates/{templateId}/install - Install template as new agent
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ templateId: string }>;
}

export async function POST(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { templateId } = await context.params;
    const body = await request.json();

    // Get template
    const { data: template, error: templateError } = await supabaseAdmin
      .from('agent_templates')
      .select('*')
      .eq('template_id', templateId)
      .single();

    if (templateError || !template) {
      return NextResponse.json(
        { error: 'Template not found' },
        { status: 404 }
      );
    }

    // Check access
    if (!template.is_public && template.creator_id !== userId) {
      return NextResponse.json(
        { error: 'Access denied' },
        { status: 403 }
      );
    }

    // Create new agent from template
    const agentName = body.instance_name || template.name;
    const systemPrompt = body.custom_system_prompt || template.system_prompt;

    const agentData = {
      account_id: userId,
      name: agentName,
      description: template.metadata?.description || '',
      system_prompt: systemPrompt,
      model: template.model,
      icon_name: template.icon_name,
      icon_color: template.icon_color,
      icon_background: template.icon_background,
      agentpress_tools: template.agentpress_tools,
      configured_mcps: template.configured_mcps || [],
      custom_mcps: template.custom_mcps || [],
      metadata: {
        installed_from_template: templateId,
        template_name: template.name,
      },
      version_count: 0,
    };

    const { data: agent, error: agentError } = await supabaseAdmin
      .from('agents')
      .insert(agentData)
      .select()
      .single();

    if (agentError) {
      console.error('Error creating agent from template:', agentError);
      return NextResponse.json(
        { error: 'Failed to install template' },
        { status: 500 }
      );
    }

    // Increment download count
    await supabaseAdmin
      .from('agent_templates')
      .update({
        download_count: (template.download_count || 0) + 1,
      })
      .eq('template_id', templateId);

    return NextResponse.json({
      status: 'success',
      instance_id: agent.agent_id,
      name: agent.name,
      message: 'Template installed successfully',
    }, { status: 201 });
  } catch (error) {
    console.error('Template installation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
