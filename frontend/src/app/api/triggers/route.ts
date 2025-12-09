/**
 * Triggers API - Automation and Scheduling
 * GET /api/triggers - List triggers for agent
 * POST /api/triggers - Create new trigger
 * GET /api/triggers/{triggerId} - Get trigger details
 * PUT /api/triggers/{triggerId} - Update trigger
 * DELETE /api/triggers/{triggerId} - Delete trigger
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyAgentAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const searchParams = request.nextUrl.searchParams;
    const agentId = searchParams.get('agent_id');

    if (!agentId) {
      return NextResponse.json(
        { error: 'agent_id query parameter is required' },
        { status: 400 }
      );
    }

    // Verify agent access
    const hasAccess = await verifyAgentAccess(agentId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Get triggers
    const { data: triggers, error } = await supabaseAdmin
      .from('agent_triggers')
      .select('*')
      .eq('agent_id', agentId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching triggers:', error);
      return NextResponse.json(
        { error: 'Failed to fetch triggers' },
        { status: 500 }
      );
    }

    return NextResponse.json({ triggers: triggers || [] });
  } catch (error) {
    console.error('Triggers fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body = await request.json();

    // Validate required fields
    if (!body.agent_id || !body.provider_id || !body.name || !body.config) {
      return NextResponse.json(
        { error: 'agent_id, provider_id, name, and config are required' },
        { status: 400 }
      );
    }

    // Verify agent access
    const hasAccess = await verifyAgentAccess(body.agent_id, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Agent not found or access denied' },
        { status: 404 }
      );
    }

    // Determine trigger type from provider
    const triggerType = body.provider_id.includes('cron') ? 'cron' : 'webhook';

    // Create trigger
    const triggerData = {
      agent_id: body.agent_id,
      trigger_type: triggerType,
      provider_id: body.provider_id,
      name: body.name,
      description: body.description || null,
      config: body.config,
      is_active: true,
    };

    const { data: trigger, error } = await supabaseAdmin
      .from('agent_triggers')
      .insert(triggerData)
      .select()
      .single();

    if (error) {
      console.error('Error creating trigger:', error);
      return NextResponse.json(
        { error: 'Failed to create trigger' },
        { status: 500 }
      );
    }

    // Generate webhook URL if webhook trigger
    if (triggerType === 'webhook') {
      const webhookUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/triggers/${trigger.trigger_id}/webhook`;

      await supabaseAdmin
        .from('agent_triggers')
        .update({ webhook_url: webhookUrl })
        .eq('trigger_id', trigger.trigger_id);

      trigger.webhook_url = webhookUrl;
    }

    return NextResponse.json({ trigger }, { status: 201 });
  } catch (error) {
    console.error('Trigger creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
