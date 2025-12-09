/**
 * Template API - Details and Installation
 * GET /api/templates/{templateId} - Get template details
 * POST /api/templates/{templateId}/install - Install template as agent
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, getOptionalUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ templateId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const { templateId } = await context.params;

    // Optional auth (public templates accessible without login)
    const userId = await getOptionalUserId(request);

    // Get template
    const { data: template, error } = await supabaseAdmin
      .from('agent_templates')
      .select('*, profiles(name)')
      .eq('template_id', templateId)
      .single();

    if (error || !template) {
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

    return NextResponse.json({ template });
  } catch (error) {
    console.error('Template fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch template' },
      { status: 500 }
    );
  }
}
