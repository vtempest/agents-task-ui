/**
 * Project API - Get, Update, Delete specific project
 * GET /api/projects/[projectId] - Get project details
 * PUT /api/projects/[projectId] - Update project
 * DELETE /api/projects/[projectId] - Delete project
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, getOptionalUserId, verifyProjectAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ projectId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const { projectId } = await context.params;

    // Try to get user_id (optional for public projects)
    const userId = await getOptionalUserId(request);

    // Fetch project
    const { data: project, error } = await supabaseAdmin
      .from('projects')
      .select('*')
      .eq('project_id', projectId)
      .single();

    if (error || !project) {
      return NextResponse.json(
        { error: 'Project not found' },
        { status: 404 }
      );
    }

    // Check if project is public or user has access
    const isPublic = project.is_public === true;
    if (!isPublic) {
      if (!userId) {
        return NextResponse.json(
          { error: 'Authentication required' },
          { status: 401 }
        );
      }
      if (project.account_id !== userId) {
        return NextResponse.json(
          { error: 'Access denied' },
          { status: 403 }
        );
      }
    }

    return NextResponse.json({ project });
  } catch (error) {
    console.error('Project fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PUT(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { projectId } = await context.params;
    const body = await request.json();

    // Verify access
    const hasAccess = await verifyProjectAccess(projectId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Project not found or access denied' },
        { status: 404 }
      );
    }

    // Prepare update data
    const updateData: any = {
      updated_at: new Date().toISOString(),
    };
    if (body.name !== undefined) updateData.name = body.name;
    if (body.description !== undefined) updateData.description = body.description;
    if (body.icon_name !== undefined) updateData.icon_name = body.icon_name;
    if (body.sandbox !== undefined) updateData.sandbox = body.sandbox;
    if (body.is_public !== undefined) updateData.is_public = body.is_public;

    // Update project
    const { data: project, error } = await supabaseAdmin
      .from('projects')
      .update(updateData)
      .eq('project_id', projectId)
      .select()
      .single();

    if (error) {
      console.error('Error updating project:', error);
      return NextResponse.json(
        { error: 'Failed to update project' },
        { status: 500 }
      );
    }

    return NextResponse.json({ project });
  } catch (error) {
    console.error('Project update error:', error);
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
    const { projectId } = await context.params;

    // Verify access
    const hasAccess = await verifyProjectAccess(projectId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Project not found or access denied' },
        { status: 404 }
      );
    }

    // Delete project
    const { error } = await supabaseAdmin
      .from('projects')
      .delete()
      .eq('project_id', projectId);

    if (error) {
      console.error('Error deleting project:', error);
      return NextResponse.json(
        { error: 'Failed to delete project' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'Project deleted successfully' });
  } catch (error) {
    console.error('Project deletion error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
