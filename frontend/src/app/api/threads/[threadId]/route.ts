/**
 * Thread API - Get, Update, Delete
 * GET /api/threads/[threadId] - Get specific thread
 * PUT /api/threads/[threadId] - Update thread
 * DELETE /api/threads/[threadId] - Delete thread
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyThreadAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ threadId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { threadId } = await context.params;

    // Verify access
    const hasAccess = await verifyThreadAccess(threadId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Fetch thread
    const { data: thread, error } = await supabaseAdmin
      .from('threads')
      .select('*')
      .eq('thread_id', threadId)
      .single();

    if (error || !thread) {
      return NextResponse.json(
        { error: 'Thread not found' },
        { status: 404 }
      );
    }

    // Fetch project if exists
    if (thread.project_id) {
      const { data: project } = await supabaseAdmin
        .from('projects')
        .select('*')
        .eq('project_id', thread.project_id)
        .single();

      if (project) {
        thread.project = project;
      }
    }

    return NextResponse.json({ thread });
  } catch (error) {
    console.error('Thread fetch error:', error);
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
    const { threadId } = await context.params;
    const body = await request.json();

    // Verify access
    const hasAccess = await verifyThreadAccess(threadId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Prepare update data
    const updateData: any = {
      updated_at: new Date().toISOString(),
    };
    if (body.metadata !== undefined) updateData.metadata = body.metadata;
    if (body.is_public !== undefined) updateData.is_public = body.is_public;
    if (body.project_id !== undefined) updateData.project_id = body.project_id;

    // Update thread
    const { data: thread, error } = await supabaseAdmin
      .from('threads')
      .update(updateData)
      .eq('thread_id', threadId)
      .select()
      .single();

    if (error) {
      console.error('Error updating thread:', error);
      return NextResponse.json(
        { error: 'Failed to update thread' },
        { status: 500 }
      );
    }

    return NextResponse.json({ thread });
  } catch (error) {
    console.error('Thread update error:', error);
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
    const { threadId } = await context.params;

    // Verify access
    const hasAccess = await verifyThreadAccess(threadId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Delete thread (cascades will handle related records)
    const { error } = await supabaseAdmin
      .from('threads')
      .delete()
      .eq('thread_id', threadId);

    if (error) {
      console.error('Error deleting thread:', error);
      return NextResponse.json(
        { error: 'Failed to delete thread' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'Thread deleted successfully' });
  } catch (error) {
    console.error('Thread deletion error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
