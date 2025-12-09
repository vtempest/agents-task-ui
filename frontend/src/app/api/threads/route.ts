/**
 * Threads API - List and Create
 * GET /api/threads - List all threads for authenticated user
 * POST /api/threads - Create a new thread
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { ThreadCreateRequest, ThreadsResponse } from '@/lib/api/types';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const searchParams = request.nextUrl.searchParams;

    // Pagination parameters
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);
    const offset = (page - 1) * limit;

    // Get total count
    const { count } = await supabaseAdmin
      .from('threads')
      .select('thread_id', { count: 'exact', head: true })
      .eq('account_id', userId);

    if (!count || count === 0) {
      return NextResponse.json({
        threads: [],
        pagination: {
          page,
          limit,
          total: 0,
          pages: 0,
        },
      });
    }

    // Fetch threads with pagination
    const { data: threads, error: threadsError } = await supabaseAdmin
      .from('threads')
      .select('thread_id, project_id, metadata, is_public, created_at, updated_at')
      .eq('account_id', userId)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (threadsError) {
      console.error('Error fetching threads:', threadsError);
      return NextResponse.json(
        { error: 'Failed to fetch threads' },
        { status: 500 }
      );
    }

    // Get unique project IDs
    const projectIds = [...new Set(
      threads
        ?.filter(t => t.project_id)
        .map(t => t.project_id) || []
    )];

    // Fetch projects in batch if needed
    let projectsById: Record<string, any> = {};
    if (projectIds.length > 0) {
      const { data: projects } = await supabaseAdmin
        .from('projects')
        .select('project_id, name, icon_name, is_public, created_at, updated_at')
        .in('project_id', projectIds);

      projectsById = (projects || []).reduce((acc, project) => {
        acc[project.project_id] = project;
        return acc;
      }, {} as Record<string, any>);
    }

    // Map threads with project data
    const mappedThreads = (threads || []).map(thread => ({
      ...thread,
      project: thread.project_id ? projectsById[thread.project_id] : undefined,
    }));

    const totalPages = Math.ceil(count / limit);

    const response: ThreadsResponse = {
      threads: mappedThreads,
      pagination: {
        page,
        limit,
        total: count,
        pages: totalPages,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Thread list error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body: ThreadCreateRequest = await request.json();

    // Prepare thread data
    const threadData = {
      account_id: userId,
      project_id: body.project_id,
      metadata: body.metadata || {},
      is_public: body.is_public || false,
    };

    // Insert thread
    const { data: thread, error } = await supabaseAdmin
      .from('threads')
      .insert(threadData)
      .select()
      .single();

    if (error) {
      console.error('Error creating thread:', error);
      return NextResponse.json(
        { error: 'Failed to create thread' },
        { status: 500 }
      );
    }

    return NextResponse.json({ thread }, { status: 201 });
  } catch (error) {
    console.error('Thread creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
