/**
 * Messages API - List and Create messages for a thread
 * GET /api/threads/[threadId]/messages - Get all messages in thread
 * POST /api/threads/[threadId]/messages - Create a new message
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId, verifyThreadAccess } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';
import type { MessageCreateRequest, MessagesResponse } from '@/lib/api/types';

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
    const searchParams = request.nextUrl.searchParams;

    // Verify access
    const hasAccess = await verifyThreadAccess(threadId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Pagination
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 1000);
    const offset = (page - 1) * limit;

    // Fetch messages
    const { data: messages, error, count } = await supabaseAdmin
      .from('messages')
      .select('*', { count: 'exact' })
      .eq('thread_id', threadId)
      .order('created_at', { ascending: true })
      .range(offset, offset + limit - 1);

    if (error) {
      console.error('Error fetching messages:', error);
      return NextResponse.json(
        { error: 'Failed to fetch messages' },
        { status: 500 }
      );
    }

    const totalPages = count ? Math.ceil(count / limit) : 0;

    const response: MessagesResponse = {
      messages: messages || [],
      pagination: {
        page,
        limit,
        total: count || 0,
        pages: totalPages,
      },
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('Messages fetch error:', error);
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
    const { threadId } = await context.params;
    const body: MessageCreateRequest = await request.json();

    // Verify access
    const hasAccess = await verifyThreadAccess(threadId, userId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: 'Thread not found or access denied' },
        { status: 404 }
      );
    }

    // Validate required fields
    if (!body.content || !body.role) {
      return NextResponse.json(
        { error: 'Message content and role are required' },
        { status: 400 }
      );
    }

    // Prepare message data
    const messageData = {
      thread_id: threadId,
      role: body.role,
      content: body.content,
      metadata: body.metadata || {},
    };

    // Insert message
    const { data: message, error } = await supabaseAdmin
      .from('messages')
      .insert(messageData)
      .select()
      .single();

    if (error) {
      console.error('Error creating message:', error);
      return NextResponse.json(
        { error: 'Failed to create message' },
        { status: 500 }
      );
    }

    // Update thread's updated_at timestamp
    await supabaseAdmin
      .from('threads')
      .update({ updated_at: new Date().toISOString() })
      .eq('thread_id', threadId);

    return NextResponse.json({ message }, { status: 201 });
  } catch (error) {
    console.error('Message creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
