/**
 * Knowledge Base API - Folder Details
 * GET /api/knowledge-base/folders/{folderId} - Get folder with entries
 * PUT /api/knowledge-base/folders/{folderId} - Update folder
 * DELETE /api/knowledge-base/folders/{folderId} - Delete folder
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

interface RouteContext {
  params: Promise<{ folderId: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const userId = await verifyAndGetUserId(request);
    const { folderId } = await context.params;

    // Get folder
    const { data: folder, error: folderError } = await supabaseAdmin
      .from('knowledge_base_folders')
      .select('*')
      .eq('folder_id', folderId)
      .eq('account_id', userId)
      .eq('is_active', true)
      .single();

    if (folderError || !folder) {
      return NextResponse.json(
        { error: 'Folder not found' },
        { status: 404 }
      );
    }

    // Get entries in folder
    const { data: entries, error: entriesError } = await supabaseAdmin
      .from('knowledge_base_entries')
      .select('entry_id, filename, summary, file_size, created_at')
      .eq('folder_id', folderId)
      .eq('is_active', true)
      .order('created_at', { ascending: false });

    if (entriesError) {
      console.error('Error fetching entries:', entriesError);
    }

    return NextResponse.json({
      folder,
      entries: entries || [],
    });
  } catch (error) {
    console.error('Folder fetch error:', error);
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
    const { folderId } = await context.params;
    const body = await request.json();

    // Verify ownership
    const { data: existing } = await supabaseAdmin
      .from('knowledge_base_folders')
      .select('folder_id')
      .eq('folder_id', folderId)
      .eq('account_id', userId)
      .eq('is_active', true)
      .single();

    if (!existing) {
      return NextResponse.json(
        { error: 'Folder not found' },
        { status: 404 }
      );
    }

    // Update folder
    const updateData: any = { updated_at: new Date().toISOString() };
    if (body.name !== undefined) updateData.name = body.name;
    if (body.description !== undefined) updateData.description = body.description;

    const { data: folder, error } = await supabaseAdmin
      .from('knowledge_base_folders')
      .update(updateData)
      .eq('folder_id', folderId)
      .select()
      .single();

    if (error) {
      console.error('Error updating folder:', error);
      return NextResponse.json(
        { error: 'Failed to update folder' },
        { status: 500 }
      );
    }

    return NextResponse.json({ folder });
  } catch (error) {
    console.error('Folder update error:', error);
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
    const { folderId } = await context.params;

    // Verify ownership
    const { data: existing } = await supabaseAdmin
      .from('knowledge_base_folders')
      .select('folder_id')
      .eq('folder_id', folderId)
      .eq('account_id', userId)
      .eq('is_active', true)
      .single();

    if (!existing) {
      return NextResponse.json(
        { error: 'Folder not found' },
        { status: 404 }
      );
    }

    // Soft delete folder
    const { error } = await supabaseAdmin
      .from('knowledge_base_folders')
      .update({ is_active: false, updated_at: new Date().toISOString() })
      .eq('folder_id', folderId);

    if (error) {
      console.error('Error deleting folder:', error);
      return NextResponse.json(
        { error: 'Failed to delete folder' },
        { status: 500 }
      );
    }

    return NextResponse.json({ message: 'Folder deleted successfully' });
  } catch (error) {
    console.error('Folder deletion error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
