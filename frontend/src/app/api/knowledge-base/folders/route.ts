/**
 * Knowledge Base API - Folder Management
 * GET /api/knowledge-base/folders - List folders
 * POST /api/knowledge-base/folders - Create folder
 * PUT /api/knowledge-base/folders/{folderId} - Update folder
 * DELETE /api/knowledge-base/folders/{folderId} - Delete folder
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);

    // Get all folders for user with entry counts
    const { data: folders, error } = await supabaseAdmin
      .from('knowledge_base_folders')
      .select('*, knowledge_base_entries(count)')
      .eq('account_id', userId)
      .eq('is_active', true)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching folders:', error);
      return NextResponse.json(
        { error: 'Failed to fetch folders' },
        { status: 500 }
      );
    }

    // Format response with entry counts
    const formattedFolders = (folders || []).map(folder => ({
      folder_id: folder.folder_id,
      name: folder.name,
      description: folder.description,
      entry_count: folder.knowledge_base_entries?.[0]?.count || 0,
      created_at: folder.created_at,
    }));

    return NextResponse.json(formattedFolders);
  } catch (error) {
    console.error('Folders fetch error:', error);
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
    if (!body.name) {
      return NextResponse.json(
        { error: 'Folder name is required' },
        { status: 400 }
      );
    }

    // Check for duplicate folder name
    const { data: existing } = await supabaseAdmin
      .from('knowledge_base_folders')
      .select('folder_id')
      .eq('account_id', userId)
      .eq('name', body.name)
      .eq('is_active', true)
      .single();

    if (existing) {
      return NextResponse.json(
        { error: 'A folder with this name already exists' },
        { status: 409 }
      );
    }

    // Create folder
    const { data: folder, error } = await supabaseAdmin
      .from('knowledge_base_folders')
      .insert({
        account_id: userId,
        name: body.name,
        description: body.description || null,
        is_active: true,
      })
      .select()
      .single();

    if (error) {
      console.error('Error creating folder:', error);
      return NextResponse.json(
        { error: 'Failed to create folder' },
        { status: 500 }
      );
    }

    return NextResponse.json({ folder }, { status: 201 });
  } catch (error) {
    console.error('Folder creation error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
