/**
 * Credentials API - Secure Credential Storage
 * GET /api/credentials - List stored credentials
 * POST /api/credentials - Store new credential
 * DELETE /api/credentials/{credentialId} - Delete credential
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const searchParams = request.nextUrl.searchParams;
    const mcpQualifiedName = searchParams.get('mcp_qualified_name');

    // Build query
    let query = supabaseAdmin
      .from('mcp_credentials')
      .select('credential_id, mcp_qualified_name, display_name, is_active, created_at, updated_at')
      .eq('account_id', userId)
      .eq('is_active', true);

    if (mcpQualifiedName) {
      query = query.eq('mcp_qualified_name', mcpQualifiedName);
    }

    const { data: credentials, error } = await query.order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching credentials:', error);
      return NextResponse.json(
        { error: 'Failed to fetch credentials' },
        { status: 500 }
      );
    }

    // Return without actual config values (security)
    const sanitizedCredentials = (credentials || []).map(cred => ({
      ...cred,
      config_keys: Object.keys(cred.encrypted_config || {}),
    }));

    return NextResponse.json({ credentials: sanitizedCredentials });
  } catch (error) {
    console.error('Credentials fetch error:', error);
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
    if (!body.mcp_qualified_name || !body.display_name || !body.config) {
      return NextResponse.json(
        { error: 'mcp_qualified_name, display_name, and config are required' },
        { status: 400 }
      );
    }

    // Note: In production, config should be encrypted before storage
    // This requires encryption key management (e.g., AWS KMS, Vault)
    // For now, we'll store as-is (not recommended for production)

    const credentialData = {
      account_id: userId,
      mcp_qualified_name: body.mcp_qualified_name,
      display_name: body.display_name,
      encrypted_config: body.config, // Should be encrypted
      is_active: true,
    };

    const { data: credential, error } = await supabaseAdmin
      .from('mcp_credentials')
      .insert(credentialData)
      .select('credential_id, mcp_qualified_name, display_name, is_active, created_at')
      .single();

    if (error) {
      console.error('Error storing credential:', error);
      return NextResponse.json(
        { error: 'Failed to store credential' },
        { status: 500 }
      );
    }

    return NextResponse.json({ credential }, { status: 201 });
  } catch (error) {
    console.error('Credential storage error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
