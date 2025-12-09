/**
 * Admin API - User Management
 * GET /api/admin/users - List all users (admin only)
 * PUT /api/admin/users/{userId} - Update user (admin only)
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

/**
 * Verify user has admin role
 */
async function verifyAdminRole(userId: string): Promise<boolean> {
  const { data, error } = await supabaseAdmin
    .from('user_roles')
    .select('role')
    .eq('user_id', userId)
    .single();

  if (error || !data) {
    return false;
  }

  return data.role === 'admin' || data.role === 'superadmin';
}

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);

    // Verify admin access
    const isAdmin = await verifyAdminRole(userId);
    if (!isAdmin) {
      return NextResponse.json(
        { error: 'Admin access required' },
        { status: 403 }
      );
    }

    const searchParams = request.nextUrl.searchParams;
    const page = parseInt(searchParams.get('page') || '1');
    const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 100);
    const offset = (page - 1) * limit;

    // Get users via Supabase Auth Admin API
    const { data: users, error } = await supabaseAdmin.auth.admin.listUsers({
      page,
      perPage: limit,
    });

    if (error) {
      console.error('Error fetching users:', error);
      return NextResponse.json(
        { error: 'Failed to fetch users' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      users: users.users,
      pagination: {
        page,
        limit,
        total: users.users.length,
      },
    });
  } catch (error) {
    console.error('Admin users fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
