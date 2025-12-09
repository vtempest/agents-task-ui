/**
 * Notifications API - Settings Management
 * GET /api/notifications/settings - Get notification settings
 * PUT /api/notifications/settings - Update notification settings
 */

import { NextRequest, NextResponse } from 'next/server';
import { verifyAndGetUserId } from '@/lib/api/auth';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);

    // Get notification settings
    const { data: settings, error } = await supabaseAdmin
      .from('notification_settings')
      .select('*')
      .eq('account_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') { // PGRST116 = no rows
      console.error('Error fetching notification settings:', error);
      return NextResponse.json(
        { error: 'Failed to fetch notification settings' },
        { status: 500 }
      );
    }

    // Create default settings if none exist
    if (!settings) {
      const defaultSettings = {
        account_id: userId,
        email_enabled: true,
        push_enabled: true,
        in_app_enabled: true,
      };

      const { data: newSettings, error: createError } = await supabaseAdmin
        .from('notification_settings')
        .insert(defaultSettings)
        .select()
        .single();

      if (createError) {
        console.error('Error creating default settings:', createError);
        return NextResponse.json(
          { error: 'Failed to create default settings' },
          { status: 500 }
        );
      }

      return NextResponse.json({ settings: newSettings });
    }

    return NextResponse.json({ settings });
  } catch (error) {
    console.error('Notification settings fetch error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}

export async function PUT(request: NextRequest) {
  try {
    const userId = await verifyAndGetUserId(request);
    const body = await request.json();

    // Build update data
    const updateData: any = {
      updated_at: new Date().toISOString(),
    };
    if (body.email_enabled !== undefined) updateData.email_enabled = body.email_enabled;
    if (body.push_enabled !== undefined) updateData.push_enabled = body.push_enabled;
    if (body.in_app_enabled !== undefined) updateData.in_app_enabled = body.in_app_enabled;

    // Update settings (upsert if not exists)
    const { data: settings, error } = await supabaseAdmin
      .from('notification_settings')
      .upsert({
        account_id: userId,
        ...updateData,
      })
      .select()
      .single();

    if (error) {
      console.error('Error updating notification settings:', error);
      return NextResponse.json(
        { error: 'Failed to update notification settings' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Notification settings updated successfully',
      settings,
    });
  } catch (error) {
    console.error('Notification settings update error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Authentication failed' },
      { status: 401 }
    );
  }
}
