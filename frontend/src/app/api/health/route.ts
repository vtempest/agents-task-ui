/**
 * Health Check API
 * GET /api/health - System health check
 */

import { NextResponse } from 'next/server';
import { supabaseAdmin } from '@/lib/api/db';

export async function GET() {
  try {
    // Check database connectivity
    const { error } = await supabaseAdmin
      .from('threads')
      .select('thread_id')
      .limit(1);

    if (error) {
      console.error('Health check database error:', error);
      return NextResponse.json(
        {
          status: 'error',
          timestamp: new Date().toISOString(),
          error: 'Database connection failed',
        },
        { status: 503 }
      );
    }

    return NextResponse.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      service: 'nextjs-api',
    });
  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 503 }
    );
  }
}
