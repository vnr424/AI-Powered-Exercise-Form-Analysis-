import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const { GoogleGenerativeAI } = await import('@google/generative-ai')
    const body = await request.json()
    const { errors, stats, exercise } = body

    const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY
    if (!apiKey) return NextResponse.json({ error: 'API key not configured' }, { status: 500 })

    const genAI = new GoogleGenerativeAI(apiKey)
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' })

    const errorSummary = Object.entries(errors)
      .sort(([,a]: any, [,b]: any) => b - a)
      .map(([type, count]) => `- ${type}: occurred ${count} time(s)`)
      .join('\n')

    const prompt = `You are an expert sports physiotherapist and strength coach analyzing a ${exercise} session captured by an AI form analysis system.

SESSION STATS:
- Total Reps: ${stats.total_reps}
- Correct Form: ${stats.correct_reps} reps (${stats.accuracy?.toFixed(1)}% accuracy)
- Average Elbow Angle: ${stats.avg_elbow}°

FORM ERRORS DETECTED ACROSS ALL REPS:
${errorSummary}

Please provide a structured coaching report with these EXACT sections using these EXACT headers:

## Overview
2-3 sentences summarizing overall performance and the most critical issues.

## Errors Explained
For each unique error type detected, explain:
- What it means biomechanically
- Why the athlete is likely making this mistake
- Which muscles/joints are being stressed or underused

## Body Impact & Injury Risk
Explain specifically how these combined errors affect the body over time, which muscles get overloaded, which joints are at risk, and what injuries could develop if uncorrected.

## Step-by-Step Correction Plan
Give 4-6 specific, actionable drills or cues the athlete should practice to fix these issues. Number each one.

## Key Takeaway
One motivating sentence summarizing the most important thing to focus on next session.

Be specific, evidence-based, and encouraging. Use clear language.`

    const result = await model.generateContent(prompt)
    const text = result.response.text()
    return NextResponse.json({ coaching: text })
  } catch (error: any) {
    console.error('Coaching error:', error)
    return NextResponse.json({ error: 'Failed to get coaching', details: error?.message }, { status: 500 })
  }
}
