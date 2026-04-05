import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const { GoogleGenerativeAI } = await import('@google/generative-ai')
    const body = await request.json()
    const { message } = body

    const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY
    
    if (!apiKey) {
      return NextResponse.json(
        { error: 'API key not configured' },
        { status: 500 }
      )
    }

    const genAI = new GoogleGenerativeAI(apiKey)
    // Using gemini-2.5-flash - the correct stable model
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-2.5-flash'
    })

    const prompt = `You are an expert AI fitness assistant for an Exercise Form Analysis System that uses MediaPipe and Random Forest ML (88.89% accuracy) to analyze bench press form in real-time with 30 FPS pose detection, 18 biomechanical features, and Grad-CAM heatmaps.

When answering fitness questions:
- Be concise (2-3 paragraphs maximum)
- Focus on biomechanics and injury prevention
- Reference the system's capabilities when relevant
- Be encouraging, educational, and evidence-based
- Use clear language without being overly technical

User question: ${message}`

    const result = await model.generateContent(prompt)
    const response = await result.response
    const text = response.text()

    return NextResponse.json({ response: text })
  } catch (error: any) {
    console.error('Chat error:', error)
    return NextResponse.json(
      { error: 'Failed to get response', details: error?.message || 'Unknown error' },
      { status: 500 }
    )
  }
}
