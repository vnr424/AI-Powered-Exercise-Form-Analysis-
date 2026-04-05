'use client'

import Hero from '@/components/Hero'
import ProblemStatement from '@/components/ProblemStatement'
import Features from '@/components/Features'
import HowItWorks from '@/components/HowItWorks'
import Demo from '@/components/Demo'
import WebcamAnalysis from '@/components/WebcamAnalysis'
import ChatBot from '@/components/ChatBot'

export default function Home() {
  return (
    <div className="bg-gray-900">
      <Hero />
      <ProblemStatement />
      <Features />
      <HowItWorks />
      <Demo />
      <WebcamAnalysis />
      <ChatBot />
    </div>
  )
}
