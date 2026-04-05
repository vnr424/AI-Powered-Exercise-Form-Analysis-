import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Exercise Form Analysis | Real-Time Biomechanical Feedback',
  description: 'Prevent injuries with AI-powered real-time exercise form analysis. Get medical-grade feedback, Grad-CAM heatmaps, and detailed biomechanical reports.',
  keywords: 'exercise form, AI fitness, biomechanics, injury prevention, MediaPipe, machine learning',
  authors: [{ name: 'Vihanga Ranaweera' }],
  openGraph: {
    title: 'AI Exercise Form Analysis System',
    description: 'Real-time biomechanical analysis with 92.3% accuracy',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={inter.className}>
        <Navbar />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
