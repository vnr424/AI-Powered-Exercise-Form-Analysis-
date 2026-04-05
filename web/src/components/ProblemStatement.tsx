'use client'

import { AlertTriangle, DollarSign, Users, TrendingDown } from 'lucide-react'

export default function ProblemStatement() {
  const problems = [
    {
      icon: AlertTriangle,
      stat: '67%',
      title: 'Incorrect Form',
      description: 'Of gym-goers perform exercises with poor technique, leading to chronic injuries',
    },
    {
      icon: TrendingDown,
      stat: '45%',
      title: 'Injury Rate',
      description: 'Annual injury rate among regular gym users due to improper form',
    },
    {
      icon: DollarSign,
      stat: '$50-150',
      title: 'Trainer Cost',
      description: 'Per session for personal training - unaffordable for most people',
    },
    {
      icon: Users,
      stat: '85%',
      title: 'No Guidance',
      description: 'Exercise without professional supervision or feedback',
    },
  ]

  return (
    <section className="py-20 relative overflow-hidden">
      <div className="absolute inset-0">
        <img src="/bg-problem.jpg" alt="Background" className="w-full h-full object-cover object-center" />
        <div className="absolute inset-0 bg-black/60"></div>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            The Problem with Exercise Form
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Poor exercise form leads to injuries, wasted effort, and long-term health issues.
            Most people lack access to professional guidance.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {problems.map((problem, index) => (
            <div
              key={index}
              className="bg-gray-900/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition transform hover:-translate-y-1"
            >
              <problem.icon className="w-12 h-12 text-red-500 mb-4" />
              <div className="text-3xl font-bold text-white mb-2">{problem.stat}</div>
              <h3 className="text-lg font-semibold text-white mb-2">{problem.title}</h3>
              <p className="text-gray-300">{problem.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-8 text-center text-white">
          <h3 className="text-2xl font-bold mb-4">Our Solution</h3>
          <p className="text-lg text-blue-100 max-w-3xl mx-auto">
            A free, AI-powered system that provides real-time biomechanical analysis,
            medical-grade feedback, and personalized coaching - accessible to anyone with a webcam.
          </p>
        </div>
      </div>
    </section>
  )
}
