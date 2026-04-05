'use client'

import { Camera, Cpu, Brain, FileCheck } from 'lucide-react'

export default function HowItWorks() {
  const steps = [
    {
      icon: Camera,
      title: 'Setup Webcam',
      description: 'Position your webcam to capture your side view during exercise',
      step: '01',
    },
    {
      icon: Cpu,
      title: 'Perform Exercise',
      description: 'Start your workout - MediaPipe tracks 33 body landmarks in real-time',
      step: '02',
    },
    {
      icon: Brain,
      title: 'Get Instant Feedback',
      description: 'Random Forest model analyzes 18 biomechanical features at 30 FPS',
      step: '03',
    },
    {
      icon: FileCheck,
      title: 'Review Report',
      description: 'View detailed analysis with heatmaps, grades, and injury risk assessment',
      step: '04',
    },
  ]

  return (
    <section className="py-20 bg-black">

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">How It Works</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            From setup to detailed analysis in four simple steps
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/4 left-full w-full h-0.5 bg-gradient-to-r from-blue-400 to-blue-200 -z-10"></div>
              )}
              <div className="bg-black/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-xl transition transform hover:-translate-y-2">
                <div className="bg-gradient-to-br from-blue-600 to-indigo-600 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 mx-auto">
                  <step.icon className="w-8 h-8 text-white" />
                </div>
                <div className="text-4xl font-bold text-gray-200 mb-2 text-center">{step.step}</div>
                <h3 className="text-xl font-bold text-white mb-3 text-center">{step.title}</h3>
                <p className="text-gray-300 text-center">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
