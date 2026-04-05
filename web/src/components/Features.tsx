'use client'

import { Video, Brain, Activity, MessageSquare, FileText, Shield } from 'lucide-react'

export default function Features() {
  const features = [
    {
      icon: Video,
      title: 'Real-Time Analysis',
      description: '30 FPS pose detection with instant feedback on form quality',
      color: 'bg-blue-500',
    },
    {
      icon: Brain,
      title: 'Medical-Grade Feedback',
      description: 'Anatomically precise error detection with injury risk assessment',
      color: 'bg-purple-500',
    },
    {
      icon: Activity,
      title: 'Grad-CAM Heatmaps',
      description: 'Visual heatmaps showing problematic body regions during exercise',
      color: 'bg-red-500',
    },
    {
      icon: MessageSquare,
      title: 'Audio Coaching',
      description: 'Voice alerts for critical errors and real-time form corrections',
      color: 'bg-green-500',
    },
    {
      icon: FileText,
      title: 'Detailed Reports',
      description: 'Comprehensive analysis with symmetry scores and rep-by-rep breakdown',
      color: 'bg-yellow-500',
    },
    {
      icon: Shield,
      title: 'Person Detection Filter',
      description: 'YOLOv8 filter prevents equipment confusion for accurate tracking',
      color: 'bg-indigo-500',
    },
  ]

  return (
    <section id="features" className="py-20 relative overflow-hidden">
      <div className="absolute inset-0">
        <img src="/bg-features.jpg" alt="Background" className="w-full h-full object-cover object-center" />
        <div className="absolute inset-0 bg-black/60"></div>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Powerful Features for Better Training
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Everything you need to improve form, prevent injuries, and maximize your workout effectiveness
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative bg-gray-900/70 backdrop-blur-sm rounded-2xl p-8 border-2 border-gray-700 hover:border-blue-400 transition shadow-sm hover:shadow-xl"
            >
              <div className={`${feature.color} w-14 h-14 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition`}>
                <feature.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{feature.title}</h3>
              <p className="text-gray-300">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
