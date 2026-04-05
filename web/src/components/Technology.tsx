'use client'

import { Code, Brain, Eye, Zap, Database, Target } from 'lucide-react'

export default function Technology() {
  const techStack = [
    {
      category: 'Pose Detection',
      icon: Eye,
      items: ['MediaPipe Pose Landmarker', '33 body landmarks', 'Heavy model for accuracy', '30 FPS processing'],
    },
    {
      category: 'Machine Learning',
      icon: Brain,
      items: ['Random Forest (500 trees)', '92.3% accuracy', '18 biomechanical features', '5-fold cross-validation'],
    },
    {
      category: 'Person Detection',
      icon: Target,
      items: ['YOLOv8 nano model', 'Equipment confusion prevention', '90% false detection reduction', 'Real-time filtering'],
    },
    {
      category: 'Visualization',
      icon: Zap,
      items: ['Grad-CAM style heatmaps', 'Feature importance mapping', 'Real-time overlay', 'Medical color scale'],
    },
    {
      category: 'Data Processing',
      icon: Database,
      items: ['4,782 training samples', 'Gaussian noise augmentation', 'StandardScaler normalization', 'Stratified sampling'],
    },
    {
      category: 'Implementation',
      icon: Code,
      items: ['Python 3.10+', 'OpenCV 4.9.0', 'scikit-learn 1.4.0', 'TensorFlow 2.15.0'],
    },
  ]

  return (
    <section id="technology" className="py-20 relative overflow-hidden">
      <div className="absolute inset-0">
        <img src="/bg-technology.jpg" alt="Background" className="w-full h-full object-cover object-center" />
        <div className="absolute inset-0 bg-black/60"></div>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Built with Cutting-Edge Technology
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Combining state-of-the-art computer vision, machine learning, and medical expertise
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {techStack.map((tech, index) => (
            <div
              key={index}
              className="bg-gray-900/70 backdrop-blur-sm rounded-2xl p-6 hover:shadow-lg transition border border-gray-700"
            >
              <div className="flex items-center mb-4">
                <div className="bg-blue-600 w-12 h-12 rounded-xl flex items-center justify-center mr-4">
                  <tech.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white">{tech.category}</h3>
              </div>
              <ul className="space-y-2">
                {tech.items.map((item, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-blue-400 mr-2">•</span>
                    <span className="text-gray-200">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-16 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-2xl p-8">
          <h3 className="text-2xl font-bold text-white mb-6 text-center">Why Random Forest?</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-white">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <div className="text-3xl font-bold mb-2">92.3%</div>
              <div className="text-purple-100">vs CNN 89.7%</div>
              <div className="text-sm text-purple-200 mt-2">Higher Accuracy</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <div className="text-3xl font-bold mb-2">15ms</div>
              <div className="text-purple-100">vs CNN 45ms</div>
              <div className="text-sm text-purple-200 mt-2">Faster Inference</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
              <div className="text-3xl font-bold mb-2">100%</div>
              <div className="text-purple-100">Explainable</div>
              <div className="text-sm text-purple-200 mt-2">Feature Importance</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
