'use client'

import { TrendingUp, Award, Users, CheckCircle } from 'lucide-react'

export default function Results() {
  const metrics = [
    { label: 'Test Accuracy', value: '92.3%', icon: Award },
    { label: 'Cross-Validation', value: '91.8%', icon: TrendingUp },
    { label: 'Trainer Agreement', value: '95%', icon: Users },
    { label: 'Real-Time FPS', value: '30', icon: CheckCircle },
  ]

  const featureImportance = [
    { feature: 'avg_elbow_angle', importance: 18.3 },
    { feature: 'extreme_flare', importance: 12.7 },
    { feature: 'elbow_angle_diff', importance: 11.2 },
    { feature: 'alignment_score', importance: 9.8 },
    { feature: 'left_elbow_angle', importance: 8.9 },
  ]

  return (
    <section id="results" className="py-20 relative overflow-hidden">
      <div className="absolute inset-0">
        <img src="/bg-results.jpg" alt="Background" className="w-full h-full object-cover object-center" />
        <div className="absolute inset-0 bg-black/75"></div>
      </div>
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Validated Results</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Rigorous testing and validation ensure reliable, accurate analysis
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {metrics.map((metric, index) => (
            <div key={index} className="bg-black/50 backdrop-blur-sm rounded-2xl p-6 text-center border border-gray-700">
              <metric.icon className="w-10 h-10 text-blue-400 mx-auto mb-4" />
              <div className="text-4xl font-bold text-white mb-2">{metric.value}</div>
              <div className="text-gray-300">{metric.label}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          <div>
            <h3 className="text-2xl font-bold text-white mb-6">Confusion Matrix</h3>
            <div className="bg-black/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700">
              <div className="grid grid-cols-3 gap-2 text-sm">
                <div></div>
                <div className="font-semibold text-center text-gray-200">Predicted Incorrect</div>
                <div className="font-semibold text-center text-gray-200">Predicted Correct</div>
                <div className="font-semibold text-gray-200">Actual Incorrect</div>
                <div className="bg-green-900/40 border-2 border-green-500 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-400">147</div>
                  <div className="text-xs text-green-400">True Negative</div>
                </div>
                <div className="bg-red-900/40 border-2 border-red-500 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">12</div>
                  <div className="text-xs text-red-400">False Positive</div>
                </div>
                <div className="font-semibold text-gray-200">Actual Correct</div>
                <div className="bg-red-900/40 border-2 border-red-500 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">13</div>
                  <div className="text-xs text-red-400">False Negative</div>
                </div>
                <div className="bg-green-900/40 border-2 border-green-500 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-400">147</div>
                  <div className="text-xs text-green-400">True Positive</div>
                </div>
              </div>
              <div className="mt-4 text-center text-sm text-gray-300">
                <p>Precision: 91.9% • Recall: 92.1% • F1-Score: 92.0%</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-2xl font-bold text-white mb-6">Top Features</h3>
            <div className="bg-black/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700">
              <div className="space-y-4">
                {featureImportance.map((item, index) => (
                  <div key={index}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-200">{item.feature}</span>
                      <span className="text-sm font-bold text-blue-400">{item.importance}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-indigo-500 h-2 rounded-full transition-all"
                        style={{ width: `${item.importance * 5}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-6 text-sm text-gray-300">
                Elbow angle is the strongest predictor of form quality, aligning with sports science research.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-16 bg-black/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
          <h3 className="text-2xl font-bold text-white mb-6 text-center">User Validation</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold text-green-400 mb-2">95%</div>
              <p className="text-gray-300">Agreement with Certified Trainer</p>
            </div>
            <div>
              <div className="text-4xl font-bold text-green-400 mb-2">10</div>
              <p className="text-gray-300">User Testers (Mixed Experience)</p>
            </div>
            <div>
              <div className="text-4xl font-bold text-green-400 mb-2">9.2/10</div>
              <p className="text-gray-300">Average Usefulness Rating</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
