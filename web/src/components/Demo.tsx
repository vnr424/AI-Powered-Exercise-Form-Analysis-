'use client'

import { Download } from 'lucide-react'

export default function Demo() {
  return (
    <section id="demo" className="py-20 bg-black text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-4">See It In Action</h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Watch real-time analysis, heatmap generation, and medical feedback
          </p>
        </div>

        <div className="bg-gray-900 rounded-2xl overflow-hidden shadow-2xl mb-12">
          <video
            className="w-full aspect-video object-cover"
            controls
            muted
            loop
            poster="/demo/heatmap.png"
          >
            <source src="/demo/demo.mov" type="video/quicktime" />
            <source src="/demo/demo.mov" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-900 rounded-xl overflow-hidden hover:scale-105 transition">
            <div className="aspect-video overflow-hidden">
              <img src="/demo/report.png" alt="Correct Form" className="w-full h-full object-cover" />
            </div>
            <div className="p-4">
              <h3 className="font-semibold mb-1">Correct Form Detection</h3>
              <p className="text-sm text-gray-400">Green indicators for proper technique</p>
            </div>
          </div>

          <div className="bg-gray-900 rounded-xl overflow-hidden hover:scale-105 transition">
            <div className="aspect-video overflow-hidden">
              <img src="/demo/heatmap.png" alt="Heatmap" className="w-full h-full object-cover" />
            </div>
            <div className="p-4">
              <h3 className="font-semibold mb-1">Heatmap Visualization</h3>
              <p className="text-sm text-gray-400">Red zones show problem areas</p>
            </div>
          </div>

          <div className="bg-gray-900 rounded-xl overflow-hidden hover:scale-105 transition">
            <div className="aspect-video overflow-hidden">
              <img src="/demo/correct-form.png" alt="Report" className="w-full h-full object-cover" />
            </div>
            <div className="p-4">
              <h3 className="font-semibold mb-1">Detailed Report</h3>
              <p className="text-sm text-gray-400">Comprehensive biomechanical analysis</p>
            </div>
          </div>
        </div>

        <div className="mt-12 text-center">
          <a href="#" className="inline-flex items-center px-8 py-4 bg-blue-600 text-white rounded-full font-semibold hover:bg-blue-700 transition shadow-lg">
            <Download className="w-5 h-5 mr-2" />
            Download Desktop App
          </a>
          <p className="text-gray-400 mt-4 text-sm">Available for Windows, macOS, and Linux</p>
        </div>
      </div>
    </section>
  )
}
