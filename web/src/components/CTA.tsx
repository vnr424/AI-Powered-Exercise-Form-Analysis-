'use client'

import { Github, Download, BookOpen, Mail } from 'lucide-react'

export default function CTA() {
  return (
    <section className="py-20 bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-700 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-4xl md:text-5xl font-bold mb-6">
          Ready to Improve Your Form?
        </h2>
        <p className="text-xl text-blue-100 mb-12 max-w-2xl mx-auto">
          Download the desktop app, explore the code, or read the documentation to get started
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-5xl mx-auto">
          <a 
            href="https://github.com/yourusername/exercise-form-analysis/releases"
            className="bg-black/10 backdrop-blur-sm hover:bg-black/20 rounded-2xl p-6 transition group"
          >
            <Download className="w-10 h-10 mx-auto mb-4 group-hover:scale-110 transition" />
            <h3 className="font-semibold mb-2">Download App</h3>
            <p className="text-sm text-blue-100">Desktop application</p>
          </a>

          <a 
            href="https://github.com/yourusername/exercise-form-analysis"
            className="bg-black/10 backdrop-blur-sm hover:bg-black/20 rounded-2xl p-6 transition group"
          >
            <Github className="w-10 h-10 mx-auto mb-4 group-hover:scale-110 transition" />
            <h3 className="font-semibold mb-2">View Source</h3>
            <p className="text-sm text-blue-100">GitHub repository</p>
          </a>

          <a 
            href="/docs"
            className="bg-black/10 backdrop-blur-sm hover:bg-black/20 rounded-2xl p-6 transition group"
          >
            <BookOpen className="w-10 h-10 mx-auto mb-4 group-hover:scale-110 transition" />
            <h3 className="font-semibold mb-2">Documentation</h3>
            <p className="text-sm text-blue-100">Setup guides & API</p>
          </a>

          <a 
            href="mailto:your.email@example.com"
            className="bg-black/10 backdrop-blur-sm hover:bg-black/20 rounded-2xl p-6 transition group"
          >
            <Mail className="w-10 h-10 mx-auto mb-4 group-hover:scale-110 transition" />
            <h3 className="font-semibold mb-2">Contact</h3>
            <p className="text-sm text-blue-100">Get in touch</p>
          </a>
        </div>
      </div>
    </section>
  )
}
