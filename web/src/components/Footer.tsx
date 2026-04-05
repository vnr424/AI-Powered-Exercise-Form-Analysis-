'use client'

import Link from 'next/link'
import { Github, Linkedin, Mail, Activity } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="bg-black text-gray-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Activity className="w-6 h-6 text-blue-500" />
              <span className="text-xl font-bold text-white">FormAI</span>
            </div>
            <p className="text-sm text-gray-400">
              AI-powered exercise form analysis for injury prevention and optimal training.
            </p>
          </div>

          <div>
            <h3 className="font-semibold text-white mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li><Link href="#features" className="hover:text-blue-400 transition">Features</Link></li>
              <li><Link href="#demo" className="hover:text-blue-400 transition">Demo</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-white mb-4">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li><a href="https://github.com/yourusername/exercise-form-analysis" className="hover:text-blue-400 transition">GitHub</a></li>
              <li><a href="#" className="hover:text-blue-400 transition">Research Paper</a></li>
              <li><a href="#" className="hover:text-blue-400 transition">API Reference</a></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-white mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a href="https://github.com/yourusername" target="_blank" rel="noopener noreferrer" className="hover:text-blue-400 transition">
                <Github className="w-6 h-6" />
              </a>
              <a href="https://linkedin.com/in/yourusername" target="_blank" rel="noopener noreferrer" className="hover:text-blue-400 transition">
                <Linkedin className="w-6 h-6" />
              </a>
              <a href="mailto:your.email@example.com" className="hover:text-blue-400 transition">
                <Mail className="w-6 h-6" />
              </a>
            </div>
            <div className="mt-4 text-sm">
              <p className="text-gray-400">Vihanga Ranaweera</p>
              <p className="text-gray-400">Final Year Project 2026</p>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-12 pt-8 text-center text-sm">
          <p>&copy; 2026 Exercise Form Analysis. Built with Next.js and Tailwind CSS.</p>
          <p className="text-gray-500 mt-2">For academic and research purposes.</p>
        </div>
      </div>
    </footer>
  )
}
