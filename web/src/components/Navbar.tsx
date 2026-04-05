'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Menu, X, Activity } from 'lucide-react'

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav className={`fixed w-full z-50 transition-all duration-300 ${
      scrolled ? 'bg-black shadow-lg' : 'bg-transparent'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="flex items-center space-x-2">
            <Activity className={`w-8 h-8 ${scrolled ? 'text-blue-600' : 'text-white'}`} />
            <span className={`text-xl font-bold ${scrolled ? 'text-white' : 'text-white'}`}>
              FormAI
            </span>
          </Link>

          <div className="hidden md:flex items-center space-x-8">
            <Link href="#features" className={`hover:text-blue-600 transition ${
              scrolled ? 'text-gray-300' : 'text-white'
            }`}>
              Features
            </Link>
            <Link href="#demo" className={`hover:text-blue-600 transition ${
              scrolled ? 'text-gray-300' : 'text-white'
            }`}>
              Demo
            </Link>
            <a 
              href="https://github.com/yourusername/exercise-form-analysis" 
              target="_blank"
              rel="noopener noreferrer"
              className="bg-blue-600 text-white px-6 py-2 rounded-full hover:bg-blue-700 transition"
            >
              View on GitHub
            </a>
          </div>

          <button 
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden"
          >
            {isOpen ? (
              <X className={scrolled ? 'text-white' : 'text-white'} />
            ) : (
              <Menu className={scrolled ? 'text-white' : 'text-white'} />
            )}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="md:hidden bg-black border-t">
          <div className="px-2 pt-2 pb-3 space-y-1">
            <Link href="#features" className="block px-3 py-2 text-gray-300 hover:bg-gray-900 rounded">
              Features
            </Link>
            <Link href="#demo" className="block px-3 py-2 text-gray-300 hover:bg-gray-900 rounded">
              Demo
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}
