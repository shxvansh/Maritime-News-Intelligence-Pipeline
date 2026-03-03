"use client"

import React, { useEffect, useState } from 'react'
import { Sidebar } from '@/components/sidebar'
import { cn } from '@/lib/utils'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${API_BASE}/db/articles`, { signal: AbortSignal.timeout(3000) })
        setApiStatus(res.ok ? 'online' : 'offline')
      } catch {
        setApiStatus('offline')
      }
    }
    checkApi()
    const interval = setInterval(checkApi, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <Sidebar apiStatus={apiStatus} />
      <main className={cn(
        "transition-all duration-300 min-h-screen",
        "ml-64" // sidebar width
      )}>
        <div className="p-6 md:p-8 max-w-7xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  )
}

export { API_BASE }
