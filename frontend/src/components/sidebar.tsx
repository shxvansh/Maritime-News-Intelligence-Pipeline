"use client"

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import {
  Database, Newspaper, BarChart3, MessageSquare,
  ChevronLeft, ChevronRight, Activity
} from 'lucide-react'

const navItems = [
  { href: '/', label: 'Ingestion Control', icon: Database },
  { href: '/feed', label: 'Intelligence Feed', icon: Newspaper },
  { href: '/analytics', label: 'Macro Analytics', icon: BarChart3 },
  { href: '/chat', label: 'RAG Query', icon: MessageSquare },
]

interface SidebarProps {
  apiStatus: 'online' | 'offline' | 'checking'
}

export function Sidebar({ apiStatus }: SidebarProps) {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside className={cn(
      "fixed left-0 top-0 z-40 h-screen flex flex-col border-r border-border bg-card transition-all duration-300",
      collapsed ? "w-16" : "w-64"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        {!collapsed && (
          <div>
            <h2 className="text-sm font-bold tracking-wide text-foreground">Maritime Intel</h2>
            <p className="text-[10px] text-muted-foreground tracking-widest uppercase">Command</p>
          </div>
        )}
        <button onClick={() => setCollapsed(!collapsed)} className="p-1 rounded hover:bg-accent">
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </Link>
          )
        })}
      </nav>

      {/* Status Footer */}
      <div className="p-3 border-t border-border space-y-2">
        <div className={cn(
          "flex items-center gap-2 px-3 py-2 rounded text-xs font-medium",
          apiStatus === 'online'
            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
            : apiStatus === 'offline'
              ? "bg-red-500/10 text-red-400 border border-red-500/20"
              : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
        )}>
          <span className="relative flex h-2 w-2">
            {apiStatus === 'online' && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            )}
            <span className={cn(
              "relative inline-flex rounded-full h-2 w-2",
              apiStatus === 'online' ? "bg-emerald-500" : apiStatus === 'offline' ? "bg-red-500" : "bg-yellow-500"
            )} />
          </span>
          {!collapsed && (
            <span>API: {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}</span>
          )}
        </div>
        {!collapsed && (
          <div className="flex items-center gap-2 px-3 text-[10px] text-muted-foreground">
            <Activity className="h-3 w-3" />
            <span>{new Date().toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' })}</span>
          </div>
        )}
      </div>
    </aside>
  )
}
