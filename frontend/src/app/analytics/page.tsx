"use client"

import React, { useState, useEffect } from 'react'
import { DashboardShell, API_BASE } from '@/components/dashboard-shell'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { BarChart3, GitBranch, Loader2, CheckCircle2, XCircle } from 'lucide-react'

export default function AnalyticsPage() {
  const [topicLoading, setTopicLoading] = useState(false)
  const [graphLoading, setGraphLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [topicHtmlUrl, setTopicHtmlUrl] = useState<string | null>(null)
  const [graphHtmlUrl, setGraphHtmlUrl] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'topic' | 'graph'>('topic')

  // On mount, check if either visualization already exists in the API
  useEffect(() => {
    const checkExisting = async () => {
      try {
        const topicRes = await fetch(`${API_BASE}/analytics/topic-model/html`)
        if (topicRes.ok) setTopicHtmlUrl(`${API_BASE}/analytics/topic-model/html`)
      } catch {}
      try {
        const graphRes = await fetch(`${API_BASE}/analytics/knowledge-graph/html`)
        if (graphRes.ok) setGraphHtmlUrl(`${API_BASE}/analytics/knowledge-graph/html`)
      } catch {}
    }
    checkExisting()
  }, [])

  const genTopicModel = async () => {
    setTopicLoading(true)
    setStatusMsg(null)
    try {
      const res = await fetch(`${API_BASE}/analytics/topic-model`, { method: 'POST' })
      if (res.ok) {
        setStatusMsg({ type: 'success', text: 'Topic model generated successfully.' })
        // Point the iframe to the HTML-serving endpoint on the API
        setTopicHtmlUrl(`${API_BASE}/analytics/topic-model/html`)
        setActiveTab('topic')
      } else {
        setStatusMsg({ type: 'error', text: `Failed: ${await res.text()}` })
      }
    } catch (e: any) {
      setStatusMsg({ type: 'error', text: `Topic modeling failed: ${e.message}` })
    } finally {
      setTopicLoading(false)
    }
  }

  const genKnowledgeGraph = async () => {
    setGraphLoading(true)
    setStatusMsg(null)
    try {
      const res = await fetch(`${API_BASE}/analytics/knowledge-graph`, { method: 'POST' })
      if (res.ok) {
        setStatusMsg({ type: 'success', text: 'Knowledge graph generated successfully.' })
        // Point the iframe to the HTML-serving endpoint on the API
        setGraphHtmlUrl(`${API_BASE}/analytics/knowledge-graph/html`)
        setActiveTab('graph')
      } else {
        setStatusMsg({ type: 'error', text: `Failed: ${await res.text()}` })
      }
    } catch (e: any) {
      setStatusMsg({ type: 'error', text: `Graph generation failed: ${e.message}` })
    } finally {
      setGraphLoading(false)
    }
  }

  return (
    <DashboardShell>
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Macro Analytics</h1>
        <p className="text-muted-foreground">Strategic trend analysis and entity relationship mapping.</p>
      </div>

      {/* Status Message */}
      {statusMsg && (
        <div className={`flex items-center gap-2 mb-6 px-4 py-3 rounded-lg border text-sm ${
          statusMsg.type === 'success'
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        }`}>
          {statusMsg.type === 'success' ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
          {statusMsg.text}
        </div>
      )}

      {/* Control Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <BarChart3 className="h-4 w-4 text-primary" /> Topic Modeling
            </CardTitle>
            <CardDescription>Run BERTopic pipeline to identify emerging macro-themes.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={genTopicModel} disabled={topicLoading} className="w-full">
              {topicLoading
                ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training...</>
                : 'Generate Topic Model'}
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <GitBranch className="h-4 w-4 text-primary" /> Knowledge Graph
            </CardTitle>
            <CardDescription>Build entity relationship graph connecting vessels, orgs, ports.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={genKnowledgeGraph} disabled={graphLoading} className="w-full">
              {graphLoading
                ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Building...</>
                : 'Generate Knowledge Graph'}
            </Button>
          </CardContent>
        </Card>
      </div>

      <Separator className="mb-6" />

      {/* Tab Switcher */}
      <div className="flex gap-1 mb-4">
        {(['topic', 'graph'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'bg-primary/10 text-primary border border-primary/20'
                : 'text-muted-foreground hover:bg-accent'
            }`}
          >
            {tab === 'topic' ? 'Topic Model' : 'Knowledge Graph'}
          </button>
        ))}
      </div>

      {/* Visualization — iframe pointing directly at the API HTML endpoint */}
      <Card>
        <CardContent className="p-0 overflow-hidden rounded-lg">
          {activeTab === 'topic' ? (
            topicHtmlUrl ? (
              <iframe
                src={topicHtmlUrl}
                className="w-full border-0 rounded-lg"
                style={{ height: 600 }}
                title="Topic Model Dashboard"
                sandbox="allow-scripts allow-same-origin"
              />
            ) : (
              <div className="p-12 text-center text-muted-foreground">
                <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p>No topic model available. Generate one using the button above.</p>
              </div>
            )
          ) : (
            graphHtmlUrl ? (
              <iframe
                src={graphHtmlUrl}
                className="w-full border-0 rounded-lg"
                style={{ height: 750 }}
                title="Knowledge Graph"
                sandbox="allow-scripts allow-same-origin"
              />
            ) : (
              <div className="p-12 text-center text-muted-foreground">
                <GitBranch className="h-12 w-12 mx-auto mb-3 opacity-30" />
                <p>No knowledge graph available. Generate one using the button above.</p>
              </div>
            )
          )}
        </CardContent>
      </Card>
    </DashboardShell>
  )
}
