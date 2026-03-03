"use client"

import React, { useState, useRef, useEffect } from 'react'
import { DashboardShell, API_BASE } from '@/components/dashboard-shell'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  MessageSquare, Send, Trash2, Loader2, User, Bot, ExternalLink
} from 'lucide-react'

interface ChatSource {
  title: string
  score: number
  risk_level?: string
  graph_context?: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  metrics?: {
    'Context Relevance'?: number
    'Faithfulness'?: number
    'Answer Relevance'?: number
  } | null
}

function MetricGauge({ label, subtitle, value }: { label: string; subtitle: string; value: number }) {
  const color = value >= 0.8 ? 'text-emerald-400' : value >= 0.5 ? 'text-amber-400' : 'text-red-400'
  const borderColor = value >= 0.8 ? 'border-emerald-500/30' : value >= 0.5 ? 'border-amber-500/30' : 'border-red-500/30'
  const bgColor = value >= 0.8 ? 'bg-emerald-500/5' : value >= 0.5 ? 'bg-amber-500/5' : 'bg-red-500/5'

  return (
    <Card className={`${bgColor} border-t-2 ${borderColor}`}>
      <CardContent className="p-4">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <p className={`text-2xl font-bold mt-1 ${color}`}>{Math.round(value * 100)}%</p>
        <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>
      </CardContent>
    </Card>
  )
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [evalLoading, setEvalLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const sendQuery = async () => {
    if (!input.trim() || loading) return
    const question = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/rag/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      const data = await res.json()
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: data.answer || 'No response generated.',
        sources: data.sources || [],
        metrics: null
      }
      setMessages(prev => [...prev, assistantMsg])

      // Auto-evaluate
      setEvalLoading(true)
      try {
        const evalRes = await fetch(`${API_BASE}/rag/evaluate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            answer: data.answer || '',
            context: data.context || ''
          })
        })
        const evalData = await evalRes.json()
        setMessages(prev => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (updated[lastIdx]?.role === 'assistant') {
            updated[lastIdx] = { ...updated[lastIdx], metrics: evalData.metrics || {} }
          }
          return updated
        })
      } catch {}
      setEvalLoading(false)
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Query failed: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  const clearHistory = () => {
    setMessages([])
  }

  // Get metrics from the last assistant msg
  const lastMetrics = [...messages].reverse().find(m => m.role === 'assistant' && m.metrics)?.metrics

  return (
    <DashboardShell>
      <div className="space-y-2 mb-6">
        <h1 className="text-3xl font-bold tracking-tight">RAG Query Interface</h1>
        <p className="text-muted-foreground">Interrogate the intelligence database using natural language.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4">
        {/* Chat Panel */}
        <Card className="flex flex-col h-[calc(100vh-200px)]">
          <CardHeader className="pb-3 border-b border-border">
            <CardTitle className="flex items-center gap-2 text-base">
              <MessageSquare className="h-4 w-4 text-primary" /> Analyst Chat
            </CardTitle>
          </CardHeader>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground py-16">
                <MessageSquare className="h-12 w-12 mx-auto mb-3 opacity-20" />
                <p className="text-sm">Ask a question about the maritime intelligence database.</p>
                <p className="text-[10px] mt-1">e.g., &ldquo;Which vessels were detained in the past month?&rdquo;</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="shrink-0 w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-3.5 w-3.5 text-primary" />
                  </div>
                )}
                <div className={`max-w-[80%] rounded-lg px-4 py-3 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-primary/10 border border-primary/20 text-foreground'
                    : 'bg-card border border-border text-foreground'
                }`}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-border space-y-1">
                      {msg.sources.map((src, j) => (
                        <p key={j} className="text-[10px] text-muted-foreground">
                          Source: {src.title} (Relevance: {src.score})
                        </p>
                      ))}
                    </div>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="shrink-0 w-7 h-7 rounded-full bg-muted flex items-center justify-center">
                    <User className="h-3.5 w-3.5 text-muted-foreground" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-3">
                <div className="shrink-0 w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-3.5 w-3.5 text-primary" />
                </div>
                <div className="bg-card border border-border rounded-lg px-4 py-3">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-border p-3">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendQuery()}
                placeholder="Enter your intelligence query..."
                className="flex-1 bg-background border border-border rounded-md px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
              <Button onClick={sendQuery} disabled={loading || !input.trim()} size="icon">
                <Send className="h-4 w-4" />
              </Button>
              <Button onClick={clearHistory} variant="ghost" size="icon">
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* Reliability Metrics Panel */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Reliability Metrics</h3>
          <p className="text-[10px] text-muted-foreground">LLM-as-a-Judge Evaluation</p>

          {evalLoading && (
            <Card>
              <CardContent className="p-4 flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Calculating metrics...
              </CardContent>
            </Card>
          )}

          {lastMetrics ? (
            <>
              <MetricGauge
                label="Context Relevance"
                subtitle="Retrieval Quality"
                value={lastMetrics['Context Relevance'] || 0}
              />
              <MetricGauge
                label="Faithfulness"
                subtitle="Hallucination Guard"
                value={lastMetrics['Faithfulness'] || 0}
              />
              <MetricGauge
                label="Answer Relevance"
                subtitle="Prompt Adherence"
                value={lastMetrics['Answer Relevance'] || 0}
              />
            </>
          ) : !evalLoading ? (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground text-xs">
                Metrics will appear after your first query.
              </CardContent>
            </Card>
          ) : null}
        </div>
      </div>
    </DashboardShell>
  )
}
