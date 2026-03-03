"use client"

import React, { useEffect, useState } from 'react'
import { DashboardShell, API_BASE } from '@/components/dashboard-shell'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Newspaper, Shield, AlertTriangle, Crosshair, Ban,
  ChevronDown, ChevronUp, ExternalLink, Loader2
} from 'lucide-react'

interface Article {
  id: string
  title: string
  classification: string
  confidence: number
  risk_level: string
  impact_scope: string
  executive_summary: string
  strategic_tags: string[]
  is_geopolitical: boolean
  has_defense_implications: boolean
  is_sanction_sensitive: boolean
  named_entities: { entities: any[] }
  url: string
}

interface EventData {
  event_id: string
  article_title: string
  article_hash: string
  event_date: string
  port: string
  country: string
  vessels_involved: string[]
  organizations_involved: string[]
  incident_type: string
  casualties: string
  cargo_type: string
  summary: string
  confidence_score: number
}

function MetricCard({ value, label, icon, color }: { value: number; label: string; icon: React.ReactNode; color: string }) {
  return (
    <Card>
      <CardContent className="p-4 flex items-center gap-4">
        <div className={`p-2.5 rounded-lg ${color}`}>
          {icon}
        </div>
        <div>
          <p className="text-2xl font-bold">{value}</p>
          <p className="text-[10px] text-muted-foreground uppercase tracking-widest">{label}</p>
        </div>
      </CardContent>
    </Card>
  )
}

function RiskBadge({ level }: { level: string }) {
  const l = (level || '').toLowerCase()
  const colors: Record<string, string> = {
    critical: 'bg-red-500/15 text-red-400 border-red-500/30',
    high: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    medium: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
    low: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  }
  return <Badge className={`text-[10px] border ${colors[l] || 'bg-muted text-muted-foreground'}`}>{level || 'N/A'}</Badge>
}

function ArticleCard({ article, events }: { article: Article; events: EventData[] }) {
  const [open, setOpen] = useState(false)

  return (
    <Card className="overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left p-4 flex items-center justify-between hover:bg-muted/30 transition-colors"
      >
        <div className="flex-1 min-w-0 pr-4">
          <p className="text-sm font-semibold truncate">{article.title}</p>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <RiskBadge level={article.risk_level} />
            <Badge variant="secondary" className="text-[10px]">{article.classification}</Badge>
            {article.is_geopolitical && <Badge className="text-[10px] bg-indigo-500/15 text-indigo-400 border border-indigo-500/30">GEO</Badge>}
            {article.has_defense_implications && <Badge className="text-[10px] bg-slate-500/15 text-slate-400 border border-slate-500/30">DEF</Badge>}
            {article.is_sanction_sensitive && <Badge className="text-[10px] bg-rose-500/15 text-rose-400 border border-rose-500/30">SANCTION</Badge>}
          </div>
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <span className="text-[10px]">{article.impact_scope}</span>
          {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </button>

      {open && (
        <div className="border-t border-border px-4 pb-4 pt-3 space-y-4 bg-muted/5">
          {/* Strategic Tags */}
          {article.strategic_tags?.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {article.strategic_tags.map((t, i) => (
                <Badge key={i} variant="outline" className="text-[10px]">{t}</Badge>
              ))}
            </div>
          )}

          {/* Summary */}
          <div>
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Executive Summary</h4>
            <p className="text-sm leading-relaxed">{article.executive_summary || 'No summary available.'}</p>
          </div>

          {article.url && (
            <a href={article.url} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
              <ExternalLink className="h-3 w-3" /> Source Article
            </a>
          )}

          {/* Events */}
          {events.length > 0 && (
            <>
              <Separator />
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Event Details</h4>
              {events.map((ev, i) => (
                <Card key={i} className="border-l-2 border-l-primary">
                  <CardContent className="p-3 space-y-2">
                    <div className="flex justify-between items-start">
                      <p className="text-xs font-mono text-muted-foreground">{ev.event_id}</p>
                      <span className="text-[10px] text-muted-foreground">Confidence: {ev.confidence_score}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <p><span className="text-muted-foreground">Date:</span> {ev.event_date || 'N/A'}</p>
                      <p><span className="text-muted-foreground">Type:</span> {ev.incident_type || 'N/A'}</p>
                      <p><span className="text-muted-foreground">Location:</span> {ev.port || 'Unknown'}, {ev.country || 'Unknown'}</p>
                      <p><span className="text-muted-foreground">Casualties:</span> {ev.casualties || 'N/A'}</p>
                      <p><span className="text-muted-foreground">Vessels:</span> {ev.vessels_involved?.join(', ') || 'N/A'}</p>
                      <p><span className="text-muted-foreground">Orgs:</span> {ev.organizations_involved?.join(', ') || 'N/A'}</p>
                    </div>
                    <p className="text-xs text-muted-foreground">{ev.summary}</p>
                  </CardContent>
                </Card>
              ))}
            </>
          )}
        </div>
      )}
    </Card>
  )
}

export default function FeedPage() {
  const [articles, setArticles] = useState<Article[]>([])
  const [events, setEvents] = useState<EventData[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('All')

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [artRes, evRes] = await Promise.all([
          fetch(`${API_BASE}/db/articles`),
          fetch(`${API_BASE}/db/events`)
        ])
        if (artRes.ok) setArticles(await artRes.json())
        if (evRes.ok) setEvents(await evRes.json())
      } catch {
        // fallback — try local JSON
        try {
          const res = await fetch('/processed_test_results.json')
          if (res.ok) {
            const raw = await res.json()
            const arts: Article[] = raw.map((item: any) => {
              const cl = item.classification || item.zero_shot_classification || {}
              const en = item.llm_structured_output?.enrichment || {}
              return {
                id: item.hash || item.id,
                title: item.title || 'Unknown',
                classification: cl.label || 'Unknown',
                confidence: cl.score || 0,
                risk_level: en.risk_level,
                impact_scope: en.impact_scope,
                executive_summary: en.executive_summary || '',
                strategic_tags: en.strategic_relevance_tags || [],
                is_geopolitical: en.is_geopolitical || false,
                has_defense_implications: en.has_defense_implications || false,
                is_sanction_sensitive: en.is_sanction_sensitive || false,
                named_entities: { entities: item.gliner_entities || [] },
                url: item.source_url || item.url || ''
              }
            })
            setArticles(arts)
          }
        } catch {}
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const classifications = Array.from(new Set(articles.map(a => a.classification))).sort()
  const filtered = filter === 'All' ? articles : articles.filter(a => a.classification === filter)

  const total = articles.length
  const critical = articles.filter(a => ['Critical', 'CRITICAL'].includes(a.risk_level)).length
  const high = articles.filter(a => ['High', 'HIGH'].includes(a.risk_level)).length
  const defense = articles.filter(a => a.has_defense_implications).length
  const sanction = articles.filter(a => a.is_sanction_sensitive).length

  if (loading) {
    return (
      <DashboardShell>
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </DashboardShell>
    )
  }

  return (
    <DashboardShell>
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Intelligence Feed</h1>
        <p className="text-muted-foreground">Classified and enriched maritime event reports.</p>
      </div>

      {/* Metrics Bar */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
        <MetricCard value={total} label="Total Reports" icon={<Newspaper className="h-4 w-4" />} color="bg-primary/10 text-primary" />
        <MetricCard value={critical} label="Critical" icon={<AlertTriangle className="h-4 w-4" />} color="bg-red-500/10 text-red-400" />
        <MetricCard value={high} label="High Risk" icon={<Shield className="h-4 w-4" />} color="bg-amber-500/10 text-amber-400" />
        <MetricCard value={defense} label="Defense Flagged" icon={<Crosshair className="h-4 w-4" />} color="bg-indigo-500/10 text-indigo-400" />
        <MetricCard value={sanction} label="Sanction Flagged" icon={<Ban className="h-4 w-4" />} color="bg-rose-500/10 text-rose-400" />
      </div>

      <Separator className="mb-6" />

      {/* Filter */}
      <div className="mb-4">
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-card border border-border rounded-md px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
        >
          <option value="All">All Classifications</option>
          {classifications.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {/* Article List */}
      {filtered.length === 0 ? (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            No processed data available. Run the extraction pipeline first.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {filtered.map(article => (
            <ArticleCard
              key={article.id}
              article={article}
              events={events.filter(e =>
                (e.article_title && e.article_title === article.title) ||
                (e.article_hash && e.article_hash === article.id)
              )}
            />
          ))}
        </div>
      )}
    </DashboardShell>
  )
}
