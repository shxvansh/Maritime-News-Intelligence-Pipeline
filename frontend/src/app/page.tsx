"use client"

import React, { useState } from 'react'
import { DashboardShell, API_BASE } from '@/components/dashboard-shell'
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Database, Cpu, Upload, Loader2, CheckCircle2, XCircle } from 'lucide-react'

interface ProcessingResult {
  title: string
  classification: string
  risk_level: string
  events_extracted: number
}

export default function IngestionPage() {
  const [scraperLoading, setScrapeLoading] = useState(false)
  const [pipelineLoading, setPipelineLoading] = useState(false)
  const [ingestLoading, setIngestLoading] = useState(false)
  const [batchSize, setBatchSize] = useState(20)
  const [results, setResults] = useState<ProcessingResult[]>([])
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  const runScraper = async () => {
    setScrapeLoading(true)
    setStatusMsg(null)
    try {
      const res = await fetch(`${API_BASE}/pipeline/scrape`, { method: 'POST' })
      const data = await res.json()
      setStatusMsg({ type: 'success', text: `Acquisition complete. ${data.articles_fetched || 0} articles retrieved.` })
    } catch (e: any) {
      setStatusMsg({ type: 'error', text: `Scraper failed: ${e.message}` })
    } finally {
      setScrapeLoading(false)
    }
  }

  const runPipeline = async () => {
    setPipelineLoading(true)
    setStatusMsg(null)
    setResults([])
    try {
      const res = await fetch(`${API_BASE}/pipeline/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch_size: batchSize })
      })
      const data = await res.json()
      setStatusMsg({ type: 'success', text: `Pipeline complete. ${data.articles_processed || 0} articles processed.` })
      if (data.details) setResults(data.details)
    } catch (e: any) {
      setStatusMsg({ type: 'error', text: `Pipeline failed: ${e.message}` })
    } finally {
      setPipelineLoading(false)
    }
  }

  const runIngest = async () => {
    setIngestLoading(true)
    setStatusMsg(null)
    try {
      const res = await fetch(`${API_BASE}/rag/ingest`, { method: 'POST' })
      if (res.ok) {
        setStatusMsg({ type: 'success', text: 'Vector ingestion complete.' })
      } else {
        setStatusMsg({ type: 'error', text: `Ingestion failed: ${await res.text()}` })
      }
    } catch (e: any) {
      setStatusMsg({ type: 'error', text: `Ingestion request failed: ${e.message}` })
    } finally {
      setIngestLoading(false)
    }
  }

  const riskColor = (level: string) => {
    const l = (level || '').toLowerCase()
    if (l === 'critical') return 'bg-red-500/15 text-red-400 border-red-500/30'
    if (l === 'high') return 'bg-amber-500/15 text-amber-400 border-amber-500/30'
    if (l === 'medium') return 'bg-blue-500/15 text-blue-400 border-blue-500/30'
    return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
  }

  return (
    <DashboardShell>
      <div className="space-y-2 mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Ingestion Control Center</h1>
        <p className="text-muted-foreground">Initiate data acquisition and intelligence extraction operations.</p>
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

      {/* Control Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Database className="h-4 w-4 text-primary" /> Data Acquisition
            </CardTitle>
            <CardDescription>Fetch latest articles from MarineTraffic GraphQL endpoint.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={runScraper} disabled={scraperLoading} className="w-full">
              {scraperLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Scraping...</> : 'Execute Scraper'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Cpu className="h-4 w-4 text-primary" /> NLP + LLM Extraction
            </CardTitle>
            <CardDescription>Run preprocessing, NER, classification, and LLM-based event extraction.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground">Batch Size</label>
              <input
                type="number"
                min={1}
                max={100}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full mt-1 px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <Button onClick={runPipeline} disabled={pipelineLoading} className="w-full">
              {pipelineLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...</> : 'Execute Pipeline'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="h-4 w-4 text-primary" /> Vector Ingestion
            </CardTitle>
            <CardDescription>Embed articles and push hybrid vectors into Qdrant for RAG operations.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={runIngest} disabled={ingestLoading} className="w-full">
              {ingestLoading ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Ingesting...</> : 'Ingest into Qdrant'}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Live Extraction Results */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Extraction Results</CardTitle>
            <CardDescription>Recently processed articles from this pipeline run.</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <ScrollArea className="h-[350px]">
              <div className="divide-y divide-border">
                {results.map((r, i) => (
                  <div key={i} className="flex items-center justify-between p-4 hover:bg-muted/30 transition-colors">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{r.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-[10px]">{r.classification}</Badge>
                        <span className="text-[10px] text-muted-foreground">{r.events_extracted} events</span>
                      </div>
                    </div>
                    <Badge className={`text-[10px] ${riskColor(r.risk_level)}`}>
                      {r.risk_level || 'N/A'}
                    </Badge>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </DashboardShell>
  )
}
