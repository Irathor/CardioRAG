import { Loader2, Send, Square } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { SourceCard } from '@/components/app/SourceCard'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { Textarea } from '@/components/ui/textarea'
import { ApiError, streamQuery, type CitationWarning, type SourceInfo } from '@/lib/api'
import { DEFAULT_RETRIEVE_K, DEFAULT_TOP_K, EXAMPLE_QUESTIONS } from '@/lib/constants'

type Phase = 'idle' | 'retrieving' | 'streaming' | 'done' | 'error'

export function QueryPanel() {
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)
  const [retrieveK, setRetrieveK] = useState(DEFAULT_RETRIEVE_K)

  const [phase, setPhase] = useState<Phase>('idle')
  const [sources, setSources] = useState<SourceInfo[]>([])
  const [answer, setAnswer] = useState('')
  const [citationWarnings, setCitationWarnings] = useState<CitationWarning[]>([])
  const [latencyMs, setLatencyMs] = useState<number | null>(null)

  const abortRef = useRef<AbortController | null>(null)

  const isBusy = phase === 'retrieving' || phase === 'streaming'

  async function runQuery(q: string) {
    if (!q.trim() || isBusy) return

    const controller = new AbortController()
    abortRef.current = controller

    setQuestion(q)
    setPhase('retrieving')
    setSources([])
    setAnswer('')
    setCitationWarnings([])
    setLatencyMs(null)

    try {
      await streamQuery(
        q,
        { topK, retrieveK },
        {
          onSources: (s) => {
            setSources(s)
            setPhase('streaming')
          },
          onToken: (text) => setAnswer((prev) => prev + text),
          onDone: (payload) => {
            setCitationWarnings(payload.citation_warnings)
            setLatencyMs(payload.latency_ms)
            setPhase('done')
          },
          onError: (detail) => {
            toast.error('Generation failed', { description: detail })
            setPhase('error')
          },
        },
        controller.signal,
      )
    } catch (err) {
      if (controller.signal.aborted) {
        setPhase('idle')
        return
      }
      const detail = err instanceof ApiError ? err.message : 'Could not reach the API.'
      toast.error('Request failed', { description: detail })
      setPhase('error')
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <Textarea
          placeholder="Ask a question about the indexed cardiovascular MRI / AI literature…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={isBusy}
          rows={3}
          className="resize-none text-base"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runQuery(question)
          }}
        />

        <div className="flex flex-wrap gap-2">
          {EXAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              disabled={isBusy}
              onClick={() => runQuery(q)}
              className="rounded-full border border-border bg-secondary px-3 py-1 text-xs text-secondary-foreground transition-colors hover:bg-accent disabled:opacity-50"
            >
              {q.length > 64 ? `${q.slice(0, 64)}…` : q}
            </button>
          ))}
        </div>

        <div className="grid gap-4 rounded-lg border border-border bg-card/50 p-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <Label>Final sources (top_k)</Label>
              <span className="text-xs tabular-nums text-muted-foreground">{topK}</span>
            </div>
            <Slider
              value={[topK]}
              min={1}
              max={10}
              step={1}
              disabled={isBusy}
              onValueChange={([v]) => setTopK(v)}
            />
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <Label>Candidates before reranking (retrieve_k)</Label>
              <span className="text-xs tabular-nums text-muted-foreground">{retrieveK}</span>
            </div>
            <Slider
              value={[retrieveK]}
              min={topK}
              max={50}
              step={1}
              disabled={isBusy}
              onValueChange={([v]) => setRetrieveK(v)}
            />
          </div>
        </div>

        <div className="flex justify-end gap-2">
          {isBusy ? (
            <Button variant="outline" onClick={stop}>
              <Square className="size-4" />
              Stop
            </Button>
          ) : (
            <Button onClick={() => runQuery(question)} disabled={!question.trim()}>
              <Send className="size-4" />
              Ask
            </Button>
          )}
        </div>
      </div>

      {phase === 'retrieving' && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
          Retrieving and reranking evidence…
        </div>
      )}

      {sources.length > 0 && (
        <div className="flex flex-col gap-3">
          <Separator />
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-muted-foreground">
              Answer{latencyMs !== null && ` · ${(latencyMs / 1000).toFixed(1)}s`}
            </h2>
            {phase === 'streaming' && (
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="size-3 animate-spin" />
                generating…
              </span>
            )}
          </div>

          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {answer}
            {phase === 'streaming' && <span className="animate-pulse">▍</span>}
          </p>

          {citationWarnings.length > 0 && (
            <Alert variant="destructive">
              <AlertTitle>Possible fabricated citation</AlertTitle>
              <AlertDescription>
                <ul className="list-disc space-y-1 pl-4">
                  {citationWarnings.map((w, i) => (
                    <li key={i}>
                      [Source {w.source_number}] quoted text not found in that source (match
                      ratio {w.match_ratio.toFixed(2)}): "{w.quoted_text}"
                    </li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <Separator />
          <h2 className="flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            Sources
            <Badge variant="outline">{sources.length}</Badge>
          </h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {sources.map((s, i) => (
              <SourceCard key={s.chunk_id} index={i} source={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
