import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { getHealth, type HealthResponse } from '@/lib/api'

type Status = 'loading' | 'ok' | 'unreachable'

export function HealthBadge() {
  const [status, setStatus] = useState<Status>('loading')
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    getHealth()
      .then((data) => {
        if (cancelled) return
        setHealth(data)
        setStatus('ok')
      })
      .catch(() => {
        if (!cancelled) setStatus('unreachable')
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (status === 'loading') {
    return (
      <Badge variant="outline" className="gap-1.5 text-muted-foreground">
        <span className="size-1.5 rounded-full bg-muted-foreground/50" />
        Checking API…
      </Badge>
    )
  }

  if (status === 'unreachable') {
    return (
      <Badge variant="destructive" className="gap-1.5">
        <span className="size-1.5 rounded-full bg-destructive-foreground" />
        API unreachable
      </Badge>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className="gap-1.5 border-emerald-600/30 text-emerald-700 dark:text-emerald-400">
        <span className="size-1.5 rounded-full bg-emerald-500" />
        {health!.num_chunks} chunks indexed
      </Badge>
      {!health!.llm_provider_configured && (
        <Badge variant="destructive">No LLM provider configured</Badge>
      )}
    </div>
  )
}
