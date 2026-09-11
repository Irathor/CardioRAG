import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import type { SourceInfo } from '@/lib/api'

export function SourceCard({ index, source }: { index: number; source: SourceInfo }) {
  const [expanded, setExpanded] = useState(false)
  const pages = source.pages.length > 1 ? `pp. ${source.pages.join(', ')}` : `p. ${source.pages[0]}`

  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-4">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="shrink-0">
              Source {index + 1}
            </Badge>
            <span className="text-xs text-muted-foreground">{pages}</span>
          </div>
          <span className="text-xs tabular-nums text-muted-foreground">
            score {source.score.toFixed(3)}
          </span>
        </div>
        <p className="text-sm font-medium leading-snug">{source.title ?? 'Untitled document'}</p>
        {source.doi && (
          <a
            href={`https://doi.org/${source.doi}`}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-muted-foreground underline-offset-2 hover:underline"
          >
            doi:{source.doi}
          </a>
        )}
      </CardHeader>
      <CardContent className="px-4">
        <p className={`text-sm text-muted-foreground ${expanded ? '' : 'line-clamp-3'}`}>
          {source.text}
        </p>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-1 text-xs font-medium text-primary hover:underline"
        >
          {expanded ? 'Show less' : 'Show full excerpt'}
        </button>
      </CardContent>
    </Card>
  )
}
