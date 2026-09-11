import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { getDocuments, type DocumentInfo } from '@/lib/api'

export function DocumentsPanel() {
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getDocuments()
      .then(setDocuments)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load'))
  }, [])

  if (error) {
    return <p className="text-sm text-destructive">Could not load documents: {error}</p>
  }

  if (!documents) {
    return (
      <div className="flex flex-col gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {documents.map((doc) => (
        <Card key={doc.document_id} className="py-3">
          <CardContent className="flex items-center justify-between gap-4 px-4">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{doc.title ?? doc.filename}</p>
              <p className="text-xs text-muted-foreground">
                {doc.filename}
                {doc.doi && ` · doi:${doc.doi}`}
              </p>
            </div>
            <Badge variant="secondary" className="shrink-0">
              {doc.num_chunks} chunks
            </Badge>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
