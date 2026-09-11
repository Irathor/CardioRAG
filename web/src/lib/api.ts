// Typed client for the CardioRAG FastAPI backend (src/cardiorag/api/).
// Mirrors src/cardiorag/api/schemas.py field-for-field - if a field is
// renamed there, it must be renamed here too, there's no shared source of truth.

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(
  /\/$/,
  '',
)
// A shared secret is inherently visible in a browser's network tab/bundle -
// this only makes sense for the same small-trusted-deployment threat model
// api/auth.py itself documents, never as real protection against an end user.
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined

export interface SourceInfo {
  title: string | null
  pages: number[]
  doi: string | null
  chunk_id: string
  document_id: string
  score: number
  text: string
}

export interface CitationWarning {
  source_number: number
  quoted_text: string
  match_ratio: number
}

export interface HealthResponse {
  status: string
  index_loaded: boolean
  num_chunks: number
  llm_provider_configured: boolean
}

export interface DocumentInfo {
  document_id: string
  title: string | null
  doi: string | null
  filename: string
  num_chunks: number
}

export interface RetrieveResponse {
  question: string
  results: SourceInfo[]
  latency_ms: number
}

export interface QueryResponse {
  question: string
  answer: string
  sources: SourceInfo[]
  latency_ms: number
  citation_warnings: CitationWarning[]
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return API_KEY ? { ...extra, 'X-API-Key': API_KEY } : extra
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response))
  return response.json()
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const response = await fetch(`${API_BASE_URL}/documents`, { headers: authHeaders() })
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response))
  const body = (await response.json()) as { documents: DocumentInfo[] }
  return body.documents
}

export async function retrieve(question: string, topK: number): Promise<RetrieveResponse> {
  const response = await fetch(`${API_BASE_URL}/retrieve`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ question, top_k: topK }),
  })
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response))
  return response.json()
}

export interface QueryStreamCallbacks {
  onSources?: (sources: SourceInfo[]) => void
  onToken?: (text: string) => void
  onDone?: (payload: {
    answer: string
    citation_warnings: CitationWarning[]
    latency_ms: number
  }) => void
  onError?: (detail: string) => void
}

/** POST /query/stream, an SSE response - EventSource can't send a POST body,
 * so this reads the response stream by hand and splits it on blank lines,
 * the same "event: X\ndata: Y\n\n" framing api/main.py's _sse_event() writes. */
export async function streamQuery(
  question: string,
  options: { topK: number; retrieveK: number },
  callbacks: QueryStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      question,
      top_k: options.topK,
      retrieve_k: options.retrieveK,
    }),
    signal,
  })

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await parseErrorDetail(response))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      dispatchEvent(buffer.slice(0, boundary), callbacks)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')
    }
  }
}

function dispatchEvent(rawEvent: string, callbacks: QueryStreamCallbacks): void {
  const lines = rawEvent.split('\n')
  const eventLine = lines.find((line) => line.startsWith('event: '))
  const dataLine = lines.find((line) => line.startsWith('data: '))
  if (!eventLine || !dataLine) return

  const eventType = eventLine.slice('event: '.length)
  const data = JSON.parse(dataLine.slice('data: '.length))

  switch (eventType) {
    case 'sources':
      callbacks.onSources?.(data.sources as SourceInfo[])
      break
    case 'token':
      callbacks.onToken?.(data.text as string)
      break
    case 'done':
      callbacks.onDone?.(data)
      break
    case 'error':
      callbacks.onError?.(data.detail as string)
      break
  }
}
