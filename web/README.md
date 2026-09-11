# CardioRAG web client

React + TypeScript + Tailwind + [shadcn/ui](https://ui.shadcn.com), talking to the CardioRAG
FastAPI backend (`../src/cardiorag/api/`) over plain HTTP and Server-Sent Events. A static single-page
app with no server of its own - every real capability (retrieval, reranking, generation, citation
checking) lives in the API, not here.

Replaces the earlier Streamlit UI (`app/streamlit_app.py`, removed) - same feature set (ask a
question, see a streamed grounded answer with sources and citation warnings, browse indexed
documents), a different rendering stack. Light/dark theme toggle (top right), persisted in
`localStorage` via `next-themes`.

## Develop

```bash
npm install
cp .env.example .env.local   # point VITE_API_BASE_URL at your running API
npm run dev
```

Requires the API running separately (`uvicorn cardiorag.api.main:app`, see the root README).

## Build

```bash
npm run build   # outputs static files to dist/
```

## Notes

- `src/lib/api.ts` is a typed client mirroring `api/schemas.py` field-for-field, including a
  hand-rolled SSE reader for `POST /query/stream` (the browser's native `EventSource` can't send a
  POST body, so a GET-only API wouldn't work here).
- The backend must have CORS enabled to be reachable from a browser at a different origin - see
  `CORS_ALLOWED_ORIGINS` in the root `.env.example`.
- `VITE_API_KEY`, if set, is sent as `X-API-Key` on every request. It is **not** actually secret
  once built into a client bundle - visible to anyone who opens the page - fine for this project's
  small-trusted-deployment threat model, not a substitute for real per-user auth.
- `components/ui/alert.tsx` adds a third `warning` variant (amber text, same neutral card
  background as `default`/`destructive`) beyond shadcn's stock two - used for the standing
  "research tool, not medical advice" disclaimer, which needed to read as a notice without the
  alarm of `destructive`.
