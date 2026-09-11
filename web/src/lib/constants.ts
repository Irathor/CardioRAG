// Four real, verified questions from the Phase 11 evaluation dataset
// (data/evaluation/eval_dataset.jsonl, ids q001/q010/q019/q031) - not
// invented sample text - so a first-time user has something grounded to
// try immediately. Kept in sync manually; there is no build step that reads
// the dataset file for this static, client-only app.
export const EXAMPLE_QUESTIONS = [
  'What undersampling acceleration factors have deep learning methods achieved for cardiac cine imaging?',
  "On the Natural Questions open-domain QA benchmark, how did RAG-Sequence's exact match score compare to REALM and T5-11B+SSM?",
  'What three quality dimensions does the Ragas framework evaluate in a RAG pipeline?',
  'What is the diagnostic accuracy of AI-based echocardiography for detecting valvular heart disease?',
] as const

export const DEFAULT_TOP_K = 5
export const DEFAULT_RETRIEVE_K = 20
