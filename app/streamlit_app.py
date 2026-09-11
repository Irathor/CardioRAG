"""CardioRAG research interface (Phase 16).

Calls the FastAPI backend (Phase 15) over HTTP rather than re-implementing
the pipeline in-process: this UI is a presentation layer only, so the
embedding model/index/reranker load once in the API process, not once per
Streamlit session, and both processes can be started, stopped, and scaled
independently - a deliberately separated, production-style topology
rather than a quick single-process demo script.

Run alongside the API:
    uvicorn cardiorag.api.main:app --port 8000
    streamlit run app/streamlit_app.py
"""

import httpx
import streamlit as st

from cardiorag.config import settings
from cardiorag.evaluation.dataset import load_evaluation_dataset

st.set_page_config(page_title="CardioRAG", page_icon="🫀", layout="wide")

EXAMPLE_QUESTIONS_PATH = "data/evaluation/eval_dataset.jsonl"


@st.cache_data(ttl=300)
def _load_example_questions() -> list[str]:
    """A few real, verified questions from the Phase 11 evaluation dataset -
    not invented sample text - so a first-time user has something grounded
    to try immediately."""
    try:
        examples = load_evaluation_dataset(EXAMPLE_QUESTIONS_PATH)
    except FileNotFoundError:
        return []
    # A small, deliberately varied sample: factual, comparison, and one
    # no-evidence question, so users immediately see the refusal behavior too.
    wanted_ids = {"q001", "q010", "q019", "q031"}
    return [e.question for e in examples if e.id in wanted_ids]


def _call_api(method: str, path: str, **kwargs):
    url = f"{settings.api_base_url}{path}"
    # Sent even when settings.api_key is None (as an absent header, which
    # require_api_key ignores) - simpler than branching on whether auth is
    # configured, and correct either way (Phase 20 fix #6).
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    try:
        response = httpx.request(method, url, timeout=60.0, headers=headers, **kwargs)
    except httpx.ConnectError:
        st.error(
            f"Could not reach the API at {settings.api_base_url}. Is it running?\n\n"
            "Start it with: `uvicorn cardiorag.api.main:app --port 8000`"
        )
        return None
    except httpx.TimeoutException:
        st.error("The API took too long to respond (60s timeout exceeded).")
        return None

    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(f"API error ({response.status_code}): {detail}")
        return None

    return response.json()


st.title("🫀 CardioRAG")
st.caption(
    "Retrieval-augmented question answering over cardiovascular magnetic resonance (CMR) and "
    "cardiovascular AI literature."
)
st.warning(
    "**Research and educational tool - not a medical diagnostic system.** "
    "Answers are grounded in a small, fixed corpus of papers and must never be treated as "
    "medical advice or a clinical recommendation.",
    icon="⚕️",
)

with st.sidebar:
    st.header("Advanced controls")
    top_k = st.slider(
        "Final sources (top_k)", min_value=1, max_value=10, value=5,
        help="How many sources are kept after reranking and shown to the LLM.",
    )
    retrieve_k = st.slider(
        "Candidates before reranking (retrieve_k)", min_value=top_k, max_value=50, value=20,
        help="How many chunks dense retrieval pulls before the cross-encoder reranks them down "
        "to top_k. Must be at least top_k.",
    )
    st.divider()
    health = _call_api("GET", "/health")
    if health:
        st.metric("Indexed chunks", health["num_chunks"])
        if not health["llm_provider_configured"]:
            st.error("No LLM provider configured on the server - /query will fail.")

example_questions = _load_example_questions()
if example_questions:
    selected_example = st.selectbox(
        "Or try a real question from the evaluation dataset:",
        options=[""] + example_questions,
        format_func=lambda q: "Choose an example..." if q == "" else q,
    )
else:
    selected_example = ""

question = st.text_area(
    "Your question",
    value=selected_example,
    placeholder="e.g. How does artificial intelligence improve cardiac MRI segmentation?",
    height=80,
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence and generating a grounded answer..."):
        result = _call_api(
            "POST", "/query",
            json={"question": question, "top_k": top_k, "retrieve_k": retrieve_k},
        )

    if result:
        st.subheader("Answer")
        st.write(result["answer"])
        st.caption(f"Generated in {result['latency_ms']} ms using {len(result['sources'])} source(s).")

        st.subheader("Sources")
        st.caption(
            "Every source below is exactly what was given to the LLM as evidence - inspect it "
            "to judge the answer's grounding yourself, rather than taking it on faith."
        )
        if not result["sources"]:
            st.info("No sources were retrieved for this question.")
        for i, source in enumerate(result["sources"], start=1):
            pages = ", ".join(str(p) for p in source["pages"]) if source["pages"] else "unknown"
            with st.expander(f"[Source {i}] {source['title'] or 'Untitled'} (score: {source['score']:.3f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Page(s):** {pages}")
                with col2:
                    st.markdown(f"**DOI:** {source['doi'] or 'N/A'}")
                st.markdown(f"**Chunk ID:** `{source['chunk_id']}`")
                st.markdown("**Retrieved text:**")
                st.text(source["text"])
