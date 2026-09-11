"""Phase 11: generate the evaluation dataset (data/evaluation/eval_dataset.jsonl).

This is a one-time authoring script, not a repeatable pipeline step: every
question and reference answer below was manually written after actually
reading the corpus text (see the Phase 11 process notes in the project
history) and spot-verifying facts against the extracted page text. Re-running
this script just re-serializes the same hand-authored content to JSON - it
does not re-derive facts from the corpus, and must not be treated as
regeneratable the way scripts/build_index.py is.

Document IDs are the content-hashes produced by compute_document_id() for
the 8 PDFs in data/corpus/ as of this dataset's creation - if the corpus
changes, these must be re-verified.
"""

from pathlib import Path

from cardiorag.models import EvaluationCategory, EvaluationExample

# document_id (sha256[:16] of file bytes) -> filename, for reference
DOC_1 = "4ee13c3bf6c0ec17"  # Improving efficiency/accuracy of CMR with AI (Zhang et al., JCMR 2024)
DOC_2 = "ec06edd265a572de"  # Screening/diagnosis of CVD using AI-enabled CMR (Wang et al., Nat Med 2024)
DOC_3 = "c9fa960ca5cf2c24"  # AI-enabled myocardial characterisation in CMR (Frontiers, mini-review)
DOC_4 = "23e3249e9a1e7541"  # Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al. 2020)
DOC_5 = "53cb82be5d04c056"  # RAG for LLMs in healthcare: a systematic review (Amugongo et al. 2025)
DOC_6 = "3adec62865ac7948"  # Improving LLM applications in biomedicine with RAG (systematic review + meta-analysis)
DOC_7 = "6c24762a3f277fd1"  # Ragas: Automated Evaluation of RAG
DOC_8 = "f37f6fabe0fe0d8c"  # Lost in the Middle: How LMs Use Long Contexts

EXAMPLES: list[EvaluationExample] = [
    # --- Factual / comparison / numeric, single document (3 per paper) ---
    EvaluationExample(
        id="q001", category=EvaluationCategory.FACTUAL,
        question="What undersampling acceleration factors have deep learning methods achieved for cardiac cine imaging?",
        expected_document_ids=[DOC_1], expected_pages=[6],
        reference_answer="Some proposed deep learning methods for cine imaging have achieved 12-fold and 13-fold undersampling for accelerated acquisition, though many such studies were limited to retrospective undersampling, single-coil data, and healthy subjects.",
        answerable=True,
    ),
    EvaluationExample(
        id="q002", category=EvaluationCategory.FACTUAL,
        question="How much did the cardiac multitasking low-rank tensor approach reduce reconstruction time compared to conventional iterative methods?",
        expected_document_ids=[DOC_1], expected_pages=[10],
        reference_answer="Cardiac multitasking (low-rank tensor approach) achieved image and T1 map quality similar to conventional iterative methods while reducing reconstruction time by more than 3000-fold.",
        answerable=True,
    ),
    EvaluationExample(
        id="q003", category=EvaluationCategory.FACTUAL,
        question="What does the review conclude is essential for translating AI-CMR research into clinical practice?",
        expected_document_ids=[DOC_1], expected_pages=[15],
        reference_answer="Interdisciplinary collaboration between MR physicists, clinicians, AI scientists, and industry professionals, combined with rigorous regulatory and ethical considerations, is essential for responsible clinical deployment.",
        answerable=True,
    ),
    EvaluationExample(
        id="q004", category=EvaluationCategory.COMPARISON,
        question="How did the AI diagnostic model's performance compare to physicians with more than 10 years of CMR reading experience?",
        expected_document_ids=[DOC_2], expected_pages=[3],
        reference_answer="The AI model achieved comparable performance (F1 score 0.931 vs. 0.927) but interpreted the 500-subject test set far faster (1.94 minutes vs. 418 minutes).",
        answerable=True,
    ),
    EvaluationExample(
        id="q005", category=EvaluationCategory.FACTUAL,
        question="What AUC did the two-stage screening and diagnostic AI models achieve for cardiovascular disease from CMR?",
        expected_document_ids=[DOC_2], expected_pages=[1],
        reference_answer="The screening model achieved an AUC of 0.988 (+/- 0.3%) and the diagnostic model achieved an AUC of 0.991 (+/- 0.0%) across internal and external test datasets.",
        answerable=True,
    ),
    EvaluationExample(
        id="q006", category=EvaluationCategory.FACTUAL,
        question="What is the current gold-standard diagnostic method for pulmonary arterial hypertension, and what risk does it carry?",
        expected_document_ids=[DOC_2], expected_pages=[8],
        reference_answer="Right heart catheterization is the gold standard for diagnosing PAH, but it is invasive and can cause complications including hematoma, pneumothorax, arrhythmias and hypotensive episodes.",
        answerable=True,
    ),
    EvaluationExample(
        id="q007", category=EvaluationCategory.FACTUAL,
        question="What is the standard imaging method for detecting and quantifying myocardial iron overload?",
        expected_document_ids=[DOC_3], expected_pages=[2],
        reference_answer="T2* mapping is the standard method for detecting and quantifying myocardial iron overload, used to guide chelation therapy.",
        answerable=True,
    ),
    EvaluationExample(
        id="q008", category=EvaluationCategory.COMPARISON,
        question="How do the two leading synthetic LGE techniques, virtual native enhancement (VNE) and cine-generated enhancement (CGE), differ in approach?",
        expected_document_ids=[DOC_3], expected_pages=[4],
        reference_answer="VNE generates synthetic LGE-like scar images from native T1 maps plus cine MRI, applied to chronic infarction and HCM; CGE identifies LGE from cine MRI alone, applied to acute MI.",
        answerable=True,
    ),
    EvaluationExample(
        id="q009", category=EvaluationCategory.FACTUAL,
        question="What fairness concern was identified with cine segmentation models trained on UK Biobank data?",
        expected_document_ids=[DOC_3], expected_pages=[6],
        reference_answer="Since over 80% of UK Biobank participants are White, models trained on this data perform less well on more diverse populations, raising data imbalance and fairness concerns.",
        answerable=True,
    ),
    EvaluationExample(
        id="q010", category=EvaluationCategory.COMPARISON,
        question="On the Natural Questions open-domain QA benchmark, how did RAG-Sequence's exact match score compare to REALM and T5-11B+SSM?",
        expected_document_ids=[DOC_4], expected_pages=[6],
        reference_answer="RAG-Sequence achieved 44.5 Exact Match on Natural Questions, versus 40.4 for REALM and 36.6 for T5-11B+SSM, setting a new state of the art.",
        answerable=True,
    ),
    EvaluationExample(
        id="q011", category=EvaluationCategory.FACTUAL,
        question="What is the difference between the RAG-Sequence and RAG-Token model formulations?",
        expected_document_ids=[DOC_4], expected_pages=[1, 3],
        reference_answer="RAG-Sequence retrieves one set of top-K documents and uses the same documents to generate the whole output sequence; RAG-Token can draw a different latent document per target token, allowing the generator to combine content from several documents.",
        answerable=True,
    ),
    EvaluationExample(
        id="q012", category=EvaluationCategory.FACTUAL,
        question="In the RAG paper's index hot-swapping experiment, how accurate were answers when the Wikipedia index matched vs. mismatched the year being asked about?",
        expected_document_ids=[DOC_4], expected_pages=[8],
        reference_answer="Using the matching index, RAG answered correctly 70% (2016 index/2016 leaders) and 68% (2018 index/2018 leaders) of the time; accuracy dropped to 12% and 4% with mismatched index/year pairs.",
        answerable=True,
    ),
    EvaluationExample(
        id="q013", category=EvaluationCategory.FACTUAL,
        question="How much did adding RAG improve GPT-4's accuracy on multi-choice medical QA questions?",
        expected_document_ids=[DOC_5], expected_pages=[9],
        reference_answer="RAG increased GPT-4's average accuracy on multi-choice medical questions from 73.44% to 79.97%.",
        answerable=True,
    ),
    EvaluationExample(
        id="q014", category=EvaluationCategory.FACTUAL,
        question="How many studies were included in the systematic review of RAG in healthcare, and from how many initially retrieved articles?",
        expected_document_ids=[DOC_5], expected_pages=[4],
        reference_answer="70 studies were included, selected from an initial pool of 2,139 articles retrieved from Google Scholar and PubMed.",
        answerable=True,
    ),
    EvaluationExample(
        id="q015", category=EvaluationCategory.COMPARISON,
        question="How did prompt-RAG compare to traditional vector-embedding-based RAG in a Korean medicine QA chatbot?",
        expected_document_ids=[DOC_5], expected_pages=[12],
        reference_answer="Prompt-RAG, which uses direct natural language prompts instead of vector embeddings for retrieval, outperformed both ChatGPT and traditional vector-embedding-based RAG models on relevance ratings from doctors.",
        answerable=True,
    ),
    EvaluationExample(
        id="q016", category=EvaluationCategory.FACTUAL,
        question="What was the pooled effect size (odds ratio) found when comparing RAG-enhanced LLMs to baseline LLMs in the biomedicine meta-analysis?",
        expected_document_ids=[DOC_6], expected_pages=[1],
        reference_answer="The pooled odds ratio was 1.35 (95% CI: 1.19-1.53, P = .001), indicating a statistically significant improvement from adding RAG.",
        answerable=True,
    ),
    EvaluationExample(
        id="q017", category=EvaluationCategory.COMPARISON,
        question="Did adding RAG increase or decrease response time and cost in the gastrointestinal radiology case study?",
        expected_document_ids=[DOC_6], expected_pages=[8],
        reference_answer="RAG increased both: mean response time rose from 15.7s (LLM alone) to 29.8s (LLM+RAG), and cost rose from $0.02 to $0.15 per case.",
        answerable=True,
    ),
    EvaluationExample(
        id="q018", category=EvaluationCategory.FACTUAL,
        question="How accurate was GPT-4 Turbo at extracting table data from documents, according to a study cited in the biomedicine RAG review?",
        expected_document_ids=[DOC_6], expected_pages=[7],
        reference_answer="A preliminary study found GPT-4 Turbo had only 16% accuracy extracting table data, highlighting the need for multi-modal document preprocessing beyond LLM-only text extraction.",
        answerable=True,
    ),
    EvaluationExample(
        id="q019", category=EvaluationCategory.FACTUAL,
        question="What three quality dimensions does the Ragas framework evaluate in a RAG pipeline?",
        expected_document_ids=[DOC_7], expected_pages=[3],
        reference_answer="Ragas evaluates Faithfulness (is the answer grounded in the retrieved context), Answer Relevance (does the answer address the question), and Context Relevance (is the retrieved context focused and free of irrelevant information).",
        answerable=True,
    ),
    EvaluationExample(
        id="q020", category=EvaluationCategory.FACTUAL,
        question="How is the faithfulness score computed in Ragas?",
        expected_document_ids=[DOC_7], expected_pages=[3],
        reference_answer="Faithfulness F is computed as the number of statements from the answer that an LLM verifies as supported by the context (V), divided by the total number of statements extracted from the answer (S): F = V / S.",
        answerable=True,
    ),
    EvaluationExample(
        id="q021", category=EvaluationCategory.COMPARISON,
        question="How did Ragas's metrics compare to the GPT Score and GPT Ranking baselines in agreement with human judgments?",
        expected_document_ids=[DOC_7], expected_pages=[5],
        reference_answer="Ragas achieved higher agreement with human annotators on all three dimensions, e.g. 0.95 vs. 0.72/0.54 for faithfulness, showing its metrics align more closely with human judgment than the baselines.",
        answerable=True,
    ),
    EvaluationExample(
        id="q022", category=EvaluationCategory.FACTUAL,
        question="What performance pattern do language models show as the position of relevant information changes within a long input context?",
        expected_document_ids=[DOC_8], expected_pages=[2],
        reference_answer="Models show a U-shaped performance curve: performance is highest when relevant information is at the beginning or end of the context, and degrades significantly when it is in the middle.",
        answerable=True,
    ),
    EvaluationExample(
        id="q023", category=EvaluationCategory.COMPARISON,
        question="How did Claude-1.3 compare to other models on the synthetic key-value retrieval task?",
        expected_document_ids=[DOC_8], expected_pages=[7],
        reference_answer="Claude-1.3 and Claude-1.3 (100K) performed nearly perfectly at all tested context lengths, while models like GPT-3.5-Turbo and MPT-30B-Instruct struggled, especially with key-value pairs placed in the middle of the context.",
        answerable=True,
    ),
    EvaluationExample(
        id="q024", category=EvaluationCategory.FACTUAL,
        question="In the open-domain QA case study, how much does retrieving more than 20 documents improve reader accuracy?",
        expected_document_ids=[DOC_8], expected_pages=[10],
        reference_answer="Reader accuracy improves only marginally beyond 20 retrieved documents - about 1.5% for GPT-3.5-Turbo and about 1% for Claude-1.3 - indicating readers don't effectively use the extra retrieved context.",
        answerable=True,
    ),

    # --- Synthesis: combining multiple facts/sections, possibly one paper ---
    EvaluationExample(
        id="q025", category=EvaluationCategory.SYNTHESIS,
        question="What common translational challenges do the AI-CMR myocardial characterisation review identify across the different techniques it covers (segmentation, mapping, synthetic imaging)?",
        expected_document_ids=[DOC_3], expected_pages=[2, 4, 6],
        reference_answer="Across LGE segmentation, parametric mapping, and synthetic contrast-free imaging, the review repeatedly raises data availability/generalisability, fairness (e.g. demographic imbalance in training data), interpretability, and regulatory approval as shared barriers to clinical adoption.",
        answerable=True,
    ),
    EvaluationExample(
        id="q026", category=EvaluationCategory.SYNTHESIS,
        question="Based on the RAG paper's retrieval ablation and index hot-swap experiments, what does this suggest about why dense retrieval is useful even though it can be replaced by simpler methods for some tasks?",
        expected_document_ids=[DOC_4], expected_pages=[7, 8],
        reference_answer="Dense retrieval outperforms fixed BM25 on most tasks (especially open-domain QA) and, unlike a fixed retriever, its non-parametric index can be swapped at test time to update world knowledge without retraining - shown by the world-leader experiment where accuracy tracked whichever index matched the query's time period.",
        answerable=True,
    ),
    EvaluationExample(
        id="q027", category=EvaluationCategory.SYNTHESIS,
        question="What do the FDA-related mentions across the CMR papers suggest about the regulatory state of AI in cardiac imaging?",
        expected_document_ids=[DOC_1, DOC_3], expected_pages=[14, 6],
        reference_answer="Over 500 AI/ML medical devices had FDA marketing authorization as of October 2022, and the FDA has issued Good Machine Learning Practice (GMLP) guidelines - together suggesting an active but still-maturing regulatory framework specifically built around AI's need for transparency and quality control.",
        answerable=True,
    ),

    # --- Multi-paper: requires evidence explicitly from 2+ documents ---
    EvaluationExample(
        id="q028", category=EvaluationCategory.MULTI_PAPER,
        question="Two different reviews in this corpus quantify RAG's benefit for biomedical LLM tasks - what numbers do they each report?",
        expected_document_ids=[DOC_5, DOC_6], expected_pages=[9, 1],
        reference_answer="One review reports RAG raising GPT-4's multi-choice medical QA accuracy from 73.44% to 79.97%; a separate systematic review and meta-analysis reports a pooled odds ratio of 1.35 (95% CI 1.19-1.53) for RAG-enhanced LLMs vs. baseline across biomedical tasks generally.",
        answerable=True,
    ),
    EvaluationExample(
        id="q029", category=EvaluationCategory.MULTI_PAPER,
        question="How do the two systematic reviews on RAG in healthcare differ in their literature search scope?",
        expected_document_ids=[DOC_5, DOC_6], expected_pages=[3, 9],
        reference_answer="One review used PRISMA and included English-language articles from January 2020 to February 2025 (70 studies from 2,139 retrieved). The other was limited to peer-reviewed publications in PubMed, Embase, and PsycINFO, excluding preprints, non-English studies, and sources like IEEE Xplore or Google Scholar.",
        answerable=True,
    ),
    EvaluationExample(
        id="q030", category=EvaluationCategory.MULTI_PAPER,
        question="How does the 'Lost in the Middle' finding about retrieval count relate to the reader-saturation behavior described for RAG systems elsewhere in this corpus?",
        expected_document_ids=[DOC_8, DOC_4], expected_pages=[10, 7],
        reference_answer="'Lost in the Middle' finds that retrieving more than 20 documents only marginally improves reader accuracy (~1-1.5%) because readers under-use context placed in the middle; this is consistent with the original RAG paper's finding that a well-chosen (not necessarily large) retrieved set, via a strong dense retriever, is what drives performance gains rather than sheer retrieval volume.",
        answerable=True,
    ),

    # --- No evidence: real, plausible questions the corpus does not cover ---
    EvaluationExample(
        id="q031", category=EvaluationCategory.NO_EVIDENCE,
        question="What is the diagnostic accuracy of AI-based echocardiography for detecting valvular heart disease?",
        expected_document_ids=[], expected_pages=[],
        reference_answer="Not covered: this corpus's cardiac-imaging papers concern cardiac MRI (CMR), not echocardiography. No evidence about echocardiography-based AI diagnostic accuracy is present.",
        answerable=False,
    ),
    EvaluationExample(
        id="q032", category=EvaluationCategory.NO_EVIDENCE,
        question="What FDA-approved AI algorithm exists for automated coronary artery calcium scoring on CT scans?",
        expected_document_ids=[], expected_pages=[],
        reference_answer="Not covered: the corpus discusses CMR (magnetic resonance), not CT-based calcium scoring, and does not name any specific FDA-approved algorithm for this purpose.",
        answerable=False,
    ),
    EvaluationExample(
        id="q033", category=EvaluationCategory.NO_EVIDENCE,
        question="What is the recommended pediatric dosing protocol for gadolinium contrast agents in cardiac MRI?",
        expected_document_ids=[], expected_pages=[],
        reference_answer="Not covered: none of the corpus papers discuss contrast agent dosing protocols, pediatric or otherwise.",
        answerable=False,
    ),
    EvaluationExample(
        id="q034", category=EvaluationCategory.NO_EVIDENCE,
        question="According to this corpus, what is the cost-effectiveness of AI-based CMR screening compared to invasive coronary angiography in a US healthcare system?",
        expected_document_ids=[], expected_pages=[],
        reference_answer="Not covered: none of the reviewed papers perform a health-economics or cost-effectiveness analysis comparing AI-CMR screening to invasive angiography.",
        answerable=False,
    ),

    # --- Misleading: false premise embedded in the question ---
    EvaluationExample(
        id="q035", category=EvaluationCategory.MISLEADING,
        question="Since CMR AI models are reported to generalize equally well across all patient populations, which specific ethnic groups were used to validate this claim?",
        expected_document_ids=[DOC_2, DOC_3], expected_pages=[8, 6],
        reference_answer="The premise is false: the corpus explicitly flags generalizability as a limitation, not a validated strength. One paper's cohort was entirely from eastern Asian institutions with generalizability to other ethnicities untested; another notes UK Biobank-trained models (>80% White participants) perform worse on more diverse populations.",
        answerable=True,
    ),
    EvaluationExample(
        id="q036", category=EvaluationCategory.MISLEADING,
        question="Given that the Ragas paper reports 100% agreement between its metrics and human judges, what accounts for this perfect agreement?",
        expected_document_ids=[DOC_7], expected_pages=[5],
        reference_answer="The premise is false: Ragas did not report 100% agreement. It reported accuracy scores such as 0.95 for faithfulness, 0.78 for answer relevance, and 0.70 for context relevance versus human judgments - higher than the baselines compared, but not perfect.",
        answerable=True,
    ),
    EvaluationExample(
        id="q037", category=EvaluationCategory.MISLEADING,
        question="Since retrieval-augmented generation has been shown in this corpus to completely eliminate hallucination in biomedical LLMs, what mechanism accounts for this total elimination?",
        expected_document_ids=[DOC_4, DOC_6], expected_pages=[10, 1],
        reference_answer="The premise is false: no paper in this corpus claims RAG eliminates hallucination entirely. The original RAG paper notes its non-parametric source (Wikipedia) is never fully factual or bias-free, and the biomedicine meta-analysis reports RAG as an improvement (pooled OR 1.35) over baseline, not a guarantee of correctness.",
        answerable=True,
    ),
    EvaluationExample(
        id="q038", category=EvaluationCategory.MISLEADING,
        question="Given that RAG-Token always requires more retrieval computation per query than RAG-Sequence, how much slower is it in practice?",
        expected_document_ids=[DOC_4], expected_pages=[3],
        reference_answer="The premise conflates two different things: RAG-Sequence and RAG-Token differ in how retrieved documents are used during generation (one document set for the whole sequence vs. a possibly different one per token), not in a stated retrieval-computation or speed comparison - the corpus does not report RAG-Token being categorically slower.",
        answerable=True,
    ),
]


def main() -> None:
    out_path = Path("data/evaluation/eval_dataset.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for example in EXAMPLES:
            f.write(example.model_dump_json())
            f.write("\n")
    print(f"Wrote {len(EXAMPLES)} evaluation examples to {out_path}")


if __name__ == "__main__":
    main()
