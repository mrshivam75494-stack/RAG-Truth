"""
verifier.py
-----------
Core faithfulness-verification engine.

Pipeline per claim:
  1. Embed the claim + all source chunks with a bi-encoder (fast, cheap).
  2. Retrieve the top-K most semantically similar source chunks. Chunks are
     built with a sliding window (stride 1 sentence) so evidence spanning a
     chunk boundary in the non-overlapping version isn't missed.
  3. Run each (evidence_chunk, claim) pair through a cross-encoder NLI model.
  4. Take the evidence chunk with the highest entailment probability as the
     claim's "best evidence".
  5. Apply a deterministic numeric-consistency guard: if the claim states a
     number/date/amount that never appears anywhere in the retrieved
     evidence, override the verdict to Hallucinated regardless of what NLI
     said. This catches a well-known NLI blind spot (number substitution
     reads as near-identical text to a semantic model).

Labels:
  - "entailment"    -> claim is Supported by the source (GREEN)
  - "neutral"       -> claim is not addressed by the source -> Hallucinated (RED)
  - "contradiction" -> claim conflicts with the source -> Hallucinated (RED)
  - "numeric_mismatch" -> NLI said entailment, but a number in the claim
                          doesn't appear anywhere in the evidence -> RED
"""

from dataclasses import dataclass, field

import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from claim_utils import split_into_claims, split_source_into_chunks
from numeric_guard import check_numeric_consistency

BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Two NLI options, traded off between speed and accuracy. DeBERTa-v3-base is
# meaningfully more accurate at genuine entailment/contradiction judgments
# (fewer false "supported" calls on subtly reworded claims) at the cost of
# being slower and a larger download (~370MB vs ~90MB).
CROSS_ENCODER_OPTIONS = {
    "Accurate (DeBERTa-v3-base, recommended)": "cross-encoder/nli-deberta-v3-base",
    "Fast (MiniLM)": "cross-encoder/nli-MiniLM2-L6-H768",
}
DEFAULT_CROSS_ENCODER_LABEL = "Accurate (DeBERTa-v3-base, recommended)"

# NLI cross-encoders from sentence-transformers output logits in this order.
NLI_LABELS = ["contradiction", "entailment", "neutral"]

DEFAULT_TOP_K_EVIDENCE = 4
DEFAULT_ENTAILMENT_THRESHOLD = 0.5

# If the contradiction probability for the winning chunk is also above this,
# treat the claim as hallucinated even if entailment nominally cleared the
# threshold -- a close entailment/contradiction race is a sign the model is
# unsure, not a confident "supported".
CONTRADICTION_VETO = 0.3


@dataclass
class ClaimResult:
    claim: str
    label: str  # "supported" | "hallucinated"
    nli_label: str  # entailment / neutral / contradiction / numeric_mismatch
    confidence: float  # probability of the winning NLI label
    best_evidence: str
    evidence_candidates: list[str] = field(default_factory=list)
    numeric_flag: set = field(default_factory=set)  # mismatched numbers, if any


@st.cache_resource(show_spinner=False)
def load_models(cross_encoder_name: str):
    """Load and cache both models for the lifetime of the Streamlit session."""
    bi_encoder = SentenceTransformer(BI_ENCODER_NAME)
    cross_encoder = CrossEncoder(cross_encoder_name)
    return bi_encoder, cross_encoder


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sliding_window_chunks(source_doc: str, sentences_per_chunk: int) -> list[str]:
    """
    Build overlapping chunks (stride 1 sentence) instead of non-overlapping
    ones. This means a fact spanning e.g. sentence 3-4 is still captured
    whole in at least one chunk, rather than risking a split at sentence 4/5.
    """
    sentences = split_into_claims(source_doc)
    if not sentences:
        return []
    if len(sentences) <= sentences_per_chunk:
        return [" ".join(sentences)]

    chunks = []
    for i in range(0, len(sentences) - sentences_per_chunk + 1):
        chunks.append(" ".join(sentences[i : i + sentences_per_chunk]))
    return chunks


def verify_answer(
    source_doc: str,
    ai_answer: str,
    sentences_per_chunk: int = 2,
    top_k_evidence: int = DEFAULT_TOP_K_EVIDENCE,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    cross_encoder_name: str = CROSS_ENCODER_OPTIONS[DEFAULT_CROSS_ENCODER_LABEL],
    enable_numeric_guard: bool = True,
):
    """
    Run the full pipeline and return:
      - claims: list[ClaimResult]
      - faithfulness_score: float (0-100)
    """
    bi_encoder, cross_encoder = load_models(cross_encoder_name)

    claims = split_into_claims(ai_answer)
    source_chunks = _sliding_window_chunks(source_doc, sentences_per_chunk)

    if not claims:
        return [], 0.0
    if not source_chunks:
        results = [
            ClaimResult(
                claim=c,
                label="hallucinated",
                nli_label="neutral",
                confidence=1.0,
                best_evidence="(no source text provided)",
            )
            for c in claims
        ]
        return results, 0.0

    chunk_embeddings = bi_encoder.encode(source_chunks, convert_to_numpy=True, normalize_embeddings=True)
    claim_embeddings = bi_encoder.encode(claims, convert_to_numpy=True, normalize_embeddings=True)

    results: list[ClaimResult] = []

    for claim, claim_emb in zip(claims, claim_embeddings):
        sims = chunk_embeddings @ claim_emb
        top_k = min(top_k_evidence, len(source_chunks))
        top_idx = np.argsort(-sims)[:top_k]
        candidate_chunks = [source_chunks[i] for i in top_idx]

        pairs = [(chunk, claim) for chunk in candidate_chunks]
        raw_scores = cross_encoder.predict(pairs)

        best_entailment_prob = -1.0
        best_contradiction_prob = 0.0
        best_label = "neutral"
        best_confidence = 0.0
        best_chunk = candidate_chunks[0]

        for chunk, logits in zip(candidate_chunks, raw_scores):
            probs = _softmax(np.array(logits))
            label_idx = int(np.argmax(probs))
            label = NLI_LABELS[label_idx]
            entail_prob = float(probs[NLI_LABELS.index("entailment")])
            contra_prob = float(probs[NLI_LABELS.index("contradiction")])

            if entail_prob > best_entailment_prob:
                best_entailment_prob = entail_prob
                best_contradiction_prob = contra_prob
                best_label = label
                best_confidence = float(probs[label_idx])
                best_chunk = chunk

        nli_says_supported = (
            best_entailment_prob >= entailment_threshold
            and best_contradiction_prob < CONTRADICTION_VETO
        )

        numeric_flag: set = set()
        final_supported = nli_says_supported
        final_nli_label = "entailment" if nli_says_supported else best_label

        if enable_numeric_guard and nli_says_supported:
            is_consistent, mismatched = check_numeric_consistency(claim, candidate_chunks)
            if not is_consistent:
                final_supported = False
                final_nli_label = "numeric_mismatch"
                numeric_flag = mismatched

        results.append(
            ClaimResult(
                claim=claim,
                label="supported" if final_supported else "hallucinated",
                nli_label=final_nli_label,
                confidence=best_entailment_prob if final_supported else best_confidence,
                best_evidence=best_chunk,
                evidence_candidates=candidate_chunks,
                numeric_flag=numeric_flag,
            )
        )

    supported_count = sum(1 for r in results if r.label == "supported")
    faithfulness_score = round(100 * supported_count / len(results), 1)

    return results, faithfulness_score
