# RAG-TRUTH— AI Answer Faithfulness Verifier

An MVP tool that catches hallucinated AI answers before they cause liability
in legal, campus-policy, or enterprise deployments. It takes a **Query**,
**Source Document**, and **AI Answer**, breaks the answer into individual
claims, verifies each one against the source, and renders a color-coded
faithfulness report with an aggregate 0–100% score.

## How it works

```
AI Answer ──► split into claims (sentences)
                     │
                     ▼
Source Doc ──► overlapping sliding-window chunks (stride 1 sentence)
                     │
        bi-encoder embeds claim + all chunks
                     │
        top-K most similar chunks retrieved per claim
                     │
        cross-encoder NLI scores (chunk, claim) pairs
                     │
        entailment ≥ threshold AND contradiction < veto  → tentatively Supported
                     │
        numeric/date consistency guard (deterministic, non-ML)
                     │
        entailment/contradiction/neutral/numeric_mismatch
                     │
        🟢 Supported / 🔴 Hallucinated
                     │
        Faithfulness Score = % claims supported
```

- **Bi-encoder** (`all-MiniLM-L6-v2`): fast semantic search to find the
  relevant slice of the source doc for each claim — avoids feeding the whole
  document to the slower cross-encoder.
- **Cross-encoder NLI**: choose between two models in Advanced settings —
  `cross-encoder/nli-deberta-v3-base` (default, more accurate at genuine
  entailment/contradiction judgments) or `cross-encoder/nli-MiniLM2-L6-H768`
  (faster, smaller, less precise on subtly reworded claims).
- **Contradiction veto**: a claim is only called Supported if entailment
  clears the threshold *and* contradiction probability stays low. A close
  entailment/contradiction race means the model is unsure — that's treated
  as a red flag, not a pass.
- **Numeric/date consistency guard** (`numeric_guard.py`, deterministic,
  no ML): NLI models are trained mostly on paraphrase/implication and are
  weak at catching pure number substitution — "10%" vs "15%" often still
  reads as entailment because the sentences are otherwise identical. This
  guard extracts numbers, percentages, and currency amounts from the claim
  and checks they actually appear somewhere in the retrieved evidence,
  overriding NLI's verdict when they don't. It's intentionally conservative
  (normalizes `Rs. 500` / `₹500` / `500` as equivalent) to avoid false
  alarms on reformatted-but-correct numbers.
- **Overlapping evidence chunks**: chunks use a sliding window (stride 1
  sentence) instead of fixed non-overlapping blocks, so a fact spanning a
  chunk boundary is still captured whole in at least one candidate.

All models together are 250MB–450MB depending on which NLI model you pick,
and run comfortably on CPU or a 4GB GPU.

## Setup

```bash
cd faithcheck
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

First run downloads the two models from Hugging Face (~250MB total,
cached afterward in `~/.cache/huggingface`).

## Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Using it

1. Paste your **Source Document** (the policy, contract, or reference text).
2. Paste the **AI Answer** you want to verify.
3. (Optional) Add the original **Query** for context.
4. Click **Analyze Faithfulness**.
5. Read the highlighted answer (green = supported, red = flagged) and expand
   any claim in the breakdown to see exactly which sentence(s) from the
   source were used as evidence and the model's confidence.


