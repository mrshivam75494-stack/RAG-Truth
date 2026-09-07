# FaithCheck — AI Answer Faithfulness Verifier

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

## Known limitations (be upfront about these in a demo/pitch)

- **Sentence-level granularity only** — a claim that's half-right and
  half-wrong within one sentence will be scored as a whole.
- **NLI ≠ ground truth** — even DeBERTa-v3-base can misjudge highly
  paraphrased claims. The numeric guard closes the most common gap (number
  substitution) but doesn't catch every kind of factual drift — e.g. a
  claim that swaps *who* is responsible for an obligation without changing
  any numbers won't be caught by the guard, only by NLI itself.
- **Retrieval-limited** — if the true supporting evidence is spread across
  more chunks than top-K, the claim may be wrongly flagged. Raise the
  "Evidence candidates" slider for longer, denser documents.
- **English-tuned models** — for multilingual source documents, swap both
  models for multilingual equivalents (e.g.
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for
  retrieval and `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` for NLI).
- **Numeric guard is deterministic, not learned** — it only fires on
  claims containing numbers/currency/percentages, and only when the number
  is genuinely absent from every retrieved chunk. It won't catch a number
  that's wrong but coincidentally appears elsewhere in the document.

## Extending toward a stronger v2

- Swap the binary Green/Red for a 3-way **Supported / Contradicted /
  Unverifiable** signal — the raw NLI label is already captured per claim in
  `ClaimResult.nli_label`, so the UI change is small.
- Add a lightweight **named-entity consistency guard** alongside the
  numeric one (e.g. spaCy's small English model) to catch swapped
  people/organizations/dates the same way `numeric_guard.py` catches
  swapped numbers.
- Add per-claim source-sentence citation numbers so a legal/campus reviewer
  can jump straight to the exact clause.
- Batch-process multiple AI answers against the same source doc for a
  side-by-side model comparison (e.g. GPT-4 vs Claude vs Llama faithfulness
  on the same policy questions).
- Fine-tune the cross-encoder on domain-specific contradiction pairs (legal
  clauses, academic policy wording) for higher precision than the
  general-purpose MNLI-trained checkpoint used here.
- Run a small labeled eval set (claims manually marked supported/
  hallucinated) through both NLI model options to measure precision/recall
  before picking a default threshold for your specific domain.
