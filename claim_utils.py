"""
claim_utils.py
--------------
Lightweight sentence/claim splitting utilities.

We deliberately avoid heavy dependencies (spaCy models, nltk downloads) so the
app has zero extra downloads beyond the two HF models used for
embedding/NLI. This is a rule-based sentence splitter tuned for typical
policy/legal/academic prose. It is not perfect (no splitter is), but it
handles common abbreviations, decimals, and enumerations well enough for an
MVP.
"""

import re

# Abbreviations that should NOT trigger a sentence break after their period.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "fig", "eq", "no", "vol", "approx", "u.s", "u.k", "inc", "ltd", "co",
    "st", "ave", "dept", "univ", "govt", "art", "sec", "para",
    "rs", "inr",  # currency abbreviations, e.g. "Rs. 500"
}

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_into_claims(text: str) -> list[str]:
    """
    Split a block of text (typically the AI answer) into individual
    claim/sentence units.

    Rules:
    - Bullet points / numbered list items are each treated as one claim,
      regardless of internal punctuation.
    - Remaining prose is split on sentence boundaries, guarding against
      common abbreviations and decimal numbers (e.g. "3.5%").
    - Empty fragments and pure whitespace are dropped.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    claims: list[str] = []
    for line in lines:
        # Treat obvious list items as atomic claims (don't re-split them
        # further unless they contain multiple full sentences).
        is_bullet = bool(re.match(r"^(\d+[.)]|[-*•])\s+", line))
        stripped_line = re.sub(r"^(\d+[.)]|[-*•])\s+", "", line)

        candidate_sentences = _split_sentences(stripped_line)
        if is_bullet and len(candidate_sentences) <= 1:
            claims.append(stripped_line)
        else:
            claims.extend(candidate_sentences)

    # Final cleanup: drop empties / very short fragments (e.g. stray "-")
    claims = [c.strip() for c in claims if len(c.strip()) > 2]
    return claims


def _split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter with abbreviation + decimal guards."""
    if not text:
        return []

    # Protect decimals like "3.5" or "Rs.5" from being split.
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", text)

    # Protect known abbreviations "Dr." "e.g." etc.
    def _protect_abbrev(match: re.Match) -> str:
        word = match.group(0)
        return word.replace(".", "<ABBR>")

    abbrev_pattern = r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\."
    protected = re.sub(abbrev_pattern, _protect_abbrev, protected, flags=re.IGNORECASE)

    raw_sentences = _SENTENCE_END_RE.split(protected)

    restored = []
    for s in raw_sentences:
        s = s.replace("<DECIMAL>", ".").replace("<ABBR>", ".")
        s = s.strip()
        if s:
            restored.append(s)
    return restored


def split_source_into_chunks(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """
    Split the source document into overlapping-free chunks of N sentences.
    Small multi-sentence chunks give the retriever/NLI model enough context
    per unit without diluting relevance the way whole-paragraph chunks would.
    """
    sentences = split_into_claims(text)  # reuse the same splitter for the source
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks
