"""
numeric_guard.py
-----------------
Cross-encoder NLI models are trained mostly on natural-language entailment
(paraphrase, implication) and are notoriously weak at catching *numeric*
substitution errors: "the penalty is 10%" vs "the penalty is 15%" reads as
almost the same sentence to a semantic model, so NLI often scores it as
entailment even though it's factually wrong.

This module is a cheap, deterministic guard that runs *after* NLI: it
extracts numbers/percentages/dates/currency amounts from the claim and from
the evidence pool, and flags a claim as inconsistent if it states a number
that never appears anywhere in the retrieved evidence.

This is intentionally conservative (it only fires when the claim has a
number the evidence plainly lacks) so it catches clear-cut hallucinated
figures without generating excessive false positives on claims that simply
paraphrase numbers differently (e.g. "half" vs "50%" -- these are NOT
caught by this guard and still rely on NLI).
"""

import re

_NUMBER_RE = re.compile(
    r"""
    (?:₹|rs\.?|\$|inr|usd)?\s*          # optional currency prefix
    \d[\d,]*(?:\.\d+)?                  # the number itself, with optional commas/decimals
    \s*%?                               # optional percent sign
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_number(raw: str) -> str:
    """Strip currency symbols, commas, and whitespace so '10 %' == '10%' == 'Rs. 10'."""
    s = raw.lower()
    s = re.sub(r"[₹$,]|rs\.?|inr|usd", "", s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def extract_numbers(text: str) -> set[str]:
    matches = _NUMBER_RE.findall(text)
    return {_normalize_number(m) for m in matches if _normalize_number(m)}


def check_numeric_consistency(claim: str, evidence_pool: list[str]) -> tuple[bool, set[str]]:
    """
    Returns (is_consistent, mismatched_numbers).

    is_consistent = False only when the claim contains at least one number
    that does not appear (in normalized form) in ANY of the evidence
    candidates -- i.e. a number the source simply never mentions.
    """
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return True, set()  # nothing numeric to check

    evidence_numbers: set[str] = set()
    for chunk in evidence_pool:
        evidence_numbers |= extract_numbers(chunk)

    mismatched = claim_numbers - evidence_numbers
    return (len(mismatched) == 0), mismatched
