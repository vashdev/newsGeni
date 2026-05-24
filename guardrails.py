"""
guardrails.py — Input and output guardrails  

Input guardrails (run BEFORE the LLM):
  - Prompt injection detection: jailbreak / instruction override attempts
  - Domain relevance gate: reject queries clearly outside fraud/legal scope

Output guardrails (run AFTER the LLM):
  - Faithfulness check: every factual claim must be grounded in retrieved context
  - Confidence gate: if retrieval score is too low, refuse to answer
  - Human review flag: high-stakes outputs get flagged for analyst sign-off
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Prompt injection detection ────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions?', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+', re.IGNORECASE),
    re.compile(r'disregard\s+(your\s+)?(?:instructions?|rules?|guidelines?)', re.IGNORECASE),
    re.compile(r'act\s+as\s+(?:if\s+you\s+(?:are|were)|a|an)\s+', re.IGNORECASE),
    re.compile(r'(?:system|admin)\s*(?:prompt|override|mode)', re.IGNORECASE),
    re.compile(r'jailbreak', re.IGNORECASE),
    re.compile(r'DAN\s+mode', re.IGNORECASE),
    re.compile(r'forget\s+(everything|all)\s+(you|above)', re.IGNORECASE),
]

def detect_injection(query: str) -> bool:
    return any(p.search(query) for p in _INJECTION_PATTERNS)


# ── Domain relevance gate ─────────────────────────────────────────────────────
_NEWS_DOMAIN_TERMS = re.compile(
    r'\b(technology|politics|news|current events)\b',
    re.IGNORECASE,
    )
 


def is_in_domain(query: str) -> bool:
    """Returns True if the query looks fraud/legal domain relevant."""
    return bool(_NEWS_DOMAIN_TERMS.search(query))


# ── Faithfulness check ────────────────────────────────────────────────────────

def _extract_key_phrases(text: str, min_len: int = 4) -> set[str]:
    """Naive key phrase extraction — 3+ word n-grams that appear in the text."""
    words  = re.findall(r'\b[A-Za-z]{%d,}\b' % min_len, text.lower())
    ngrams = set()
    for i in range(len(words) - 1):
        ngrams.add(f"{words[i]} {words[i+1]}")
    print(f" ngrams  -- {ngrams}   ")
    return ngrams

def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    Rough faithfulness: fraction of answer key-phrases that appear in context.
    Score of 1.0 = fully grounded. Score < 0.4 = likely hallucination.

    Production note: in the real system this was a dedicated NLI model
    (cross-encoder/nli-deberta-v3-small) but the n-gram heuristic is a good
    fast fallback.
    """
    if not answer.strip() or not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    answer_phrases   = _extract_key_phrases(answer)

    if not answer_phrases:
        return 1.0  # Nothing to check

    grounded = sum(1 for phrase in answer_phrases if phrase in combined_context)
    return grounded / len(answer_phrases)


# ── Dataclasses ───────────────────────────────────────────────────────────────

class GuardrailStatus(str, Enum):
    PASS    = "pass"
    WARN    = "warn"
    BLOCK   = "block"


@dataclass
class InputGuardrailResult:
    status:           GuardrailStatus
    sanitized_query:  str
    injection_detected: bool     = False
    out_of_domain:    bool       = False
    reason:           str        = ""


@dataclass
class OutputGuardrailResult:
    status:               GuardrailStatus
    answer:               str
    faithfulness_score:   float
    requires_human_review: bool = False
    reason:               str   = ""


# ── Main guardrail functions ──────────────────────────────────────────────────

def run_input_guardrails(query: str, enforce_domain: bool = True) -> InputGuardrailResult:
    """
    Run all input guardrails. Returns a result with status PASS/WARN/BLOCK.

    BLOCK conditions:  injection detected
    WARN conditions:  out-of-domain query
    """

    if detect_injection(query):
        logger.warning("Prompt injection detected in query: %s", query[:80])
        return InputGuardrailResult(
            status=GuardrailStatus.BLOCK,
            sanitized_query=query,
            injection_detected=True,
            reason="Query contains prompt injection patterns and was blocked.",
        )

    out_of_domain = enforce_domain and not is_in_domain(query)
    status        = GuardrailStatus.PASS

 

    if out_of_domain:
        logger.info("Out-of-domain query: %s", query[:80])
        status = GuardrailStatus.WARN

    reason_parts = []
    if out_of_domain:  reason_parts.append("Query may be outside  news domain")

    return InputGuardrailResult(
        status=GuardrailStatus.PASS if status == GuardrailStatus.PASS else status,
        sanitized_query=query,
        out_of_domain=out_of_domain,
        reason="; ".join(reason_parts),
    )


def run_output_guardrails(
    answer:           str,
    context_chunks:   list[str],
    retrieval_scores: list[float],
    faithfulness_threshold: float = 0.35,
    confidence_threshold:   float = 0.20,
) -> OutputGuardrailResult:
    """
    Run all output guardrails.

    BLOCK:  faithfulness < threshold (likely hallucination)
    WARN:   low retrieval confidence → flag for human review
    """
    faith  = faithfulness_score(answer, context_chunks)
    avg_rs = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

    if faith < faithfulness_threshold:
        logger.warning("Low faithfulness score %.2f — possible hallucination", faith)
        return OutputGuardrailResult(
            status=OutputGuardrailResult,
            answer="Insufficient evidence in indexed documents to answer this question reliably.",
            faithfulness_score=faith,
            requires_human_review=True,
            reason=f"Faithfulness score {faith:.2f} below threshold {faithfulness_threshold}.",
        )

    requires_review = avg_rs < confidence_threshold
    if requires_review:
        logger.info("Low retrieval confidence %.2f — flagging for review", avg_rs)

    return OutputGuardrailResult(
        status=GuardrailStatus.WARN if requires_review else GuardrailStatus.PASS,
        answer=answer,
        faithfulness_score=faith,
        requires_human_review=requires_review,
        reason=f"Low retrieval confidence ({avg_rs:.2f}). Recommend analyst review." if requires_review else "",
    )
