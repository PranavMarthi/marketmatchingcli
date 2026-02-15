"""Simple CLI to score confidence between two market questions."""

import argparse
import math
import re

from marketmap.services.embeddings import compute_embeddings

STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "by",
    "and",
    "or",
    "is",
    "be",
    "with",
    "than",
    "from",
}


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in STOPWORDS and not t.isdigit()
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_entities(text: str) -> set[str]:
    entities: set[str] = set()

    person = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    for p in person:
        entities.add(f"person:{p.strip().lower()}")

    org_terms = [
        "Fed",
        "Federal Reserve",
        "SEC",
        "FBI",
        "NATO",
        "EU",
        "UN",
        "WHO",
        "OpenAI",
        "Anthropic",
        "Tesla",
        "SpaceX",
        "Apple",
        "Google",
        "Microsoft",
    ]
    for org in org_terms:
        if re.search(rf"\b{re.escape(org)}\b", text, re.IGNORECASE):
            entities.add(f"org:{org.lower()}")

    geo_terms = [
        "United States",
        "US",
        "USA",
        "China",
        "Russia",
        "Ukraine",
        "Israel",
        "Iran",
        "Taiwan",
        "India",
        "UK",
        "Japan",
    ]
    for g in geo_terms:
        if re.search(rf"\b{re.escape(g)}\b", text, re.IGNORECASE):
            entities.add(f"gpe:{g.lower()}")

    return entities


def _logical_score(q1: str, q2: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    t1 = _tokens(q1)
    t2 = _tokens(q2)
    token_j = _jaccard(t1, t2)

    neg1 = bool(re.search(r"\b(not|no|without|fail|won't|will not)\b", q1.lower()))
    neg2 = bool(re.search(r"\b(not|no|without|fail|won't|will not)\b", q2.lower()))
    if token_j >= 0.65 and neg1 != neg2:
        score += 0.6
        reasons.append("complementary_wording")

    pattern = r"(less than|under|below|more than|over|at least|at most)\s+(\d[\d,]*(?:\.\d+)?)"
    m1 = re.search(pattern, q1.lower())
    m2 = re.search(pattern, q2.lower())
    if m1 and m2 and m1.group(1) != m2.group(1):
        s1 = re.sub(pattern, "", q1.lower()).strip()
        s2 = re.sub(pattern, "", q2.lower()).strip()
        if _jaccard(_tokens(s1), _tokens(s2)) >= 0.6:
            score += 0.35
            reasons.append("threshold_chain")

    win_re = r"^will\s+(.+?)\s+win\b"
    w1 = re.search(win_re, q1.lower())
    w2 = re.search(win_re, q2.lower())
    if w1 and w2 and w1.group(1) != w2.group(1):
        score += 0.25
        reasons.append("mutex_candidates")

    return min(1.0, score), reasons


def score_pair(question_1: str, question_2: str) -> dict[str, object]:
    embeddings = compute_embeddings([question_1, question_2])
    semantic = float(max(0.0, min(1.0, embeddings[0].dot(embeddings[1]))))

    e1 = _extract_entities(question_1)
    e2 = _extract_entities(question_2)
    entity_overlap = _jaccard(e1, e2)

    logical, logical_reasons = _logical_score(question_1, question_2)

    token_overlap = _jaccard(_tokens(question_1), _tokens(question_2))
    template_penalty = 0.0
    penalty_reasons: list[str] = []
    if semantic >= 0.9 and entity_overlap < 0.1:
        if token_overlap >= 0.9:
            template_penalty += 0.2
            penalty_reasons.append("near_duplicate_template")
        elif token_overlap >= 0.8:
            template_penalty += 0.1
            penalty_reasons.append("high_template_similarity")

    confidence = 0.7 * semantic + 0.2 * logical + 0.1 * entity_overlap - template_penalty
    confidence = float(max(0.0, min(1.0, confidence)))

    return {
        "confidence": round(confidence, 4),
        "semantic_score": round(semantic, 4),
        "logical_score": round(logical, 4),
        "entity_overlap_score": round(entity_overlap, 4),
        "template_penalty": round(template_penalty, 4),
        "logical_reasons": logical_reasons,
        "template_penalty_reasons": penalty_reasons,
        "mode": "pairwise_question_estimate",
        "note": "Uses semantic+logical+entity signals only (no price-series stat/propagation).",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score confidence between two market questions.")
    parser.add_argument("--q1", type=str, help="First market question")
    parser.add_argument("--q2", type=str, help="Second market question")
    args = parser.parse_args()

    q1 = args.q1 or input("Question 1: ").strip()
    q2 = args.q2 or input("Question 2: ").strip()

    if not q1 or not q2:
        raise SystemExit("Both questions are required.")

    result = score_pair(q1, q2)
    print("\nConfidence score")
    print(f"  confidence:          {result['confidence']}")
    print(f"  semantic_score:      {result['semantic_score']}")
    print(f"  logical_score:       {result['logical_score']}")
    print(f"  entity_overlap:      {result['entity_overlap_score']}")
    print(f"  template_penalty:    {result['template_penalty']}")
    if result["logical_reasons"]:
        print(f"  logical_reasons:     {', '.join(result['logical_reasons'])}")
    if result["template_penalty_reasons"]:
        print(f"  penalty_reasons:     {', '.join(result['template_penalty_reasons'])}")
    print(f"  note:                {result['note']}")


if __name__ == "__main__":
    main()
