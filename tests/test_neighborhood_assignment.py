from collections import Counter

from marketmap.neighborhoods.assign import choose_neighborhood_from_tags


def test_neighborhood_assignment_is_deterministic() -> None:
    tags = ["sports", "nba", "playoffs"]
    df = Counter({"sports": 300, "nba": 120, "playoffs": 80})

    first = choose_neighborhood_from_tags(
        tags=tags,
        title="Will the Celtics win?",
        category="Sports",
        doc_freq=df,
        total_docs=7000,
    )
    second = choose_neighborhood_from_tags(
        tags=tags,
        title="Will the Celtics win?",
        category="Sports",
        doc_freq=df,
        total_docs=7000,
    )

    assert first == second
    assert first[0].startswith("sports::")
