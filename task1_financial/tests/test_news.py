"""Offline tests for headline ranking. No network access.

Run: python -m task1_financial.tests.test_news
"""

from task1_financial.src import news


def test_aliases_ignore_corporate_suffixes():
    a = news.company_aliases("NVDA", "NVIDIA Corporation")
    assert "nvda" in a and "nvidia" in a
    assert "corporation" not in a, "generic suffixes would match unrelated companies"

    b = news.company_aliases("AAPL", "Apple Inc.")
    assert b == {"aapl", "apple"}
    return 4


def test_relevance_filters_off_topic():
    aliases = news.company_aliases("NVDA", "NVIDIA Corporation")
    on = "Nvidia Just Put $3.5 Billion Behind Its Next AI Expansion"
    off = "Tech stocks today: Apple stock slips as Tim Cook prepares to step down"
    assert news._relevance(on, aliases) > 0
    assert news._relevance(off, aliases) == 0
    # Possessives survive normalisation.
    assert news._relevance("Nvidia's Q2 earnings beat", aliases) > 0
    return 3


def test_ranking_is_binary_not_by_mention_count():
    aliases = news.company_aliases("NVDA", "NVIDIA Corporation")
    padded = "NVIDIA Corporation NVDA stock position trimmed (NASDAQ:NVDA)"
    editorial = "Nvidia just put $3.5 billion behind its next AI expansion"
    # An aggregator headline mentions the name more often, but must not outrank
    # an editorial headline, otherwise three sources collapse into one.
    assert news._relevance(padded, aliases) > news._relevance(editorial, aliases)
    key = lambda t: news._relevance(t, aliases) == 0
    assert key(padded) == key(editorial), "sort key must not separate these two"
    return 2


def test_dedupe_is_case_and_punctuation_insensitive():
    batches = [
        [{"headline": "Nvidia beats on earnings"}],
        [{"headline": "NVIDIA  beats, on earnings!"}, {"headline": "Different story"}],
    ]
    out = news._dedupe(batches)
    assert len(out) == 2, f"expected 2 unique, got {len(out)}"
    assert out[0]["headline"] == "Nvidia beats on earnings"
    return 2


def main():
    checks = [
        test_aliases_ignore_corporate_suffixes,
        test_relevance_filters_off_topic,
        test_ranking_is_binary_not_by_mention_count,
        test_dedupe_is_case_and_punctuation_insensitive,
    ]
    total = 0
    for check in checks:
        n = check()
        total += n
        print(f"  PASS  {check.__name__:46s} ({n} assertions)")
    print(f"\n{len(checks)} tests passed, {total} assertions.")


if __name__ == "__main__":
    main()
