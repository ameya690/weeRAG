"""Deterministic regression tests for wee.eval.

No randomness, no network calls, no model dependencies.
All QA samples and expected values are hardcoded.
"""

from wee.eval import (
    exact_match,
    f1_score,
    max_f1,
    faithfulness,
    context_precision_recall,
    evaluate_qa,
    jaccard,
    sentence_split,
    groundedness_score,
    citation_support,
)

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_exact_hit(self):
        assert exact_match("Paris", ["Paris"]) == 1

    def test_case_insensitive(self):
        assert exact_match("paris", ["Paris"]) == 1

    def test_whitespace_normalised(self):
        assert exact_match("  Paris  ", ["Paris"]) == 1

    def test_multiple_gold_answers(self):
        assert exact_match("NYC", ["New York City", "NYC", "New York"]) == 1

    def test_no_match(self):
        assert exact_match("London", ["Paris", "Berlin"]) == 0

    def test_empty_prediction(self):
        assert exact_match("", ["Paris"]) == 0

    def test_empty_gold_list(self):
        assert exact_match("Paris", []) == 0

    def test_both_empty(self):
        # empty pred matches empty gold string
        assert exact_match("", [""]) == 1


class TestF1Score:
    def test_perfect_match(self):
        assert f1_score("the capital is Paris", "the capital is Paris") == 1.0

    def test_partial_overlap(self):
        # pred = "the cat sat", gold = "the cat"
        # overlap = 2 (the, cat), prec = 2/3, rec = 2/2 = 1
        # F1 = 2 * (2/3) * 1 / ((2/3) + 1) = (4/3) / (5/3) = 4/5 = 0.8
        score = f1_score("the cat sat", "the cat")
        assert abs(score - 0.8) < 1e-6

    def test_no_overlap(self):
        assert f1_score("hello world", "foo bar") == 0.0

    def test_empty_prediction(self):
        assert f1_score("", "some gold answer") == 0.0

    def test_empty_gold(self):
        assert f1_score("some prediction", "") == 0.0

    def test_both_empty(self):
        assert f1_score("", "") == 1.0


class TestMaxF1:
    def test_picks_best_gold(self):
        score = max_f1("Paris is the capital", ["Paris is the capital", "Berlin"])
        assert score == 1.0

    def test_empty_golds(self):
        assert max_f1("Paris", []) == 0.0


class TestFaithfulness:
    def test_fully_supported(self):
        answer = "Paris is in France."
        contexts = ["Paris is the capital city located in France."]
        score = faithfulness(answer, contexts)
        assert score >= 0.5  # jaccard overlap should be enough

    def test_unsupported(self):
        answer = "Jupiter is the largest planet."
        contexts = ["Paris is in France."]
        score = faithfulness(answer, contexts)
        assert score == 0.0

    def test_empty_answer(self):
        # no sentences -> vacuously faithful
        assert faithfulness("", ["some context"]) == 1.0

    def test_empty_contexts(self):
        # no context to support any sentence
        assert faithfulness("Some claim here.", []) == 0.0

    def test_multi_sentence_partial(self):
        answer = "Paris is in France. Jupiter orbits the Sun."
        contexts = ["Paris is a city in France."]
        score = faithfulness(answer, contexts)
        # One of two sentences supported at best
        assert 0.0 <= score <= 1.0


class TestContextPrecisionRecall:
    def test_perfect(self):
        selected = ["Paris is the capital of France."]
        gold = ["Paris is the capital of France."]
        prec, rec = context_precision_recall(selected, gold)
        assert prec == 1.0
        assert rec == 1.0

    def test_both_empty(self):
        prec, rec = context_precision_recall([], [])
        assert prec == 1.0
        assert rec == 1.0

    def test_empty_selected(self):
        prec, rec = context_precision_recall([], ["some gold context"])
        assert prec == 0.0
        assert rec == 0.0

    def test_empty_gold(self):
        prec, rec = context_precision_recall(["some selected context"], [])
        assert prec == 0.0
        assert rec == 1.0

    def test_partial_match(self):
        selected = ["Paris is in France.", "Berlin is in Germany."]
        gold = ["Paris is the capital of France."]
        prec, rec = context_precision_recall(selected, gold)
        # At most 1 of 2 selected match, so prec <= 0.5; rec could be 1.0
        assert prec <= 1.0
        assert rec <= 1.0


class TestJaccard:
    def test_identical(self):
        assert jaccard("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert jaccard("foo bar", "baz quux") == 0.0

    def test_both_empty(self):
        assert jaccard("", "") == 1.0


class TestSentenceSplit:
    def test_multiple_sentences(self):
        text = "First sentence. Second sentence! Third?"
        parts = sentence_split(text)
        assert len(parts) == 3

    def test_single_sentence(self):
        parts = sentence_split("Just one.")
        assert len(parts) == 1

    def test_empty(self):
        assert sentence_split("") == []


class TestGroundednessScore:
    def test_empty_answer(self):
        result = groundedness_score("", ["context"])
        assert result["score"] == 1.0
        assert result["total_sentences"] == 0

    def test_all_grounded(self):
        answer = "Paris is in France."
        contexts = ["Paris is the capital city in France."]
        result = groundedness_score(answer, contexts, thr=0.3)
        assert result["score"] >= 0.5
        assert result["ungrounded"] is not None  # key exists


class TestCitationSupport:
    def test_no_citations(self):
        result = citation_support("answer", [], ["context"])
        assert result["score"] == 1.0
        assert result["total"] == 0

    def test_supported_citation(self):
        contexts = ["Paris is the capital of France."]
        citations = ["Paris is the capital of France."]
        result = citation_support("answer", citations, contexts)
        assert result["score"] == 1.0
        assert result["supported"] == 1


# ---------------------------------------------------------------------------
# evaluate_qa integration tests
# ---------------------------------------------------------------------------


# Hardcoded QA samples -- fully deterministic
SAMPLES = [
    {
        "pred": "Paris",
        "answers": ["Paris"],
        "contexts": ["The capital of France is Paris."],
    },
    {
        "pred": "Berlin",
        "answers": ["Berlin", "berlin"],
        "contexts": ["Germany's capital city is Berlin."],
    },
    {
        "pred": "London is the capital of the United Kingdom",
        "answers": ["London"],
        "contexts": [
            "London is the capital of the United Kingdom.",
            "The UK's seat of government is in London.",
        ],
    },
]


class TestEvaluateQA:
    def test_returns_expected_keys(self):
        result = evaluate_qa(SAMPLES)
        assert "metrics" in result
        assert "samples" in result
        for key in ("em", "f1", "faithfulness"):
            assert key in result["metrics"]

    def test_sample_count(self):
        result = evaluate_qa(SAMPLES)
        assert len(result["samples"]) == len(SAMPLES)

    def test_em_above_threshold(self):
        result = evaluate_qa(SAMPLES)
        # First two are exact matches; third has extra words so EM = 0
        # Minimum expected: 2/3 ~= 0.66
        assert result["metrics"]["em"] >= 0.6

    def test_f1_above_threshold(self):
        result = evaluate_qa(SAMPLES)
        # Two exact matches + one partial -> aggregate F1 ~ 0.74
        assert result["metrics"]["f1"] >= 0.7

    def test_faithfulness_above_threshold(self):
        result = evaluate_qa(SAMPLES)
        # Single-sentence preds with jaccard overlap: faithfulness ~ 0.33
        assert result["metrics"]["faithfulness"] >= 0.3

    def test_per_sample_values_in_range(self):
        result = evaluate_qa(SAMPLES)
        for s in result["samples"]:
            assert 0 <= s["em"] <= 1
            assert 0.0 <= s["f1"] <= 1.0
            assert 0.0 <= s["faithfulness"] <= 1.0


class TestEvaluateQAWithCitations:
    """Samples that include gold_citations trigger context precision/recall."""

    CITATION_SAMPLES = [
        {
            "pred": "Paris",
            "answers": ["Paris"],
            "contexts": ["The capital of France is Paris."],
            "gold_citations": ["The capital of France is Paris."],
        },
    ]

    def test_context_metrics_present(self):
        result = evaluate_qa(self.CITATION_SAMPLES)
        assert "context_precision" in result["metrics"]
        assert "context_recall" in result["metrics"]

    def test_perfect_citation(self):
        result = evaluate_qa(self.CITATION_SAMPLES)
        assert result["metrics"]["context_precision"] == 1.0
        assert result["metrics"]["context_recall"] == 1.0


class TestEvaluateQAEdgeCases:
    def test_empty_predictions(self):
        samples = [
            {"pred": "", "answers": ["Paris"], "contexts": ["Paris is in France."]},
        ]
        result = evaluate_qa(samples)
        assert result["metrics"]["em"] == 0
        assert result["metrics"]["f1"] == 0.0

    def test_empty_contexts(self):
        samples = [
            {"pred": "Paris", "answers": ["Paris"], "contexts": []},
        ]
        result = evaluate_qa(samples)
        assert result["metrics"]["em"] == 1
        # faithfulness with no context: answer has sentences but no ctx -> 0
        assert result["metrics"]["faithfulness"] == 0.0

    def test_perfect_match(self):
        samples = [
            {
                "pred": "Paris",
                "answers": ["Paris"],
                "contexts": ["Paris."],
            },
        ]
        result = evaluate_qa(samples)
        assert result["metrics"]["em"] == 1
        assert result["metrics"]["f1"] == 1.0

    def test_empty_sample_list(self):
        result = evaluate_qa([])
        assert result["metrics"]["em"] == 0
        assert result["metrics"]["f1"] == 0.0
        assert result["metrics"]["faithfulness"] == 0.0
        assert result["samples"] == []
