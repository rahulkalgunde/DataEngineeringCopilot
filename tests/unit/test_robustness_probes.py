from data_engineering_copilot.evaluation.robustness_probes import build_probes


def _qa_rows():
    return [
        {
            "id": "qa-a",
            "question": "What is A?",
            "contexts": ["A is alpha."],
            "ground_truth": "alpha",
            "source_name": "SrcOne",
        },
        {
            "id": "qa-b",
            "question": "What is B?",
            "contexts": ["B is beta."],
            "ground_truth": "beta",
            "source_name": "SrcTwo",
        },
        {
            "id": "qa-c",
            "question": "What is C?",
            "contexts": ["C is gamma."],
            "ground_truth": "gamma",
            "source_name": "SrcThree",
        },
    ]


def test_deterministic_output():
    p1 = build_probes(_qa_rows())
    p2 = build_probes(_qa_rows())
    assert p1 == p2


def test_noise_probe_appends_two_cross_source_distractors():
    probes = [p for p in build_probes(_qa_rows()) if p["probe"] == "noise"]
    row = next(p for p in probes if p["id"] == "qa-a-noise")
    assert len(row["contexts"]) == 3
    assert row["expect_refusal"] is False


def test_rejection_probe_replaces_contexts():
    probes = [p for p in build_probes(_qa_rows()) if p["probe"] == "rejection"]
    row = next(p for p in probes if p["id"] == "qa-b-rejection")
    assert len(row["contexts"]) == 2
    assert all("beta" not in c for c in row["contexts"])
    assert row["expect_refusal"] is True


def test_ids_are_stable_slugs():
    for p in build_probes(_qa_rows()):
        assert "-" in p["id"] and "_" not in p["id"]
