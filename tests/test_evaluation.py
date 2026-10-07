"""Fails CI if answer accuracy, grounding or sandbox behaviour regresses."""
from evaluation.run_eval import run


def test_evaluation_suite_is_perfect():
    results = run()
    failures = [d for items in results.values() for ok, d in items if not ok]
    assert not failures, "\n".join(failures)
    assert sum(len(v) for v in results.values()) >= 80
