from typing import Dict, Any

import pytest

from avr.wandbapi.run import RunPredicateBuilder, RunPredicate


class MyRun:
    def __init__(self, cfg: Dict[str, Any]):
        self._attrs = {"config": cfg}


@pytest.fixture
def pgm_and_na2_predicate() -> RunPredicate:
    return (
        RunPredicateBuilder.true_()
        .and_(RunPredicate.has_config_attr("avr.dataset", "pgm"))
        .and_(RunPredicate.has_config_attr("num_answers", 2))
        .build()
    )


@pytest.fixture
def pgm_or_raven_predicate() -> RunPredicate:
    return (
        RunPredicateBuilder.true_()
        .and_(RunPredicate.has_config_attr("avr.dataset", "pgm"))
        .or_(RunPredicate.has_config_attr("avr.dataset", "raven"))
        .build()
    )


def test_pgm_and_na2_predicate_matches(pgm_and_na2_predicate: RunPredicate):
    run = MyRun({"avr": {"dataset": "pgm"}, "num_answers": 2})
    assert pgm_and_na2_predicate(run) == True


@pytest.mark.parametrize(
    "run",
    [
        # Different n_a
        MyRun({"avr": {"dataset": "pgm"}, "num_answers": 4}),
        # Different dataset
        MyRun({"avr": {"dataset": "raven"}, "num_answers": 2}),
        # No n_a
        MyRun({"avr": {"dataset": "pgm"}}),
        # No dataset
        MyRun({"num_answers": 2}),
    ],
)
def test_pgm_and_na2_predicate_doesnt_match(
    pgm_and_na2_predicate: RunPredicate, run: MyRun
):
    assert pgm_and_na2_predicate(run) == False


@pytest.mark.parametrize(
    "run",
    [
        MyRun({"avr": {"dataset": "pgm"}, "num_answers": 2}),
        MyRun({"avr": {"dataset": "raven"}, "num_answers": 2}),
    ],
)
def test_pgm_or_raven_predicate_matches(
    pgm_or_raven_predicate: RunPredicate, run: MyRun
):
    assert pgm_or_raven_predicate(run) == True


@pytest.mark.parametrize(
    "run",
    [
        MyRun({"avr": {"dataset": "other"}, "num_answers": 2}),
        MyRun({"num_answers": 2}),
    ],
)
def test_pgm_or_raven_predicate_doesnt_match(
    pgm_or_raven_predicate: RunPredicate, run: MyRun
):
    assert pgm_or_raven_predicate(run) == False
