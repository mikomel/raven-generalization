from typing import Dict, Optional, Callable, Any

from wandb.apis.public import Run


def safe_dict_get(d: Dict, key: str) -> Optional:
    value = d
    for node in key.split("."):
        if value and hasattr(value, "get"):
            value = value.get(node)
    return value


class RunPredicate:
    def __init__(self, fn: Callable[[Run], bool]):
        self.fn = fn

    @staticmethod
    def always_true() -> "RunPredicate":
        fn = lambda run: True
        return RunPredicate(fn)

    @staticmethod
    def has_config_attr(key: str, value: Any) -> "RunPredicate":
        def fn(run: Run) -> bool:
            return value == safe_dict_get(run._attrs["config"], key)

        return RunPredicate(fn)

    def __call__(self, run: Run) -> bool:
        return self.fn(run)


class RunPredicateBuilder:
    def __init__(self, predicate: RunPredicate):
        self._predicate = predicate

    @staticmethod
    def true_() -> "RunPredicateBuilder":
        predicate = RunPredicate.always_true()
        return RunPredicateBuilder(predicate)

    def and_(self, other: RunPredicate) -> "RunPredicateBuilder":
        current = self._predicate
        fn = lambda run: current(run) and other(run)
        self._predicate = RunPredicate(fn)
        return self

    def or_(self, other: RunPredicate) -> "RunPredicateBuilder":
        current = self._predicate
        fn = lambda run: current(run) or other(run)
        self._predicate = RunPredicate(fn)
        return self

    def build(self) -> RunPredicate:
        return self._predicate
