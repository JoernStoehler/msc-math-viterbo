from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence as TypingSequence

class Features(dict[str, Any]): ...

class Dataset:
    column_names: TypingSequence[str]
    num_rows: int
    features: Features

    @staticmethod
    def from_list(
        data: TypingSequence[Mapping[str, Any]],
        *,
        features: Features | None = ...,
        info: Any | None = ...,
        split: Any | None = ...,
    ) -> Dataset: ...
    @staticmethod
    def from_dict(
        data: Mapping[str, TypingSequence[Any]],
        *,
        features: Features | None = ...,
        info: Any | None = ...,
        split: Any | None = ...,
    ) -> Dataset: ...
    def save_to_disk(self, path: str, **kwargs: Any) -> None: ...
    @staticmethod
    def load_from_disk(path: str, **kwargs: Any) -> Dataset: ...
    def map(
        self,
        function: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        features: Features | None = ...,
        **kwargs: Any,
    ) -> Dataset: ...
    def to_pandas(self) -> Any: ...

def Value(type_name: str) -> Any: ...
def Sequence(feature: Any) -> Any: ...
def concatenate_datasets(datasets: TypingSequence[Dataset]) -> Dataset: ...
