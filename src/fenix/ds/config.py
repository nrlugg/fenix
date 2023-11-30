from typing import Literal, TypedDict

Metric = Literal["cosine", "dot", "l2"]


class Select(TypedDict):
    query: list[str]
    index: list[str]


class Config(TypedDict, total=False):
    limit: int
    filter: str | None
    select: Select | None
    column: str
    metric: Metric


def default_config() -> Config:
    return {
        "limit": 10,
        "filter": None,
        "select": None,
        "column": "vector",
        "metric": "l2",
    }
