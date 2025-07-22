import abc
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass
import enum
import functools
import operator
from types import EllipsisType
from typing import Any, ClassVar, Self, Type, TypeVar, overload

import pyarrow.acero as ac
import pyarrow.compute as pc

T = TypeVar("T")

type FieldRef = str | pc.Expression
type FilterExpr = FieldRef
type SelectExpr = object | FieldRef | EllipsisType


@overload
def normalize_sequence(s: str | Sequence[str]) -> list[str]: ...


@overload
def normalize_sequence(s: T | Sequence[T]) -> list[T]: ...


def normalize_sequence(s: str | T | Sequence[T]) -> list[str] | list[T]:
    if isinstance(s, str):
        return [s]

    if not isinstance(s, Sequence):
        return [s]

    return list(s)


def normalize_args(*args: T | Sequence[T] | None) -> list[T]:
    return functools.reduce(
        operator.add,
        (normalize_sequence(s) for s in args if s is not None),
    )


def normalize_maps(*maps: Mapping[str, T] | None) -> Mapping[str, T]:
    return functools.reduce(
        operator.or_,
        (m for m in maps if m is not None),
    )


class NodeType(enum.StrEnum):
    TABLE_SOURCE = enum.auto()
    SCAN = enum.auto()
    FILTER = enum.auto()
    PROJECT = enum.auto()
    AGGREGATE = enum.auto()
    ORDER_BY = enum.auto()
    HASHJOIN = enum.auto()
    ASOFJOIN = enum.auto()

    @enum.property
    def node_options(self) -> Type[ac.ExecNodeOptions]:
        import pyarrow.acero

        node_options = str.join("", map(str.capitalize, str.split(self, "_")))
        node_options = str.format("{}NodeOptions", node_options)

        return getattr(pyarrow.acero, node_options)


@dataclass(frozen=True)
class Op(abc.ABC):
    type: ClassVar[NodeType]

    def __init_subclass__(cls, type: NodeType):
        cls.type = NodeType(type)

    @property
    def options(self) -> ac.ExecNodeOptions:
        return self.type.node_options(*astuple(self))

    def __call__(self, *inputs: ac.Declaration) -> ac.Declaration:
        return ac.Declaration(self.type, self.options, list(inputs))

    @classmethod
    @abc.abstractmethod
    def normalize(cls, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError()
