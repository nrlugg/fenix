from collections.abc import Mapping
from dataclasses import dataclass
import itertools
from types import EllipsisType
import typing

import pyarrow.compute as pc

from fenix.flow.ops.op import NodeType, Op, SelectExpr, normalize_maps


def normalize_expression(name: str, expr: SelectExpr) -> tuple[str, pc.Expression]:
    if isinstance(expr, pc.Expression):
        return name, expr

    if isinstance(expr, EllipsisType):
        return name, pc.field(name)

    return name, pc.scalar(expr)


@dataclass(frozen=True)
class Select(Op, type=NodeType.PROJECT):
    expressions: list[pc.Expression]
    names: list[str]

    @classmethod
    def normalize(
        cls, projections: Mapping[str, SelectExpr] | None = None, **more_projections: SelectExpr
    ) -> typing.Self:
        projections = normalize_maps(projections, more_projections)

        names, expressions = zip(
            *itertools.starmap(normalize_expression, Mapping.items(more_projections)),
        )

        return cls(list(expressions), list(names))


project = select = Select.normalize
