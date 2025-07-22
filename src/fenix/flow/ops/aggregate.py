from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
from types import NoneType
from typing import Any, NotRequired, Self, TypedDict

import pyarrow.compute as pc

from fenix.flow.ops.op import FieldRef, NodeType, Op, normalize_args, normalize_maps

type FuncOps = Mapping[str, Any] | pc.FunctionOptions | None
type AggrDef = tuple[list[FieldRef], str, pc.FunctionOptions | None, str]


class FuncDef(TypedDict):
    function_name: str
    function_args: FieldRef | Sequence[FieldRef]
    function_options: NotRequired[FuncOps]


def normalize_func_ops(name: str, args: FuncOps) -> pc.FunctionOptions | None:
    if isinstance(args, (NoneType, pc.FunctionOptions)):
        return args

    cls = pc.get_function(name)
    cls = getattr(pc, cls._doc.options_class)

    return cls(*args)


def normalize_aggr_def(name: str, args: FuncDef) -> AggrDef:
    return (
        normalize_args(args["function_args"]),
        function_name := args["function_name"],
        normalize_func_ops(function_name, args["function_options"]),
        name,
    )


@dataclass(frozen=True)
class Aggregate(Op, type=NodeType.AGGREGATE):
    aggregates: list[AggrDef]
    keys: Sequence[FieldRef] | None = None

    @classmethod
    def normalize(
        cls,
        aggregates: Mapping[str, FuncDef] | None = None,
        group_keys: FieldRef | Sequence[FieldRef] | None = None,
        **more_aggregates: FuncDef,
    ) -> Self:
        aggregates = normalize_maps(aggregates, more_aggregates)
        group_keys = normalize_args(group_keys)

        return cls(
            aggregates=list(
                itertools.starmap(normalize_aggr_def, Mapping.items(aggregates)),
            ),
            keys=group_keys,
        )


aggr = aggregate = group_by = Aggregate.normalize
