from collections.abc import Sequence
from dataclasses import dataclass
import enum
import functools
import operator
from typing import Self

from fenix.flow.ops.op import FieldRef, NodeType, Op, normalize_args

type JoinKey[X] = X | tuple[X, X]


class JoinType(str, enum.Enum):
    INNER = "inner"

    LEFT_SEMI = "left semi"
    LEFT_ANTI = "left anti"
    LEFT_OUTER = "left outer"

    RIGHT_SEMI = "right semi"
    RIGHT_ANTI = "right anti"
    RIGHT_OUTER = "right outer"

    FULL_OUTER = "full outer"

    # Aliases
    LEFT = LEFT_OUTER
    RIGHT = RIGHT_OUTER
    OUTER = FULL_OUTER


def normalize_keys[X](keys: JoinKey[X]) -> tuple[X, X]:
    if isinstance(keys, tuple):
        return keys

    return keys, keys


@dataclass(frozen=True)
class HashJoin(Op, type=NodeType.HASHJOIN):
    join_type: JoinType
    left_keys: list[FieldRef]
    right_keys: list[FieldRef]
    left_output: list[FieldRef] | None = None
    right_output: list[FieldRef] | None = None
    output_suffix_for_left: str = ""
    output_suffix_for_right: str = ""

    @classmethod
    def normalize(
        cls,
        keys: JoinKey[FieldRef] | Sequence[JoinKey[FieldRef]] | None = None,
        type: JoinType = JoinType.INNER,
        outs: JoinKey[FieldRef | Sequence[FieldRef]] | None = None,
        suffix: JoinKey[str] | None = None,
        **more_keys: FieldRef,
    ) -> Self:
        keys = operator.concat(normalize_args(keys), list(dict.items(more_keys)))

        left_keys, right_keys = map(list, zip(*map(normalize_keys, keys)))

        left_output, right_output = normalize_keys(outs)
        left_output = normalize_args(left_output) or None
        right_output = normalize_args(right_output) or None

        left_suffix, right_suffix = normalize_keys(suffix or "")

        return cls(
            type, left_keys, right_keys, left_output, right_output, left_suffix, right_suffix
        )


join = hash_join = HashJoin.normalize

left_join = functools.partial(join, type=JoinType.LEFT)
right_join = functools.partial(join, type=JoinType.RIGHT)
inner_join = functools.partial(join, type=JoinType.INNER)
