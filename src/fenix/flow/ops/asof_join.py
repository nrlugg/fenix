from collections.abc import Sequence
from dataclasses import dataclass
import typing

from fenix.flow.ops.hash_join import JoinKey, normalize_keys
from fenix.flow.ops.op import FieldRef, NodeType, Op, normalize_args


@dataclass(frozen=True)
class AsofJoin(Op, type=NodeType.ASOFJOIN):
    left_on: list[FieldRef]
    left_by: list[FieldRef]
    right_on: list[FieldRef]
    right_by: list[FieldRef]
    tolerance: int

    @classmethod
    def normalize(
        cls,
        on: JoinKey[FieldRef] | Sequence[JoinKey[FieldRef]],
        by: JoinKey[FieldRef] | Sequence[JoinKey[FieldRef]],
        tolerance: int,
    ) -> typing.Self:
        on = normalize_args(on)
        by = normalize_args(by)

        left_on, right_on = zip(*map(normalize_keys, on))
        left_by, right_by = zip(*map(normalize_keys, by))

        return cls(list(left_on), list(left_by), list(right_on), list(right_by), tolerance)


join_asof = asof_join = AsofJoin.normalize
