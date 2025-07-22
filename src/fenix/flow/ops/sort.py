from collections.abc import Sequence
from dataclasses import dataclass
import enum
import functools
import re
from typing import Self

from fenix.flow.ops.op import NodeType, Op, normalize_args


class NullPlacement(enum.StrEnum):
    AT_START = enum.auto()
    AT_END = enum.auto()

    # Aliases
    START = AT_START
    FIRST = AT_START
    END = AT_END
    LAST = AT_END


class SortDirection(enum.StrEnum):
    ASCENDING = enum.auto()
    DESCENDING = enum.auto()

    # Aliases
    ASC = ASCENDING
    DESC = DESCENDING


type SortKey = str | tuple[str, SortDirection]


REGEX_PATTER: str = r"""
^(
    (?P<key>\w+)
    (
        \s+
        (?P<dir>(ASC|DESC))
    )?
)$
"""


def normalize_keys(
    key: SortKey, *, sort_direction: SortDirection | None = None, check: bool = False
) -> SortKey:
    if isinstance(key, str):
        assert (match := re.match(REGEX_PATTER, key, flags=re.VERBOSE)) is not None

        key, dir = match.group("key", "dir")

        if (dir := dir or sort_direction) is None:
            return key

    if isinstance(key, tuple):
        key, dir = key

    if check:
        raise ValueError()

    key = normalize_keys(key, check=True)
    dir = getattr(SortDirection, str.upper(dir))

    assert isinstance(key, str)
    return key, dir


asc = ascending = functools.partial(normalize_keys, sort_direction=SortDirection.ASC)
desc = descending = functools.partial(normalize_keys, sort_direction=SortDirection.DESC)


@dataclass(frozen=True)
class Sort(Op, type=NodeType.ORDER_BY):
    keys: list[SortKey]
    null_placement: NullPlacement = NullPlacement.LAST

    @classmethod
    def normalize(
        cls,
        *more_keys: SortKey | Sequence[SortKey],
        keys: SortKey | Sequence[SortKey] | None = None,
        sort_direction: SortDirection = SortDirection.ASC,
        null_placement: NullPlacement = NullPlacement.END,
    ) -> Self:
        norm = functools.partial(normalize_keys, sort_direction=sort_direction)
        keys = normalize_args(*more_keys, keys)

        return cls(
            keys=list(map(norm, keys)),
            null_placement=null_placement,
        )


sort = order_by = Sort.normalize
