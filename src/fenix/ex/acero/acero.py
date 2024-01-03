from typing import Any, Literal

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc


def source(table: pa.Table) -> ac.Declaration:
    return ac.Declaration("table_source", ac.TableSourceNodeOptions(table))


def filter(expression: pc.Expression) -> ac.Declaration:
    return ac.Declaration("filter", ac.FilterNodeOptions(expression))


def select(*columns: str | pc.Expression, **named_columns: str | pc.Expression) -> ac.Declaration:
    named_columns = {c: c for c in columns} | named_columns
    named_columns = {k: pc.field(v) if isinstance(v, str) else v for k, v in named_columns.items()}
    return ac.Declaration(
        "project",
        ac.ProjectNodeOptions(list(named_columns.values()), names=list(named_columns.keys())),
    )


def order_by(columns: list[tuple[str, Literal["ascending", "descending"]]]) -> ac.Declaration:
    return ac.Declaration("order_by", ac.OrderByNodeOptions(columns))


def map(func_name: str, args: list[pc.Expression]) -> pc.Expression:
    return pc.Expression._call(func_name, args)


def col(column_name: str) -> pc.Expression:
    return pc.field(column_name)


def lit(value: Any) -> pc.Expression:
    return pc.scalar(value)


def from_sequence(*nodes: ac.Declaration) -> ac.Declaration:
    return ac.Declaration.from_sequence(list(nodes))
