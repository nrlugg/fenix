from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import operator
from typing import Self

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as ds

import fenix.flow.ops as ops
from fenix.types import DatasetLike, FileFormat, FileSystem, TabularLike


@dataclass(frozen=True)
class Flow:
    exec_plan: ac.Declaration

    @classmethod
    def from_op(cls, op: ops.Op, *inputs: ac.Declaration) -> Self:
        return cls(
            exec_plan=operator.call(op, *inputs),
        )

    @classmethod
    def from_tabluar(cls, source: TabularLike, schema: pa.Schema | None = None) -> Self:
        return cls.from_op(
            ops.tabular_source(source, schema),
        )

    @classmethod
    def from_dataset(
        cls,
        source: DatasetLike,
        schema: pa.Schema | None = None,
        format: FileFormat = "parquet",
        filesystem: str | FileSystem | None = None,
        partitioning: str | ds.Partitioning | None = None,
    ) -> Self:
        return cls.from_op(
            ops.dataset_source(source, schema, format, filesystem, partitioning),
        )

    def select(
        self,
        projections: Mapping[str, ops.SelectExpr] | None = None,
        **more_projections: ops.SelectExpr,
    ) -> Self:
        op = ops.select(projections, **more_projections)

        return self.from_op(op, self.exec_plan)

    def filter(
        self,
        *more_expressions: ops.FilterExpr | Sequence[ops.FilterExpr],
        filter_expressions: ops.FilterExpr | Sequence[ops.FilterExpr] | None = None,
    ) -> Self:
        op = ops.filter(*more_expressions, filter_expressions=filter_expressions)

        return self.from_op(op, self.exec_plan)

    def group_by(
        self,
        aggregates: Mapping[str, ops.FuncDef] | None = None,
        group_keys: ops.FieldRef | Sequence[ops.FieldRef] | None = None,
        **more_aggregates: ops.FuncDef,
    ) -> Self:
        op = ops.group_by(aggregates, group_keys, **more_aggregates)

        return self.from_op(op, self.exec_plan)

    def order_by(
        self,
        *more_keys: ops.SortKey | Sequence[ops.SortKey],
        keys: ops.SortKey | Sequence[ops.SortKey] | None = None,
        sort_direction: ops.SortDirection = ops.SortDirection.ASC,
        null_placement: ops.NullPlacement = ops.NullPlacement.END,
    ) -> Self:
        op = ops.order_by(
            *more_keys, keys=keys, sort_direction=sort_direction, null_placement=null_placement
        )

        return self.from_op(op, self.exec_plan)

    def join(
        source: Self,
        target: Self,
        keys: ops.JoinKey[ops.FieldRef] | Sequence[ops.JoinKey[ops.FieldRef]] | None = None,
        type: ops.JoinType = ops.JoinType.INNER,
        output_fields: ops.JoinKey[ops.FieldRef | Sequence[ops.FieldRef]] | None = None,
        output_suffix: ops.JoinKey[str] | None = None,
        **more_keys: ops.FieldRef,
    ) -> Self:
        op = ops.hash_join(keys, type, output_fields, output_suffix, **more_keys)

        return source.from_op(op, source.exec_plan, target.exec_plan)

    def join_asof(
        source: Self,
        target: Self,
        on: ops.JoinKey[ops.FieldRef] | Sequence[ops.JoinKey[ops.FieldRef]],
        by: ops.JoinKey[ops.FieldRef] | Sequence[ops.JoinKey[ops.FieldRef]],
        tolerance: int,
    ) -> Self:
        op = ops.join_asof(on, by, tolerance)

        return source.from_op(op, source.exec_plan, target.exec_plan)

    def to_table(self) -> pa.Table:
        return self.exec_plan.to_table()

    def to_reader(self, *, batch_size: int | None = None) -> pa.RecordBatchReader:
        reader = self.exec_plan.to_reader()

        if batch_size:
            reader = ds.Scanner.from_batches(reader, batch_size=batch_size).to_reader()

        return reader

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        yield from self.to_reader()
