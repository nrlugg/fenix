from collections.abc import Iterator
from dataclasses import dataclass
import os
from typing import Literal, Self

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pyarrow.interchange as ic

from fenix.flow.ops.op import NodeType, Op
from fenix.types import ArrowStream, DataFrame, DatasetLike, FileFormat, FileSystem, TabularLike


@dataclass(frozen=True)
class TabularSource(Op, type=NodeType.TABLE_SOURCE):
    source: pa.Table

    @classmethod
    def normalize(cls, source: TabularLike, schema: pa.Schema | None = None) -> Self:
        if isinstance(source, ds.Scanner):
            source = source.to_table()

        if isinstance(source, DataFrame):
            source = ic.from_dataframe(source)

        if isinstance(source, Iterator) and not isinstance(source, ArrowStream):
            source = pa.Table.from_batches(source, schema=schema)

        if isinstance(source, ArrowStream) and not isinstance(source, pa.Table):
            source = pa.table(source, schema=schema)

        if not isinstance(source, pa.Table):
            raise TypeError(f"{type(source)=} cannot be converted to an arrow_c_stream")

        return cls(source)


@dataclass(frozen=True)
class DatasetSource(Op, type=NodeType.SCAN):
    source: ds.Dataset

    @classmethod
    def normalize(
        cls,
        source: DatasetLike,
        schema: pa.Schema | None = None,
        format: FileFormat = "parquet",
        filesystem: FileSystem | str | None = None,
        partitioning: ds.Partitioning | Literal["hive"] | None = "hive",
    ) -> ds.Dataset:
        if isinstance(source, ds.Dataset):
            return cls(source)

        if isinstance(filesystem, str):
            filesystem, path = fs.FileSystem.from_ur(filesystem)
            filesystem = fs.SubTreeFileSystem(path, filesystem)

        if isinstance(partitioning, str):
            partitioning = ds.partitioning(flavor=partitioning)

        if isinstance(source, (str, os.PathLike)):
            source = ds.dataset(source, schema, format, filesystem, partitioning)

        return cls(source)


from_tabular = tabular_source = TabularSource.normalize
from_dataset = dataset_source = DatasetSource.normalize
