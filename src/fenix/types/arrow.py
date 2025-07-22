from collections.abc import Iterator
import os
from typing import Literal, Protocol, runtime_checkable

import fsspec
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs


@runtime_checkable
class ArrowStream(Protocol):
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


@runtime_checkable
class DataFrame(Protocol):
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> object: ...


type DatasetLike = str | os.PathLike | ds.Dataset
type TabularLike = Iterator[pa.RecordBatch] | ds.Scanner | ArrowStream | DataFrame
type PyArrowLike = DatasetLike | TabularLike


type FileFormat = Literal["arrow", "csv", "ipc", "parquet"]
type FileSystem = str | pa.fs.FileSystem | fsspec.AbstractFileSystem
