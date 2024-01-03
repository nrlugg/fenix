import functools
import os
import uuid
from dataclasses import dataclass
from os.path import join
from typing import Iterator, Sequence, TypedDict

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
import xxhash
from torch import Tensor

import fenix.ex.acero as ac
import fenix.io as io
import fenix.vq as vq

BATCH_SIZE: int = 2**20
TABLE_DIR: str = "data"
INDEX_DIR: str = "indx"
INDEX_COL: str = "__INDEX__"
SCORE_COL: str = "__SCORE__"

Select = str | Sequence[str] | None
Filter = pc.Expression | None


class IndexConfig(TypedDict):
    k: int
    n: int
    metric: str
    sample_size: int
    num_samples: int


class SavedIndex(TypedDict):
    tensor: Tensor
    column: str
    config: IndexConfig


def index_field() -> pa.Field:
    return pc.field(INDEX_COL)


@dataclass
class Dataset(ds.Dataset):
    uri: str

    def __post_init__(self) -> None:
        self.filesystem.makedirs(self._table_uri(), exist_ok=True)
        self.filesystem.makedirs(self._index_uri(), exist_ok=True)

        for version, encoder in self.list_indexes().items():
            self.register_encoder(version, encoder)
            self.register_distance(encoder["column"], encoder["config"]["metric"])

    def register_encoder(self, vers: int, data: SavedIndex) -> "Dataset":
        name = self._index_encoder_name(vers)
        type = self.schema.field(data["column"]).type

        if name in pc.list_functions():
            return self

        def func(
            ctx: pc.UdfContext, v: pa.FixedShapeTensorArray, k: pa.Int64Scalar
        ) -> pa.ListArray:
            x = vq.arrow_to_torch(v)
            i = vq.encode(x, data["tensor"], k.as_py(), data["config"]["metric"])
            return pa.array(iter(i.numpy()))

        pc.register_scalar_function(
            func,
            name,
            {"summary": "", "description": ""},
            {"v": type, "k": pa.int64()},
            pa.list_(pa.int64()),
        )

        return self

    def register_distance(self, column: str, metric: str) -> "Dataset":
        type = self.schema.field(column).type
        size = type.shape[0]
        name = f"list_{metric}_distance:{type.value_type}:{size}"

        def dist(
            ctx: pc.UdfContext, v: pa.FixedSizeListArray, q: pa.FixedSizeListScalar
        ) -> pa.FloatArray:
            return pa.array(
                vq.distance(
                    vq.arrow_to_torch(q).unsqueeze(0),
                    vq.arrow_to_torch(v),
                    metric=metric,
                )
                .squeeze(0)
                .numpy()
            )

        pc.register_scalar_function(
            dist,
            name,
            {"summary": "", "description": ""},
            {"a": type, "b": type.storage_type},
            type.value_type,
        )

        return self

    @property
    def name(self) -> str:
        return os.path.basename(self.uri)

    @property
    def hash(self) -> str:
        return xxhash.xxh32_hexdigest(
            os.path.abspath(self.uri).encode(),
        )

    def _table_uri(self, name: str | None = None) -> str:
        uri = join(self.uri, TABLE_DIR)

        if name is not None:
            uri = join(uri, name)

        return uri

    def _index_uri(self, vers: int | None = None) -> str:
        uri = join(self.uri, INDEX_DIR)

        if vers is not None:
            uri = join(uri, str(vers))

        return uri

    def _index_encoder_name(self, vers: int) -> str:
        return f"encoder/{self.hash}/{vers}"

    @functools.cached_property
    def filesystem(self) -> fsspec.AbstractFileSystem:
        return fsspec.filesystem("file")

    def to_table(self, index: int | None = None) -> pa.Table:
        t = io.arrow.from_ipc(self._table_uri())

        if index is not None:
            i = io.arrow.from_ipc(self._index_uri(index))
            t = t.append_column(
                INDEX_COL,
                i.rename_columns([INDEX_COL]).column(INDEX_COL),
            )

        return t

    def to_pyarrow(self) -> ds.Dataset:
        return ds.dataset(self.to_table())

    def count_rows(self, filter: Filter = None) -> int:
        return self.to_pyarrow().count_rows(filter=filter)

    @property
    def schema(self) -> pa.Schema:
        return self.to_table().schema

    def filter(self, expression: pc.Expression) -> "Dataset":
        raise NotImplementedError()

    def get_fragments(self, filter: pc.Expression | None = None) -> Iterator[ds.Fragment]:
        raise NotADirectoryError()

    def head(self, num_rows: int, columns: Select = None, filter: Filter = None) -> pa.Table:
        return self.to_pyarrow().head(num_rows, columns=columns, filter=filter)

    def join(self, *args, **kwargs) -> "Dataset":
        raise NotADirectoryError()

    def replace_schema(self, schema: pa.Schema) -> pa.Schema:
        raise NotADirectoryError()

    def scanner(self, columns: Select = None, filter: Filter = None) -> ds.Scanner:
        return self.to_pyarrow().scanner(columns=columns, filter=filter)

    def sort_by(self, sorting: str | list[tuple[str, str]]) -> "Dataset":
        raise NotADirectoryError()

    def take(
        self,
        indices: Sequence[int] | np.ndarray | pa.Array,
        columns: Select = None,
        filter: Filter = None,
    ) -> pa.Array:
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        mask = np.zeros((self.count_rows(),), dtype=np.bool_)
        np.put(mask, indices, True)

        unique = np.unique(indices)
        indices = np.searchsorted(unique, indices)

        t = self.to_table()

        if columns is not None:
            t = t.Select(columns)

        t = t.filter(mask).take(indices)

        if filter is not None:
            t = t.filter(filter)

        return t

    def to_batches(self, columns: Select = None, filter: Filter = None) -> list[pa.RecordBatch]:
        return self.to_pyarrow().to_batches(columns=columns, filter=filter)

    def to_reader(self, columns: Select = None, filter: Filter = None) -> Iterator[pa.RecordBatch]:
        return self.to_pyarrow().to_reader(columns=columns, filter=filter)

    def sample(self, size: int) -> pa.Table:
        return self.to_table().filter(
            np.random.permutation(self.count_rows()) < size,
        )

    def insert_table(self, data: pa.Table | pa.RecordBatchReader) -> "Dataset":
        if isinstance(data, pa.Table):
            data = data.to_reader(BATCH_SIZE)

        path = self._table_uri(str(uuid.uuid4())) + ".arrow"

        io.arrow.to_ipc(path, data, data.schema)

        return self.update_index()

    def create_index(self, column: str, config: IndexConfig) -> "Dataset":
        t = self.to_table()
        v = t.column(column)
        q = vq.kmeans(v, **config)

        vers = max(0, *self.list_indexes(), 0) + 1
        path = self._index_uri(vers)

        self.filesystem.makedirs(path, exist_ok=True)

        with self.filesystem.open(join(path, ".indx.torch"), "wb") as f:
            data: SavedIndex = {"tensor": q, "column": column, "config": config}
            torch.save(data, f)

        return self.register_encoder(vers, data).update_index(vers)

    def update_index(self, vers: int | None = None) -> "Dataset":
        for version, encoder in self.list_indexes().items():
            if vers is not None and version != vers:
                continue

            encoder_name = self._index_encoder_name(version)

            for path in self.list_fragments():
                data = io.arrow.from_ipc(path)
                path = join(self._index_uri(version), os.path.basename(path))

                if self.filesystem.exists(path):
                    continue

                indx = ac.from_sequence(
                    ac.source(data),
                    ac.select(
                        {INDEX_COL: ac.map(encoder_name, [ac.col(encoder["column"]), ac.lit(1)])},
                    ),
                    ac.select(
                        {INDEX_COL: ac.map("list_element", [ac.col(INDEX_COL), ac.lit(0)])},
                    ),
                ).to_reader()

                io.arrow.to_ipc(path, indx, indx.schema)

        return self

    def list_indexes(self) -> dict[int, SavedIndex]:
        return {
            vers: torch.load(join(path, ".indx.torch"))
            for path in sorted(self.filesystem.glob(join(self.uri, INDEX_DIR, "*")))
            for vers in [int(os.path.basename(path))]
        }

    def list_fragments(self) -> Iterator[str]:
        yield from sorted(
            self.filesystem.glob(join(self._table_uri(), "*.arrow")),
        )

    def search_index(
        self,
        query: pa.Array | pa.ChunkedArray | np.ndarray | Tensor,
        index: int,
        select: Sequence[str] | None = None,
        probes: int = 8,
    ) -> pa.RecordBatchReader:
        encoder = self.list_indexes().get(index)
        assert encoder is not None

        metric = encoder["config"]["metric"]
        column = encoder["column"]
        type = self.schema.field(column).type
        size = type.shape[0]
        func = f"list_{metric}_distance:{type.value_type}:{size}"

        if isinstance(query, pa.ChunkedArray):
            query = query.combine_chunks()

        if isinstance(query, pa.Array):
            query = query.to_numpy()

        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query)

        t = self.to_table(index)
        q = pa.scalar(query.numpy(), type=type.storage_type)
        i = pa.array(
            vq.encode(query, encoder["tensor"], probes, metric).squeeze().numpy(),
        )

        if select is None:
            select = [f.name for f in t.schema if not isinstance(f.type, pa.FixedShapeTensorType)]

        return ac.from_sequence(
            ac.source(t),
            ac.filter(ac.col(INDEX_COL).isin(i)),
            ac.select(
                *select,
                **{SCORE_COL: ac.map(func, [ac.col(column), q])},
            ),
            ac.order_by([(SCORE_COL, "ascending")]),
        ).to_reader()
