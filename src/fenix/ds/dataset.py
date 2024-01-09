import os
import uuid
from dataclasses import dataclass
from os.path import join
from typing import Iterator, NotRequired, Sequence, TypedDict

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import xxhash
from torch import Tensor

import fenix.ds.acero as ac
import fenix.io as io
import fenix.vq as vq

BATCH_SIZE: int = 2**20
TABLE_PATH: str = "data"
INDEX_PATH: str = "indx"
INDEX_FILE: str = ".torch"
INDEX_NAME: str = "__INDEX__"
SCORE_NAME: str = "__SCORE__"


class Index(TypedDict):
    column: str
    metric: str
    version: NotRequired[str]


class IndexConfig(TypedDict):
    k: int
    n: int
    sample_size: int
    num_samples: int


class IndexState(TypedDict):
    tensor: Tensor
    config: IndexConfig


class IndexSpec(TypedDict):
    index: Index
    state: IndexState


def index_name(index: Index) -> str:
    return join(index["column"], index["metric"], index["version"])


def index_path(uri: str, index: Index) -> str:
    return join(uri, INDEX_PATH, index_name(index))


def index_data(uri: str, index: Index) -> IndexState:
    with fsspec.open(join(index_path(uri, index), INDEX_FILE)) as f:
        return torch.load(f, map_location="cpu")


def index_topk(hash: str, index: Index) -> str:
    return join("topk", hash, index_name(index))


def index_top1(hash: str, index: Index) -> str:
    return join("top1", hash, index_name(index))


def index_dist(type: pa.DataType, index: Index) -> str:
    return join(f"list_{index['metric']}_distance", str(type.value_type), str(type.list_size))


@dataclass(frozen=True)
class Dataset(pa.dataset.Dataset):
    uri: str

    def __post_init__(self) -> None:
        fs = fsspec.filesystem("file")
        fs.makedirs(join(self.uri, TABLE_PATH), exist_ok=True)
        fs.makedirs(join(self.uri, INDEX_PATH), exist_ok=True)

        for spec in self.list_indexes():
            register_distance(self, spec["index"])
            register_encoder(self, spec["index"])
            self.update_index(spec["index"])

    @property
    def name(self) -> str:
        return os.path.basename(self.uri)

    @property
    def hash(self) -> str:
        return xxhash.xxh32_hexdigest(
            os.path.abspath(self.uri).encode(),
        )

    def to_table(self, index: Index | None = None) -> pa.Table:
        t = io.arrow.from_ipc(
            join(self.uri, TABLE_PATH),
        )

        if index is not None and "version" in index:
            i = io.arrow.from_ipc(index_path(self.uri, index))
            t = t.append_column(INDEX_NAME, i.column(INDEX_NAME))

        return t

    @property
    def schema(self) -> pa.Schema:
        return self.to_table().schema

    def scanner(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> pa.dataset.Scanner:
        return pa.dataset.Scanner.from_batches(
            self.to_table().to_reader(BATCH_SIZE),
            columns=columns,
            filter=filter,
        )

    def count_rows(self, filter: pc.Expression | None = None) -> int:
        return self.scanner().count_rows(filter=filter)

    def filter(self, expression: pc.Expression) -> "Dataset":
        raise NotImplementedError()

    def get_fragments(self, filter: pc.Expression | None = None) -> Iterator[pa.dataset.Fragment]:
        raise NotADirectoryError()

    def head(
        self,
        num_rows: int,
        columns: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> pa.Table:
        return self.scanner().head(num_rows, columns=columns, filter=filter)

    def join(self, *args, **kwargs) -> "Dataset":
        raise NotADirectoryError()

    def replace_schema(self, schema: pa.Schema) -> pa.Schema:
        raise NotADirectoryError()

    def sort_by(self, sorting: str | list[tuple[str, str]]) -> "Dataset":
        raise NotADirectoryError()

    def take(
        self,
        indices: Sequence[int] | np.ndarray | pa.Array,
        columns: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> pa.Array:
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        mask = np.zeros((self.count_rows(),), dtype=np.bool_)
        np.put(mask, indices, True)

        unique = np.unique(indices)
        indices = np.searchsorted(unique, indices)

        t = self.to_table()

        if columns is not None:
            t = t.select(columns)

        t = t.filter(mask).take(indices)

        if filter is not None:
            t = t.filter(filter)

        return t

    def to_batches(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> list[pa.RecordBatch]:
        return self.scanner().to_batches(columns=columns, filter=filter)

    def to_reader(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> pa.RecordBatchReader:
        return self.scanner().to_reader(columns=columns, filter=filter)

    def sample(self, size: int) -> pa.Table:
        return self.to_table().filter(
            np.random.permutation(self.count_rows()) < size,
        )

    def insert_table(self, data: pa.Table | pa.RecordBatchReader) -> "Dataset":
        if isinstance(data, pa.Table):
            data = data.to_reader(BATCH_SIZE)

        path = join(self.uri, TABLE_PATH, str(uuid.uuid4()) + ".arrow")

        io.arrow.to_ipc(path, data, data.schema)

        return Dataset(self.uri)

    @property
    def fragments(self) -> list[str]:
        fs = fsspec.filesystem("file")
        return sorted(
            fs.glob(
                join(self.uri, TABLE_PATH, "*.arrow"),
            ),
        )

    def search_index(
        self,
        query: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
        index: Index,
        limit: int,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        probes: int | None = None,
    ) -> pa.Table:
        type = self.schema.field(index["column"]).type
        dist = index_dist(type, index)

        table = self.to_table(index)

        if isinstance(query, pa.ChunkedArray):
            query = query.combine_chunks()

        if isinstance(query, pa.Array):
            query = query.to_numpy()

        if isinstance(query, Tensor):
            query = query.numpy()

        if isinstance(query, np.ndarray):
            query = pa.scalar(query, type=type)

        if "version" in index and probes is not None:
            func = index_topk(self.hash, index)
            mask = pc.field(INDEX_NAME).isin(
                pc.call_function(func, [pa.array([query]), probes]).values,
            )

            if filter is None:
                filter = mask
            else:
                filter = filter & mask

        if select is None:
            select = [f.name for f in self.schema if not isinstance(f.type, pa.FixedSizeListType)]

        reader = ac.from_sequence(
            ac.source(table),
            ac.filter(filter),
            ac.select(
                *select,
                **{SCORE_NAME: ac.map(dist, [ac.col(index["column"]), query])},
            ),
            ac.order_by([(SCORE_NAME, "ascending")]),
        ).to_reader()

        return pa.dataset.Scanner.from_batches(reader).take(np.arange(limit))

    def create_index(self, index: Index, config: IndexConfig) -> "Dataset":
        fs = fsspec.filesystem("file")

        path = join(self.uri, INDEX_PATH, index["column"], index["metric"])

        try:
            vers = 1 + max(
                [
                    int(vers)
                    for path in fs.ls(path)
                    for vers in [os.path.basename(path)]
                    if vers.isnumeric()
                ],
            )
        except FileNotFoundError:
            vers = 1

        path = join(path, str(vers))

        fs.makedirs(path, exist_ok=True)

        v = self.to_table().column(index["column"])
        q = vq.kmeans(v, metric=index["metric"], **config)

        with fs.open(join(path, INDEX_FILE), "wb") as f:
            torch.save({"tensor": q, "config": config}, f)

        return Dataset(self.uri)

    def update_index(self, index: Index) -> "Dataset":
        fs = fsspec.filesystem("file")

        func = index_top1(self.hash, index)

        for path in self.fragments:
            data = io.arrow.from_ipc(path)
            path = join(index_path(self.uri, index), os.path.basename(path))

            if fs.exists(path):
                continue

            indx = ac.from_sequence(
                ac.source(data),
                ac.select(
                    **{INDEX_NAME: ac.map(func, [ac.col(index["column"])])},
                ),
            ).to_table()

            io.arrow.to_ipc(path, indx, indx.schema)

        return self

    def list_indexes(self) -> Iterator[IndexSpec]:
        fs = fsspec.filesystem("file")
        for column_path in fs.ls(join(self.uri, INDEX_PATH)):
            column = os.path.basename(column_path)

            for metric_path in fs.ls(column_path):
                metric = os.path.basename(metric_path)

                for version_path in fs.ls(metric_path):
                    version = os.path.basename(version_path)

                    index: Index = {"column": column, "metric": metric, "version": version}
                    state: IndexState = index_data(self.uri, index)

                    yield {"index": index, "state": state}


def register_encoder(dataset: Dataset, index: Index) -> None:
    type = dataset.schema.field(index["column"]).type
    name = index_topk(dataset.hash, index)

    if name in pc.list_functions():
        return

    data = index_data(dataset.uri, index)

    def topk(ctx: pc.UdfContext, v: pa.FixedSizeListType, k: pa.Int64Scalar) -> pa.ListType:
        x = io.arrow.to_torch(v)

        i = vq.encode(x, data["tensor"], k.as_py(), index["metric"]).numpy()
        i = pa.ListArray.from_arrays(
            pa.array(np.arange(0, i.size, k.as_py())),
            pa.array(i.flatten()),
        )

        return i

    def top1(ctx: pc.UdfContext, v: pa.FixedSizeListArray) -> pa.Int64Array:
        return pc.list_element(topk(ctx, v, pa.scalar(1)), 0, memory_pool=ctx.memory_pool)

    pc.register_scalar_function(
        topk,
        name,
        {"summary": "", "description": ""},
        {"v": type, "k": pa.int64()},
        pa.list_(pa.int64()),
    )

    pc.register_scalar_function(
        top1,
        name.replace("topk", "top1"),
        {"summary": "", "description": ""},
        {"v": type},
        pa.int64(),
    )


def register_distance(dataset: Dataset, index: Index) -> None:
    type = dataset.schema.field(index["column"]).type
    name = index_dist(type, index)

    if name in pc.list_functions():
        return None

    def dist(
        ctx: pc.UdfContext, v: pa.FixedSizeListType, q: pa.FixedSizeListType
    ) -> pa.FloatArray:
        return pa.array(
            vq.distance(
                io.arrow.to_torch(q).unsqueeze(0),
                io.arrow.to_torch(v),
                metric=index["metric"],
            )
            .squeeze(0)
            .numpy()
        )

    pc.register_scalar_function(
        dist,
        name,
        {"summary": "", "description": ""},
        {"a": type, "b": type},
        type.value_type,
    )
