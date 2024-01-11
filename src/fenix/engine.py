import functools
import os
import uuid
from os.path import join
from typing import Self, Sequence, TypedDict

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import xxhash
from torch import Tensor

import fenix.io as io
import fenix.vq as vq

BATCH_SIZE: int = 2**20
SOURCE_ROOT: str = "source"
CODING_ROOT: str = "coding"
CODING_DATA: str = ".coding.torch"
CODING_NAME: str = "__GROUP_ID__"
METRIC_NAME: str = "__DISTANCE__"


class CodingConfig(TypedDict):
    codebook_size: int
    num_codebooks: int
    num_epochs: int
    batch_size: int


class CodingState(TypedDict):
    tensor: Tensor
    config: CodingConfig


class Source(msgspec.Struct, frozen=True, dict=True):
    source: str

    @functools.cached_property
    def source_path(self) -> str:
        return join(self.source, SOURCE_ROOT)

    def to_pyarrow(self) -> pa.Table:
        return io.arrow.from_ipc(self.source_path)

    @functools.cached_property
    def schema(self) -> pa.Schema:
        return self.to_pyarrow().schema

    def insert(self, data: pa.Table | pa.RecordBatchReader) -> Self:
        fs = fsspec.filesystem("file")
        fs.makedirs(self.source_path, exist_ok=True)

        if isinstance(data, pa.Table):
            data = data.to_reader(BATCH_SIZE)

        name = str(uuid.uuid4()) + ".arrow"
        path = join(self.source_path, name)

        io.arrow.to_ipc(path, data, data.schema)

        return self

    def list_fragments(self) -> list[str]:
        fs = fsspec.filesystem("file")
        return sorted(
            fs.glob(join(self.source_path, "*.arrow")),
        )


class Engine(Source, frozen=True, dict=True):
    column: str
    metric: str
    coding: str | None = None

    @property
    def id(self) -> str:
        id = join(os.path.abspath(self.source), self.column, self.metric)

        if self.coding is not None:
            id = join(id, self.coding)

        return xxhash.xxh32_hexdigest(id)

    @functools.cached_property
    def coding_root(self) -> str:
        return join(self.source, CODING_ROOT, self.column, self.metric)

    @functools.cached_property
    def coding_path(self) -> str:
        assert self.coding is not None
        return join(self.coding_root, self.coding)

    @functools.cached_property
    def coding_data(self) -> CodingState:
        with fsspec.open(join(self.coding_path, CODING_DATA)) as f:
            return torch.load(f, map_location="cpu")

    @functools.cached_property
    def coding_func(self) -> str:
        return f"encode:{self.id}"

    @functools.cached_property
    def column_type(self) -> pa.DataType:
        return super().to_pyarrow().schema.field(self.column).type

    @functools.cached_property
    def metric_func(self) -> str:
        type = self.column_type
        return f"list_{self.metric}_distance:{type.value_type}:{type.list_size}"

    def __post_init__(self) -> None:
        try:
            self.register_metric()
        except FileNotFoundError:
            pass

        try:
            self.register_coding()
            self.update()
        except AssertionError:
            pass

    def to_pyarrow(
        self, select: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> pa.Table:
        t = super().to_pyarrow()

        if self.coding is not None:
            i = io.arrow.from_ipc(self.coding_path)
            t = t.append_column(CODING_NAME, i.column(CODING_NAME))

        if filter is not None:
            t = t.filter(filter)

        if select is not None:
            t = t.select(select)

        return t

    def search(
        self,
        query: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
        limit: int,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        probes: int | None = None,
    ) -> pa.Table:
        table = self.to_pyarrow()

        if isinstance(query, pa.ChunkedArray):
            query = query.combine_chunks()

        if isinstance(query, pa.Array):
            query = query.to_numpy()

        if isinstance(query, Tensor):
            query = query.numpy()

        if isinstance(query, np.ndarray):
            query = pa.scalar(query, type=self.column_type)

        if self.coding is not None and probes is not None:
            codes = pc.field(CODING_NAME).isin(
                pc.call_function(self.coding_func, [pa.array([query]), probes]).values,
            )

            if filter is None:
                filter = codes
            else:
                filter = filter & codes

        if select is None:
            select = [f.name for f in self.schema if not isinstance(f.type, pa.FixedSizeListType)]

        else:
            select = list(select)

        select += [METRIC_NAME]

        table = table.filter(filter) if filter is not None else table
        table = table.append_column(
            METRIC_NAME, pc.call_function(self.metric_func, [table[self.column], query])
        )

        return table.select(select).take(
            pc.select_k_unstable(table, limit, [(METRIC_NAME, "ascending")])
        )

    def encode(self, name: str, config: CodingConfig) -> "Engine":
        fs = fsspec.filesystem("file")

        path = join(self.coding_root, name)

        if fs.exists(path):
            raise ValueError(f"Coding '{self.column}/{self.metric}/{name}' already exists.")

        fs.makedirs(path, exist_ok=True)

        v = self.to_pyarrow().column(self.column)
        q = vq.kmeans(v, self.metric, **config)

        with fs.open(join(path, CODING_DATA), "wb") as f:
            torch.save({"tensor": q, "config": config}, f)

        return Engine(self.source, self.column, self.metric, name)

    def update(self) -> Self:
        fs = fsspec.filesystem("file")

        for path in self.list_fragments():
            data = io.arrow.from_ipc(path)
            path = join(self.coding_path, os.path.basename(path))

            if fs.exists(path):
                continue

            code = pc.call_function(self.coding_func, [data[self.column], pa.scalar(1)])
            code = pc.list_element(code, 0)
            code = pa.table({CODING_NAME: code})

            io.arrow.to_ipc(path, code, code.schema)

        return self

    def register_coding(self) -> None:
        if self.coding_func in pc.list_functions():
            return

        q = self.coding_data["tensor"]

        def topk(ctx: pc.UdfContext, v: pa.FixedSizeListType, k: pa.Int64Scalar) -> pa.ListType:
            x = io.arrow.to_torch(v)
            i = vq.encode(x, q, k.as_py(), self.metric).numpy()
            return pa.array(iter(i), type=pa.list_(pa.int64()))

        pc.register_scalar_function(
            topk,
            self.coding_func,
            {"summary": "", "description": ""},
            {"v": self.column_type, "k": pa.int64()},
            pa.list_(pa.int64()),
        )

    def register_metric(self) -> None:
        if self.metric_func in pc.list_functions():
            return None

        def dist(
            ctx: pc.UdfContext, v: pa.FixedSizeListType, q: pa.FixedSizeListType
        ) -> pa.FloatArray:
            return pa.array(
                vq.distance(
                    io.arrow.to_torch(q).unsqueeze(0),
                    io.arrow.to_torch(v),
                    metric=self.metric,
                )
                .squeeze(0)
                .numpy()
            )

        pc.register_scalar_function(
            dist,
            self.metric_func,
            {"summary": "", "description": ""},
            {"a": self.column_type, "b": self.column_type},
            self.column_type.value_type,
        )
