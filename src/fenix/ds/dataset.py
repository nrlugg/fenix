import functools
import os
from os.path import join
from typing import TypedDict

import ibis
import msgspec
import numpy as np
import pyarrow as pa
import torch
from ibis import deferred as col
from ibis import selectors as sel
from ibis import udf
from torch import Tensor

import fenix.vq as vq

DISTANCE_COLUMN: str = "__DISTANCE__"
GROUP_ID_COLUMN: str = "__GROUP_ID__"
TABLE_ID_COLUMN: str = "id"


class IndexConfig(TypedDict):
    k: int
    n: int
    f: int
    column: str
    metric: str
    epochs: int
    sample: int


class SavedIndex(TypedDict):
    data: Tensor
    conf: IndexConfig


@udf.scalar.builtin
def list_inner_product(u: list[float], v: list[float]) -> float:  # type:ignore
    ...


@udf.scalar.builtin
def list_distance(u: list[float], v: list[float]) -> float:  # type:ignore
    ...


@udf.scalar.builtin
def list_cosine_similarity(u: list[float], v: list[float]) -> float:  # type:ignore
    ...


class Dataset(msgspec.Struct, frozen=True, dict=True):
    uri: str

    def __post_init__(self) -> None:
        os.makedirs(self.indexes_uri, exist_ok=True)

    @property
    def dataset_uri(self) -> str:
        return join(self.uri, "fenix.db")

    @property
    def indexes_uri(self) -> str:
        return join(self.uri, "index")

    @functools.cached_property
    def client(self) -> ibis.BaseBackend:
        return ibis.duckdb.connect(self.dataset_uri)

    def list_tables(self) -> list[str]:
        return [name for name in self.client.list_tables() if not name.endswith("-index")]

    def create_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name not in self.list_tables()

        if isinstance(data, pa.Table):
            self.client.create_table(name, data)

        elif isinstance(data, pa.RecordBatchReader):
            batch = self.client.create_table(name, data.read_next_batch())

            for batch in data:
                self.client.insert(name, batch)

        else:
            raise ValueError()

    def update_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name in self.list_tables()

        if isinstance(data, pa.Table):
            self.client.insert(name, data)

        elif isinstance(data, pa.RecordBatchReader):
            for batch in data:
                self.client.insert(name, batch)

    def remove_table(self, name: str) -> None:
        if name not in self.list_tables():
            return

        self.client.drop_table(name)

    def create_index(self, name: str, conf: IndexConfig) -> None:
        path = join(self.indexes_uri, f"{name}.bin")

        if os.path.exists(path):
            raise FileExistsError()

        t = self.client.table(name)
        f = min(conf["sample"] / t.count().execute(), 1.0)

        v = t.sample(f)[conf["column"]].to_pyarrow()
        x = torch.from_numpy(
            np.stack(v.to_numpy(zero_copy_only=False)),
        )

        q = vq.build_quantization(
            x,
            k=conf["k"],
            n=conf["n"],
            f=conf["f"],
            epochs=conf["epochs"],
        )

        torch.save(
            {
                "conf": conf,
                "data": q,
            },
            path,
        )

        q = vq.apply_quantization(x, q, 1).squeeze(-1)

        self.create_table(
            f"{name}-index",
            pa.table(
                {
                    "table.id": t.id.to_pyarrow(),
                    GROUP_ID_COLUMN: q.numpy(),
                }
            ),
        )

    def update_index(self, name: str, conf: IndexConfig) -> None:
        self.remove_index(name)
        self.create_index(name, conf)

    def remove_index(self, name: str) -> None:
        path = join(self.indexes_uri, f"{name}.pt")
        if os.path.exists(path):
            os.unlink(path)

        name = f"{name}-index"
        if name in self.client.list_tables():
            self.client.drop_table(f"{name}-index")

    def table(self, name: str) -> pa.RecordBatchReader:
        return self.client.table(name).to_pyarrow_batches()

    def index(self, name: str) -> SavedIndex:
        return torch.load(
            join(self.indexes_uri, f"{name}.bin"),
            map_location="cpu",
        )

    def search_index(
        self,
        table: str,
        query: pa.Table,
        limit: int,
        probes: int,
    ) -> pa.Table:
        index = self.index(table)
        metric = index["conf"]["metric"]
        column = index["conf"]["column"]

        x = torch.from_numpy(
            np.stack(query[column].to_numpy(zero_copy_only=False)),
        )

        i = vq.apply_quantization(x, index["data"], probes, metric)

        q = (
            self.client.read_in_memory(
                query.append_column(
                    GROUP_ID_COLUMN,
                    pa.array(list(i.numpy()), type=pa.list_(pa.int64())),
                )
            )
            .select(~sel.c(GROUP_ID_COLUMN), col[GROUP_ID_COLUMN].unnest())
            .rename(lambda name: f"query.{name}" if name != GROUP_ID_COLUMN else name)
        )

        t = self.client.table(table).rename(
            lambda name: f"table.{name}" if name != GROUP_ID_COLUMN else name
        )

        u = col[f"query.{column}"]
        v = col[f"table.{column}"]
        d = (
            1 - list_cosine_similarity(u, v)
            if metric == "cosine"
            else -list_inner_product(u, v)
            if metric == "dot"
            else list_distance(u, v)
        )

        return (
            q.inner_join(
                self.client.table(f"{table}-index"),
                predicates=GROUP_ID_COLUMN,
            )
            .inner_join(t, predicates="table.id")
            .select(
                sel.startswith("query.") & ~sel.endswith(f".{column}"),
                sel.startswith("table.") & ~sel.endswith(f".{column}"),
                d.name(DISTANCE_COLUMN),
                GROUP_ID_COLUMN,
            )
            .mutate(
                rank=(
                    ibis.row_number()
                    .over(group_by="query.id", order_by=DISTANCE_COLUMN)
                    .name("rank")
                )
            )
            .filter(col["rank"] <= limit)
            .drop("rank")
            .order_by(["query.id", DISTANCE_COLUMN])
            .to_pyarrow()
        )

    def search_table(
        self,
        table: str,
        query: pa.Table,
        limit: int,
        column: str,
        metric: str,
    ) -> pa.Table:
        q = self.client.read_in_memory(query).rename(
            lambda name: f"query.{name}" if name != GROUP_ID_COLUMN else name
        )

        t = self.client.table(table).rename(
            lambda name: f"table.{name}" if name != GROUP_ID_COLUMN else name
        )

        u = col[f"query.{column}"]
        v = col[f"table.{column}"]
        d = (
            1 - list_cosine_similarity(u, v)
            if metric == "cosine"
            else -list_inner_product(u, v)
            if metric == "dot"
            else list_distance(u, v)
        )

        return (
            q.cross_join(t)
            .select(
                sel.startswith("query.") & ~sel.endswith(f".{column}"),
                sel.startswith("table.") & ~sel.endswith(f".{column}"),
                d.name(DISTANCE_COLUMN),
            )
            .mutate(
                rank=(
                    ibis.row_number()
                    .over(group_by="query.id", order_by=DISTANCE_COLUMN)
                    .name("rank")
                )
            )
            .filter(col["rank"] < limit)
            .drop("rank")
            .order_by(["query.id", DISTANCE_COLUMN])
            .to_pyarrow()
        )
