import os
from typing import TypedDict, TypeVar

import duckdb
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

import fenix.vq as vq

TABLE_NAME: str = "dataset"
INDEX_NAME: str = "hashing"
QUERY_NAME: str = "query"
VECTOR_COLUMN: str = "vector"
DISTANCE_COLUMN: str = "__DISTANCE__"
GROUP_ID_COLUMN: str = "__GROUP_ID__"


Array = TypeVar("Array", pa.Array, np.ndarray, Tensor)


class HashingConfig(TypedDict, total=False):
    k: int
    n: int
    f: int
    metric: str
    epochs: int


def default_hashing_config() -> HashingConfig:
    return dict(
        k=256,
        n=2,
        f=256,
        metric="cosine",
        epochs=50,
    )


class Hashing(msgspec.Struct, frozen=True, dict=True):
    q: Tensor
    k: int
    n: int
    f: int
    metric: str
    epochs: int

    @staticmethod
    def from_array(x: pa.Array | np.ndarray | Tensor, config: HashingConfig) -> "Hashing":
        config = default_hashing_config() | config

        if isinstance(x, pa.Array):
            x = np.stack(x.to_numpy(zero_copy_only=False))

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        return Hashing(vq.build_quantization(x, **config), **config)

    @staticmethod
    def from_table(t: pa.Table, config: HashingConfig) -> "Hashing":
        return Hashing.from_array(t[VECTOR_COLUMN].combine_chunks(), config=config)

    @staticmethod
    def from_connection(c: duckdb.DuckDBPyConnection) -> "Hashing":
        t = c.table(INDEX_NAME).fetch_arrow_table()

        q = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([t["n"].to_numpy(), t["k"].to_numpy()])),
            torch.from_numpy(np.stack(t["q"].to_numpy())),
        ).to_dense()

        k = pc.max(t["k"]).as_py() + 1
        n = pc.max(t["n"]).as_py() + 1
        [f] = t["f"].unique().to_pylist()
        [metric] = t["metric"].unique().to_pylist()
        [epochs] = t["epochs"].unique().to_pylist()

        return Hashing(q, k, n, f, metric, epochs)

    def apply(self, x: Array, k: int) -> Array:
        if isinstance(x, pa.Array):
            return pa.array(
                list(
                    vq.apply_quantization(
                        torch.from_numpy(np.stack(x.to_numpy(zero_copy_only=False))),
                        self.q,
                        k,
                        self.metric,
                    ).numpy()
                )
            )

        if isinstance(x, np.ndarray):
            return vq.apply_quantization(
                torch.from_numpy(x),
                self.q,
                k,
                self.metric,
            ).numpy()

        return vq.apply_quantization(torch.from_numpy(x), self.q, k, self.metric)

    def to_table(self) -> pa.Table:
        q = self.q
        return pa.Table.from_pylist(
            [
                {
                    "q": q.numpy(),
                    "n": n,
                    "k": k,
                    "f": self.f,
                    "metric": self.metric,
                    "epochs": self.epochs,
                }
                for n, q in enumerate(q.unbind(0))
                for k, q in enumerate(q.unbind(0))
            ],
        )


class Dataset(msgspec.Struct, frozen=True, dict=True):
    conn: duckdb.DuckDBPyConnection

    @staticmethod
    def from_uri(uri: str) -> "Dataset":
        os.makedirs(os.path.dirname(uri), exist_ok=True)
        conn = duckdb.connect(uri)
        return Dataset(conn)

    def insert(self, data: pa.Table | pa.RecordBatchReader) -> "Dataset":
        if isinstance(data, pa.Table):
            return self.insert(data.to_reader())

        assert isinstance(data, pa.RecordBatchReader)

        for batch in data:
            batch = pa.Table.from_batches([batch])

            try:
                count = self.conn.table(TABLE_NAME).count("*").fetch_arrow_table()[0][0].as_py()
                index = np.arange(count, count + batch.num_rows)
                batch = batch.add_column(0, "id", pa.array(index))

                group = pa.array([None] * batch.num_rows, type=pa.int64())
                batch = batch.append_column(GROUP_ID_COLUMN, group)

                self.conn.from_arrow(batch).insert_into(TABLE_NAME)

            except (duckdb.CatalogException, duckdb.InvalidInputException):
                count = 0
                index = np.arange(count, count + batch.num_rows)
                batch = batch.add_column(0, "id", pa.array(index))

                group = pa.array([None] * batch.num_rows, type=pa.int64())
                batch = batch.append_column(GROUP_ID_COLUMN, group)

                self.conn.from_arrow(batch).create(TABLE_NAME)

        return self.update_index()

    def create_index(self, config: HashingConfig, sample: int | None = None) -> "Dataset":
        if sample is not None:
            t = self.conn.sql(
                f"SELECT {VECTOR_COLUMN} FROM {TABLE_NAME} USING SAMPLE {sample} ROWS"
            ).fetch_arrow_table()

        else:
            t = self.conn.table(TABLE_NAME).fetch_arrow_table()

        h = Hashing.from_table(t, config=config)

        self.conn.from_arrow(h.to_table()).create(INDEX_NAME)

        return self.update_index()

    def update_index(self) -> "Dataset":
        h = self.to_index()

        if h is None:
            return self

        def index_hash(v: pa.ListArray) -> pa.Int64Array:
            return pc.list_element(h.apply(v.combine_chunks(), 1), 0)

        self.conn.create_function(  # type:ignore
            "index_hash",
            index_hash,
            [duckdb.dtype("FLOAT[]")],
            duckdb.dtype("BIGINT"),
            type="arrow",
        )

        self.conn.sql(
            f"""
            UPDATE {TABLE_NAME} AS source SET {GROUP_ID_COLUMN} = (
                SELECT
                    COALESCE(target.{GROUP_ID_COLUMN}, index_hash(target.{VECTOR_COLUMN}))
                FROM
                    {TABLE_NAME} AS target
                WHERE
                    source.id = target.id
            )
            """
        )

        def query_hash(v: pa.ListArray, k: pa.Int64Array) -> pa.ListArray:
            return h.apply(v.combine_chunks(), k[0].as_py())

        self.conn.create_function(  # type:ignore
            "query_hash",
            query_hash,
            [duckdb.dtype("FLOAT[]"), duckdb.dtype("BIGINT")],
            duckdb.dtype("BIGINT[]"),
            type="arrow",
        )

        return self

    def to_table(self) -> pa.RecordBatchReader:
        return self.conn.table(TABLE_NAME).fetch_arrow_reader()

    def to_index(self) -> Hashing | None:
        try:
            return Hashing.from_connection(self.conn)
        except duckdb.CatalogException:
            return None

    def search_index(
        self,
        query: pa.Table,
        limit: int,
        probes: int,
    ) -> pa.Table:
        query = query.add_column(0, "id", pa.array(np.arange(0, query.num_rows)))
        query = query.rename_columns([f"target_{c}" for c in query.column_names])

        h = self.to_index()
        assert h is not None

        (
            self.conn.from_arrow(query)
            .project(
                duckdb.StarExpression(),  # type:ignore
                duckdb.FunctionExpression(  # type:ignore
                    "unnest",
                    duckdb.FunctionExpression(
                        "query_hash",
                        duckdb.ColumnExpression(f"target_{VECTOR_COLUMN}"),
                        duckdb.ConstantExpression(probes),
                    ),
                ).alias(GROUP_ID_COLUMN),
            )
            .create(QUERY_NAME)
        )

        match h.metric:
            case "cosine":
                metric = "0.5 - 0.5 * list_cosine_similarity"
            case "dot":
                metric = "-list_inner_product"
            case "l2":
                metric = "list_distance"
            case _:
                raise ValueError()

        r = self.conn.sql(
            f"""
            SELECT
              target.* EXCLUDE(target_{VECTOR_COLUMN}, {GROUP_ID_COLUMN})
            , source.* EXCLUDE({VECTOR_COLUMN}, {GROUP_ID_COLUMN})
            , {GROUP_ID_COLUMN}
            , {metric}(target.target_{VECTOR_COLUMN}, source.{VECTOR_COLUMN}) AS {DISTANCE_COLUMN}
            FROM {QUERY_NAME} AS target
                INNER JOIN {TABLE_NAME} AS source USING ({GROUP_ID_COLUMN})
            QUALIFY ROW_NUMBER() OVER(PARTITION BY target.target_id ORDER BY {DISTANCE_COLUMN}) < {limit}
            ORDER BY target.target_id, {DISTANCE_COLUMN}
            """
        ).fetch_arrow_table()

        self.conn.sql(f"DROP TABLE {QUERY_NAME}")

        return r.rename_columns(
            [
                f"source_{c}"
                if not c.startswith("target_") and c not in {GROUP_ID_COLUMN, DISTANCE_COLUMN}
                else c
                for c in r.column_names
            ]
        )

    def search_table(
        self,
        query: pa.Table,
        limit: int,
        metric: str,
    ) -> pa.Table:
        query = query.add_column(0, "id", pa.array(np.arange(0, query.num_rows)))
        query = query.rename_columns([f"target_{c}" for c in query.column_names])

        self.conn.from_arrow(query).create(QUERY_NAME)

        match metric:
            case "cosine":
                metric = "0.5 - 0.5 * list_cosine_similarity"
            case "dot":
                metric = "-list_inner_product"
            case "l2":
                metric = "list_distance"
            case _:
                raise ValueError()

        r = self.conn.sql(
            f"""
            SELECT
              target.* EXCLUDE(target_{VECTOR_COLUMN})
            , source.* EXCLUDE({VECTOR_COLUMN})
            , {metric}(target.target_{VECTOR_COLUMN}, source.{VECTOR_COLUMN}) AS {DISTANCE_COLUMN}
            FROM
              {QUERY_NAME} AS target
            , {TABLE_NAME} AS source
            QUALIFY ROW_NUMBER() OVER(PARTITION BY target.target_id ORDER BY {DISTANCE_COLUMN}) < {limit}
            ORDER BY target.target_id, {DISTANCE_COLUMN}
            """
        ).fetch_arrow_table()

        self.conn.sql(f"DROP TABLE {QUERY_NAME}")

        return r.rename_columns(
            [
                f"source_{c}"
                if not c.startswith("target_") and c not in {GROUP_ID_COLUMN, DISTANCE_COLUMN}
                else c
                for c in r.column_names
            ]
        )
