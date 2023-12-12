import os
from os.path import join
from typing import TypedDict

import duckdb
import msgspec
import numpy as np
import pyarrow as pa
import torch

import fenix.vq as vq


class IndexConfig(TypedDict):
    k: int
    n: int
    f: int
    column: str
    metric: str
    sample: int
    epochs: int


class SearchSelect(TypedDict):
    query: list[str]
    table: list[str]


class SeachParams(TypedDict, total=False):
    limit: int
    filter: str | None
    select: SearchSelect | None
    column: str
    metric: str


def default_search_params() -> SeachParams:
    return {
        "limit": 10,
        "filter": None,
        "select": None,
        "column": "vector",
        "metric": "index",
    }


class Dataset(msgspec.Struct, frozen=True):
    uri: str = "./data/random"

    def __post_init__(self) -> None:
        os.makedirs(self.indexes_uri, exist_ok=True)
        with self.connect():
            pass

    @property
    def dataset_uri(self) -> str:
        return join(self.uri, "fenix.db")

    @property
    def indexes_uri(self) -> str:
        return join(self.uri, "index")

    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.dataset_uri, read_only=read_only)

    def list_tables(self) -> list[str]:
        with self.connect(read_only=True) as conn:
            return [name for (name,) in conn.execute("SHOW TABLES").fetchall()]

    def create_table(self, name: str, data: pa.Table) -> "Dataset":
        assert name not in self.list_tables()

        with self.connect() as conn:
            table = conn.from_arrow(data)
            table.create(name)

        return self

    def update_table(self, name: str, data: pa.Table) -> "Dataset":
        assert name in self.list_tables()

        with self.connect() as conn:
            table = conn.from_arrow(data)
            table.insert_into(name)

        return self

    def remove_table(self, name: str) -> "Dataset":
        if name not in self.list_tables():
            return self

        with self.connect() as conn:
            conn.sql(f"DROP TABLE IF EXISTS {name}")

        return self

    def to_pyarrow(self, name: str, batch_size: int = 32_000) -> pa.RecordBatchReader:
        with duckdb.connect(self.uri, read_only=True) as conn:
            return conn.table(name).record_batch(batch_size)

    def create_index(self, name: str, conf: IndexConfig) -> "Dataset":
        with self.connect() as conn:
            t = conn.sql(
                f"SELECT * FROM {name} USING SAMPLE {conf['sample']} ROWS"
            ).to_arrow_table()

            v = t[conf["column"]]

            x = torch.from_numpy(
                np.stack(v.to_numpy(zero_copy_only=False)),
            )

            q = vq.build_quantization(
                x,
                k=conf["k"],
                n=conf["n"],
                f=conf["f"],
                metric=conf["metric"],
                epochs=conf["epochs"],
            )

            torch.save(
                {
                    "conf": conf,
                    "data": q,
                },
                join(self.indexes_uri, f"{name}.pt"),
            )

            q = vq.apply_quantization(x, q, metric=conf["metric"])
            q = vq.index_quantization(q, k=conf["k"])

            i = pa.table({"id": q.numpy()})  # noqa

            conn.sql(
                f"""
                CREATE OR REPLACE TABLE {name} AS
                  SELECT
                    {name}.*
                  , i.id AS group_id
                  FROM {name} POSITIONAL JOIN i 
                """
            )

        return self

    def search(
        self,
        query: pa.Table,
        table: str,
        params: SeachParams | None = None,
    ) -> pa.Table:
        params = default_search_params() | params if params else default_search_params()

        query = query.select(params["select"]["query"]) if params["select"] is not None else query

        if params["metric"] == "index":
            index_uri = join(self.indexes_uri, f"{table}.pt")
            assert os.path.exists(index_uri)
            q = torch.load(index_uri, map_location="cpu")
            x = torch.from_numpy(
                np.stack(query[q["conf"]["column"]].to_numpy(zero_copy_only=False)),
            )

            i = vq.apply_quantization(x, q["data"], metric=q["conf"]["metric"])
            i = vq.index_quantization(i, k=q["conf"]["k"])

            query = query.append_column("group_id", pa.array(i.numpy()))

            params["metric"] = q["conf"]["metric"]
            params["column"] = q["conf"]["column"]
            filter = "query_.group_id = table_.group_id"
            params["filter"] = (
                filter if params["filter"] is None else " AND ".join([filter, params["filter"]])
            )

        query = query.rename_columns(
            [f"query_{name}" if name != "group_id" else name for name in query.column_names]
        )

        func = (
            "0.5 + 0.5 * list_cosine_similarity"
            if params["metric"] == "cosine"
            else "-list_inner_product"
            if params["metric"] == "dot"
            else "list_distance"
        )

        self.create_table("query_", query)

        with self.connect(read_only=True) as conn:
            limit = params["limit"]
            column = params["column"]
            filter = f"WHERE {params['filter']}" if params["filter"] else ""
            select = [
                *[name for name in conn.table("query_").columns if name != "group_id"],
                *[
                    f"{name} AS index_{name}" if name != "group_id" else name
                    for name in conn.table(table).columns
                    if params["select"] is None or name in params["select"]["table"]
                ],
                "distance",
            ]

            print(conn.sql("SELECT * FROM query_"))

            data = (
                conn.sql(
                    f"""
                    SELECT
                        query_.*
                    ,   table_.*
                    FROM query_, LATERAL(
                            SELECT
                                *
                            ,   {func}(query_.query_{column}, {table}.{column}) AS distance
                            FROM
                                {table}
                            ORDER BY distance ASC
                            OFFSET 1
                            LIMIT {limit}
                        ) AS table_
                    ORDER BY distance ASC
                    """
                )
                .project(", ".join(select))
                .to_arrow_table()
            )

        self.remove_table("query_")

        return data
