import os
from os.path import join
from typing import TypedDict

import duckdb
import msgspec
import numpy as np
import pyarrow as pa
import torch
import xxhash

import fenix.vq as vq


class IndexConfig(TypedDict):
    k: int
    n: int
    f: int
    column: str
    metrix: str
    epochs: int
    sample: int


class SearchSelect(TypedDict):
    query: list[str]
    table: list[str]


class SearchParams(TypedDict, total=False):
    limit: int
    probes: int
    filter: str | None
    select: SearchSelect | None
    column: str
    metric: str


def default_search_params() -> SearchParams:
    return {
        "limit": 10,
        "probes": 32,
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

    def create_table(self, name: str, data: pa.Table) -> None:
        assert name not in self.list_tables()

        with self.connect() as conn:
            table = conn.from_arrow(data)
            table.create(name)

    def update_table(self, name: str, data: pa.Table) -> None:
        assert name in self.list_tables()

        with self.connect() as conn:
            table = conn.from_arrow(data)
            table.insert_into(name)

    def remove_table(self, name: str) -> None:
        if name not in self.list_tables():
            return

        with self.connect() as conn:
            conn.sql(f"DROP TABLE IF EXISTS {name}")

    def to_pyarrow(self, name: str, batch_size: int = 32_000) -> pa.RecordBatchReader:
        with duckdb.connect(self.uri, read_only=True) as conn:
            return conn.table(name).record_batch(batch_size)

    def create_index(self, name: str, conf: IndexConfig) -> None:
        path = join(self.indexes_uri, f"{name}.pt")

        if os.path.exists(path):
            raise FileExistsError()

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

    def remove_index(self, name: str) -> None:
        path = join(self.indexes_uri, f"{name}.pt")
        if os.path.exists(path):
            os.unlink(path)

        with self.connect() as conn:
            if "group_id" in conn.table(name).columns:
                conn.sql(f"ALTER TABLE {name} DROP group_id")

    def search(
        self,
        table: str,
        query: pa.Table,
        params: SearchParams | None = None,
    ) -> pa.Table:
        params = default_search_params() | params if params else default_search_params()

        QUERY = xxhash.xxh64(msgspec.msgpack.encode(params)).hexdigest()
        QUERY = f"query_{table}_{QUERY}"

        query = query.select(params["select"]["query"]) if params["select"] is not None else query

        if params["metric"] == "index":
            index_uri = join(self.indexes_uri, f"{table}.pt")
            assert os.path.exists(index_uri)
            q = torch.load(index_uri, map_location="cpu")
            x = torch.from_numpy(
                np.stack(query[q["conf"]["column"]].to_numpy(zero_copy_only=False)),
            )

            i = vq.apply_quantization(x, q["data"], params["probes"])

            query = query.append_column("group_id", pa.array(list(i.numpy())))
            query = duckdb.sql(
                "SELECT * EXCLUDE(group_id), UNNEST(group_id) AS group_id FROM query"
            ).to_arrow_table()

            params["metric"] = q["conf"]["metric"]
            params["column"] = q["conf"]["column"]
            filter = f"{QUERY}.group_id = {table}.group_id"
            params["filter"] = (
                filter if params["filter"] is None else " AND ".join([filter, params["filter"]])
            )

        query = query.rename_columns(
            [f"query_{name}" if name != "group_id" else name for name in query.column_names]
        )

        func = (
            "1 - list_cosine_similarity"
            if params["metric"] == "cosine"
            else "-list_inner_product"
            if params["metric"] == "dot"
            else "list_distance"
        )

        print(func)

        self.create_table(QUERY, query)

        with self.connect(read_only=True) as conn:
            limit = params["limit"]
            column = params["column"]
            filter = f"{params['filter']}" if params["filter"] else ""
            select = [
                *[name for name in conn.table(QUERY).columns if name != "group_id"],
                *[
                    f"{name} AS index_{name}" if name != "group_id" else name
                    for name in conn.table(table).columns
                    if params["select"] is None or name in params["select"]["table"]
                ],
                "distance",
            ]

            print(select)

            data = (
                conn.sql(
                    f"""
                    SELECT
                      {QUERY}.*
                    , {table}.*
                    , {func}({QUERY}.query_{column}, {table}.{column}) AS distance
                    FROM {QUERY} INNER JOIN {table} ON {filter}
                    QUALIFY ROW_NUMBER() OVER(PARTITION BY {QUERY}.query_id ORDER BY distance ASC) <= {limit}
                    ORDER BY {QUERY}.query_id ASC, distance ASC
                    """
                )
                .project(", ".join(select))
                .to_arrow_table()
            )

        self.remove_table(QUERY)

        return data
