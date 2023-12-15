import os
from os.path import join
from typing import TypedDict

import duckdb
import msgspec
import numpy as np
import pyarrow as pa
import torch
import xxhash
from torch import Tensor

import fenix.vq as vq


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


class SearchSelect(TypedDict, total=False):
    target: list[str]
    source: list[str]


class SearchParams(TypedDict, total=False):
    limit: int
    probes: int
    filter: str | None
    select: SearchSelect | None
    column: str
    metric: str | None


def default_search_params() -> SearchParams:
    return {
        "limit": 10,
        "probes": 32,
        "filter": None,
        "select": None,
        "column": "vector",
        "metric": None,
    }


class Dataset(msgspec.Struct, frozen=True):
    uri: str

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

    def create_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name not in self.list_tables()

        with self.connect() as conn:
            if isinstance(data, pa.Table):
                table = conn.from_arrow(data)
                table.create(name)

            elif isinstance(data, pa.RecordBatchReader):
                batch = conn.from_arrow(data.read_next_batch())
                batch.create(name)

                for batch in map(conn.from_arrow, data):
                    batch.insert_into(name)

            else:
                raise ValueError()

    def update_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name in self.list_tables()

        with self.connect() as conn:
            if isinstance(data, pa.Table):
                table = conn.from_arrow(data)
                table.insert_into(name)

            elif isinstance(data, pa.RecordBatchReader):
                for batch in map(conn.from_arrow, data):
                    batch.insert_into(name)

    def remove_table(self, name: str) -> None:
        if name not in self.list_tables():
            return

        with self.connect() as conn:
            conn.sql(f"DROP TABLE IF EXISTS {name}")

        self.remove_index(name)

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

    def update_index(self, name: str, conf: IndexConfig) -> None:
        self.remove_index(name)
        self.create_index(name, conf)

    def remove_index(self, name: str) -> None:
        path = join(self.indexes_uri, f"{name}.pt")
        if os.path.exists(path):
            os.unlink(path)

        with self.connect() as conn:
            if "group_id" in conn.table(name).columns:
                conn.sql(f"ALTER TABLE {name} DROP group_id")

    def table(self, name: str, batch_size: int = 32_000) -> pa.RecordBatchReader:
        with duckdb.connect(self.dataset_uri, read_only=True) as conn:
            return conn.table(name).record_batch(batch_size)

    def index(self, name: str) -> SavedIndex:
        return torch.load(
            join(self.indexes_uri, f"{name}.pt"),
            map_location="cpu",
        )

    def search(
        self,
        target: pa.Table,
        source: str,
        params: SearchParams | None = None,
    ) -> pa.Table:
        params = default_search_params() | params if params else default_search_params()

        SOURCE = source
        TARGET = xxhash.xxh64(msgspec.msgpack.encode(params)).hexdigest()
        TARGET = f"target_{SOURCE}_{TARGET}"

        if params["metric"] is None:
            q = self.index(SOURCE)
            x = torch.from_numpy(
                np.stack(target[q["conf"]["column"]].to_numpy(zero_copy_only=False)),
            )

            i = vq.apply_quantization(x, q["data"], params["probes"])

            target = target.append_column("group_id", pa.array(list(i.numpy())))
            target = duckdb.sql(
                "SELECT * EXCLUDE(group_id), UNNEST(group_id) AS group_id FROM target"
            ).to_arrow_table()

            params["metric"] = q["conf"]["metric"]
            params["column"] = q["conf"]["column"]
            filter = f"{TARGET}.group_id = {SOURCE}.group_id"
            params["filter"] = (
                filter if params["filter"] is None else " AND ".join([filter, params["filter"]])
            )

        target = target.rename_columns(
            [f"target.{name}" if name != "group_id" else name for name in target.column_names]
        )

        func = (
            "1 - list_cosine_similarity"
            if params["metric"] == "cosine"
            else "-list_inner_product"
            if params["metric"] == "dot"
            else "list_distance"
        )

        self.create_table(TARGET, target)

        with self.connect(read_only=True) as conn:
            limit = params["limit"]
            column = params["column"]
            filter = f"{params['filter']}" if params["filter"] else ""
            select = [
                *[
                    f'"{name}"'
                    for name in conn.table(TARGET).columns
                    if name != "group_id"
                    and (
                        params["select"] is None
                        or name.removeprefix("target.") in params["select"]["target"]
                    )
                ],
                *[
                    f'"{name}" AS "source.{name}"'
                    for name in conn.table(SOURCE).columns
                    if name != "group_id"
                    and (params["select"] is None or name in params["select"]["source"])
                ],
                "group_id AS _group_id",
                "distance AS _distance",
            ]

            data = (
                conn.sql(
                    f"""
                    SELECT
                      {TARGET}.*
                    , {SOURCE}.*
                    , {func}({TARGET}."target.{column}", {SOURCE}.{column}) AS distance
                    FROM {TARGET} INNER JOIN {SOURCE} ON {filter}
                    QUALIFY ROW_NUMBER() OVER(PARTITION BY {TARGET}."target.id" ORDER BY distance ASC) <= {limit}
                    ORDER BY {TARGET}."target.id" ASC, distance ASC
                    """
                )
                .project(", ".join(select))
                .to_arrow_table()
            )

        self.remove_table(TARGET)

        return data
