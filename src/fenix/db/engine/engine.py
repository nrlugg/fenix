import os

import duckdb
import msgspec
import pyarrow as pa

from fenix.db.config import Metric, Select


class Engine(msgspec.Struct, frozen=True):
    path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        conn = duckdb.connect(self.path)
        conn.close()

    def list_tables(self) -> list[str]:
        conn = duckdb.connect(self.path, read_only=True)
        names = [name for (name,) in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        return names

    def put(self, name: str, data: pa.Table) -> "Engine":
        conn = duckdb.connect(self.path)
        table = conn.from_arrow(data)

        names = [name for (name,) in conn.execute("SHOW TABLES").fetchall()]
        if name in names:
            table.insert_into(name)
        else:
            table.create(name)

        conn.close()

        return self

    def get(self, name: str) -> pa.RecordBatchReader:
        conn = duckdb.connect(self.path, read_only=True)
        data = conn.table(name).record_batch()
        conn.close()
        return data

    def drop(self, name: str) -> "Engine":
        conn = duckdb.connect(self.path)
        conn.sql(f"DROP TABLE IF EXISTS {name}")
        conn.close()
        return self

    def search(
        self,
        query: pa.Table | duckdb.DuckDBPyRelation,
        index: str,
        limit: int = 10,
        filter: str | None = None,
        select: Select | None = None,
        column: str = "vector",
        metric: Metric = "l2",
    ) -> pa.Table:
        conn = duckdb.connect(self.path)
        func = (
            "0.5 + 0.5 * list_cosine_similarity"
            if metric == "cosine"
            else "-list_inner_product"
            if metric == "dot"
            else "list_distance"
        )

        query = conn.from_arrow(query)
        names = [f"{name} as query_{name}" for name in query.columns]
        query = query.project(", ".join(names))
        query.create("query_")

        filter = f"WHERE {filter}" if filter else ""

        data = (
            conn.sql(
                f"""
                SELECT
                    query_.*
                ,   index_.*
                FROM query_, LATERAL(
                        SELECT
                            *
                        ,   {func}(query_.query_{column}, {index}.{column}) AS distance
                        FROM
                            {index}
                        {filter}
                        ORDER BY distance ASC
                        OFFSET 1
                        LIMIT {limit}
                    ) AS index_
                ORDER BY distance ASC
                """
            )
            .project(
                ", ".join(
                    [
                        *[
                            name
                            for name in conn.table("query_").columns
                            if select is None or name.removeprefix("query_") in select["query"]
                        ],
                        *[
                            f"{name} AS index_{name}"
                            for name in conn.table(index).columns
                            if select is None or name in select["index"]
                        ],
                        "distance",
                    ]
                )
            )
            .to_arrow_table()
        )

        conn.sql("DROP TABLE query_")
        conn.close()

        return data
