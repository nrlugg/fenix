import os

import duckdb
import pyarrow as pa

from fenix.db.client import Client
from fenix.db.config import Config, default_config


class Engine(Client, frozen=True):
    uri: str = "./data/fenix.ddb"

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.uri), exist_ok=True)
        conn = duckdb.connect(self.uri)
        conn.close()

    def list_tables(self) -> list[str]:
        with duckdb.connect(self.uri, read_only=True) as conn:
            return [name for (name,) in conn.execute("SHOW TABLES").fetchall()]

    def put(self, name: str, data: pa.Table) -> "Engine":
        with duckdb.connect(self.uri) as conn:
            table = conn.from_arrow(data)

            names = [name for (name,) in conn.execute("SHOW TABLES").fetchall()]

            if name in names:
                table.insert_into(name)
            else:
                table.create(name)

        return self

    def get(self, name: str) -> pa.RecordBatchReader:
        with duckdb.connect(self.uri, read_only=True) as conn:
            return conn.table(name).record_batch()

    def drop(self, name: str) -> "Engine":
        with duckdb.connect(self.uri) as conn:
            conn.sql(f"DROP TABLE IF EXISTS {name}")

        return self

    def search(
        self,
        query: pa.Table | duckdb.DuckDBPyRelation,
        index: str,
        config: Config | None = None,
    ) -> pa.Table:
        config = default_config() | config if config else default_config()

        func = (
            "0.5 + 0.5 * list_cosine_similarity"
            if config["metric"] == "cosine"
            else "-list_inner_product"
            if config["metric"] == "dot"
            else "list_distance"
        )

        with duckdb.connect(self.uri) as conn:
            query = conn.from_arrow(query)
            names = [f"{name} as query_{name}" for name in query.columns]
            query = query.project(", ".join(names))
            query.create("query_")

            limit = config["limit"]
            column = config["column"]
            filter = f"WHERE {config['filter']}" if config["filter"] else ""
            select = [
                *[
                    name
                    for name in conn.table("query_").columns
                    if config["select"] is None
                    or name.removeprefix("query_") in config["select"]["query"]
                ],
                *[
                    f"{name} AS index_{name}"
                    for name in conn.table(index).columns
                    if config["select"] is None or name in config["select"]["index"]
                ],
                "distance",
            ]

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
                .project(", ".join(select))
                .to_arrow_table()
            )

            conn.sql("DROP TABLE query_")

        return data
