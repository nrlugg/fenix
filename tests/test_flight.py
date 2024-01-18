from typing import Iterator

import fenix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

VECTOR_SIZE: int = 256
NUM_VECTORS: int = 100_000
BATCH_SIZE: int = 1_000

VECTOR = pa.list_(pa.float32(), list_size=VECTOR_SIZE)
SCHEMA = pa.schema({"id": pa.int64(), "vector": VECTOR})


def generate_batches() -> Iterator[pa.RecordBatch]:
    NUM_ROWS = 0

    for _ in range(NUM_VECTORS // BATCH_SIZE):
        x = np.random.randn(BATCH_SIZE, VECTOR_SIZE).astype(np.float32)
        x = x + 10 * x[0, :]

        a = pa.FixedSizeListArray.from_arrays(
            pa.array(x.flatten()),
            list_size=x.shape[-1],
        )

        i = pa.array(
            np.arange(NUM_ROWS, NUM_ROWS + BATCH_SIZE),
        )

        NUM_ROWS += BATCH_SIZE

        yield pa.record_batch([i, a], names=["id", "vector"])


class TestFlight:
    server: fenix.Server = fenix.Server("data/fenix")
    source: pa.Table = pa.Table.from_batches(generate_batches(), SCHEMA)

    def test_make_table(self) -> None:
        SOURCE = "test/table"

        flight = fenix.Flight()
        flight = flight.make_table(SOURCE, self.source.to_reader())

        table = flight.read_table(SOURCE).read_all()

        assert self.source == table

    @pytest.mark.parametrize(
        "metric",
        [
            "cosine",
            "dot",
            "inner_product",
            "l2",
            "euclidean",
        ],
    )
    def test_make_index(self, metric: str) -> None:
        SOURCE = "test/table"
        CODING = f"test/{metric}"
        COLUMN = "vector"

        flight = fenix.Flight()
        flight = flight.make_index(
            name=CODING,
            source=SOURCE,
            column=COLUMN,
            config={
                "metric": metric,
                "codebook_size": 8,
                "num_codebooks": 2,
                "batch_size": 2560,
                "num_epochs": 5,
            },
        )

        table = flight.read_table(SOURCE, CODING, COLUMN).read_all()

        assert self.source == table.drop(["__CODED_ID__"])
        assert table.schema == pa.schema(
            [*self.source.schema, pa.field("__CODED_ID__", pa.int64())]
        )

    @pytest.mark.parametrize(
        "metric",
        [
            "cosine",
            "dot",
            "inner_product",
            "l2",
            "euclidean",
        ],
    )
    def test_search_without_index(self, metric: str) -> None:
        SOURCE = "test/table"
        COLUMN = "vector"

        flight = fenix.Flight()
        result = flight.search(
            target=pc.random(VECTOR_SIZE).cast(VECTOR.value_type),
            source=SOURCE,
            column=COLUMN,
            metric=metric,
            maxval=10,
        )

        assert result.num_rows == 10
        assert result.schema == pa.schema(
            [*self.source.schema, pa.field("__DISTANCE__", VECTOR.value_type)]
        )

    @pytest.mark.parametrize(
        "metric",
        [
            "cosine",
            "dot",
            "inner_product",
            "l2",
            "euclidean",
        ],
    )
    def test_search_with_index(self, metric: str) -> None:
        SOURCE = "test/table"
        CODING = f"test/{metric}"
        COLUMN = "vector"

        flight = fenix.Flight()
        result = flight.search(
            target=pc.random(VECTOR_SIZE).cast(VECTOR.value_type),
            source=SOURCE,
            column=COLUMN,
            metric=metric,
            coding=CODING,
            maxval=10,
            probes=16,
        )

        assert result.num_rows == 10
        assert result.schema == pa.schema(
            [
                *self.source.schema,
                pa.field("__CODED_ID__", pa.int64()),
                pa.field("__DISTANCE__", VECTOR.value_type),
            ]
        )

    def test_remove(self) -> None:
        flight = fenix.Flight()
        flight.remove()
        self.server.shutdown()
