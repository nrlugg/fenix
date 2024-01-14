import os
from typing import Iterator, Literal, Sequence

import fsspec
import pyarrow as pa

import fenix.io.arrow

LOCATION: str = "sources"


def load(root: str, name: str | Sequence[str]) -> pa.Table:
    if isinstance(name, str):
        path = os.path.join(root, LOCATION, name + ".arrow")

        return fenix.io.arrow.load(path)

    assert isinstance(name, Sequence) and not isinstance(name, str)
    return join(
        *[load(root, name) for name in name],
    )


def make(root: str, name: str, data: pa.RecordBatchReader) -> pa.Table:
    path = os.path.join(root, LOCATION, name + ".arrow")
    return fenix.io.arrow.make(path, data)


def join(*data: pa.Table, axis: Literal[0, 1] = 0) -> pa.Table:
    if len(data) == 1:
        return data[0]

    match axis:
        case 0:
            return pa.concat_tables(data)

        case 1:
            return pa.table(
                {c: t.column(c) for t in data for c in t.column_names},
            )

        case _:
            raise ValueError()


def list(root: str) -> Iterator[str]:
    for path in fsspec.get_mapper(os.path.join(root, LOCATION)):
        if path.endswith(".arrow"):
            yield path.removesuffix(".arrow")


def drop(root: str, name: str) -> None:
    path = os.path.join(root, LOCATION, name + ".arrow")

    if os.path.exists(path):
        os.unlink(path)
