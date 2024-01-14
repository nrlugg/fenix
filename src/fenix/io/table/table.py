import os
from typing import Literal

import pyarrow as pa

import fenix.io.arrow

LOCATION: str = "sources"


def load(root: str, name: str | list[str]) -> pa.Table:
    if isinstance(name, list):
        return join(
            *[load(root, name) for name in name],
        )

    path = os.path.join(root, LOCATION, name + ".arrow")

    return fenix.io.arrow.load(path)


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
