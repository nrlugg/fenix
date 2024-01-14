import os
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from torch import Tensor

import fenix.io.arrow
import fenix.io.coder
import fenix.io.table

CODE_COL: str = "__CODED_ID__"
DIST_COL: str = "__DISTANCE__"
LOCATION: str = "indexes"


def load(root: str, name: str, data: str | list[str], column: str) -> pa.Table:
    if isinstance(data, list):
        return fenix.io.table.join(
            *[load(root, name, data, column) for data in data],
        )

    assert isinstance(data, str)

    fenix.io.coder.load(root, name)

    path = os.path.join(root, LOCATION, name, data, column + ".arrow")

    return fenix.io.table.join(
        fenix.io.table.load(root, data),
        fenix.io.arrow.load(path),
        axis=1,
    )


def make(root: str, name: str, data: str | list[str], column: str) -> pa.Table:
    if isinstance(data, list):
        return fenix.io.table.join(
            *[make(root, name, data, column) for data in data],
        )

    assert isinstance(data, str)

    path = os.path.join(root, LOCATION, name, data, column + ".arrow")

    os.makedirs(os.path.dirname(path), exist_ok=False)

    fenix.io.coder.load(root, name)

    code = fenix.io.table.load(root, data).column(column)
    code = pc.call_function(name, [code, pa.scalar(1)])
    code = pc.list_element(code, 0)
    code = pa.table({CODE_COL: code})
    code = fenix.io.arrow.make(path, code)

    return load(root, name, data, column)


def call(
    root: str,
    name: str | None,
    data: str | list[str] | pa.Table,
    column: str,
    target: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
    metric: str | None = None,
    select: Sequence[str] | None = None,
    filter: pc.Expression | None = None,
    maxval: int | None = None,
    probes: int | None = None,
) -> pa.Table:
    if not isinstance(data, pa.Table):
        if name is not None:
            data = load(root, name, data, column)
        else:
            data = fenix.io.table.load(root, data)

    source = data.column(column)

    if isinstance(target, pa.ChunkedArray):
        target = target.combine_chunks()

    if isinstance(target, pa.Array):
        target = target.to_numpy()

    if isinstance(target, Tensor):
        target = target.numpy()

    if isinstance(target, np.ndarray):
        target = pa.scalar(target, type=type)

    if name is not None and probes is not None:
        code = fenix.io.coder.load(root, name)

        if metric is None:
            metric = code["config"]["metric"]

        mask = pc.field(CODE_COL).isin(
            pc.call_function(name, [pa.array([target]), probes]).values,
        )

        if filter is None:
            filter = mask
        else:
            filter = filter & mask

    select = list(select) if select is not None else data.column_names
    select = select + [DIST_COL]

    assert metric is not None

    func = f"distance:{metric}:{source.type.value_type}:{source.type.list_size}"

    data = data.filter(filter) if filter is not None else data
    data = data.append_column(DIST_COL, pc.call_function(func, [source, target]))
    data = data.select(select)
    data = data.take(
        pc.select_k_unstable(data, maxval, [(DIST_COL, "ascending")]),
    )

    return data
