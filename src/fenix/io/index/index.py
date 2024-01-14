import os
from typing import Iterator, Sequence

import fsspec
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


def load(root: str, name: str, data: str | Sequence[str], column: str) -> pa.Table:
    if isinstance(data, str):
        fenix.io.coder.load(root, name)

        path = os.path.join(root, LOCATION, data, column, name + ".arrow")

        return fenix.io.table.join(
            fenix.io.table.load(root, data),
            fenix.io.arrow.load(path),
            axis=1,
        )

    assert isinstance(data, Sequence) and not isinstance(data, str)
    return fenix.io.table.join(
        *[load(root, name, data, column) for data in data],
    )


def make(root: str, name: str, data: str | Sequence[str], column: str) -> pa.Table:
    if isinstance(data, str):
        path = os.path.join(root, LOCATION, data, column, name + ".arrow")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        fenix.io.coder.load(root, name)

        def record_batch_generator() -> Iterator[pa.RecordBatch]:
            for code in fenix.io.table.load(root, data).to_reader():
                code = code.column(column)
                code = pc.call_function(name, [code, pa.scalar(1)])
                code = pc.list_element(code, 0)
                yield pa.record_batch([code], names=[CODE_COL])

        fenix.io.arrow.make(
            path,
            pa.RecordBatchReader.from_batches(
                pa.schema({CODE_COL: pa.int64()}),
                record_batch_generator(),
            ),
        )

        return load(root, name, data, column)

    assert isinstance(data, Sequence) and not isinstance(data, str)
    return fenix.io.table.join(
        *[make(root, name, data, column) for data in data],
    )


def list(root: str) -> Iterator[str]:
    for path in fsspec.get_mapper(os.path.join(root, LOCATION)):
        if path.endswith(".arrow"):
            yield path.removesuffix(".arrow")


def drop(root: str, name: str, data: str, column: str) -> None:
    path = os.path.join(root, LOCATION, data, column, name + ".arrow")

    if os.path.exists(path):
        os.unlink(path)


def call(
    root: str,
    name: str | None,
    data: str | Sequence[str] | pa.Table,
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

    type = data.schema.field(column).type

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

    select = [*select] if select is not None else data.column_names
    select = select + [DIST_COL]

    assert metric is not None

    func = f"distance:{metric}:{type.value_type}:{type.list_size}"

    if func not in pc.list_functions():

        def dist(
            ctx: pc.UdfContext, x: pa.FixedSizeListArray, q: pa.FixedSizeListScalar
        ) -> pa.FloatArray:
            return pa.array(
                (
                    fenix.io.coder.distance(
                        fenix.io.torch.from_arrow(q).unsqueeze(0),
                        fenix.io.torch.from_arrow(x),
                        metric=metric,
                    )
                    .squeeze(0)
                    .numpy()
                ),
                type=type.value_type,
            )

        pc.register_scalar_function(
            dist,
            func,
            {"summary": "", "description": ""},
            {"x": type, "q": type},
            type.value_type,
        )

    data = data.filter(filter) if filter is not None else data
    data = data.append_column(DIST_COL, pc.call_function(func, [data.column(column), target]))
    data = data.select(select)

    if maxval is not None and len(data) > maxval:
        data = data.take(
            pc.select_k_unstable(data, maxval, [(DIST_COL, "ascending")]),
        )

    return data.combine_chunks()
