import os

import fsspec
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch import Tensor

BATCH_SIZE: int = 2**18


def from_ipc(path: str, schema: pa.Schema | None = None) -> pa.Table:
    if os.path.isdir(path):
        return pa.concat_tables(
            from_ipc(f) for f in ds.dataset(path, format="ipc", schema=schema).files
        )

    with pa.memory_map(path, "rb") as source:
        return pa.ipc.open_file(source).read_all()


def to_ipc(path: str, data: pa.Table | pa.RecordBatchReader, schema: pa.Schema) -> None:
    assert path.endswith(".arrow")

    fs = fsspec.filesystem("file")
    fs.makedirs(os.path.dirname(path), exist_ok=True)

    data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader(BATCH_SIZE)

    with pa.OSFile(path, "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            for batch in data:
                writer.write_batch(batch)


def to_torch(v: pa.FixedSizeListArray | pa.FixedSizeListScalar) -> Tensor:
    if isinstance(v, pa.FixedSizeListScalar):
        return torch.from_numpy(
            v.values.to_numpy(),
        )

    return torch.from_numpy(
        v.values.to_numpy(),
    ).view(-1, v.type.list_size)
