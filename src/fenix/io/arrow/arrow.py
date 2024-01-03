import os

import pyarrow as pa
import pyarrow.dataset as ds

BATCH_SIZE: int = 2**20


def from_ipc(path: str, schema: pa.Schema | None = None) -> pa.Table:
    if os.path.isdir(path):
        return pa.concat_tables(
            from_ipc(f) for f in ds.dataset(path, format="ipc", schema=schema).files
        )

    with pa.memory_map(path, "rb") as source:
        return pa.ipc.open_file(source).read_all()


def to_ipc(
    path: str, data: pa.Table | ds.Dataset | pa.RecordBatchReader, schema: pa.Schema | None = None
) -> None:
    assert path.endswith(".arrow")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader(BATCH_SIZE)

    with pa.OSFile(path, "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            for batch in data:
                writer.write_batch(batch)
