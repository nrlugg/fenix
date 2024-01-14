import os

import pyarrow as pa


def load(path: str) -> pa.Table:
    with pa.memory_map(path, "rb") as source:
        return pa.ipc.open_stream(source).read_all()


def make(path: str, data: pa.RecordBatchReader) -> pa.Table:
    assert path.endswith(".arrow")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with pa.OSFile(path, "wb") as sink:
        with pa.ipc.new_stream(sink, data.schema) as writer:
            for batch in data:
                writer.write_batch(batch)

    return load(path)
