import msgspec
import pyarrow as pa

from fenix.ds.config import Config


class Engine(msgspec.Struct, frozen=True, dict=True):
    uri: str

    def list_tables(self) -> list[str]:
        raise NotImplementedError()

    def put(self, name: str, data: pa.RecordBatchReader) -> "Engine":
        raise NotImplementedError()

    def get(self, name: str) -> pa.RecordBatchReader:
        raise NotImplementedError()

    def drop(self, name: str) -> "Engine":
        raise NotImplementedError()

    def search(self, query: pa.Table, index: str, config: Config | None = None) -> pa.Table:
        raise NotImplementedError()
