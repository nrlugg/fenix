import msgspec
import pyarrow as pa

from fenix.db.config import Config


class Client(msgspec.Struct, frozen=True, dict=True):
    uri: str

    def list_tables(self) -> list[str]:
        raise NotImplementedError()

    def put(self, name: str, data: pa.RecordBatchReader) -> "Client":
        raise NotImplementedError()

    def get(self, name: str) -> pa.RecordBatchReader:
        raise NotImplementedError()

    def drop(self, name: str) -> "Client":
        raise NotImplementedError()

    def search(self, query: pa.Table, index: str, config: Config | None = None) -> pa.Table:
        raise NotImplementedError()
