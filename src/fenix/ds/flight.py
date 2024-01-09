import functools
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import Iterator, Sequence, TypedDict

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.flight as fl
from torch import Tensor

import fenix.ds.dataset as ds


class DatasetDescriptor(TypedDict):
    table: str
    index: ds.Index | None


@dataclass
class DatasetServer(fl.FlightServerBase):
    database: str
    location: str | None = None

    def __post_init__(self) -> None:
        super().__init__(location=self.location)

        fs = fsspec.filesystem("file")

        self.database = os.path.abspath(self.database)
        self.datasets = {
            name: ds.Dataset(path)
            for path, dirs, _ in fs.walk(self.database)
            if set(dirs) == {ds.TABLE_PATH, ds.INDEX_PATH}
            for name in [path.removeprefix(self.database).removeprefix("/")]
        }

        if self.location is None:
            self.location = f"grpc://0.0.0.0:{self.port}"

    def get_flight_info(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
    ) -> fl.FlightInfo:
        body = msgspec.json.decode(descriptor.command)
        name = body["table"]
        data = self.datasets[name].to_table(index=body["index"])

        if body["index"] is not None:
            name = f"{name}:{ds.index_name(body['index'])}"

        return fl.FlightInfo(
            schema=data.schema,
            descriptor=descriptor,
            endpoints=[fl.FlightEndpoint(name, [self.location])],
            total_records=data.num_rows,
            total_bytes=data.nbytes,
        )

    def list_flights(
        self,
        ctx: fl.ServerCallContext,
        criteria: bytes,
    ) -> Iterator[fl.FlightDescriptor]:
        for name in self.datasets:
            yield self.get_flight_info(
                ctx,
                fl.FlightDescriptor.for_command(
                    msgspec.json.encode({"table": name, "index": None})
                ),
            )

            for spec in self.datasets[name].list_indexes():
                yield self.get_flight_info(
                    ctx,
                    fl.FlightDescriptor.for_command(
                        msgspec.json.encode({"table": name, "index": spec["index"]})
                    ),
                )

    def do_put(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> None:
        body = msgspec.json.decode(descriptor.command)
        name = body["table"]
        path = join(self.database, name)

        if name not in self.datasets:
            self.datasets[name] = ds.Dataset(path)

        self.datasets[name] = self.datasets[name].insert_table(
            pa.RecordBatchReader.from_batches(reader.schema, reader.to_reader()),
        )

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        name = ticket.ticket.decode()

        index: ds.Index | None
        if ":" in name:
            name, spec = ticket.ticket.decode().split(":")

            column, metric, version = spec.split("/")
            index = {"column": column, "metric": metric, "version": version}

        else:
            index = None

        data = self.datasets[name].to_table(index=index)

        return fl.GeneratorStream(data.schema, data.to_reader())

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        body = msgspec.json.decode(descriptor.command)
        name = body["table"]

        if body["param"]["filter"] is not None:
            body["param"]["filter"] = pickle.loads(body["param"]["filter"])

        data = self.datasets[name].search_index(
            reader.read_all().column("vector").combine_chunks(),
            **body["param"],
        )

        writer.begin(data.schema)
        writer.write_table(data)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        body = msgspec.json.decode(action.body.to_pybytes())
        name = body.pop("table")
        data = self.datasets[name]

        match action.type:
            case "create-index":
                data.create_index(**body["param"])

            case _:
                raise ValueError()


@dataclass
class RemoteDataset(pa.dataset.Dataset):
    uri: str

    @property
    def host(self) -> str:
        host, *ame = self.uri.removeprefix("grpc://").split("/")
        return join("grpc://", host)

    @property
    def name(self) -> str:
        return self.uri.removeprefix(self.host).removeprefix("/")

    @functools.cached_property
    def conn(self) -> fl.FlightClient:
        return fl.connect(self.host)

    def __del__(self) -> None:
        self.conn.close()

    def to_table(self, index: ds.Index | None = None) -> pa.RecordBatchReader:
        ticket = self.name

        if index is not None:
            ticket = f"{ticket}:{ds.index_name(index)}"

        return self.conn.do_get(fl.Ticket(ticket)).to_reader()

    def scanner(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> pa.dataset.Scanner:
        return pa.dataset.Scanner.from_batches(
            self.to_table(),
            columns=columns,
            filter=filter,
        )

    def count_rows(self, filter: pc.Expression | None = None) -> int:
        return self.scanner().count_rows(filter=filter)

    def filter(self, expression: pc.Expression) -> "RemoteDataset":
        raise NotImplementedError()

    def get_fragments(self, filter: pc.Expression | None = None) -> Iterator[pa.dataset.Fragment]:
        raise NotImplementedError()

    def head(
        self,
        num_rows: int,
        columns: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> pa.Table:
        return self.scanner().head(num_rows, columns=columns, filter=filter)

    def join(self, *args, **kwargs) -> "RemoteDataset":
        raise NotImplementedError()

    def replace_schema(self, schema: pa.Schema) -> pa.Schema:
        raise NotImplementedError()

    def sort_by(self, sorting: str | list[tuple[str, str]]) -> "RemoteDataset":
        raise NotImplementedError()

    def take(
        self,
        indices: Sequence[int] | np.ndarray | pa.Array,
        columns: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> pa.Array:
        return self.scanner().take(indices, columns=columns, filter=filter)

    def to_batches(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> list[pa.RecordBatch]:
        return self.scanner().to_batches(columns=columns, filter=filter)

    def to_reader(
        self, columns: Sequence[str] | None = None, filter: pc.Expression | None = None
    ) -> Iterator[pa.RecordBatch]:
        return self.scanner().to_reader(columns=columns, filter=filter)

    def insert_table(self, data: pa.Table | pa.RecordBatchReader) -> "RemoteDataset":
        data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader()

        desc = fl.FlightDescriptor.for_command(
            msgspec.json.encode({"table": self.name, "index": None}),
        )

        writer, reader = self.conn.do_put(desc, data.schema)

        with writer:
            for batch in data:
                writer.write_batch(batch)

        return RemoteDataset(self.uri)

    def search_index(
        self,
        query: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
        index: ds.Index,
        limit: int,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        probes: int | None = None,
    ) -> pa.Table:
        descriptor = fl.FlightDescriptor.for_command(
            msgspec.json.encode(
                {
                    "table": self.name,
                    "param": {
                        "index": index,
                        "limit": limit,
                        "select": select,
                        "filter": filter if filter is None else pickle.dumps(filter),
                        "probes": probes,
                    },
                }
            )
        )

        writer, reader = self.conn.do_exchange(descriptor)

        if isinstance(query, Tensor):
            query = query.numpy()

        if isinstance(query, np.ndarray):
            query = pa.array(query)

        query = pa.table({"vector": query})

        with writer:
            writer.begin(query.schema)
            writer.write_table(query)
            writer.done_writing()

            return reader.read_all()

    def create_index(self, index: ds.Index, config: ds.IndexConfig) -> "RemoteDataset":
        action = fl.Action(
            "create-index",
            msgspec.json.encode(
                {
                    "table": self.name,
                    "param": {
                        "index": index,
                        "config": config,
                    },
                },
            ),
        )

        self.conn.do_action(action)

        return RemoteDataset(self.uri)
