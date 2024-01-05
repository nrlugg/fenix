import functools
import os
import pickle
from dataclasses import dataclass
from os.path import join
from typing import Iterator, Sequence

import fsspec
import msgspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.flight as fl

from fenix.ds.dataset import INDEX_DIR, TABLE_DIR, Dataset, IndexConfig


@dataclass
class Server(fl.FlightServerBase):
    database: str
    location: str | None = None

    def __post_init__(self) -> None:
        super().__init__(location=self.grpc_uri)

        self.database = os.path.abspath(self.database)

        if self.location is None:
            self.location = f"grpc://0.0.0.0:{self.port}"

    @functools.cached_property
    def filesystem(self) -> fsspec.AbstractFileSystem:
        return fsspec.filesystem("file")

    @staticmethod
    def descriptor_to_name(desc: fl.FlightDescriptor) -> str:
        return join(*map(bytes.decode, desc.path))

    def descriptor_to_path(self, desc: fl.FlightDescriptor) -> str:
        return join(self.database, self.descriptor_to_name(desc))

    def descriptor_to_data(self, desc: fl.FlightDescriptor) -> Dataset:
        return Dataset(self.descriptor_to_path(desc))

    def get_flight_info(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
    ) -> fl.FlightInfo:
        name = self.descriptor_to_name(descriptor)
        data = self.descriptor_to_data(descriptor).to_table()
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
        for path, dirs, _ in self.filesystem.walk(self.database):
            if set(dirs) == {TABLE_DIR, INDEX_DIR}:
                name = path.removeprefix(self.database).removeprefix(os.path.sep)
                desc = fl.FlightDescriptor.for_path(*name.split(os.path.sep))
                yield self.get_flight_info(ctx, desc)

    def do_put(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> None:
        self.descriptor_to_data(descriptor).insert_table(
            pa.RecordBatchReader(reader.schema, reader)
        )

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        desc = fl.FlightDescriptor.for_path(*ticket.ticket)
        data = self.descriptor_to_data(desc)
        return fl.GeneratorStream(data.schema, data.to_reader())

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        param = msgspec.json.decode(descriptor.command)

        if param["args"]["filter"] is not None:
            param["args"]["filter"] = pickle.loads(param["args"]["filter"])

        query = reader.read_all()[0].combine_chunks()
        table = Dataset(join(self.database, param["name"]))

        match param["type"]:
            case "index":
                r = table.search_index(query, **param["args"])

            case "table":
                r = table.search_table(query, **param["args"])

            case _:
                raise ValueError()

        writer.begin(r.schema)

        for batch in r:
            writer.write_batch(batch)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        body = msgspec.json.decode(action.body.to_pybytes())

        match action.type:
            case "create-index":
                self.dataset.create_index(**body)

            case _:
                raise ValueError()


@dataclass
class Client:
    uri: str

    @functools.cached_property
    def conn(self) -> fl.FlightClient:
        return fl.connect(self.uri)

    def __del__(self) -> None:
        self.conn.close()

    def to_table(self, name: str) -> pa.RecordBatchReader:
        flight = self.conn.get_flight_info(fl.FlightDescriptor.for_path(*name.split(os.path.sep)))
        ticket = flight.endpoints[0].ticket
        reader = self.conn.do_get(ticket)
        return reader.to_reader()

    def insert_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> "Client":
        data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader()
        desc = fl.FlightDescriptor.for_path(*name.split(os.path.sep))

        writer, reader = self.conn.do_put(desc, data.schema)

        for batch in data:
            writer.write_batch(batch)

        return self

    def create_index(self, name: str, column: str, config: IndexConfig) -> "Client":
        self.conn.do_action(
            fl.Action(
                "create-index",
                msgspec.json.encode({"name": name, "column": column, "config": config}),
            )
        )

        return self

    def search_index(
        self,
        query: pa.Table,
        table: str,
        index: int,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        probes: int = 8,
    ) -> pa.RecordBatchReader:
        descriptor = fl.FlightDescriptor.for_command(
            msgspec.json.encode(
                {
                    "name": table,
                    "type": "index",
                    "args": {
                        "index": index,
                        "select": select,
                        "filter": filter if filter is None else pickle.dumps(filter),
                        "probes": probes,
                    },
                }
            )
        )

        writer, reader = self.conn.do_exchange(descriptor)

        with writer:
            writer.begin(query.schema)
            writer.write_table(query)
            writer.done_writing()

        return pa.RecordBatchReader.from_batches(reader)


@dataclass
class RemoteDataset(Dataset):
    ...
