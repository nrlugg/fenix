import functools
from typing import Iterator

import msgspec
import pyarrow as pa
import pyarrow.flight as fl

from fenix.ds.config import Config, default_config
from fenix.ds.dataset import Dataset
from fenix.ds.engine import Engine


class FlightServer(fl.FlightServerBase):
    def __init__(
        self,
        database: str = "./data/fenix.ddb",
        location: str | None = None,
    ) -> None:
        super().__init__(location=location)

        self.dataset = Dataset(database)
        self.location = location if location is not None else f"grpc://0.0.0.0:{self.port}"

    def get_flight_info(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
    ) -> fl.FlightInfo:
        name = descriptor.path[0].decode()
        data = self.dataset.get(name)

        return fl.FlightInfo(
            schema=data.schema,
            descriptor=descriptor,
            endpoints=[fl.FlightEndpoint(name, [self.location])],
            total_records=sum(len(batch) for batch in data),
            total_bytes=sum(batch.nbytes for batch in data),
        )

    def list_flights(
        self,
        ctx: fl.ServerCallContext,
        criteria: bytes,
    ) -> Iterator[fl.FlightDescriptor]:
        for name in self.dataset.list_tables():
            yield self.get_flight_info(
                ctx,
                fl.FlightDescriptor.for_path(name),
            )

    def do_put(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> None:
        self.dataset.put(
            descriptor.path[0].decode(),
            reader.read_all(),
        )

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        name = ticket.ticket.decode()
        data = self.dataset.get(name)

        return fl.GeneratorStream(data.schema, data)

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        data = self.dataset.search(
            query=reader.read_all(),
            index=descriptor.path[0].decode(),
            config=msgspec.json.decode(descriptor.path[1]),
        )

        writer.begin(data.schema)
        writer.write_table(data)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        if action.type == "drop":
            self.dataset.drop(
                action.body.to_pybytes().decode(),
            )


class FlightDataset(Engine, frozen=True, dict=True):
    uri: str

    @functools.cached_property
    def client(self) -> fl.FlightClient:
        return fl.connect(self.uri)

    def list_tables(self) -> list[str]:
        return [
            flight.endpoints[0].ticket.ticket.decode() for flight in self.client.list_flights()
        ]

    def put(self, name: str, data: pa.RecordBatchReader) -> "FlightDataset":
        descriptor = fl.FlightDescriptor.for_path(name)
        writer, reader = self.client.do_put(descriptor, data.schema)

        for batch in data:
            writer.write_batch(batch)

        writer.close()

        return self

    def get(self, name: str) -> pa.RecordBatchReader:
        flight = self.client.get_flight_info(fl.FlightDescriptor.for_path(name))
        ticket = flight.endpoints[0].ticket
        reader = self.client.do_get(ticket)
        return reader.to_reader()

    def drop(self, name: str) -> "FlightDataset":
        self.client.do_action(
            fl.Action("drop", name.encode()),
        )

        return self

    def search(
        self,
        query: pa.Table,
        index: str,
        config: Config | None = None,
    ) -> pa.Table:
        descriptor = fl.FlightDescriptor.for_path(
            index,
            msgspec.json.encode(config if config else default_config()),
        )

        writer, reader = self.client.do_exchange(descriptor)

        with writer:
            writer.begin(query.schema)
            writer.write_table(query)
            writer.done_writing()

            result = reader.read_all()

        return result
