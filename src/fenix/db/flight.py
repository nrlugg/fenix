import functools
from typing import Iterator

import msgspec
import pyarrow as pa
import pyarrow.flight as fl

from fenix.db.client import Client
from fenix.db.config import Config, default_config
from fenix.db.engine import Engine


class Flight(fl.FlightServerBase):
    def __init__(
        self,
        database: str = "./data/fenix.ddb",
        location: str | None = None,
    ) -> None:
        super().__init__(location=location)

        self.engine = Engine(database)
        self.location = location if location is not None else f"grpc://0.0.0.0:{self.port}"

    def get_flight_info(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
    ) -> fl.FlightInfo:
        name = descriptor.path[0].decode()
        data = self.engine.get(name)

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
        for name in self.engine.list_tables():
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
        self.engine.put(
            descriptor.path[0].decode(),
            reader.read_all(),
        )

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        name = ticket.ticket.decode()
        data = self.engine.get(name)

        return fl.GeneratorStream(data.schema, data)

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        data = self.engine.search(
            query=reader.read_all(),
            index=descriptor.path[0].decode(),
            config=msgspec.json.decode(descriptor.path[1]),
        )

        writer.begin(data.schema)
        writer.write_table(data)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        if action.type == "drop":
            self.engine.drop(
                action.body.to_pybytes().decode(),
            )


class Remote(Client, frozen=True, dict=True):
    uri: str

    @functools.cached_property
    def flight(self) -> fl.FlightClient:
        return fl.connect(self.uri)

    def list_tables(self) -> list[str]:
        return [
            flight.endpoints[0].ticket.ticket.decode() for flight in self.flight.list_flights()
        ]

    def put(self, name: str, data: pa.RecordBatchReader) -> "Remote":
        descriptor = fl.FlightDescriptor.for_path(name)
        writer, reader = self.flight.do_put(descriptor, data.schema)

        for batch in data:
            writer.write_batch(batch)

        writer.close()

        return self

    def get(self, name: str) -> pa.RecordBatchReader:
        flight = self.flight.get_flight_info(fl.FlightDescriptor.for_path(name))
        ticket = flight.endpoints[0].ticket
        reader = self.flight.do_get(ticket)
        return reader.to_reader()

    def drop(self, name: str) -> "Remote":
        self.flight.do_action(
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

        writer, reader = self.flight.do_exchange(descriptor)

        with writer:
            writer.begin(query.schema)
            writer.write_table(query)
            writer.done_writing()

            result = reader.read_all()

        return result
