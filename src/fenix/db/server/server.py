from typing import Iterator

import msgspec
import pyarrow.flight as fl

from fenix.db.engine import Engine


class Server(fl.FlightServerBase):
    def __init__(
        self,
        path: str = "./data/fenix.ddb",
        host: str = "0.0.0.0",
        port: int = 8815,
    ) -> None:
        LOCATION = f"grpc://{host}:{port}"
        super().__init__(LOCATION)

        self.engine = Engine(path)
        self.location = LOCATION

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
            **msgspec.json.decode(descriptor.path[1]),
        )

        writer.begin(data.schema)
        writer.write_table(data)
