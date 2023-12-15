import functools
from typing import Iterator

import msgspec
import pyarrow as pa
import pyarrow.flight as fl
from torch import Tensor

from fenix.ds.dataset import Dataset, IndexConfig, SearchParams


class FlightServer(fl.FlightServerBase):
    def __init__(
        self,
        database: str,
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
        data = self.dataset.table(name)

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
        name = descriptor.path[0].decode()

        if name in self.dataset.list_tables():
            self.dataset.update_table(name, reader.read_all())
        else:
            self.dataset.create_table(name, reader.read_all())

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        name = ticket.ticket.decode()
        data = self.dataset.table(name)

        return fl.GeneratorStream(data.schema, data)

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        body = msgspec.json.decode(descriptor.command)
        data = self.dataset.search(
            target=reader.read_all(),
            source=body["source"],
            params=body["params"],
        )

        writer.begin(data.schema)
        writer.write_table(data)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        body = msgspec.json.decode(action.body.to_pybytes())

        match action.type:
            case "remove-table":
                self.dataset.remove_table(**body)

            case "create-index":
                self.dataset.create_index(**body)

            case "remove-index":
                self.dataset.remove_index(**body)

            case "update-index":
                self.dataset.remove_index(**body)

            case _:
                raise NotImplementedError()


class FlightDataset(msgspec.Struct, frozen=True, dict=True):
    uri: str

    @functools.cached_property
    def connect(self) -> fl.FlightClient:
        return fl.connect(self.uri)

    def list_tables(self) -> list[str]:
        return [
            flight.endpoints[0].ticket.ticket.decode() for flight in self.connect.list_flights()
        ]

    def _insert_data(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader()

        descriptor = fl.FlightDescriptor.for_path(name)
        writer, reader = self.connect.do_put(descriptor, data.schema)

        for batch in data:
            writer.write_batch(batch)

        writer.close()

    def create_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name not in self.list_tables()
        self._insert_data(name, data)

    def update_table(self, name: str, data: pa.Table | pa.RecordBatchReader) -> None:
        assert name in self.list_tables()
        self._insert_data(name, data)

    def remove_table(self, name: str) -> None:
        assert name in self.list_tables()
        self.connect.do_action(
            fl.Action(
                "remove-table",
                msgspec.json.encode({"name": name}),
            )
        )

    def create_index(self, name: str, conf: IndexConfig) -> None:
        assert name in self.list_tables()
        self.connect.do_action(
            fl.Action(
                "create-index",
                msgspec.json.encode({"name": name, "conf": conf}),
            )
        )

    def update_index(self, name: str, conf: IndexConfig) -> None:
        assert name in self.list_tables()
        self.connect.do_action(
            fl.Action(
                "update-index",
                msgspec.json.encode({"name": name, "conf": conf}),
            )
        )

    def remove_index(self, name: str) -> None:
        assert name in self.list_tables()
        self.connect.do_action(
            fl.Action(
                "remove-index",
                msgspec.json.encode({"name": name}),
            )
        )

    def table(self, name: str) -> pa.RecordBatchReader:
        flight = self.connect.get_flight_info(fl.FlightDescriptor.for_path(name))
        ticket = flight.endpoints[0].ticket
        reader = self.connect.do_get(ticket)
        return reader.to_reader()

    def index(self, name: str) -> Tensor:
        raise NotImplementedError()

    def search(
        self,
        target: pa.Table,
        source: str,
        params: SearchParams | None = None,
    ) -> pa.Table:
        descriptor = fl.FlightDescriptor.for_command(
            msgspec.json.encode(
                {
                    "source": source,
                    "params": params,
                }
            )
        )

        writer, reader = self.connect.do_exchange(descriptor)

        with writer:
            writer.begin(target.schema)
            writer.write_table(target)
            writer.done_writing()

            result = reader.read_all()

        return result
