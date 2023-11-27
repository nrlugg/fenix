import functools

import msgspec
import pyarrow as pa
import pyarrow.flight as fl

from fenix.db.config import Config


class Client(msgspec.Struct, frozen=True, dict=True):
    location: str

    @functools.cached_property
    def flight(self) -> fl.FlightClient:
        return fl.connect(self.location)

    def list_tables(self) -> list[str]:
        return [
            flight.endpoints[0].ticket.ticket.decode() for flight in self.flight.list_flights()
        ]

    def put(self, name: str, data: pa.RecordBatchReader) -> "Client":
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

    def search(
        self,
        query: pa.Table,
        index: str,
        config: Config | None = None,
    ) -> pa.Table:
        descriptor = fl.FlightDescriptor.for_path(
            index,
            msgspec.json.encode(config if config else {}),
        )

        writer, reader = self.flight.do_exchange(descriptor)

        with writer:
            writer.begin(query.schema)
            writer.write_table(query)
            writer.done_writing()

            result = reader.read_all()

        return result
