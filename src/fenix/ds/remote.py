import functools
import os
import pickle
from os.path import join
from typing import Iterator, Self, Sequence

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.flight as fl
from torch import Tensor

from fenix.ds.engine import CODING_DATA, SOURCE_ROOT, CodingConfig, Engine, Source


class Server(fl.FlightServerBase):
    def __init__(self, root: str, host: str = "0.0.0.0", port: int = 9001) -> None:
        self.data_location = os.path.abspath(root)
        self.grpc_location = f"grpc://{host}/{port}"

        super().__init__(location=self.grpc_location)

    @staticmethod
    def source_from_descriptor(desc: fl.FlightDescriptor) -> Source:
        return Source(join(*map(bytes.decode, desc.path)))

    @staticmethod
    def engine_from_descriptor(desc: fl.FlightDescriptor) -> Engine:
        return Engine(**msgspec.json.decode(desc.command))

    def get_flight_info(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
    ) -> fl.FlightInfo:
        source = (
            self.source_from_descriptor(descriptor)
            if descriptor.path is not None
            else self.engine_from_descriptor(descriptor)
        )

        name = os.path.abspath(source.source)
        name = name.removeprefix(self.data_location)
        name = name.removeprefix("/")

        if isinstance(source, Engine) and source.coding is not None:
            name = f"{name}:{source.column}/{source.metric}/{source.coding}"

        data = source.to_pyarrow()

        return fl.FlightInfo(
            schema=data.schema,
            descriptor=descriptor,
            endpoints=[fl.FlightEndpoint(name, [self.grpc_location])],
            total_records=data.num_rows,
            total_bytes=data.nbytes,
        )

    def list_flights(
        self,
        ctx: fl.ServerCallContext,
        criteria: bytes,
    ) -> Iterator[fl.FlightDescriptor]:
        fs = fsspec.filesystem("file")

        for path, _, files in fs.walk(self.data_location):
            if path.endswith(SOURCE_ROOT) and all(file.endswith(".arrow") for file in files):
                descriptor = fl.FlightDescriptor.for_path(
                    *(
                        path.removesuffix(SOURCE_ROOT)
                        .removesuffix("/")
                        .removeprefix(self.grpc_location)
                        .removeprefix("/")
                        .split("/")
                    )
                )

                yield self.get_flight_info(ctx, descriptor)

            if CODING_DATA in files:
                *source, _, column, metric, coding = path.split(os.path.sep)

                source = join(*source).removeprefix(self.data_location).removeprefix("/")
                descriptor = fl.FlightDescriptor.for_command(
                    msgspec.json.encode(
                        {
                            "source": source,
                            "column": column,
                            "metric": metric,
                            "coding": coding,
                        }
                    )
                )

                yield self.get_flight_info(ctx, descriptor)

    def do_put(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> None:
        self.source_from_descriptor(descriptor).insert(
            pa.RecordBatchReader.from_batches(reader.schema, reader.to_reader()),
        )

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        data: Source | Engine

        if ":" in (name := ticket.ticket.decode()):
            name, spec = name.split(":")
            column, metric, coding = spec.split("/")

            data = Engine(name, column, metric, coding)

        else:
            data = Source(name)

        return fl.GeneratorStream(data.schema, data.to_pyarrow().to_reader())

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        config = msgspec.json.decode(descriptor.command)
        engine = Engine(**config["engine"])

        if config["search"]["filter"] is not None:
            config["search"]["filter"] = pickle.loads(config["search"]["filter"])

        result = engine.search(
            reader.read_all().column("vector").combine_chunks(),
            **config["search"],
        )

        writer.begin(result.schema)
        writer.write_table(result)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        config = msgspec.json.decode(action.body.to_pybytes())
        engine = Engine(**config["engine"])

        match action.type:
            case "encode":
                engine.encode(**config["encode"])

            case _:
                raise ValueError()


class Remote(msgspec.Struct, frozen=True, dict=True):
    source: str
    column: str
    metric: str
    coding: str | None = None

    def __post_init__(self) -> None:
        for flight in self.conn.list_flights():
            spec = msgspec.json.decode(flight.descriptor.command)
            name = spec["table"]
            if name < self.name:
                raise ValueError("Cannot nest datasets")

    @functools.cached_property
    def host(self) -> str:
        host, *name = self.source.removeprefix("grpc://").split("/")
        return join("grpc://", host)

    @functools.cached_property
    def name(self) -> str:
        return self.source.removeprefix(self.host).removeprefix("/")

    @functools.cached_property
    def conn(self) -> fl.FlightClient:
        return fl.connect(self.host)

    def __del__(self) -> None:
        self.conn.close()

    def to_pyarrow(self) -> pa.RecordBatchReader:
        ticket = self.name

        if self.coding is not None:
            ticket = f"{ticket}:{self.column}/{self.metric}/{self.coding}"

        return self.conn.do_get(fl.Ticket(ticket)).to_reader()

    def insert(self, data: pa.Table | pa.RecordBatchReader) -> Self:
        data = data if isinstance(data, pa.RecordBatchReader) else data.to_reader()
        desc = fl.FlightDescriptor.for_path(*self.name.split(os.path.sep))

        writer, reader = self.conn.do_put(desc, data.schema)

        with writer:
            for batch in data:
                writer.write_batch(batch)

        return self

    def search(
        self,
        query: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
        limit: int,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        probes: int | None = None,
    ) -> pa.Table:
        descriptor = fl.FlightDescriptor.for_command(
            msgspec.json.encode(
                {
                    "engine": {
                        "source": self.name,
                        "column": self.column,
                        "metric": self.metric,
                        "coding": self.coding,
                    },
                    "search": {
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

    def encode(self, name: str, config: CodingConfig) -> "Remote":
        action = fl.Action(
            "encode",
            msgspec.json.encode(
                {
                    "engine": {
                        "source": self.name,
                        "column": self.column,
                        "metric": self.metric,
                    },
                    "encode": {
                        "name": name,
                        "config": config,
                    },
                }
            ),
        )

        self.conn.do_action(action)

        return Remote(self.source, self.column, self.metric, name)
