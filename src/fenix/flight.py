import functools
import os
import pickle
import shutil
from typing import Iterator, Self, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.flight as fl
from pydantic.dataclasses import dataclass
from torch import Tensor

import fenix.io as io


class Server(fl.FlightServerBase):
    def __init__(self, root: str, host: str = "0.0.0.0", port: int = 9001) -> None:
        self.root = os.path.abspath(root)
        self.grpc = f"grpc://{host}:{port}"

        super().__init__(location=self.grpc)

    def get_flight_info(
        self, ctx: fl.ServerCallContext, descriptor: fl.FlightDescriptor
    ) -> fl.FlightInfo:
        raise NotImplementedError()

    def list_flights(
        self, ctx: fl.ServerCallContext, criteria: bytes
    ) -> Iterator[fl.FlightDescriptor]:
        raise NotImplementedError()

    def do_put(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.FlightMetadataWriter,
    ) -> None:
        name = descriptor.path[0].decode()
        data = reader.to_reader()

        io.table.make(self.root, name, data)

    def do_get(self, ctx: fl.ServerCallContext, ticket: fl.Ticket):
        source = ticket.ticket.decode().split(":")

        if hasattr(self, "coding") and hasattr(self, "column"):
            data = io.index.load(self.root, self.coding, source, self.column)
        else:
            data = io.table.load(self.root, source)

        if hasattr(self, "filter"):
            data = data.filter(self.filter)

        if hasattr(self, "select"):
            data = data.select(self.select)

        return fl.GeneratorStream(data.schema, data.to_reader())

    def do_exchange(
        self,
        ctx: fl.ServerCallContext,
        descriptor: fl.FlightDescriptor,
        reader: fl.MetadataRecordBatchReader,
        writer: fl.MetadataRecordBatchWriter,
    ) -> None:
        config = pickle.loads(descriptor.command)

        config["target"] = reader.read_all().column("target").combine_chunks()
        config["filter"] = pickle.loads(config["filter"])

        data = io.index.call(self.root, **config)

        writer.begin(data.schema)
        writer.write_table(data)

    def do_action(self, ctx: fl.ServerCallContext, action: fl.Action) -> None:
        config = pickle.loads(action.body.to_pybytes())

        match action.type:
            case "make-coder":
                io.coder.make(self.root, **config)

            case "make-index":
                io.index.make(self.root, **config)

            case "drop-table":
                io.table.drop(self.root, **config)

            case "drop-index":
                io.coder.drop(self.root, **config)

                for path in io.index.list(self.root):
                    path = os.path.basename(path)

                    if path.endswith(config["name"]):
                        *_, source, column, coding = path.split("/")
                        io.index.drop(self.root, coding, source, column)

            case "remove":
                shutil.rmtree(self.root)

            case "set-coding":
                self.coding = config["coding"]

            case "del-coding":
                if hasattr(self, "coding"):
                    delattr(self, "coding")

            case "set-column":
                self.column = config["column"]

            case "del-column":
                if hasattr(self, "column"):
                    delattr(self, "column")

            case "set-filter":
                self.filter = config["filter"]

            case "del-filter":
                if hasattr(self, "filter"):
                    delattr(self, "filter")

            case "set-select":
                self.select = config["select"]

            case "del-select":
                if hasattr(self, "select"):
                    delattr(self, "select")

            case _:
                raise ValueError()


@dataclass(frozen=True)
class Flight:
    host: str = "0.0.0.0"
    port: int = 9001

    @functools.cached_property
    def conn(self) -> fl.FlightClient:
        return fl.connect(f"grpc://{self.host}:{self.port}")

    def __del__(self) -> None:
        self.conn.close()

    def make_table(self, name: str, data: pa.RecordBatchReader) -> Self:
        descriptor = fl.FlightDescriptor.for_path(name)

        writer, reader = self.conn.do_put(descriptor, data.schema)

        with writer:
            for batch in data:
                writer.write_batch(batch)

        return self

    def read_table(
        self,
        source: str | Sequence[str],
        coding: str | None = None,
        column: str | None = None,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
    ) -> pa.RecordBatchReader:
        if coding is not None and column is not None:
            self.conn.do_action(fl.Action("set-coding", pickle.dumps({"coding": coding})))
            self.conn.do_action(fl.Action("set-column", pickle.dumps({"column": column})))

        if select is not None:
            self.conn.do_action(
                fl.Action("set-select", pickle.dumps({"select": select})),
            )

        if filter is not None:
            self.conn.do_action(
                fl.Action("set-filter", pickle.dumps({"filter": filter})),
            )

        source = ":".join(source) if not isinstance(source, str) else source
        ticket = fl.Ticket(source)
        reader = self.conn.do_get(ticket).to_reader()

        self.conn.do_action(fl.Action("del-coding", pickle.dumps({})))
        self.conn.do_action(fl.Action("del-column", pickle.dumps({})))
        self.conn.do_action(fl.Action("del-select", pickle.dumps({})))
        self.conn.do_action(fl.Action("del-filter", pickle.dumps({})))

        return reader

    def drop_table(self, name: str) -> Self:
        self.conn.do_action(
            fl.Action("drop-table", pickle.dumps({"name": name})),
        )

        return self

    def make_index(
        self, name: str, source: str | Sequence[str], column: str, config: io.coder.Config
    ) -> Self:
        self.conn.do_action(
            fl.Action(
                "make-coder",
                pickle.dumps(
                    {
                        "name": name,
                        "source": source,
                        "column": column,
                        "config": config,
                    }
                ),
            )
        )

        return self.sync_index(name, source, column)

    def sync_index(self, name: str, source: str | Sequence[str], column: str) -> Self:
        self.conn.do_action(
            fl.Action(
                "make-index",
                pickle.dumps(
                    {
                        "name": name,
                        "source": source,
                        "column": column,
                    }
                ),
            )
        )

        return self

    def drop_index(self, name: str) -> Self:
        self.conn.do_action(
            fl.Action("drop-index", pickle.dumps({"name": name})),
        )

        return self

    def search(
        self,
        target: pa.Array | pa.ChunkedArray | pa.FixedSizeListScalar | np.ndarray | Tensor,
        source: str | Sequence[str],
        column: str,
        metric: str,
        coding: str | None = None,
        select: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        maxval: int | None = None,
        probes: int | None = None,
    ) -> pa.Table:
        METRICS: set[str] = {"cosine", "dot", "inner_product", "l2", "euclidean"}

        assert metric in METRICS

        descriptor = fl.FlightDescriptor.for_command(
            pickle.dumps(
                {
                    "coding": coding,
                    "source": source,
                    "column": column,
                    "metric": metric,
                    "select": select,
                    "filter": pickle.dumps(filter),
                    "maxval": maxval,
                    "probes": probes,
                }
            )
        )

        if isinstance(target, Tensor):
            target = target.numpy()

        if isinstance(target, np.ndarray):
            target = pa.array(target)

        target = pa.table({"target": target})

        writer, reader = self.conn.do_exchange(descriptor)

        with writer:
            writer.begin(target.schema)
            writer.write_table(target)
            writer.done_writing()

            return reader.read_all()

    def remove(self) -> Self:
        self.conn.do_action(fl.Action("remove", pickle.dumps({})))
        return self
