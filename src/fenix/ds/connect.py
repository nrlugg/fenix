from fenix.ds.dataset import Dataset
from fenix.ds.engine import Engine
from fenix.ds.flight import FlightDataset


def connect(uri: str) -> Engine:
    if uri.startswith("grpc://"):
        return FlightDataset(uri)

    return Dataset(uri)
