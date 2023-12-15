from fenix.ds.dataset import Dataset
from fenix.ds.flight import FlightDataset


def connect(uri: str) -> Dataset | FlightDataset:
    if uri.startswith("grpc://"):
        return FlightDataset(uri)

    return Dataset(uri)
