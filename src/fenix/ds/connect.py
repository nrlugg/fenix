from fenix.ds.dataset import Dataset
from fenix.ds.flight import RemoteDataset


def connect(uri: str) -> Dataset | RemoteDataset:
    if uri.startswith("grpc://"):
        return RemoteDataset(uri)

    return Dataset(uri)
