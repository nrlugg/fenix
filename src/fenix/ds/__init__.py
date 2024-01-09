from .dataset import Dataset
from .flight import DatasetServer, RemoteDataset


def dataset(uri: str) -> Dataset | RemoteDataset:
    if uri.startswith("grpc://"):
        return RemoteDataset(uri)

    return Dataset(uri)
