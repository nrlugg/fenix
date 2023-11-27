from fenix.db.client import Client
from fenix.db.engine import Engine
from fenix.db.flight import Remote


def connect(uri: str) -> Client:
    if uri.startswith("grpc://"):
        return Remote(uri)

    return Engine(uri)
