from . import ds, ex, io, vq


def launch(
    source: str, column: str, metric: str, coding: str | None = None
) -> ds.Engine | ds.Remote:
    if source.startswith("grpc://"):
        return ds.Remote(source, column, metric, coding)

    return ds.Engine(source, column, metric, coding)
