import logging

import typer

from fenix.flight import Server

logging.basicConfig()
LOGGER = logging.getLogger("fenix")
LOGGER.setLevel(level=logging.INFO)


def launch(root: str, host: str = "0.0.0.0", port: int = 9001):
    server = Server(root, host, port)

    LOGGER.info(f" Server started at {server.grpc}")

    server.serve()


if __name__ == "__main__":
    typer.run(launch)
