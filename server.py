import fenix

if __name__ == "__main__":
    fenix.ds.DatasetServer("./data/fenix", "grpc://0.0.0.0:8888").serve()
