import os
import warnings
from typing import Iterator, Sequence, TypedDict

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import fenix.io.batch
import fenix.io.table
import fenix.io.torch

warnings.simplefilter("ignore", UserWarning)


LOCATION: str = "codings"


class Config(TypedDict):
    metric: str
    codebook_size: int
    num_codebooks: int
    batch_size: int
    num_epochs: int


class Coding(TypedDict):
    tensor: Tensor
    column: pa.DataType
    config: Config


def distance(u: Tensor, v: Tensor, metric: str) -> Tensor:
    if metric in {"euclidean", "l2"}:
        return torch.cdist(u, v)

    if metric in {"cosine"}:
        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)
        return 0.5 - 0.5 * u @ v.transpose(-1, -2)

    if metric in {"dot", "inner_product"}:
        return -u @ v.transpose(-1, -2)

    raise ValueError()


def update(q: Tensor, v: Tensor, metric: str) -> Tensor:
    if metric == "cosine":
        q = F.normalize(q, dim=-1)
        v = F.normalize(v, dim=-1)

    d = distance(v, q, metric=metric)
    i = torch.argmin(d, dim=-1)
    q = torch.index_reduce(q, 0, i, v, reduce="mean")

    if metric == "cosine":
        q = F.normalize(q, dim=-1)

    return q


def load(root: str, name: str) -> Coding:
    path = os.path.join(root, LOCATION, name + ".torch")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "rb") as f:
        data: Coding = torch.load(f, map_location="cpu")

    column = data["column"]

    if name not in pc.list_functions():

        def func(ctx: pc.UdfContext, x: pa.FixedSizeListArray, k: pa.Int64Scalar) -> pa.ListArray:
            return call(x, data, k.as_py())

        pc.register_scalar_function(
            func,
            name,
            {"summary": "", "description": ""},
            {"x": column, "k": pa.int64()},
            pa.list_(pa.int64()),
        )

    return data


def make(root: str, name: str, source: str | Sequence[str], column: str, config: Config) -> Coding:
    func = torch.compile(torch.vmap(update))
    data = fenix.io.table.load(root, source)

    filter = (
        np.random.permutation(data.num_rows) < config["codebook_size"] * config["num_codebooks"]
    )

    coding = fenix.io.torch.from_arrow(
        data.filter(filter).column(column).combine_chunks(),
    ).view(config["num_codebooks"], config["codebook_size"], -1)

    for _ in range(config["num_epochs"]):
        batch_size = config["num_codebooks"] * config["batch_size"]
        batch_rows = np.random.permutation(data.num_rows)
        batch_rows = batch_rows[: batch_rows.size // batch_size * batch_size]

        for indices in tqdm(np.array_split(batch_rows, batch_rows.size // batch_size)):
            np.put(filter := np.zeros(data.num_rows, dtype=np.bool_), indices, True)

            sample = fenix.io.torch.from_arrow(
                data.filter(filter).column(column).combine_chunks()
            ).view(config["num_codebooks"], config["batch_size"], -1)

            coding = func(coding, sample, metric=config["metric"])

    path = os.path.join(root, LOCATION, name + ".torch")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        torch.save({"tensor": coding, "column": data.column(column).type, "config": config}, f)

    return load(root, name)


def list(root: str) -> Iterator[str]:
    for path in fsspec.get_mapper(os.path.join(root, LOCATION)):
        if path.endswith(".torch"):
            yield path.removesuffix(".torch")


def drop(root: str, name: str) -> None:
    path = os.path.join(root, LOCATION, name + ".torch")

    if os.path.exists(path):
        os.unlink(path)


def call(
    target: np.ndarray | Tensor | pa.FixedSizeListArray | pa.Table,
    coding: Coding | tuple[str, str],
    maxval: int | None = None,
) -> Tensor | pa.ListArray:
    return_torch = isinstance(target, Tensor)
    return_numpy = isinstance(target, np.ndarray)

    if isinstance(coding, tuple):
        coding = load(*coding)

    n = coding["config"]["num_codebooks"]
    k = coding["config"]["codebook_size"]
    column = coding["column"]
    metric = coding["config"]["metric"]
    tensor = coding["tensor"]

    if isinstance(target, pa.Table):
        target = target.column(column).combine_chunks()

    if isinstance(target, pa.Array):
        target = fenix.io.torch.from_arrow(target)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    assert isinstance(target, Tensor)

    data = distance(target, tensor.flatten(end_dim=-2), metric=metric).view(-1, n, k)
    data = sum(
        [
            data[:, j, i]
            for j in range(n)
            for i in [
                torch.arange(0, k).repeat_interleave(k ** (n - j - 1)).repeat(k**j),
            ]
        ],
        torch.tensor(0),
    )

    if maxval is not None:
        data = torch.topk(data, maxval, largest=False).indices
    else:
        data = torch.argsort(data, descending=False)

    if return_torch:
        return data

    if return_numpy:
        return data.numpy()

    return pa.array(iter(data.numpy()), type=pa.list_(pa.int64()))
