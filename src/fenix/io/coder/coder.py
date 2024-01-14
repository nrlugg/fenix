import os
import warnings
from typing import TypedDict

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


LOCATION: str = "indexes"


class Config(TypedDict):
    column: pa.DataType
    metric: str
    codebook_size: int
    num_codebooks: int
    batch_size: int
    num_epochs: int


class Coding(TypedDict):
    tensor: Tensor
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


def kmeans(q: Tensor, v: Tensor, metric: str) -> Tensor:
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

    with open(path, "rb") as f:
        data: Coding = torch.load(f, map_location="cpu")

    column = data["config"]["column"]
    metric = data["config"]["metric"]

    if name not in pc.list_functions():

        def topk(ctx: pc.UdfContext, x: pa.FixedSizeListArray, k: pa.Int64Scalar) -> pa.ListArray:
            return call(data, x, k.as_py())

        pc.register_scalar_function(
            topk,
            name,
            {"summary": "", "description": ""},
            {"x": column, "k": pa.int64()},
            pa.list_(pa.int64()),
        )

    name = f"distance:{metric}:{column.value_type}:{column.list_size}"

    if name not in pc.list_functions():

        def dist(
            ctx: pc.UdfContext, x: pa.FixedSizeListArray, q: pa.FixedSizeListScalar
        ) -> pa.FloatArray:
            return pa.array(
                (
                    distance(
                        fenix.io.torch.from_arrow(q).unsqueeze(0),
                        fenix.io.torch.from_arrow(x),
                        metric=metric,
                    )
                    .squeeze(0)
                    .numpy()
                ),
                type=column.value_type,
            )

        pc.register_scalar_function(
            dist,
            name,
            {"summary": "", "description": ""},
            {"x": column, "q": column},
            column.value_type,
        )

    return data


def make(root: str, name: str, data: str | list[str], column: str, config: Config) -> Coding:
    update = torch.compile(torch.vmap(kmeans))
    source = fenix.io.table.load(root, data)

    filter = (
        np.random.permutation(source.num_rows) < config["codebook_size"] * config["num_codebooks"]
    )

    coding = fenix.io.torch.from_arrow(
        source.filter(filter).column(column).combine_chunks(),
    ).view(config["num_codebooks"], config["codebook_size"], -1)

    for _ in range(config["num_epochs"]):
        batch_size = config["num_codebooks"] * config["batch_size"]
        batch_rows = np.random.permutation(source.num_rows)
        batch_rows = batch_rows[: batch_rows.size // batch_size * batch_size]

        for indices in tqdm(np.array_split(batch_rows, batch_rows.size // batch_size)):
            np.put(filter := np.zeros(source.num_rows, dtype=np.bool_), indices, True)

            sample = fenix.io.torch.from_arrow(
                source.filter(filter).column(column).combine_chunks()
            ).view(config["num_codebooks"], config["batch_size"], -1)

            coding = update(coding, sample, metric=config["metric"])

    path = os.path.join(root, LOCATION, name + ".torch")

    os.makedirs(os.path.dirname(path), exist_ok=False)

    with open(path, "wb") as f:
        torch.save({"tensor": coding, "config": config}, f)

    return load(root, name)


def call(
    code: Coding | tuple[str, str],
    data: Tensor | pa.FixedSizeListArray | pa.Table,
    topk: int | None = None,
) -> Tensor | pa.ListArray:
    return_torch = isinstance(data, Tensor)

    if isinstance(code, tuple):
        code = load(*code)

    n = code["config"]["num_codebooks"]
    k = code["config"]["codebook_size"]
    column = code["config"]["column"]
    metric = code["config"]["metric"]
    coding = code["tensor"]

    if isinstance(data, pa.Table):
        data = data.column(column)

    if not isinstance(data, Tensor):
        data = fenix.io.torch.from_arrow(data)

    data = distance(data, coding.flatten(end_dim=-2), metric=metric).view(-1, n, k)
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

    if topk is not None:
        data = torch.topk(data, topk, largest=False).indices
    else:
        data = torch.argsort(data, descending=False)

    if return_torch:
        return data

    return pa.array(iter(data.numpy()), type=pa.list_(pa.int64()))
