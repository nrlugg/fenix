import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import fenix.io as io


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


def update(v: Tensor, q: Tensor, metric: str) -> Tensor:
    if metric == "cosine":
        q = F.normalize(q, dim=-1)
        v = F.normalize(v, dim=-1)

    d = distance(v, q, metric=metric)
    i = torch.argmin(d, dim=-1)
    q = torch.index_reduce(q, 0, i, v, reduce="mean")

    if metric == "cosine":
        q = F.normalize(q, dim=-1)

    return q


@torch.inference_mode()
def kmeans(
    vector: pa.FixedShapeTensorArray,
    metric: str,
    codebook_size: int,
    num_codebooks: int,
    batch_size: int,
    num_epochs: int,
) -> Tensor:
    f = torch.compile(torch.vmap(update))

    q = io.arrow.to_torch(
        pc.array_filter(
            vector,
            np.random.permutation(len(vector)) < codebook_size * num_codebooks,
        ).combine_chunks()
    ).view(num_codebooks, codebook_size, -1)

    for _ in range(num_epochs):
        size = num_codebooks * batch_size
        indx = np.random.permutation(len(vector))
        indx = indx[: indx.size // size * size]

        for i in tqdm(np.array_split(indx, indx.size // size)):
            np.put(filter := np.zeros(len(vector), dtype=np.bool_), i, True)

            v = io.arrow.to_torch(
                pc.array_filter(vector, filter).combine_chunks(),
            ).view(num_codebooks, batch_size, -1)

            q = f(v, q, metric=metric)

    return q


@torch.inference_mode()
def encode(x: Tensor, q: Tensor, limit: int, metric: str) -> Tensor:
    n = q.size(0)
    k = q.size(1)

    d = distance(x, q.flatten(end_dim=-2), metric=metric).view(-1, *q.shape[:-1])
    d = sum(
        [
            d[:, j, i]
            for j in range(n)
            for i in [torch.arange(0, k).repeat_interleave(k ** (n - j - 1)).repeat(k**j)]
        ],
        torch.tensor(0),
    )

    return torch.topk(d, limit, largest=False).indices
