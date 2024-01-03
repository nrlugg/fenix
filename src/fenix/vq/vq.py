from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
from torch import Tensor

ARROW_TO_TORCH_DTYPE: dict[pa.DataType, torch.dtype] = {
    pa.float16(): torch.float16,
    pa.float32(): torch.float32,
    pa.float64(): torch.float64,
}


def arrow_to_torch(
    v: pa.FixedShapeTensorArray | pa.FixedSizeListArray | pa.FixedSizeListScalar
) -> Tensor:
    if isinstance(v, pa.FixedSizeListScalar):
        return torch.from_numpy(v.values.to_numpy())

    if isinstance(v, pa.FixedSizeListArray):
        return torch.from_numpy(v.values.to_numpy()).view(-1, v.type.list_size)

    if isinstance(v, pa.FixedShapeTensorArray):
        return torch.from_numpy(v.to_numpy_ndarray())

    raise TypeError()


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


def sample(data: pa.FixedShapeTensorArray, size: Sequence[int]) -> Tensor:
    mask = np.random.permutation(len(data)) < np.prod(size)
    data = pc.array_filter(data, mask).combine_chunks()
    return arrow_to_torch(data).view(*size, -1)


@torch.inference_mode()
def kmeans(
    x: pa.FixedShapeTensorArray,
    k: int,
    n: int,
    metric: str,
    sample_size: int,
    num_samples: int,
) -> Tensor:
    f = torch.compile(torch.vmap(update))

    q = sample(x, (n, k))

    for _ in range(num_samples):
        v = sample(x, (n, sample_size))
        q = f(v, q, metric=metric)

    return q


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
