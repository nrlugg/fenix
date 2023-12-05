import torch
import torch.nn.functional as F
from torch import Tensor


def distance(u: Tensor, v: Tensor, metric: str = "l2") -> Tensor:
    if metric == "l2":
        d = torch.cdist(u, v)

    if metric == "cosine":
        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)
        d = u @ v.transpose(-1, -2)
        d = 0.5 * (1 + d)

    if metric == "dot":
        d = -u @ v.transpose(-1, -2)

    return d


def build_quantization(
    x: Tensor,
    k: int = 256,
    p: int = 1,
    f: int = 256,
    metric: str = "l2",
    epochs: int = 50,
) -> Tensor:
    if p > 1 and metric == "cosine":
        raise ValueError("Cannot use product quantization (p > 1) with 'cosine' metric")

    x = x.view(x.size(0), p, x.size(-1) // p)
    x = x.transpose(0, 1)

    i = torch.randperm(x.size(1))[:k]
    q = x[:, i]

    for _ in range(epochs):
        i = torch.randperm(x.size(1))[: k * f]
        v = x[:, i]

        d = distance(v, q, metric=metric)

        i = torch.arange(p).unsqueeze(-1)
        j = torch.argmin(d, dim=-1)
        q = torch.index_put(q, (i, j), v, accumulate=True)

        n = j[:, None, :] == torch.arange(k)[None, :, None]
        n = torch.sum(n, dim=-1, keepdim=True) + 1

        q = q / n

    return q


def apply_quantization(x: Tensor, q: Tensor, metric: str = "l2") -> Tensor:
    p = q.size(0)
    k = q.size(1)

    x = x.view(x.size(0), p, x.size(-1) // p)
    x = x.transpose(0, 1)

    i = torch.argmin(distance(x, q, metric=metric), dim=-1)

    return torch.arange(k**p).reshape(p * (k,))[i.unbind(0)]
