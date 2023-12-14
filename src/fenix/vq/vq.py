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
        d = 0.5 - 0.5 * d

    if metric == "dot":
        d = -u @ v.transpose(-1, -2)

    return d


def build_quantization(
    x: Tensor,
    k: int,
    n: int = 1,
    f: int = 256,
    metric: str = "l2",
    epochs: int = 50,
) -> Tensor:
    i = torch.randperm(x.size(0))[:k]
    q = x[i]

    for _ in range(epochs):
        i = torch.randperm(x.size(0))[: k * f]
        v = x[i]

        if metric == "cosine":
            q = F.normalize(q, dim=-1)
            v = F.normalize(v, dim=-1)

        d = distance(v, q, metric=metric)
        i = torch.argmin(d, dim=-1)

        q = torch.scatter_reduce(
            q,
            dim=0,
            index=i.unsqueeze(-1).repeat_interleave(x.size(-1), dim=-1),
            src=v,
            reduce="mean",
        )

        if metric == "cosine":
            q = F.normalize(q, dim=-1)

    q = torch.unsqueeze(q, dim=0)

    if n > 1:
        q = torch.concatenate(
            [
                q,
                build_quantization(x, k, n - 1, f, metric, epochs),
            ],
            dim=0,
        )

    return q


def apply_quantization(x: Tensor, q: Tensor, k: int, metric: str = "l2") -> Tensor:
    n = q.size(0)
    d = torch.stack(
        [distance(x, q[i], metric=metric) for i in range(n)],
        dim=0,
    )

    dim = (n - 1) * [None]
    s = d[0, ..., *dim]

    for i in range(1, n):
        l = i * [None]
        r = (n - i - 1) * [None]

        s = s + d[i, :, *l, :, *r]

    return torch.topk(
        s.flatten(1),
        k=k,
        dim=-1,
        largest=False,
    ).indices
