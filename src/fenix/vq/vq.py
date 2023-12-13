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
    k: int,
    n: int = 1,
    f: int = 256,
    metric: str = "l2",
    epochs: int = 50,
) -> Tensor:
    x = x.dequantize()

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
        q = torch.index_put_(q, (i,), v, accumulate=True)

        if metric == "cosine":
            q = F.normalize(q, dim=-1)

        else:
            c = i[None, :] == torch.arange(k)[:, None]
            c = torch.sum(c, dim=-1, keepdim=True) + 1

            q = q / c

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


def apply_quantization(x: Tensor, q: Tensor, metric: str = "l2") -> Tensor:
    x = x.dequantize()
    q = q.dequantize()
    n = q.size(0)

    d = distance(x, q[0], metric=metric).unsqueeze(0)

    if n > 1:
        d = torch.concatenate(
            [
                d,
                apply_quantization(x, q[1:], metric=metric),
            ],
            dim=0,
        )

    return d


def index_quantization(d: Tensor, k: int, metric: str = "l2") -> Tensor:
    n = d.size(0)

    dim = (n - 1) * [None]
    s = d[0, ..., *dim]

    for i in range(1, n):
        l = i * [None]
        r = (n - i - 1) * [None]

        s = s + d[i, :, *l, :, *r]

    return torch.topk(
        torch.flatten(s, start_dim=1),
        k=k,
        dim=-1,
        largest=False,
    ).indices
