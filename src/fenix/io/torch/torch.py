import pyarrow as pa
import torch
from torch import Tensor


def from_arrow(x: pa.FixedSizeListArray | pa.FixedSizeListScalar) -> Tensor:
    if isinstance(x, pa.FixedSizeListScalar):
        return torch.from_numpy(x.values.to_numpy())

    return torch.from_numpy(x.values.to_numpy()).view(-1, x.type.list_size)
