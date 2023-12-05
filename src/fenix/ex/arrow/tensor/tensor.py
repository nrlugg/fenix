from typing import Sequence, Type

import msgspec
import numpy as np
import pyarrow as pa
import torch
from torch import Tensor


class TensorType(pa.ExtensionType):
    def __init__(self, dtype: pa.DataType, shape: Sequence[int]) -> None:
        self.shape = tuple(shape)
        super().__init__(dtype, "tensor")

    def __arrow_ext_serialize__(self) -> bytes:
        return msgspec.msgpack.encode(self.shape)

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "TensorType":
        shape = msgspec.msgpack.decode(serialized)
        return TensorType(storage_type, shape)

    def __arrow_ext_class__(self) -> Type["TensorArray"]:
        return TensorArray

    def __arrow_ext_scalar_class__(self) -> Type["TensorScalar"]:
        return TensorScalar


class TensorArray(pa.ExtensionArray):
    @staticmethod
    def from_numpy(tensor: np.ndarray) -> "TensorArray":
        dtype = pa.from_numpy_dtype(tensor.dtype)
        num_rows, *shape = tensor.shape
        storage_data = tensor.reshape(num_rows, -1)
        storage_type = pa.list_(dtype, storage_data.shape[-1])
        return pa.ExtensionArray.from_storage(
            TensorType(storage_type, shape),
            pa.array(list(storage_data), storage_type),
        )

    @staticmethod
    def from_torch(tensor: Tensor) -> "TensorArray":
        return TensorArray.from_numpy(
            tensor.detach().cpu().numpy(),
        )

    def to_numpy(self) -> np.ndarray:
        return self.storage.flatten().to_numpy(zero_copy_only=False).reshape(-1, *self.type.shape)

    def to_torch(self) -> Tensor:
        return torch.from_numpy(self.to_numpy())


class TensorScalar(pa.ExtensionScalar):
    @staticmethod
    def from_numpy(tensor: np.ndarray) -> "TensorScalar":
        dtype = pa.from_numpy_dtype(tensor.dtype)
        shape = tuple(tensor.shape)
        tensor = tensor.reshape(-1)
        scalar = pa.scalar(tensor, pa.list_(dtype, tensor.shape[-1]))
        return pa.ExtensionScalar.from_storage(
            TensorType(scalar.type, shape),
            scalar,
        )

    @staticmethod
    def from_torch(tensor: Tensor) -> "TensorScalar":
        return TensorScalar.from_numpy(
            tensor.detach().cpu().numpy(),
        )

    def to_numpy(self) -> np.ndarray:
        return self.value.values.to_numpy(zero_copy_only=False).reshape(*self.type.shape)

    def to_torch(self) -> Tensor:
        return torch.from_numpy(self.to_numpy())


def from_numpy(tensor: np.ndarray) -> TensorArray:
    return TensorArray.from_numpy(tensor)


def from_torch(tensor: Tensor) -> TensorArray:
    return TensorArray.from_torch(tensor)
