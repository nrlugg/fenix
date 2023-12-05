from typing import Sequence, Type

import msgspec
import numpy as np
import pyarrow as pa
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor


class QUInt8NDArray(np.ndarray):
    scale: float
    shift: int

    def __new__(cls, array: NDArray[np.uint8], scale: float, shift: int) -> "QUInt8NDArray":
        q = array.view(cls)
        q.scale = scale
        q.shift = shift

        return q

    @staticmethod
    def from_torch(tensor: Tensor) -> "QUInt8NDArray":
        if not tensor.is_quantized:
            tensor = torch.quantize_per_tensor_dynamic(
                tensor,
                dtype=torch.quint8,
                reduce_range=True,
            )

        return QUInt8NDArray(
            tensor.int_repr().numpy(),
            tensor.q_scale(),
            tensor.q_zero_point(),
        )

    @staticmethod
    def from_float(array: ArrayLike) -> "QUInt8NDArray":
        return QUInt8NDArray.from_torch(
            torch.from_numpy(
                np.asarray(array, dtype=np.float32),
            )
        )

    def to_float(self) -> NDArray[np.float32]:
        return self.scale * (self.astype(np.float32) - self.shift)

    def to_torch(self) -> Tensor:
        return torch._make_per_tensor_quantized_tensor(
            torch.from_numpy(self),
            self.scale,
            self.shift,
        )


class QUInt8TensorType(pa.ExtensionType):
    def __init__(self, shape: Sequence[int], scale: float, shift: int) -> None:
        STORAGE_TYPE: pa.DataType = pa.list_(pa.uint8(), np.prod(shape))

        self.shape = tuple(shape)
        self.scale = scale
        self.shift = shift

        super().__init__(STORAGE_TYPE, "tensor::qint8")

    def __arrow_ext_serialize__(self) -> bytes:
        return msgspec.msgpack.encode(
            {
                "shape": self.shape,
                "scale": self.scale,
                "shift": self.shift,
            }
        )

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "QUInt8TensorType":
        return QUInt8TensorType(
            **msgspec.msgpack.decode(serialized),
        )

    def __arrow_ext_class__(self) -> Type["QUInt8TensorArray"]:
        return QUInt8TensorArray

    def __arrow_ext_scalar_class__(self) -> Type["QUInt8TensorScalar"]:
        return QUInt8TensorScalar


class QUInt8TensorArray(pa.ExtensionArray):
    @staticmethod
    def from_torch(tensor: Tensor) -> "QUInt8TensorArray":
        if not tensor.is_quantized:
            tensor = torch.quantize_per_tensor_dynamic(
                tensor,
                dtype=torch.quint8,
                reduce_range=True,
            )

        BATCH = tensor.shape[0]
        SHAPE = tensor.shape[1:]
        SCALE = tensor.q_scale()
        SHIFT = tensor.q_zero_point()

        child = pa.Array.from_buffers(
            pa.uint8(),
            tensor.numel(),
            [None, pa.py_buffer(tensor.int_repr().numpy())],
        )

        array = pa.Array.from_buffers(
            pa.list_(pa.uint8(), np.prod(SHAPE)),
            BATCH,
            [None],
            children=[child],
        )

        return pa.ExtensionArray.from_storage(
            QUInt8TensorType(SHAPE, SCALE, SHIFT),
            array,
        )

    @staticmethod
    def from_numpy(tensor: np.ndarray) -> "QUInt8TensorArray":
        if isinstance(tensor, QUInt8NDArray):
            return QUInt8TensorArray.from_torch(
                tensor.to_torch(),
            )

        return QUInt8TensorArray.from_float(tensor)

    def to_torch(self) -> Tensor:
        *_, buff = self.buffers()
        return torch._make_per_tensor_quantized_tensor(
            torch.frombuffer(buff, dtype=torch.uint8).view(len(self), *self.type.shape),
            self.type.scale,
            self.type.shift,
        )

    def to_numpy(self) -> QUInt8NDArray:
        return QUInt8NDArray.from_torch(
            self.to_torch(),
        )


class QUInt8TensorScalar(pa.ExtensionScalar):
    @staticmethod
    def from_torch(tensor: Tensor) -> "QUInt8TensorScalar":
        if not tensor.is_quantized:
            tensor = torch.quantize_per_tensor_dynamic(
                tensor,
                dtype=torch.quint8,
                reduce_range=True,
            )

        SHAPE = tensor.shape
        SCALE = tensor.q_scale()
        SHIFT = tensor.q_zero_point()

        value = pa.scalar(
            tensor.int_repr().view(-1).numpy(),
            pa.list_(pa.uint8(), np.prod(SHAPE)),
        )

        return pa.ExtensionScalar.from_storage(
            QUInt8TensorType(SHAPE, SCALE, SHIFT),
            value,
        )

    @staticmethod
    def from_numpy(self, tensor: np.ndarray) -> "QUInt8TensorScalar":
        if isinstance(tensor, QUInt8NDArray):
            return QUInt8TensorScalar.from_torch(
                tensor.to_torch(),
            )

        return QUInt8TensorScalar.from_torch(
            torch.from_numpy(tensor),
        )

    def to_torch(self) -> Tensor:
        offset = self.value.values.offset
        *_, buff = self.value.values.buffers()
        buff = buff[offset : offset + np.prod(self.type.shape)]
        return torch._make_per_tensor_quantized_tensor(
            torch.frombuffer(buff, dtype=torch.uint8).view(*self.type.shape),
            self.type.scale,
            self.type.shift,
        )

    def to_numpy(self) -> QUInt8NDArray:
        return QUInt8NDArray(
            self.value.values.to_numpy().reshape(*self.type.shape),
            self.scale,
            self.shift,
        )


def from_numpy(tensor: np.ndarray) -> QUInt8TensorArray:
    return QUInt8TensorArray.from_numpy(tensor)


def from_torch(tensor: Tensor) -> QUInt8TensorArray:
    return QUInt8TensorArray.from_torch(tensor)
