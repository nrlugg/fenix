from typing import Type, Union

import jmespath
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from fenix.ex.arrow.tensor.tensor import TensorArray, TensorScalar, TensorType

NumpyNestedTensor = dict[str, Union[np.ndarray, "NumpyNestedTensor"]]
TorchNestedTensor = dict[str, Union[Tensor, "TorchNestedTensor"]]
NestedTensorShape = dict[str, Union[tuple[int, ...], "NestedTensorShape"]]


class NestedTensorType(pa.ExtensionType):
    def __init__(self, dtype: pa.DataType, shape: NestedTensorShape) -> None:
        self.shape = shape
        super().__init__(dtype, "NestedTensor")

    def __arrow_ext_serialize__(self) -> bytes:
        return msgspec.msgpack.encode(self.shape)

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "TensorType":
        shape = msgspec.msgpack.decode(serialized)
        return NestedTensorType(storage_type, shape)

    def __arrow_ext_class__(self) -> Type["NestedTensorArray"]:
        return NestedTensorArray

    def __arrow_ext_scalar_class__(self) -> Type["NestedTensorScalar"]:
        return NestedTensorScalar


class NestedTensorArray(pa.ExtensionArray):
    @staticmethod
    def from_numpy(NestedTensor: NumpyNestedTensor) -> "NestedTensorArray":
        def to_shapes(NestedTensor: NumpyNestedTensor) -> NestedTensorShape:
            return {
                k: to_shapes(v) if isinstance(v, dict) else tuple(v.shape[1:])
                for k, v in NestedTensor.items()
            }

        def to_struct(NestedTensor: NumpyNestedTensor) -> pa.StructArray:
            return pa.StructArray.from_arrays(
                [
                    pa.array(
                        list(t.reshape(t.shape[0], -1)),
                        type=pa.list_(
                            pa.from_numpy_dtype(t.dtype),
                            np.prod(t.shape[1:]).item(),
                        ),
                    )
                    if isinstance(t, np.ndarray)
                    else to_struct(t)
                    for t in NestedTensor.values()
                ],
                names=list(NestedTensor),
            )

        shapes = to_shapes(NestedTensor)
        struct = to_struct(NestedTensor)
        return pa.ExtensionArray.from_storage(
            NestedTensorType(struct.type, shapes),
            struct,
        )

    @staticmethod
    def from_torch(NestedTensor: TorchNestedTensor) -> "NestedTensorArray":
        def to_numpy(NestedTensor: TorchNestedTensor) -> NumpyNestedTensor:
            return {
                k: t.detach().numpy() if isinstance(t, Tensor) else to_numpy(t)
                for k, t in NestedTensor.items()
            }

        return NestedTensorArray.from_numpy(to_numpy(NestedTensor))

    def to_numpy(self) -> NumpyNestedTensor:
        def to_numpy(array: pa.StructArray, shape: NestedTensorShape) -> NumpyNestedTensor:
            return {
                k: (
                    to_numpy(a, shape)
                    if type(a.type) == pa.StructType and isinstance(shape, dict)
                    else a.flatten().to_numpy(zero_copy_only=False).reshape(-1, *shape)
                )
                for k, shape in shape.items()
                for a in [pc.struct_field(array, k)]
            }

        return to_numpy(self.storage, self.type.shape)

    def to_torch(self) -> TorchNestedTensor:
        def from_numpy(NestedTensor: NumpyNestedTensor) -> TorchNestedTensor:
            return {
                k: torch.from_numpy(t) if isinstance(t, np.ndarray) else from_numpy(t)
                for k, t in NestedTensor.items()
            }

        return from_numpy(self.to_numpy())

    def to_field(self, *keys) -> Union["NestedTensorArray", "TensorArray"]:
        array = pc.struct_field(self.storage, keys)
        shape = jmespath.search(".".join(keys), self.type.shape)

        if isinstance(shape, dict):
            return pa.ExtensionArray.from_storage(
                NestedTensorType(array.type, shape),
                array,
            )

        if isinstance(shape, (tuple, list)):
            return pa.ExtensionArray.from_storage(
                TensorType(array.type, list(shape)),
                array,
            )

        raise ValueError()


class NestedTensorScalar(pa.ExtensionScalar):
    def to_numpy(self) -> NumpyNestedTensor:
        def to_numpy(array: pa.StructScalar, shape: NestedTensorShape) -> NumpyNestedTensor:
            return {
                k: (
                    to_numpy(a, shape)
                    if type(a.type) == pa.StructType and isinstance(shape, dict)
                    else a.values.to_numpy(zero_copy_only=False).reshape(*shape)
                )
                for k, shape in shape.items()
                for a in [pc.struct_field(array, k)]
            }

        return to_numpy(self.value, self.type.shape)

    def to_torch(self) -> TorchNestedTensor:
        def from_numpy(NestedTensor: NumpyNestedTensor) -> TorchNestedTensor:
            return {
                k: torch.tensor(t) if isinstance(t, np.ndarray) else from_numpy(t)
                for k, t in NestedTensor.items()
            }

        return from_numpy(self.to_numpy())

    def to_field(self, *keys) -> Union["NestedTensorScalar", "TensorScalar"]:
        array = pc.struct_field(self.value, keys)
        shape = jmespath.search(".".join(keys), self.type.shape)

        if isinstance(shape, dict):
            return pa.ExtensionScalar.from_storage(
                NestedTensorType(array.type, shape),
                array,
            )

        if isinstance(shape, (tuple, list)):
            return pa.ExtensionScalar.from_storage(
                TensorType(array.type, list(shape)),
                array,
            )

        raise ValueError()


def from_numpy(tensor: NumpyNestedTensor) -> NestedTensorArray:
    return NestedTensorArray.from_numpy(tensor)


def from_torch(tensor: TorchNestedTensor) -> NestedTensorArray:
    return NestedTensorArray.from_torch(tensor)
