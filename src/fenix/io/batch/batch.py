from typing import Callable, Iterator, TypeVar

import numpy as np
import psutil
import pyarrow as pa
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader, IterableDataset

import fenix.io.table

X = TypeVar("X")
Y = TypeVar("Y")


@dataclass(frozen=True)
class RandomBatchIterator(IterableDataset):
    root: str
    name: str | list[str]
    size: int

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        table = fenix.io.table.load(self.root, self.name)

        index = np.random.permutation(table.num_rows)
        index = index[: table.num_rows // self.size * self.size]

        for rowids in np.array_split(index, index.size // self.size):
            filter = np.zeros(table.num_rows, dtype=np.bool_)
            np.put(filter, rowids, True)

            yield table.filter(filter)


def imap(
    root: str,
    name: str | list[str],
    size: int,
    func: Callable[[pa.RecordBatch], pa.RecordBatch] | None = None,
) -> Iterator[pa.RecordBatch]:
    yield from DataLoader(
        RandomBatchIterator(root, name, size),
        batch_size=None,
        shuffle=False,
        num_workers=psutil.cpu_count(logical=False),
        collate_fn=func,
    )
