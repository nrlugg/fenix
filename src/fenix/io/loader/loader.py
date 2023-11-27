from typing import Callable, Generic, Iterator, TypeVar, final

import msgspec
import multiprocess as mp
import psutil

X = TypeVar("X")
Y = TypeVar("Y")


def iter_batches(batch_iter: Iterator[X], semaphore: mp.Semaphore) -> Iterator[X]:
    for batch in batch_iter:
        semaphore.acquire()
        yield batch


class Loader(msgspec.Struct, Generic[X]):
    def __iter_batches__(self) -> Iterator[X]:
        raise NotImplementedError()

    @final
    def map_batches(
        self,
        func: Callable[[X], Y],
        num_workers: int | None = None,
        buffer_size: int = 1,
    ) -> Iterator[Y]:
        num_workers = num_workers if num_workers is not None else psutil.cpu_count(logical=False)

        with mp.Manager() as m:
            semaphore = m.Semaphore(num_workers * buffer_size)

            with mp.Pool(num_workers) as p:
                iter = iter_batches(self.__iter_batches__(), semaphore)

                for batch in p.imap_unordered(func, iter):
                    yield batch
                    semaphore.release()
