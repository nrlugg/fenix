from typing import Callable, Iterator, TypeVar

import multiprocess as mp
import psutil

X = TypeVar("X")
Y = TypeVar("Y")


def iter_batches(batch_iter: Iterator[X], semaphore: mp.Semaphore) -> Iterator[X]:
    for batch in batch_iter:
        semaphore.acquire()
        yield batch


def map(
    func: Callable[[X], Y],
    iter: Iterator[X] | Callable[[], Iterator[X]],
    *,
    num_workers: int | None = None,
    buffer_size: int = 2,
) -> Iterator[Y]:
    if callable(iter):
        iter = iter()

    num_workers = num_workers if num_workers is not None else psutil.cpu_count(logical=False)

    with mp.Manager() as m:
        semaphore = m.Semaphore(num_workers * buffer_size)

        with mp.Pool(num_workers) as p:
            for batch in p.imap_unordered(func, iter_batches(iter, semaphore)):
                yield batch
                semaphore.release()
