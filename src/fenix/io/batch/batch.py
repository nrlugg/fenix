from typing import Callable, Iterable, Iterator, TypeVar

import multiprocess as mp
import psutil

X = TypeVar("X")
Y = TypeVar("Y")


def map(
    func: Callable[[X], Y],
    iter: Iterable[X],
    *,
    num_workers: int | None = None,
    buffer_size: int = 1,
) -> Iterator[Y]:
    num_workers = num_workers if num_workers is not None else psutil.cpu_count(logical=False)

    with mp.Manager() as manager:
        semaphore = manager.Semaphore(num_workers * buffer_size)

        def iterator() -> Iterator[X]:
            for item in iter:
                semaphore.acquire()
                yield item

        with mp.Pool(num_workers) as pool:
            for item in pool.imap_unordered(func, iterator()):
                yield item
                semaphore.release()
