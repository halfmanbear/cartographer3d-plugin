from __future__ import annotations

import multiprocessing
import os
from typing import TYPE_CHECKING, Callable, TypeVar, final

from typing_extensions import ParamSpec, override

from cartographer.interfaces.multiprocessing import TaskExecutor

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from cartographer.schedulers.klipper import KlipperScheduler

P = ParamSpec("P")
R = TypeVar("R")


def _reset_worker_priority() -> None:
    """
    Reset scheduling to normal priority after fork.
    Klipper runs with RT priority for stepper timing - workers must not inherit this.
    """
    try:
        # Reset to normal scheduling class (SCHED_OTHER, priority 0)
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except (OSError, AttributeError):
        pass

    try:
        # Set positive nice value - lower priority than default
        os.nice(10)
    except OSError:
        pass

    try:
        # Clear any CPU affinity restrictions - let kernel schedule freely
        cpu_count = os.cpu_count() or 1
        os.sched_setaffinity(0, set(range(cpu_count)))
    except (OSError, AttributeError):
        pass


@final
class MultiprocessingExecutor(TaskExecutor):
    """Execute tasks in a separate process."""

    def __init__(self, scheduler: KlipperScheduler) -> None:
        self._scheduler = scheduler

    @override
    def run(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        def worker(child_conn: Connection) -> None:
            # MUST reset priority before any work - we inherited Klipper's RT priority
            _reset_worker_priority()

            try:
                result = fn(*args, **kwargs)
                child_conn.send((False, result))
            except Exception as e:
                child_conn.send((True, e))
            finally:
                child_conn.close()

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(target=worker, args=(child_conn,), daemon=True)
        proc.start()
        child_conn.close()

        pipe_fd = parent_conn.fileno()
        sentinel_fd = proc.sentinel

        ready = self._scheduler.wait_for_fds([pipe_fd, sentinel_fd])

        if pipe_fd in ready:
            try:
                is_error, payload = parent_conn.recv()
            finally:
                parent_conn.close()
                proc.join()
            if is_error:
                raise payload from None
            return payload

        proc.join()
        exit_code = proc.exitcode
        parent_conn.close()
        raise RuntimeError(
            f"Worker process terminated unexpectedly with exit code {exit_code}"
        )
