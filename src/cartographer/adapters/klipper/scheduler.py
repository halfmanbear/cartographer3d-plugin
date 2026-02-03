from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, final, overload

from typing_extensions import override

from cartographer.interfaces.multiprocessing import Scheduler

if TYPE_CHECKING:
    from reactor import Reactor


@final
class KlipperScheduler(Scheduler):
    """Klipper-specific scheduler using the reactor pattern."""

    def __init__(self, reactor: Reactor) -> None:
        self._reactor = reactor

    @override
    def sleep(self, seconds: float) -> None:
        eventtime = self._reactor.monotonic()
        _ = self._reactor.pause(eventtime + seconds)

    def wait_for_fds(
        self,
        fds: Sequence[int],
        timeout: float | None = None,
    ) -> list[int]:
        """
        Block until at least one fd is readable. Returns list of ready fds.
        Uses reactor's native select() integration - zero CPU while waiting.
        """
        completion = self._reactor.completion()
        ready_fds: list[int] = []
        handles = []

        def make_callback(fd: int):
            def on_readable(eventtime: float) -> None:
                if fd not in ready_fds:
                    ready_fds.append(fd)
                # Complete on first ready fd - wakes the waiting greenlet
                if not completion.test():
                    completion.complete(eventtime)
            return on_readable

        # Register all fds with reactor's select() mechanism
        for fd in fds:
            handle = self._reactor.register_fd(fd, make_callback(fd))
            handles.append(handle)

        try:
            # Block until completion (fd ready) or timeout
            # This yields to reactor which uses select() - no CPU burn
            result = completion.wait(
                self._reactor.monotonic() + timeout if timeout else self._reactor.NEVER,
            )
            # If multiple fds ready simultaneously, check them all
            if result is not None:
                for fd in fds:
                    if fd not in ready_fds:
                        # Additional readability check for race conditions
                        import select
                        r, _, _ = select.select([fd], [], [], 0)
                        if r:
                            ready_fds.append(fd)
            return ready_fds
        finally:
            for handle in handles:
                self._reactor.unregister_fd(handle)

    # Keep wait_until for backward compat, but prefer wait_for_fds
    @overload
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: None = None,
        poll_interval: float = 0.1,
    ) -> None: ...

    @overload
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float,
        poll_interval: float = 0.1,
    ) -> bool: ...

    @override
    def wait_until(
        self,
        condition: Callable[[], bool],
        timeout: float | None = None,
        poll_interval: float = 0.1,
    ) -> bool | None:
        eventtime = self._reactor.monotonic()
        end_time = eventtime + timeout if timeout is not None else float("inf")
        while not condition():
            eventtime = self._reactor.pause(eventtime + poll_interval)
            if eventtime >= end_time:
                return False
        return None if timeout is None else True
