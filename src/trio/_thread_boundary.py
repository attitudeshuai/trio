"""The single execution boundary used for pushing synchronous work into threads.

Every place in Trio that needs to run a blocking, synchronous function in a
worker thread goes through a `ThreadBoundary`:

* :func:`trio.to_thread.run_sync` (the public entry point),
* the file I/O wrappers in :mod:`trio._file_io`,
* subprocess exit waiting/cleanup (``waitid`` threads and the Windows
  ``WaitForSingleObject`` wrapper),
* and the other internal call sites (``trio.open_file``, ``trio.Path``,
  ``trio.socket`` name resolution, ``trio.lowlevel.open_process``, ...).

The boundary declares:

* a concurrency *capacity* (how many items may run at once),
* a *queue capacity* (how many items may wait for a slot),
* an *overflow policy* (`OverflowPolicy.WAIT` or `OverflowPolicy.REJECT`)
  deciding what happens when both the slots and the queue are full,
* optional *source* tags, which make the queue fair across sources: the same
  source submitting thousands of items cannot indefinitely starve the other
  sources – within a source items still run in FIFO order, across sources a
  free slot is handed out round-robin.

It also tracks precise accounting on every path (cancellation while queued,
cancellation while running, worker exceptions, shutdown), exposes statistics
(occupancy, queue length, rejection count and per-source waiting times) and
supports an explicit `~ThreadBoundary.shutdown` with either draining or
fast-fail semantics that reports the items that did not finish instead of
silently dropping them.
"""

from __future__ import annotations

import contextlib
import enum
import math
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

import attrs
import outcome

from . import _core
from ._core import (
    Abort,
    BusyResourceError,
    ClosedResourceError,
    RaiseCancelT,
    enable_ki_protection,
)
from ._sync import AsyncContextManagerMixin, Event
from ._util import final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ._core import Task

# Default concurrency, kept identical to the historical default thread
# limiter. The default queue is unbounded and overflowing callers wait, so
# declaring nothing reproduces the previous behavior exactly.
DEFAULT_CAPACITY = 40


class OverflowPolicy(str, enum.Enum):
    """What a boundary does when every slot is taken and the queue is full.

    Attributes:
      WAIT: The caller still waits for a slot, in declared (FIFO per source)
        order. Nothing is ever rejected and ``queue_capacity`` does not cause
        submissions to fail. This is the default.
      REJECT: The caller is rejected immediately with
        `ThreadBoundaryRejected`, instead of waiting.
    """

    WAIT = "wait"
    REJECT = "reject"


class ThreadBoundaryRejected(BusyResourceError):
    """Raised when a submission cannot get a slot or a queue position.

    This is raised by `ThreadBoundary.submit_on_behalf_of` (and thus by
    :func:`trio.to_thread.run_sync`) when the boundary's capacity and queue
    are both exhausted and the overflow policy is
    `~OverflowPolicy.REJECT`.

    It is distinct from the ordinary cancellation/blocking exceptions so
    callers can tell "the boundary was full and refused me" apart from "my
    own work raised" or "I was cancelled". It is a subclass of
    :exc:`trio.BusyResourceError` because it is the "would block" style of
    failure.
    """


class ThreadBoundaryClosed(ClosedResourceError):
    """Raised when work is submitted to a shut-down boundary.

    Also raised inside tasks that were still queued when
    :meth:`ThreadBoundary.shutdown` was called with ``drain=False``: those
    items are failed fast and included in the returned
    `ThreadBoundaryShutdown` report.
    """


@attrs.frozen
class SourceBoundaryStatistics:
    """Statistics for a single *source* tag.

    Attributes:
      source: The tag the items were submitted with (``None`` if untagged).
      in_use: Number of items from this source currently running.
      waiting: Number of items from this source currently queued.
      wait_seconds: How long each currently queued item has been waiting,
        oldest first, in Trio clock seconds.
      completed_wait_count: How many queued items from this source have
        already acquired a slot.
      total_wait_seconds: Total queueing time of the completed items.
      min_wait_seconds: Shortest observed queueing time, or ``None``.
      max_wait_seconds: Longest observed queueing time, or ``None``.
    """

    source: object
    in_use: int
    waiting: int
    wait_seconds: tuple[float, ...]
    completed_wait_count: int
    total_wait_seconds: float
    min_wait_seconds: float | None
    max_wait_seconds: float | None


@attrs.frozen
class ThreadBoundaryStatistics:
    """Snapshot of a `ThreadBoundary`'s state.

    Besides the new fields below, this also carries the four historical
    ``CapacityLimiter`` field names (``borrowed_tokens``, ``total_tokens``,
    ``borrowers`` and ``tasks_waiting``), so statistics readers written
    against the default thread limiter keep working unchanged.

    Attributes:
      capacity: Maximum number of items that may run at once.
      in_use: Number of items currently running.
      queue_capacity: Maximum number of items that may wait at once.
      waiting: Number of items currently queued.
      rejected: Total number of submissions refused with
        `ThreadBoundaryRejected` due to the overflow policy.
      closed: Whether the boundary has been shut down.
      sources: Per-source statistics, keyed by the source tag.
      borrowers: Opaque identities that currently hold a slot.
    """

    capacity: int | float
    in_use: int
    queue_capacity: int | float
    waiting: int
    rejected: int
    closed: bool
    sources: dict[object, SourceBoundaryStatistics]
    borrowers: list[object]
    # Backwards-compatible CapacityLimiterStatistics aliases.
    total_tokens: int | float
    tasks_waiting: int

    @property
    def borrowed_tokens(self) -> int:
        """Alias for `in_use`."""
        return self.in_use


@final
@attrs.define(eq=False)
class PendingThreadWork:  # type: ignore[explicit-any]
    """Description of a unit of work submitted to a `ThreadBoundary`.

    Instances are used in the `ThreadBoundaryShutdown` report. They are
    mutable while the boundary tracks the work (``state`` transitions from
    ``"waiting"`` to ``"running"``).

    Attributes:
      fn: The synchronous function that was submitted.
      args: The positional arguments it was submitted with.
      source: The source tag it was submitted under.
      state: ``"waiting"`` if queued, ``"running"`` if it holds a slot.
      enqueued_at: Trio clock time at which the item entered the boundary.
      started_at: Trio clock time at which the item acquired a slot, or
        ``None`` while still queued.
    """

    fn: Callable[..., object]  # type: ignore[explicit-any]
    args: tuple[object, ...]
    source: object
    state: str = "waiting"
    enqueued_at: float = 0.0
    started_at: float | None = None

    @property
    def is_running(self) -> bool:
        """Whether this item currently occupies a slot."""
        return self.state == "running"


@final
@attrs.frozen
class ThreadBoundaryShutdown:
    """Result of `ThreadBoundary.shutdown`.

    Attributes:
      drained: ``True`` if the call waited for every item to finish,
        ``False`` if queued items were failed fast.
      unfinished: Items that had not finished when shutdown returned. With
        ``drain=True`` this is always empty. With ``drain=False`` it contains
        the items that were still queued (failed with
        `ThreadBoundaryClosed`) followed by the items that were still
        running (threads cannot be killed, so those are reported rather
        than silently discarded).
    """

    drained: bool
    unfinished: tuple[PendingThreadWork, ...]

    @property
    def running(self) -> tuple[PendingThreadWork, ...]:
        """The items that were still running at shutdown."""
        return tuple(w for w in self.unfinished if w.state == "running")

    @property
    def waiting(self) -> tuple[PendingThreadWork, ...]:
        """The items that were still queued at shutdown."""
        return tuple(w for w in self.unfinished if w.state == "waiting")


@attrs.define(eq=False)
class _Ticket:
    # The task parked while this ticket waits for a slot.
    task: Task
    # Opaque borrower identity handed back on release (e.g. ThreadPlaceholder).
    borrower: object
    source: object
    enqueued_at: float
    work: PendingThreadWork | None


def _validate_capacity(value: float) -> None:
    if not isinstance(value, int) and value is not math.inf:
        raise TypeError("capacity must be an int or math.inf")
    if value < 0:
        raise ValueError("capacity must be >= 0")


@final
class ThreadBoundary(AsyncContextManagerMixin):
    """Declarable execution boundary for running sync work in threads.

    This is the richer successor to a plain `~trio.CapacityLimiter` for the
    worker-thread machinery. It is deliberately a drop-in: it implements the
    same ``acquire_*``/``release_*`` surface, so it can be passed as the
    ``limiter=`` argument to :func:`trio.to_thread.run_sync`.

    Args:
      capacity: Maximum number of items that may run at once. Defaults to 40,
        matching Trio's historical default thread limit.
      queue_capacity: Maximum number of items that may wait for a slot. When
        the queue is full too, ``overflow`` decides whether further
        submissions wait anyway or are rejected. Defaults to unbounded.
      overflow: `OverflowPolicy.WAIT`` (the default) or
        `OverflowPolicy.REJECT`.

    Source fairness: submit with a ``source`` tag (via
    :func:`trio.to_thread.run_sync`'s ``source`` argument or
    :meth:`submit_on_behalf_of`). Items within one source start in FIFO
    order, and a freed slot is offered to sources round-robin, so a source
    that flooded the queue cannot keep later-arriving sources waiting
    forever.

    Shutdown: `shutdown` closes the boundary. New submissions then raise
    `ThreadBoundaryClosed`. With ``drain=True`` (the default) it waits for
    every running and queued item to finish; with ``drain=False`` queued
    items are failed immediately and the still-running ones are reported
    back (Python cannot forcibly kill a thread).
    """

    def __init__(
        self,
        capacity: float = DEFAULT_CAPACITY,
        *,
        queue_capacity: float = math.inf,
        overflow: OverflowPolicy | str = OverflowPolicy.WAIT,
    ) -> None:
        _validate_capacity(capacity)
        _validate_capacity(queue_capacity)
        self._capacity: int | float = capacity
        self._queue_capacity: int | float = queue_capacity
        self.overflow = overflow

        # borrower -> source, for every item currently holding a slot
        self._borrowers: dict[object, object] = {}
        # borrower -> PendingThreadWork for items carrying a description
        self._work: dict[object, PendingThreadWork] = {}
        # source -> number of running items
        self._running_per_source: dict[object, int] = {}
        # source -> FIFO deque of parked tickets; key order is the
        # round-robin order (leftmost source is served next, then rotated
        # to the end).
        self._queued: OrderedDict[object, deque[_Ticket]] = OrderedDict()
        # source -> [completed count, total wait, min wait, max wait]
        self._wait_stats: dict[object, list[float]] = {}
        self._rejected: int = 0
        self._closed: bool = False
        self._shutdown_report: ThreadBoundaryShutdown | None = None
        # Set whenever both the running set and the queue transition to
        # empty. Recreated after firing, mirroring _track_active_thread in
        # _threads. Starts unset: it only becomes interesting after work has
        # been submitted (and shutdown only waits on it while non-empty).
        self._idle_event = Event()

    def __repr__(self) -> str:
        return (
            f"<trio.to_thread.ThreadBoundary at {id(self):#x}, "
            f"{len(self._borrowers)}/{self._capacity} running, "
            f"{self._waiting_count()} waiting, "
            f"{self._rejected} rejected>"
        )

    ################
    # Declarations
    ################

    @property
    def capacity(self) -> int | float:
        """The total number of slots; may be an int or ``math.inf``."""
        return self._capacity

    @capacity.setter
    def capacity(self, new_capacity: float) -> None:
        _validate_capacity(new_capacity)
        self._capacity = new_capacity
        self._dispatch()

    @property
    def total_tokens(self) -> int | float:
        """Historical alias for `capacity`."""
        return self._capacity

    @total_tokens.setter
    def total_tokens(self, new_total_tokens: float) -> None:
        self.capacity = new_total_tokens

    @property
    def queue_capacity(self) -> int | float:
        """Maximum number of items that may queue waiting for a slot."""
        return self._queue_capacity

    @queue_capacity.setter
    def queue_capacity(self, value: float) -> None:
        _validate_capacity(value)
        self._queue_capacity = value
        self._dispatch()

    @property
    def overflow(self) -> OverflowPolicy:
        """Policy used when slots and queue positions are exhausted."""
        return self._overflow

    @overflow.setter
    def overflow(self, value: OverflowPolicy | str) -> None:
        self._overflow = OverflowPolicy(value)

    @property
    def closed(self) -> bool:
        """Whether `shutdown` has been called."""
        return self._closed

    ################
    # Introspection / statistics
    ################

    @property
    def borrowed_tokens(self) -> int:
        """Number of slots currently in use."""
        return len(self._borrowers)

    @property
    def available_tokens(self) -> int | float:
        """Number of free slots."""
        return self._capacity - len(self._borrowers)

    @property
    def rejected(self) -> int:
        """How many submissions were refused due to overflow."""
        return self._rejected

    def _waiting_count(self) -> int:
        # Sums are at most "number of parked tasks", so this is cheap enough.
        return sum(len(dq) for dq in self._queued.values())

    def statistics(self) -> ThreadBoundaryStatistics:
        """Return an immutable snapshot of the boundary's statistics.

        See `ThreadBoundaryStatistics` and `SourceBoundaryStatistics` for
        the available fields. The snapshot includes current occupancy,
        queue length, the cumulative rejection count and waiting times
        broken down by source tag, so an operator can tell a capacity
        shortage apart from one source hogging the boundary.
        """
        now = _core.current_time()
        sources: dict[object, SourceBoundaryStatistics] = {}
        all_sources = (
            self._running_per_source.keys()
            | self._queued.keys()
            | self._wait_stats.keys()
        )
        for source in all_sources:
            dq = self._queued.get(source)
            wait_seconds = (
                tuple(now - ticket.enqueued_at for ticket in dq) if dq else ()
            )
            summary = self._wait_stats.get(source)
            if summary is None:
                completed_count = 0
                total_wait = 0.0
                min_wait: float | None = None
                max_wait: float | None = None
            else:
                completed_count = int(summary[0])
                total_wait = summary[1]
                min_wait = summary[2]
                max_wait = summary[3]
            sources[source] = SourceBoundaryStatistics(
                source=source,
                in_use=self._running_per_source.get(source, 0),
                waiting=len(dq) if dq else 0,
                wait_seconds=wait_seconds,
                completed_wait_count=completed_count,
                total_wait_seconds=total_wait,
                min_wait_seconds=min_wait,
                max_wait_seconds=max_wait,
            )
        waiting = self._waiting_count()
        return ThreadBoundaryStatistics(
            capacity=self._capacity,
            in_use=len(self._borrowers),
            queue_capacity=self._queue_capacity,
            waiting=waiting,
            rejected=self._rejected,
            closed=self._closed,
            sources=sources,
            borrowers=list(self._borrowers),
            total_tokens=self._capacity,
            tasks_waiting=waiting,
        )

    ################
    # Acquire / release (CapacityLimiter-compatible surface)
    ################

    @enable_ki_protection
    def acquire_nowait(self) -> None:
        """Acquire a slot on behalf of the current task without blocking.

        Raises:
          WouldBlock: If no slot is immediately available.
          ThreadBoundaryClosed: If the boundary was shut down.
        """
        self.acquire_on_behalf_of_nowait(_core.current_task())

    @enable_ki_protection
    def acquire_on_behalf_of_nowait(self, borrower: object) -> None:
        """Acquire a slot on behalf of ``borrower`` without blocking."""
        if self._closed:
            raise ThreadBoundaryClosed("this thread boundary has been shut down")
        if borrower in self._borrowers:
            raise RuntimeError(
                "this borrower is already holding one of this ThreadBoundary's slots",
            )
        if len(self._borrowers) < self._capacity:
            self._borrowers[borrower] = None
            self._running_per_source[None] = self._running_per_source.get(None, 0) + 1
        else:
            raise _core.WouldBlock()

    @enable_ki_protection
    async def acquire(self) -> None:
        """Acquire a slot, waiting fairly if none is available."""
        await self.acquire_on_behalf_of(_core.current_task())

    @enable_ki_protection
    async def acquire_on_behalf_of(self, borrower: object) -> None:
        """Acquire a slot on behalf of ``borrower``.

        Untagged submissions participate in the round-robin under the
        ``None`` source.
        """
        await self.submit_on_behalf_of(borrower)

    @enable_ki_protection
    async def submit_on_behalf_of(  # type: ignore[explicit-any, misc]
        self,
        borrower: object,
        *,
        source: object = None,
        work: tuple[Callable[..., object], tuple[object, ...]] | None = None,
    ) -> None:
        """Acquire a slot on behalf of ``borrower`` with boundary declarations.

        Args:
          borrower: Opaque identity recorded as holding the slot; passed
            back unchanged to `release_on_behalf_of`.
          source: Optional tag used for fair scheduling and statistics.
          work: Optional ``(fn, args)`` pair describing the work that will
            run while the slot is held; it only feeds shutdown reports.

        Raises:
          ThreadBoundaryRejected: If slots and queue are both full and the
            overflow policy is `OverflowPolicy.REJECT`.
          ThreadBoundaryClosed: If the boundary was shut down.
        """
        await _core.checkpoint_if_cancelled()

        if self._closed:
            await _core.cancel_shielded_checkpoint()
            raise ThreadBoundaryClosed("this thread boundary has been shut down")
        if borrower in self._borrowers:
            await _core.cancel_shielded_checkpoint()
            raise RuntimeError(
                "this borrower is already holding one of this ThreadBoundary's slots",
            )

        now = _core.current_time()
        descriptor = (
            PendingThreadWork(
                fn=work[0],
                args=work[1],
                source=source,
                enqueued_at=now,
            )
            if work is not None
            else None
        )

        if len(self._borrowers) < self._capacity:
            self._borrow(borrower, source, descriptor, started_at=now)
            await _core.cancel_shielded_checkpoint()
            return

        waiting = self._waiting_count()
        if waiting >= self._queue_capacity and self._overflow is OverflowPolicy.REJECT:
            self._rejected += 1
            await _core.cancel_shielded_checkpoint()
            raise ThreadBoundaryRejected(
                f"thread boundary is saturated: {len(self._borrowers)} running and "
                f"{waiting} queued",
            )

        ticket = _Ticket(
            task=_core.current_task(),
            borrower=borrower,
            source=source,
            enqueued_at=now,
            work=descriptor,
        )
        self._queued.setdefault(source, deque()).append(ticket)

        def abort(_: RaiseCancelT) -> Abort:
            # Cancelled while queued: pull the ticket out precisely. No slot
            # was ever taken, so there is nothing to release and no waiter
            # can be left behind.
            dq = self._queued.get(source)
            if dq is not None:
                # Already dispatched if the ticket is gone (the slot hand-off
                # is synchronous); then cancellation arrived too late and the
                # normal return path will run.
                with contextlib.suppress(ValueError):
                    dq.remove(ticket)
                if not dq:
                    del self._queued[source]
            return Abort.SUCCEEDED

        await _core.wait_task_rescheduled(abort)

    @enable_ki_protection
    def _borrow(
        self,
        borrower: object,
        source: object,
        descriptor: PendingThreadWork | None,
        *,
        started_at: float,
    ) -> None:
        self._borrowers[borrower] = source
        self._running_per_source[source] = self._running_per_source.get(source, 0) + 1
        if descriptor is not None:
            descriptor.state = "running"
            descriptor.started_at = started_at
            self._work[borrower] = descriptor

    @enable_ki_protection
    def _record_completed_wait(self, source: object, waited: float) -> None:
        summary = self._wait_stats.get(source)
        if summary is None:
            self._wait_stats[source] = [1, waited, waited, waited]
        else:
            summary[0] += 1
            summary[1] += waited
            if waited < summary[2]:
                summary[2] = waited
            if waited > summary[3]:
                summary[3] = waited

    @enable_ki_protection
    def _dispatch(self) -> None:
        # Hand free slots to queued tickets. Sources are served from the
        # left of the OrderedDict and rotated to the end, so each source
        # gets at most one slot per round (round-robin); inside a source
        # tickets are dequeued FIFO.
        while self._queued and len(self._borrowers) < self._capacity:
            source, dq = next(iter(self._queued.items()))
            ticket = dq.popleft()
            if dq:
                self._queued.move_to_end(source)
            else:
                del self._queued[source]
            self._record_completed_wait(
                ticket.source, _core.current_time() - ticket.enqueued_at
            )
            self._borrow(
                ticket.borrower,
                ticket.source,
                ticket.work,
                started_at=_core.current_time(),
            )
            _core.reschedule(ticket.task, outcome.Value(None))

    @enable_ki_protection
    def release(self) -> None:
        """Release a slot previously acquired by the current task."""
        self.release_on_behalf_of(_core.current_task())

    @enable_ki_protection
    def release_on_behalf_of(self, borrower: object) -> None:
        """Release the slot held by ``borrower`` and wake the next waiter.

        Always safe to call after a successful acquire, including after the
        boundary was shut down or from a worker thread whose Trio task is
        gone: the slot is returned exactly once and queued waiters are woken
        or (during fast-fail shutdown) have already been removed.
        """
        if borrower not in self._borrowers:
            raise RuntimeError(
                "this borrower isn't holding any of this ThreadBoundary's slots",
            )
        source = self._borrowers.pop(borrower)
        self._work.pop(borrower, None)
        remaining = self._running_per_source.get(source, 1) - 1
        if remaining:
            self._running_per_source[source] = remaining
        else:
            self._running_per_source.pop(source, None)
        self._dispatch()
        self._poke_idle()

    ################
    # Shutdown
    ################

    def _poke_idle(self) -> None:
        if not self._borrowers and not self._queued:
            # Everyone is gone: wake drain waiters, then arm a fresh event
            # for subsequent drains.
            self._idle_event.set()
            self._idle_event = Event()

    @enable_ki_protection
    async def shutdown(self, *, drain: bool = True) -> ThreadBoundaryShutdown:
        """Close the boundary and resolve everything still in flight.

        New submissions made after this point raise
        `ThreadBoundaryClosed`.

        Args:
          drain: If ``True`` (the default), wait until every running and
            queued item has finished; the returned report is empty. If
            ``False``, fail every queued item immediately with
            `ThreadBoundaryClosed` and do not wait for the running items –
            Python provides no way to kill a running thread, so the running
            items are listed in the returned report instead of being
            silently dropped.

        Returns:
          A `ThreadBoundaryShutdown` describing the unfinished items.

        Calling this more than once returns the same report.
        """
        # Always provide a checkpoint.
        await _core.checkpoint_if_cancelled()
        if self._shutdown_report is not None:
            await _core.cancel_shielded_checkpoint()
            return self._shutdown_report

        self._closed = True

        if drain:
            # Releases both dispatch queued tickets and re-arm _idle_event,
            # so this settles once all running and queued work is gone.
            while self._borrowers or self._queued:
                await self._idle_event.wait()
            report = ThreadBoundaryShutdown(drained=True, unfinished=())
        else:
            unfinished: list[PendingThreadWork] = []
            # Fail everything still queued; pop all tickets before
            # rescheduling so abort()/double-wake can't touch them.
            parked: Iterable[_Ticket] = (
                ticket for dq in self._queued.values() for ticket in dq
            )
            for ticket in parked:
                if ticket.work is not None:
                    unfinished.append(ticket.work)
                _core.reschedule(
                    ticket.task,
                    outcome.Error(
                        ThreadBoundaryClosed(
                            "thread boundary was shut down without draining",
                        ),
                    ),
                )
            self._queued.clear()
            # Running threads cannot be forcibly cancelled; report them.
            unfinished.extend(self._work.values())
            report = ThreadBoundaryShutdown(
                drained=False,
                unfinished=tuple(unfinished),
            )

        self._shutdown_report = report
        self._poke_idle()
        return report
