from __future__ import annotations

import threading
from functools import partial

import pytest

import trio
import trio.lowlevel

from .._thread_boundary import (
    OverflowPolicy,
    ThreadBoundary,
    ThreadBoundaryClosed,
    ThreadBoundaryRejected,
)
from .._threads import to_thread_run_sync as run_sync
from ..testing import wait_all_tasks_blocked


def test_default_boundary_shape() -> None:
    # Declaring nothing reproduces the historical default thread limiter:
    # capacity 40, unbounded queue, callers wait on overflow.
    async def main() -> None:
        boundary = trio.to_thread.current_default_thread_limiter()
        assert isinstance(boundary, ThreadBoundary)
        assert boundary.capacity == 40
        assert boundary.queue_capacity == float("inf")
        assert boundary.overflow is OverflowPolicy.WAIT
        # CapacityLimiter-compatible surface
        assert boundary.total_tokens == 40
        assert boundary.borrowed_tokens == 0
        assert boundary.available_tokens == 40
        stats = boundary.statistics()
        assert stats.borrowed_tokens == 0
        assert stats.total_tokens == 40
        assert stats.tasks_waiting == 0
        assert stats.in_use == 0
        assert stats.waiting == 0
        assert stats.rejected == 0
        assert stats.borrowers == []
        assert stats.closed is False

        boundary.total_tokens = 3
        assert boundary.capacity == 3
        assert await trio.to_thread.run_sync(lambda: "ok") == "ok"
        boundary.total_tokens = 40

    trio.run(main)


async def test_capacity_enforced_and_returned() -> None:
    boundary = ThreadBoundary(2)
    gate = threading.Event()

    def work(i: int) -> int:
        gate.wait()
        return i

    try:
        async with trio.open_nursery() as nursery:
            for i in range(5):
                nursery.start_soon(partial(run_sync, work, i, limiter=boundary))
            await wait_all_tasks_blocked()
            stats = boundary.statistics()
            assert stats.in_use == 2
            assert stats.waiting == 3
            # nothing rejected with the default WAIT policy
            assert stats.rejected == 0
            gate.set()

        assert boundary.borrowed_tokens == 0
        assert boundary.statistics().waiting == 0
    finally:
        gate.set()


async def test_reject_overflow_policy() -> None:
    boundary = ThreadBoundary(1, queue_capacity=1, overflow=OverflowPolicy.REJECT)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()

            async def try_submit() -> None:
                with pytest.raises(ThreadBoundaryRejected):
                    await run_sync(block, limiter=boundary)

            nursery.start_soon(try_submit)
            await wait_all_tasks_blocked()

            stats = boundary.statistics()
            assert stats.in_use == 1
            assert stats.waiting == 1
            assert stats.rejected == 1
            gate.set()
    finally:
        gate.set()

    # After the backlog clears, submissions work normally again.
    assert await run_sync(lambda: 7, limiter=boundary) == 7
    assert boundary.statistics().in_use == 0


async def test_rejected_is_distinguishable() -> None:
    boundary = ThreadBoundary(1, queue_capacity=0, overflow=OverflowPolicy.REJECT)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()

            async def try_submit() -> None:
                # Distinguishable rejection type ...
                with pytest.raises(ThreadBoundaryRejected) as excinfo:
                    await run_sync(block, limiter=boundary)
                # ... that is also catchable as the ordinary "would block"
                # error family.
                assert isinstance(excinfo.value, trio.BusyResourceError)
                assert not isinstance(excinfo.value, trio.Cancelled)

            nursery.start_soon(try_submit)
            await wait_all_tasks_blocked()
            gate.set()
    finally:
        gate.set()


async def test_rejected_takes_no_slot_and_leaks_nothing() -> None:
    boundary = ThreadBoundary(1, queue_capacity=0, overflow=OverflowPolicy.REJECT)
    gate = threading.Event()
    started: list[str] = []
    lock = threading.Lock()

    def work(tag: str) -> None:
        with lock:
            started.append(tag)
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, work, "running", limiter=boundary))
            await wait_all_tasks_blocked()

            async def rejected() -> None:
                with pytest.raises(ThreadBoundaryRejected):
                    await run_sync(work, "rejected", limiter=boundary)

            nursery.start_soon(rejected)
            await wait_all_tasks_blocked()
            gate.set()

        assert started == ["running"]
        assert boundary.borrowed_tokens == 0
        assert boundary.statistics().waiting == 0
    finally:
        gate.set()


async def test_wait_policy_ignores_queue_capacity() -> None:
    # Even with a small declared queue capacity, WAIT keeps waiting.
    boundary = ThreadBoundary(1, queue_capacity=2, overflow=OverflowPolicy.WAIT)
    gate = threading.Event()

    def work(i: int) -> int:
        gate.wait()
        return i

    results: list[int] = []

    async def collect(i: int) -> None:
        results.append(await run_sync(work, i, limiter=boundary))

    try:
        async with trio.open_nursery() as nursery:
            for i in range(5):
                nursery.start_soon(collect, i)
            await wait_all_tasks_blocked()
            assert boundary.statistics().waiting == 4
            gate.set()
    finally:
        gate.set()

    assert sorted(results) == [0, 1, 2, 3, 4]
    assert boundary.statistics().rejected == 0


async def test_fifo_within_source_round_robin_across_sources() -> None:
    # Record the order in which the boundary *hands the single slot* to
    # tickets, from inside the Trio thread: that decision is deterministic
    # and independent of OS thread scheduling.
    boundary = ThreadBoundary(1)
    gates = {tag: threading.Event() for tag in ("A1", "A2", "A3", "B1")}
    dispatched: list[str] = []

    def work(tag: str) -> str:
        gates[tag].wait()
        return tag

    # Trio intentionally randomizes the order in which *runnable* tasks are
    # scheduled, so we force the boundary's queue into a known shape by
    # making each submission reach (and park in) the queue before the next is
    # submitted: A2, then A3, then B1.
    async with trio.open_nursery() as nursery:
        nursery.start_soon(partial(run_sync, work, "A1", limiter=boundary, source="A"))
        await wait_all_tasks_blocked()

        nursery.start_soon(partial(run_sync, work, "A2", limiter=boundary, source="A"))
        while boundary.statistics().waiting < 1:
            await trio.lowlevel.checkpoint()
        nursery.start_soon(partial(run_sync, work, "A3", limiter=boundary, source="A"))
        while boundary.statistics().waiting < 2:
            await trio.lowlevel.checkpoint()
        nursery.start_soon(partial(run_sync, work, "B1", limiter=boundary, source="B"))
        while boundary.statistics().waiting < 3:
            await trio.lowlevel.checkpoint()

        def current_tag() -> str | None:
            running = [w.args[0] for w in boundary._work.values()]
            assert len(running) <= 1
            return running[0] if running else None

        # Dispatch order with one slot: A1 already running; among the queued
        # tickets A2 gets the next slot (FIFO within source A), then B1
        # (round-robin to source B), then A3. A's early flood cannot push B
        # behind all of its backlog.
        for expected in ("A1", "A2", "B1", "A3"):
            assert current_tag() == expected
            dispatched.append(expected)
            gates[expected].set()
            # The worker thread delivers via run_sync_soon, which is only
            # loosely synchronized with wait_all_tasks_blocked, so poll for
            # the actual slot hand-over instead of assuming one settle.
            while current_tag() == expected:
                await trio.lowlevel.checkpoint()

    assert dispatched == ["A1", "A2", "B1", "A3"]


async def test_cancel_while_queued_removes_ticket() -> None:
    boundary = ThreadBoundary(2)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()

            with trio.CancelScope() as cs:
                cs.cancel()
                with pytest.raises(trio.Cancelled):
                    await run_sync(block, limiter=boundary)
            await wait_all_tasks_blocked()
            # The cancelled waiter took no slot and left no ticket behind.
            assert boundary.statistics().in_use == 2
            assert boundary.statistics().waiting == 0

            # A fresh submission queues normally: accounting is intact.
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()
            assert boundary.statistics().in_use == 2
            assert boundary.statistics().waiting == 1
            gate.set()

        assert boundary.borrowed_tokens == 0
    finally:
        gate.set()


async def test_exception_in_worker_returns_slot() -> None:
    boundary = ThreadBoundary(1)

    def explode() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match=r"^boom$"):
        await run_sync(explode, limiter=boundary)
    assert boundary.borrowed_tokens == 0
    assert await run_sync(lambda: 1, limiter=boundary) == 1


async def test_statistics_per_source() -> None:
    boundary = ThreadBoundary(1)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, block, limiter=boundary, source="A"))
            await wait_all_tasks_blocked()
            nursery.start_soon(partial(run_sync, block, limiter=boundary, source="A"))
            nursery.start_soon(partial(run_sync, block, limiter=boundary, source="B"))
            await wait_all_tasks_blocked()

            stats = boundary.statistics()
            assert stats.sources["A"].in_use == 1
            assert stats.sources["A"].waiting == 1
            assert stats.sources["B"].in_use == 0
            assert stats.sources["B"].waiting == 1
            a_waits = stats.sources["A"].wait_seconds
            b_waits = stats.sources["B"].wait_seconds
            assert len(a_waits) == 1
            assert a_waits[0] >= 0
            assert len(b_waits) == 1
            assert b_waits[0] >= 0

            gate.set()

        stats = boundary.statistics()
        assert stats.in_use == 0
        assert stats.sources["A"].completed_wait_count == 1
        assert stats.sources["B"].completed_wait_count == 1
        assert stats.sources["A"].total_wait_seconds >= 0
        assert stats.sources["A"].min_wait_seconds is not None
        assert stats.sources["A"].max_wait_seconds is not None
    finally:
        gate.set()


async def test_nowait_and_direct_acquire_compat() -> None:
    boundary = ThreadBoundary(1)
    assert not boundary.closed

    borrower = object()
    other = object()
    boundary.acquire_on_behalf_of_nowait(borrower)
    assert boundary.borrowed_tokens == 1
    # Saturated: a different borrower is refused without blocking.
    with pytest.raises(trio.WouldBlock):
        boundary.acquire_on_behalf_of_nowait(other)
    # The same borrower twice is an error, same as CapacityLimiter.
    with pytest.raises(RuntimeError):
        boundary.acquire_on_behalf_of_nowait(borrower)
    boundary.release_on_behalf_of(borrower)
    assert boundary.borrowed_tokens == 0

    # Async-context-manager style, like CapacityLimiter: while the current
    # task holds a slot, *other* borrowers are refused (WouldBlock), and the
    # same borrower is a RuntimeError, matching CapacityLimiter.
    other2 = object()
    async with boundary:
        assert boundary.borrowed_tokens == 1
        with pytest.raises(trio.WouldBlock):
            boundary.acquire_on_behalf_of_nowait(other2)
        with pytest.raises(RuntimeError):
            boundary.acquire_nowait()
    assert boundary.borrowed_tokens == 0

    with pytest.raises(RuntimeError):
        boundary.release()


async def test_capacity_validation() -> None:
    with pytest.raises(TypeError, match="capacity must be an int"):
        ThreadBoundary(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity must be >= 0"):
        ThreadBoundary(-1)
    boundary = ThreadBoundary(2)
    with pytest.raises(TypeError, match="capacity must be an int"):
        boundary.capacity = 1.5  # type: ignore[assignment]
    with pytest.raises(ValueError, match="capacity must be >= 0"):
        boundary.capacity = -1
    with pytest.raises(ValueError, match="bogus"):
        ThreadBoundary(1, overflow="bogus")  # type: ignore[arg-type]


async def test_increasing_capacity_wakes_waiters() -> None:
    boundary = ThreadBoundary(0)
    gate = threading.Event()

    def work(i: int) -> int:
        gate.wait()
        return i

    results: list[int] = []

    async def collect(i: int) -> None:
        results.append(await run_sync(work, i, limiter=boundary))

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(collect, 1)
            nursery.start_soon(collect, 2)
            await wait_all_tasks_blocked()
            assert boundary.statistics().waiting == 2
            boundary.capacity = 1
            await wait_all_tasks_blocked()
            assert boundary.statistics().in_use == 1
            assert boundary.statistics().waiting == 1
            boundary.capacity = 2
            await wait_all_tasks_blocked()
            assert boundary.statistics().in_use == 2
            gate.set()
    finally:
        gate.set()

    assert sorted(results) == [1, 2]


async def test_shutdown_drain() -> None:
    boundary = ThreadBoundary(1)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    report_holder: list[object] = []
    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            nursery.start_soon(partial(run_sync, block, limiter=boundary))
            await wait_all_tasks_blocked()

            async def drain() -> None:
                report_holder.append(await boundary.shutdown(drain=True))

            nursery.start_soon(drain)
            await wait_all_tasks_blocked()
            assert boundary.statistics().closed is True
            gate.set()

        (report,) = report_holder
        assert report.drained is True
        assert report.unfinished == ()
        assert report.running == ()
        assert report.waiting == ()
    finally:
        gate.set()

    # Closed boundary refuses new work.
    with pytest.raises(ThreadBoundaryClosed):
        await run_sync(lambda: None, limiter=boundary)
    # Idempotent.
    assert await boundary.shutdown(drain=True) is report


async def test_shutdown_fast_fail_reports_unfinished() -> None:
    boundary = ThreadBoundary(1)
    gate = threading.Event()

    def block(tag: str) -> str:
        gate.wait()
        return tag

    closed_errors: list[str] = []

    async def queued(tag: str) -> None:
        try:
            await run_sync(block, tag, limiter=boundary, source="queued")
        except ThreadBoundaryClosed:
            closed_errors.append(tag)

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                partial(
                    run_sync,
                    block,
                    "running",
                    limiter=boundary,
                    source="running",
                )
            )
            await wait_all_tasks_blocked()
            nursery.start_soon(queued, "q1")
            nursery.start_soon(queued, "q2")
            await wait_all_tasks_blocked()
            assert boundary.statistics().waiting == 2

            report = await boundary.shutdown(drain=False)

            assert report.drained is False
            assert len(report.running) == 1
            assert len(report.waiting) == 2
            assert report.running[0].fn is block
            assert report.running[0].args == ("running",)
            assert report.running[0].is_running
            assert {w.args[0] for w in report.waiting} == {"q1", "q2"}
            assert {w.state for w in report.unfinished} == {"running", "waiting"}
            await wait_all_tasks_blocked()
            assert sorted(closed_errors) == ["q1", "q2"]

            # New submissions fail immediately.
            with pytest.raises(ThreadBoundaryClosed):
                await run_sync(block, "late", limiter=boundary)

            # Idempotent: same report.
            assert await boundary.shutdown(drain=False) is report
            gate.set()

        # The running thread still finished in the background and its slot
        # was returned precisely, even after fast-fail shutdown.
        assert boundary.borrowed_tokens == 0
    finally:
        gate.set()


async def test_run_sync_source_tag_end_to_end() -> None:
    boundary = ThreadBoundary(1)
    gate = threading.Event()

    def block() -> None:
        gate.wait()

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(
                partial(run_sync, block, limiter=boundary, source="my-source")
            )
            await wait_all_tasks_blocked()
            stats = boundary.statistics()
            assert stats.sources["my-source"].in_use == 1
            gate.set()
    finally:
        gate.set()


async def test_file_io_uses_default_boundary_with_source() -> None:
    boundary = trio.to_thread.current_default_thread_limiter()
    gate = threading.Event()

    class SlowFile:
        def __init__(self) -> None:
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            gate.wait()
            return b"data"

        def write(self, data: bytes) -> int:
            return len(data)

        def close(self) -> None:
            self.closed = True

    wrapped = trio.wrap_file(SlowFile())
    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(wrapped.read)
            await wait_all_tasks_blocked()
            stats = boundary.statistics()
            assert stats.in_use == 1
            assert "trio.file_io.read" in stats.sources
            assert stats.sources["trio.file_io.read"].in_use == 1
            gate.set()
    finally:
        gate.set()
