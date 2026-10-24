from __future__ import annotations

import contextlib
import errno
import os
import sys
from pathlib import Path as SyncPath
from unittest.mock import patch

import pytest

import trio
import trio.lowlevel
from trio import _core, _file_lock
from trio._file_io import AsyncIOWrapper, _LockedAsyncIOWrapper

from .._core._tests.tutil import slow

# A pid that must not exist on any supported platform (pids are much
# smaller in practice).
DEAD_PID = 2_147_400_000

# Make spawned children able to import the in-tree trio even when it is not
# installed (e.g. running straight from a checkout).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(trio.__file__)))
_CHILD_ENV = {
    **os.environ,
    "PYTHONPATH": _SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""),
}

CHILD_HOLD = (
    "import sys, trio\n"
    "async def m():\n"
    "    l = trio.FileLock(sys.argv[1])\n"
    "    await l.acquire()\n"
    "    sys.stdout.write('ready\\n')\n"
    "    sys.stdout.flush()\n"
    "    await trio.sleep_forever()\n"
    "trio.run(m)\n"
)

CHILD_CRASH = (
    "import os, sys, trio\n"
    "async def m():\n"
    "    l = trio.FileLock(sys.argv[1])\n"
    "    await l.acquire()\n"
    "    sys.stdout.write('ready\\n')\n"
    "    sys.stdout.flush()\n"
    "    os._exit(0)\n"
    "trio.run(m)\n"
)

# Acquires the lock, then overwrites the metadata with a record naming a
# process that does not exist -- simulating a stale holder while a *live*
# process still holds the OS lock (the "inherited descriptor" scenario).
CHILD_HOLD_FAKE_DEAD_PID = (
    "import json, sys, trio\n"
    "async def m():\n"
    "    l = trio.FileLock(sys.argv[1])\n"
    "    await l.acquire()\n"
    f"    rec = json.dumps({{'v': 1, 'pid': {DEAD_PID},\n"
    "        'time': __import__('time').time(),\n"
    "        'host': __import__('platform').node(),\n"
    "        'id': 'deadbeef', 'ex': True}).encode()\n"
    "    with open(sys.argv[1] + '.trio-lock', 'r+b') as f:\n"
    "        f.seek(0)\n"
    "        f.write(rec + b' ' * (512 - len(rec)))\n"
    "        f.flush()\n"
    "    sys.stdout.write('ready\\n')\n"
    "    sys.stdout.flush()\n"
    "    await trio.sleep_forever()\n"
    "trio.run(m)\n"
)

CHILD_STATE_COUNTER = (
    "import sys, trio\n"
    "async def m():\n"
    "    for _ in range(10):\n"
    "        async with trio.open_state_file(sys.argv[1]) as f:\n"
    "            await f.seek(0)\n"
    "            n = int((await f.read()) or b'0')\n"
    "            await f.seek(0)\n"
    "            await f.write(str(n + 1).encode())\n"
    "            await f.truncate()\n"
    "trio.run(m)\n"
)


@pytest.fixture
def target(tmp_path: SyncPath) -> str:
    return os.fspath(tmp_path / "state.bin")


async def _start_child(code: str, target: str) -> trio.Process:
    import subprocess

    proc = await trio.lowlevel.open_process(
        [sys.executable, "-u", "-c", code, target],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_CHILD_ENV,
    )
    assert proc.stdout is not None
    line = await proc.stdout.receive_some(4096)
    assert line.strip() == b"ready", line
    return proc


###############################################################################
# Same-process basics
###############################################################################


async def test_acquire_release_locked(target: str) -> None:
    lock = trio.FileLock(target)
    assert not lock.locked
    assert lock.path == target
    assert not os.path.exists(lock.sidecar_path)  # noqa: ASYNC240

    await lock.acquire()
    try:
        assert lock.locked
        assert os.path.exists(lock.sidecar_path)  # noqa: ASYNC240
    finally:
        await lock.release()
    assert not lock.locked

    # reusable
    await lock.acquire()
    await lock.aclose()
    assert not lock.locked


async def test_aclose_idempotent(target: str) -> None:
    lock = trio.FileLock(target)
    await lock.acquire()
    await lock.aclose()
    await lock.aclose()


async def test_release_without_hold_raises(target: str) -> None:
    lock = trio.FileLock(target)
    with pytest.raises(RuntimeError):
        await lock.release()


async def test_busy_and_closed_errors(target: str) -> None:
    lock = trio.FileLock(target)
    await lock.acquire()
    with pytest.raises(trio.BusyResourceError):
        await lock.acquire()
    await lock.release()

    await lock.aclose()
    with pytest.raises(trio.ClosedResourceError):
        await lock.acquire()


async def test_acquire_nowait(target: str) -> None:
    a = trio.FileLock(target)
    b = trio.FileLock(target)
    assert await a.acquire_nowait()
    try:
        assert not await b.acquire_nowait()
        assert not b.locked
    finally:
        await a.release()
    assert await b.acquire_nowait()
    await b.release()


async def test_blocking_acquire_waits_for_release(target: str) -> None:
    a = trio.FileLock(target)
    b = trio.FileLock(target)
    await a.acquire()

    async def release_soon() -> None:
        await trio.sleep(0.2)
        await a.release()

    async with trio.open_nursery() as nursery:
        nursery.start_soon(release_soon)
        await b.acquire()
    assert b.locked
    await b.release()


async def test_timeout_is_distinct(target: str) -> None:
    a = trio.FileLock(target)
    await a.acquire()
    b = trio.FileLock(target)
    with pytest.raises(trio.FileLockTimeout) as excinfo:
        await b.acquire(timeout=0.2)
    exc = excinfo.value
    # distinct from every other failure mode
    assert isinstance(exc, trio.FileLockError)
    assert not isinstance(exc, OSError)
    assert not isinstance(exc, TimeoutError)
    assert not isinstance(exc, trio.StaleLockError)
    assert exc.holder is not None
    assert exc.holder.pid == os.getpid()
    assert exc.holder.acquired_at is not None
    await a.release()


async def test_timeout_zero(target: str) -> None:
    a = trio.FileLock(target)
    await a.acquire()
    b = trio.FileLock(target)
    with pytest.raises(trio.FileLockTimeout):
        await b.acquire(timeout=0)
    await a.release()


async def test_cancel_while_waiting(target: str) -> None:
    a = trio.FileLock(target)
    b = trio.FileLock(target)
    await a.acquire()
    with trio.move_on_after(0.2):
        await b.acquire()
    assert not b.locked

    await a.release()
    # The abandoned worker self-disposes once the lock frees; a fresh
    # acquisition must then succeed.
    with trio.fail_after(5):
        while not await trio.FileLock(target).acquire_nowait():  # noqa: ASYNC110
            await trio.sleep(0.05)


async def test_holder_info(target: str) -> None:
    assert await trio.FileLock(target).current_holder() is None
    lock = trio.FileLock(target)
    await lock.acquire()
    try:
        holder = await lock.current_holder()
        assert holder is not None
        assert holder.pid == os.getpid()
        assert holder.exclusive is True
        assert holder.instance_id == lock._instance_id
        assert holder.is_local
        assert holder.age is not None
        assert holder.age >= 0
    finally:
        await lock.release()
    assert await trio.FileLock(target).current_holder() is None


async def test_shared_and_exclusive(target: str) -> None:
    s1 = trio.FileLock(target, exclusive=False)
    s2 = trio.FileLock(target, exclusive=False)
    e = trio.FileLock(target)
    assert await s1.acquire_nowait()
    assert await s2.acquire_nowait()
    assert not await e.acquire_nowait()
    await s1.release()
    await s2.release()
    assert await e.acquire_nowait()
    await e.release()


async def test_open_file_lock_context_manager(target: str) -> None:
    async with trio.open_file_lock(target, timeout=1):
        other = trio.FileLock(target)
        assert not await other.acquire_nowait()
    after = trio.FileLock(target)
    assert await after.acquire_nowait()
    await after.release()


###############################################################################
# open_file integration and lifecycle
###############################################################################


async def test_open_file_default_unchanged(target: str) -> None:
    # No lock argument: exact previous behaviour and return type.
    f = await trio.open_file(target, "w+b")
    try:
        assert type(f) is AsyncIOWrapper
        assert not hasattr(f, "file_lock")
    finally:
        await f.aclose()
    assert not os.path.exists(target + ".trio-lock")  # noqa: ASYNC240


async def test_open_file_lock_mode(target: str) -> None:
    f = await trio.open_file(target, "w+b", lock=True)
    assert isinstance(f, _LockedAsyncIOWrapper)
    assert f.file_lock.locked
    await f.write(b"payload")
    await f.aclose()
    assert f.closed
    assert not f.file_lock.locked

    assert SyncPath(target).read_bytes() == b"payload"  # noqa: ASYNC240


async def test_open_file_lock_blocks_others(target: str) -> None:
    async with await trio.open_file(target, "w+b", lock=True):
        other = trio.FileLock(target)
        assert not await other.acquire_nowait()


async def test_lock_released_on_exception(target: str) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with await trio.open_file(target, "w+b", lock=True) as f:
            assert f.file_lock.locked
            raise RuntimeError("boom")
    other = trio.FileLock(target)
    assert await other.acquire_nowait()
    await other.release()


async def test_lock_released_on_cancellation(target: str) -> None:
    with trio.move_on_after(0.1):
        async with await trio.open_file(target, "w+b", lock=True):
            await trio.sleep_forever()
    with trio.fail_after(5):
        while not await trio.FileLock(target).acquire_nowait():  # noqa: ASYNC110
            await trio.sleep(0.05)


async def test_passed_file_lock(target: str) -> None:
    lock = trio.FileLock(target)
    await lock.acquire()
    async with await trio.open_file(target, "w+b", lock=lock) as f:
        assert f.file_lock is lock
    # caller-owned lock survives close; closing it is the caller's job (and
    # the file close did not release it mid-use)
    assert lock.locked
    await lock.release()

    # an unacquired caller lock is acquired on our behalf
    lock2 = trio.FileLock(target)
    async with await trio.open_file(target, "r+b", lock=lock2):
        assert lock2.locked
    assert not lock2.locked


async def test_timed_open_file_times_out_and_closes_data_file(
    target: str,
) -> None:
    holder = trio.FileLock(target)
    await holder.acquire()
    contender = trio.FileLock(target, default_timeout=0.2)
    with pytest.raises(trio.FileLockTimeout):
        await trio.open_file(target, "w+b", lock=contender)
    assert not contender.locked
    # once the holder goes away, the same lock object can be reused
    await holder.release()
    f = await trio.open_file(target, "w+b", lock=contender)
    assert contender.locked
    await f.aclose()
    assert not contender.locked


async def test_lock_true_with_fd_rejected() -> None:
    # locking an anonymous descriptor has no path to derive a lockfile from
    r, w = os.pipe()
    try:
        with pytest.raises(ValueError, match="path"):
            await trio.open_file(r, "rb", closefd=True, lock=True)
        # the failed call still closed the file it opened
        with pytest.raises(OSError) as excinfo:  # noqa: PT011
            os.fstat(r)
        assert excinfo.value.errno == errno.EBADF
    finally:
        with contextlib.suppress(OSError):
            os.close(r)
        os.close(w)


async def test_bad_lock_argument(target: str) -> None:
    with pytest.raises(TypeError):
        await trio.open_file(target, "w+b", lock="yes")  # type: ignore[arg-type]


async def test_path_open_lock(target: str) -> None:
    p = trio.Path(target)
    async with await p.open("w+b", lock=True) as f:
        assert isinstance(f, _LockedAsyncIOWrapper)
        await f.write(b"via path")
    other = trio.FileLock(target)
    assert await other.acquire_nowait()
    await other.release()


async def test_path_open_default_unchanged(target: str) -> None:
    # exceptions and types remain identical without lock=
    missing = target + ".missing"
    with pytest.raises(FileNotFoundError):
        await trio.Path(missing).open("rb")
    f = await trio.Path(target).open("w+b")
    assert type(f) is AsyncIOWrapper
    await f.aclose()


###############################################################################
# Temp files and state files
###############################################################################


async def test_open_temp_file(target: str) -> None:
    async with trio.open_temp_file(dir=os.path.dirname(target)) as (
        f,
        name,
    ):
        assert isinstance(f, _LockedAsyncIOWrapper)
        assert f.file_lock.locked
        await f.write(b"temporary")
        await f.flush()
        assert os.path.exists(name)  # noqa: ASYNC240
        assert os.path.exists(name + ".trio-lock")  # noqa: ASYNC240
        other = trio.FileLock(name)
        assert not await other.acquire_nowait()
    assert not os.path.exists(name)  # noqa: ASYNC240
    assert not os.path.exists(name + ".trio-lock")  # noqa: ASYNC240


async def test_open_temp_file_no_lock(target: str) -> None:
    async with trio.open_temp_file(dir=os.path.dirname(target), lock=False) as (
        f,
        name,
    ):
        assert type(f) is AsyncIOWrapper
    assert not os.path.exists(name)  # noqa: ASYNC240


async def test_open_state_file_creates_and_locks(tmp_path: SyncPath) -> None:
    path = os.fspath(tmp_path / "counter.state")
    assert not os.path.exists(path)  # noqa: ASYNC240
    async with trio.open_state_file(path) as f:
        assert os.path.exists(path)  # noqa: ASYNC240
        assert f.file_lock.locked
        await f.write(b"1")
    async with trio.open_state_file(path) as f:
        assert await f.read() == b"1"
    # existing file must not be truncated
    async with trio.open_state_file(path, create=True) as f:
        assert await f.read() == b"1"


async def test_state_file_serializes_same_process_tasks(
    tmp_path: SyncPath,
) -> None:
    path = os.fspath(tmp_path / "counter.state")

    async def bump() -> None:
        async with trio.open_state_file(path) as f:
            await f.seek(0)
            n = int((await f.read()) or b"0")
            await trio.lowlevel.checkpoint()
            await f.seek(0)
            await f.write(str(n + 1).encode())
            await f.truncate()

    async with trio.open_nursery() as nursery:
        for _ in range(20):
            nursery.start_soon(bump)
    async with await trio.open_file(path, "rb") as f:
        assert await f.read() == b"20"


###############################################################################
# Platform capability
###############################################################################


async def test_unsupported_platform_raises(target: str) -> None:
    def boom(self: object, fd: int, exclusive: bool) -> bool:
        raise trio.FileLockUnsupported("simulated missing capability")

    with patch.object(type(_file_lock._backend), "try_lock", boom):
        lock = trio.FileLock(target)
        with pytest.raises(trio.FileLockUnsupported):
            await lock.acquire()
        assert not lock.locked

        with pytest.raises(trio.FileLockUnsupported):
            await trio.open_file(target, "w+b", lock=True)

    # and once the capability is back, locking works normally again
    async with trio.open_file_lock(target):
        pass


###############################################################################
# Descriptor hygiene (POSIX only)
###############################################################################


@slow
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")
async def test_no_sidecar_fd_leaks(target: str) -> None:
    sidecar = target + ".trio-lock"

    def open_sidecar_fds() -> set[str]:
        out: set[str] = set()
        for name in os.listdir("/proc/self/fd"):
            try:
                link = os.readlink(f"/proc/self/fd/{name}")
            except OSError:
                continue
            if link == sidecar:
                out.add(name)
        return out

    holder = trio.FileLock(target)
    await holder.acquire()
    contenders = [trio.FileLock(target) for _ in range(3)]
    for lock in contenders:
        with pytest.raises(trio.FileLockTimeout):
            await lock.acquire(timeout=0.05)
    for lock in contenders:
        with trio.move_on_after(0.05):
            await lock.acquire()
        assert not lock.locked
    await holder.release()
    # let abandoned workers finish and reap themselves
    for _ in range(50):
        await trio.sleep(0.05)
        if not open_sidecar_fds():
            break
    assert open_sidecar_fds() == set()


###############################################################################
# Cross-process scenarios
###############################################################################


@slow
async def test_cross_process_exclusion_holder_and_takeover(target: str) -> None:

    proc = await _start_child(CHILD_HOLD, target)
    try:
        lock = trio.FileLock(target)
        assert not await lock.acquire_nowait()
        holder = await lock.current_holder()
        assert holder is not None
        assert holder.pid == proc.pid
        assert holder.is_local

        with pytest.raises(trio.FileLockTimeout):
            await lock.acquire(timeout=0.5)
        assert not lock.locked

        proc.kill()
        await proc.wait()

        await lock.acquire(timeout=10)
        assert lock.locked
        # child was killed without releasing; the kernel freed the OS lock
        # and our acquisition diagnosed the stale record
        assert lock.last_takeover is not None
        assert lock.last_takeover.previous is not None
        assert lock.last_takeover.previous.pid == proc.pid
        assert "no process with that pid exists" in lock.last_takeover.evidence
        await lock.release()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@slow
async def test_crash_os_exit_leaves_diagnosis(tmp_path: SyncPath) -> None:

    target = os.fspath(tmp_path / "crashed.bin")
    proc = await _start_child(CHILD_CRASH, target)
    await proc.wait()

    # metadata remains, OS lock was released by the kernel
    holder = await trio.FileLock(target).current_holder()
    assert holder is not None
    assert holder.pid == proc.pid

    lock = trio.FileLock(target, stale_policy=trio.StaleLockPolicy.TAKE_OVER)
    await lock.acquire()
    assert lock.locked
    assert lock.last_takeover is not None
    assert lock.last_takeover.previous is not None
    assert lock.last_takeover.previous.pid == proc.pid
    await lock.release()


@slow
async def test_stale_policies_against_live_lock(target: str) -> None:

    proc = await _start_child(CHILD_HOLD_FAKE_DEAD_PID, target)
    try:
        err_lock = trio.FileLock(target, stale_policy=trio.StaleLockPolicy.ERROR)
        with pytest.raises(trio.StaleLockError) as excinfo:
            await err_lock.acquire(timeout=2)
        assert str(DEAD_PID) in excinfo.value.evidence
        assert excinfo.value.holder is not None
        assert excinfo.value.holder.pid == DEAD_PID

        take_lock = trio.FileLock(target, stale_policy=trio.StaleLockPolicy.TAKE_OVER)
        with pytest.raises(trio.StaleLockError, match="still held"):
            await take_lock.acquire(timeout=2)
        assert not take_lock.locked

        # WAIT just blocks; once the live holder dies it gets the lock
        wait_lock = trio.FileLock(target)
        proc.kill()
        await proc.wait()
        await wait_lock.acquire(timeout=10)
        assert wait_lock.locked
        await wait_lock.release()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@slow
async def test_subprocess_output_redirected_to_locked_file(
    tmp_path: SyncPath,
) -> None:

    out = os.fspath(tmp_path / "child.out")
    locked = await trio.open_file(out, "wb", lock=True)
    try:
        result = await trio.run_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('locked output'); sys.stdout.flush()",
            ],
            stdout=locked,
            check=False,
        )
        assert result.returncode == 0
        # run_process owns the lifecycle: flushed, closed and unlocked once
        # the child is done
        assert locked.closed
        assert not locked.file_lock.locked
    finally:
        if not locked.closed:
            await locked.aclose()

    assert SyncPath(out).read_bytes() == b"locked output"  # noqa: ASYNC240

    # and the lock is genuinely free afterwards
    again = trio.FileLock(out)
    assert await again.acquire_nowait()
    await again.release()


async def _run_process_task(*args: object, **kwargs: object) -> object:
    return await trio.run_process(*args, **kwargs)  # type: ignore[arg-type]


@slow
async def test_lock_held_while_child_writes(tmp_path: SyncPath) -> None:
    from functools import partial

    out = os.fspath(tmp_path / "child2.out")
    locked = await trio.open_file(out, "wb", lock=True)
    try:
        async with _core.open_nursery() as nursery:
            proc = await nursery.start(
                partial(
                    _run_process_task,
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "import sys, time\n"
                        "sys.stdout.write('x'); sys.stdout.flush()\n"
                        "time.sleep(1.5)\n",
                    ],
                    stdout=locked,
                    check=False,
                ),
            )
            # while the child is still running, the lock is held
            probe = trio.FileLock(out)
            assert not await probe.acquire_nowait()
            await proc.wait()
    finally:
        if not locked.closed:
            await locked.aclose()

    assert locked.closed
    assert not locked.file_lock.locked
    assert SyncPath(out).read_bytes() == b"x"  # noqa: ASYNC240


@slow
async def test_subprocess_plain_async_file_stays_caller_owned(
    tmp_path: SyncPath,
) -> None:
    # Without lock=, run_process must not take over the caller's file
    # object's lifetime (the pre-existing semantics).
    out = os.fspath(tmp_path / "plain.out")
    plain = await trio.open_file(out, "wb")
    try:
        result = await trio.run_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('plain')",
            ],
            stdout=plain,
            check=False,
        )
        assert result.returncode == 0
        assert not plain.closed
    finally:
        await plain.aclose()
    assert SyncPath(out).read_bytes() == b"plain"  # noqa: ASYNC240


@slow
async def test_state_file_multiprocess_counter(tmp_path: SyncPath) -> None:

    path = os.fspath(tmp_path / "shared.state")
    SyncPath(path).write_bytes(b"0")  # noqa: ASYNC240

    async def one_child() -> None:
        result = await trio.run_process(
            [sys.executable, "-u", "-c", CHILD_STATE_COUNTER, path],
            check=False,
            env=_CHILD_ENV,
        )
        assert result.returncode == 0

    async with trio.open_nursery() as nursery:
        nursery.start_soon(one_child)
        nursery.start_soon(one_child)

    assert SyncPath(path).read_bytes() == b"20"  # noqa: ASYNC240
