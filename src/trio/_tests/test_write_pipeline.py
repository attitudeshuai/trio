from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

import trio
from trio import _write_pipeline as wp
from trio.testing import wait_all_tasks_blocked

if TYPE_CHECKING:
    import pathlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _staging_files(directory: pathlib.Path) -> list[str]:
    return [
        name
        for name in os.listdir(directory)
        if name.startswith(".trio-write-") and name.endswith(".tmp")
    ]


def _all_pipeline_files(directory: pathlib.Path) -> list[str]:
    return [
        name for name in os.listdir(directory) if name.startswith(".trio-write-")
    ]


# ---------------------------------------------------------------------------
# Default semantics remain exactly as before
# ---------------------------------------------------------------------------


async def test_default_open_truncates_in_place(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old content")
    f = await trio.open_file(target, "w")
    try:
        # Truncation happened immediately at open, no staging file involved.
        assert target.read_text() == ""
        assert _staging_files(tmp_path) == []
        await f.write("new")
    finally:
        await f.aclose()
    assert target.read_text() == "new"
    assert _all_pipeline_files(tmp_path) == []


async def test_default_path_writes_match_pathlib(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    await trio.Path(target).write_bytes(b"abc")
    assert target.read_bytes() == b"abc"
    await trio.Path(target).write_text("text", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "text"
    await trio.Path(tmp_path / "t").touch()
    assert (tmp_path / "t").exists()
    assert _all_pipeline_files(tmp_path) == []


async def test_default_rename_existing_raises_native(
    tmp_path: pathlib.Path,
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("s")
    dst.write_text("d")
    with pytest.raises(FileExistsError):
        await trio.Path(src).rename(dst)
    # replace still overwrites
    result = await trio.Path(src).replace(dst)
    assert isinstance(result, trio.Path)
    assert dst.read_text() == "s"


async def test_direct_strength_equals_default(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old")
    await trio.Path(target).write_bytes(
        b"x", strength=trio.WriteStrength.DIRECT
    )
    assert target.read_bytes() == b"x"
    assert _staging_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Whole-content atomic/durable writes
# ---------------------------------------------------------------------------


async def test_atomic_write_bytes_and_text(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old")
    await trio.Path(target).write_bytes(b"new", strength=trio.WriteStrength.ATOMIC)
    assert target.read_bytes() == b"new"
    await trio.Path(target).write_text(
        "newer", encoding="utf-8", strength=trio.WriteStrength.ATOMIC
    )
    assert target.read_text(encoding="utf-8") == "newer"
    assert _all_pipeline_files(tmp_path) == []


async def test_target_unchanged_while_staging_open(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old")
    async with await trio.open_file(
        target, "w", strength=trio.WriteStrength.ATOMIC
    ) as f:
        assert target.read_text() == "old"
        assert _staging_files(tmp_path)
        await f.write("partial")
        assert target.read_text() == "old"
    assert target.read_text() == "partial"
    assert _all_pipeline_files(tmp_path) == []


async def test_durable_fsyncs_file_and_dir(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    fsynced_files: list[int] = []
    fsynced_dirs: list[pathlib.Path] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        fsynced_files.append(fd)
        return real_fsync(fd)

    def spy_fsync_dir(path: pathlib.Path) -> None:
        fsynced_dirs.append(path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "fsync", spy_fsync)
        mp.setattr(wp, "_fsync_parent_dir", spy_fsync_dir)
        await trio.Path(target).write_bytes(b"d", strength=trio.WriteStrength.DURABLE)

    assert target.read_bytes() == b"d"
    assert fsynced_files
    assert fsynced_dirs == [target]


async def test_atomic_touch(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "t"
    await trio.Path(target).touch(strength=trio.WriteStrength.ATOMIC)
    assert target.exists()
    assert target.read_bytes() == b""

    # existing target is replaced atomically too
    target.write_text("x")
    await trio.Path(target).touch(strength=trio.WriteStrength.DURABLE)
    assert target.read_bytes() == b""

    with pytest.raises(FileExistsError):
        existing = tmp_path / "t2"
        existing.write_text("already there")
        await trio.Path(existing).touch(
            exist_ok=False, strength=trio.WriteStrength.ATOMIC
        )
    assert existing.read_text() == "already there"
    assert _all_pipeline_files(tmp_path) == []


async def test_atomic_move(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "s"
    dst = tmp_path / "d"
    await trio.Path(src).write_text("payload", strength=trio.WriteStrength.ATOMIC)
    out = await trio.Path(src).rename(dst, strength=trio.WriteStrength.ATOMIC)
    assert isinstance(out, trio.Path)
    assert not src.exists()
    assert dst.read_text() == "payload"

    await trio.Path(dst).write_text("payload2", strength=trio.WriteStrength.ATOMIC)
    target = tmp_path / "f"
    target.write_text("old")
    await trio.Path(dst).replace(target, strength=trio.WriteStrength.DURABLE)
    assert target.read_text() == "payload2"
    assert not dst.exists()


# ---------------------------------------------------------------------------
# Failure classification and target preservation
# ---------------------------------------------------------------------------


async def test_missing_parent_is_create_error(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "missing-dir" / "f"
    with pytest.raises(trio.TempFileCreateError) as excinfo:
        await trio.Path(target).write_bytes(b"x", strength=trio.WriteStrength.ATOMIC)
    assert excinfo.value.stage == "create"
    assert isinstance(excinfo.value, OSError)
    assert excinfo.value.errno == 2
    assert not target.exists()


async def test_write_stage_error_preserves_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("keep")

    def boom(file: object, data: object) -> None:
        raise OSError(28, "No space left on device")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "_write_all", boom)
        with pytest.raises(trio.ContentWriteError) as excinfo:
            await trio.Path(target).write_bytes(b"x", strength=trio.WriteStrength.ATOMIC)
    assert excinfo.value.stage == "write"
    assert excinfo.value.errno == 28
    assert target.read_text() == "keep"
    assert _staging_files(tmp_path) == []
    # lock is released: another write works
    await trio.Path(target).write_bytes(b"after", strength=trio.WriteStrength.ATOMIC)
    assert target.read_bytes() == b"after"


async def test_flush_stage_error_preserves_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("keep")

    def boom(file: object, durable: bool) -> None:
        raise OSError(5, "flush failed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "_finish_file", boom)
        with pytest.raises(trio.FlushError) as excinfo:
            await trio.Path(target).write_text("x", strength=trio.WriteStrength.ATOMIC)
    assert excinfo.value.stage == "flush"
    assert target.read_text() == "keep"
    assert _staging_files(tmp_path) == []


async def test_replace_stage_error_preserves_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("keep")

    def boom(staging: object, dest: object, durable: bool) -> None:
        raise OSError(5, "replace failed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wp, "_replace_only", boom)
        with pytest.raises(trio.ReplaceError) as excinfo:
            await trio.Path(target).write_bytes(b"x", strength=trio.WriteStrength.DURABLE)
    assert excinfo.value.stage == "replace"
    assert target.read_text() == "keep"
    assert _staging_files(tmp_path) == []


async def test_rename_failure_is_replace_error(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "s"
    dst = tmp_path / "d"
    src.write_text("s")
    dst.write_text("d")
    with pytest.raises(trio.ReplaceError) as excinfo:
        await trio.Path(src).rename(dst, strength=trio.WriteStrength.ATOMIC)
    assert excinfo.value.stage == "replace"
    assert isinstance(excinfo.value, OSError)
    assert src.read_text() == "s"
    assert dst.read_text() == "d"


async def test_non_oserror_not_wrapped(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    with pytest.raises(ValueError, match="strength"):
        await trio.Path(target).write_bytes(b"x", strength="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Open-file pipeline
# ---------------------------------------------------------------------------


async def test_open_atomic_binary(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    async with await trio.open_file(target, "wb", strength=trio.WriteStrength.ATOMIC) as f:
        await f.write(b"bin")
    assert target.read_bytes() == b"bin"
    assert _all_pipeline_files(tmp_path) == []


async def test_open_rejects_unsupported_modes(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    for mode in ("r", "a", "r+b", "w+", "rb"):
        with pytest.raises(ValueError, match="strength"):
            await trio.open_file(target, mode, strength=trio.WriteStrength.ATOMIC)


async def test_open_rejects_fd_with_strength(tmp_path: pathlib.Path) -> None:
    fd = os.open(tmp_path / "raw", os.O_RDWR | os.O_CREAT)
    try:
        with pytest.raises(ValueError, match="file descriptor"):
            await trio.open_file(fd, "wb", strength=trio.WriteStrength.ATOMIC)
    finally:
        os.close(fd)


async def test_open_exclusive_mode(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    async with await trio.open_file(target, "x", strength=trio.WriteStrength.ATOMIC) as f:
        await f.write("first")
    assert target.read_text() == "first"
    with pytest.raises(FileExistsError):
        await trio.open_file(target, "x", strength=trio.WriteStrength.ATOMIC)


async def test_open_cancelled_before_close_discards(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "f"
    target.write_text("old")
    with trio.CancelScope() as cscope:
        f = await trio.open_file(target, "w", strength=trio.WriteStrength.ATOMIC)
        await f.write("new")
        cscope.cancel()
        with pytest.raises(trio.Cancelled):
            await f.aclose()
    assert target.read_text() == "old"
    assert _staging_files(tmp_path) == []


async def test_open_flush_failure_discards(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old")

    def boom(fd: int) -> None:
        raise OSError(5, "fsync failed")

    f = await trio.open_file(target, "wb", strength=trio.WriteStrength.DURABLE)
    await f.write(b"new")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "fsync", boom)
        with pytest.raises(trio.FlushError):
            await f.aclose()
    assert target.read_text() == "old"
    assert _staging_files(tmp_path) == []


async def test_guarded_write_failure_marks_discard(
    tmp_path: pathlib.Path,
) -> None:
    class BrokenFile:
        closed = False

        def write(self, data: object) -> int:
            raise OSError(28, "No space left on device")

        def writelines(self, lines: object) -> None:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

        def fileno(self) -> int:
            return -1

    wrapper = trio.wrap_file(BrokenFile())

    class State:
        failed = False

    state = State()
    wp.install_write_guards(wrapper, state)  # type: ignore[arg-type]
    with pytest.raises(trio.ContentWriteError) as excinfo:
        await wrapper.write(b"x")
    assert excinfo.value.stage == "write"
    assert state.failed is True


# ---------------------------------------------------------------------------
# Concurrency policy
# ---------------------------------------------------------------------------


async def test_conflict_fail_while_file_open(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("old")
    async with await trio.open_file(target, "w", strength=trio.WriteStrength.ATOMIC):
        with pytest.raises(trio.WriteConflictError):
            await trio.Path(target).write_bytes(
                b"late", strength=trio.WriteStrength.ATOMIC, conflict=trio.WriteConflict.FAIL
            )
        # same-task reentry also fails explicitly rather than deadlocking
        with pytest.raises(trio.WriteConflictError):
            await trio.open_file(
                target,
                "w",
                strength=trio.WriteStrength.ATOMIC,
                conflict=trio.WriteConflict.FAIL,
            )


async def test_conflict_wait_serializes(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("0")
    started = trio.Event()
    release = trio.Event()

    async def holder() -> None:
        async with await trio.open_file(target, "w", strength=trio.WriteStrength.ATOMIC) as f:
            started.set()
            await release.wait()
            await f.write("holder")

    async def latecomer() -> None:
        await started.wait()
        await trio.Path(target).write_bytes(
            b"latecomer", strength=trio.WriteStrength.ATOMIC
        )

    async with trio.open_nursery() as nursery:
        nursery.start_soon(holder)
        nursery.start_soon(latecomer)
        await wait_all_tasks_blocked()
        # The latecomer is waiting for the holder's staging file.
        assert target.read_text() == "0"
        release.set()

    assert target.read_text() in ("holder", "latecomer")


async def test_many_concurrent_writers_wait(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "c"
    target.write_text("start")

    async def writer(i: int) -> None:
        await trio.Path(target).write_text(
            f"writer-{i}", strength=trio.WriteStrength.ATOMIC
        )

    async with trio.open_nursery() as nursery:
        for i in range(8):
            nursery.start_soon(writer, i)

    assert target.read_text().startswith("writer-")
    assert _all_pipeline_files(tmp_path) == []


async def test_cross_process_lock_blocks_and_fails(tmp_path: pathlib.Path) -> None:
    """A lock held by another process is honoured by both policies."""

    target = tmp_path / "f"
    target.write_text("old")
    sidecar = wp._lock_path(target)
    os.makedirs(os.fspath(sidecar.parent), exist_ok=True)

    locked_fd = os.open(os.fspath(sidecar), os.O_RDWR | os.O_CREAT, 0o644)
    if sys.platform == "win32":  # pragma: no branch
        import msvcrt

        if os.fstat(locked_fd).st_size == 0:
            os.write(locked_fd, b"\0")
        os.lseek(locked_fd, 0, os.SEEK_SET)
        msvcrt.locking(locked_fd, msvcrt.LK_NBLCK, 1)
    else:  # pragma: no branch
        import fcntl

        fcntl.flock(locked_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(trio.WriteConflictError):
            await trio.Path(target).write_bytes(
                b"x",
                strength=trio.WriteStrength.ATOMIC,
                conflict=trio.WriteConflict.FAIL,
            )

        # WAIT blocks until the other process drops the lock.
        async def unlock_soon() -> None:
            await trio.sleep(0.2)
            if sys.platform == "win32":  # pragma: no branch
                import msvcrt

                os.lseek(locked_fd, 0, os.SEEK_SET)
                msvcrt.locking(locked_fd, msvcrt.LK_UNLCK, 1)
            else:  # pragma: no branch
                import fcntl

                fcntl.flock(locked_fd, fcntl.LOCK_UN)

        async with trio.open_nursery() as nursery:
            nursery.start_soon(unlock_soon)
            await trio.Path(target).write_bytes(
                b"waited", strength=trio.WriteStrength.ATOMIC
            )
        assert target.read_bytes() == b"waited"
    finally:
        os.close(locked_fd)


# ---------------------------------------------------------------------------
# Cancellation and cleanup
# ---------------------------------------------------------------------------


async def test_cancelled_write_cleans_up(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "f"
    target.write_text("stable")

    import time

    real_write_all = wp._write_all

    def slow_write_all(file: object, data: object) -> None:
        time.sleep(1)
        return real_write_all(file, data)  # type: ignore[arg-type]

    with trio.move_on_after(0.3):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(wp, "_write_all", slow_write_all)
            await trio.Path(target).write_bytes(
                b"late", strength=trio.WriteStrength.ATOMIC
            )

    assert target.read_text() == "stable"
    assert _staging_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Subprocess output redirection
# ---------------------------------------------------------------------------


async def test_subprocess_direct_redirect(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    await trio.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.write('direct')"],
        stdout=out,
    )
    assert out.read_text() == "direct"


async def test_subprocess_atomic_redirect(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    out.write_text("old")
    await trio.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.write('atomic')"],
        stdout=out,
        redirect_strength=trio.WriteStrength.ATOMIC,
    )
    assert out.read_text() == "atomic"
    assert _all_pipeline_files(tmp_path) == []


async def test_subprocess_durable_stderr_redirect(tmp_path: pathlib.Path) -> None:
    err = tmp_path / "err"
    await trio.run_process(
        [sys.executable, "-c", "import sys; sys.stderr.write('error')"],
        stderr=err,
        redirect_strength=trio.WriteStrength.DURABLE,
        check=False,
    )
    assert err.read_text() == "error"
    assert _all_pipeline_files(tmp_path) == []


async def test_subprocess_redirect_conflict_fail(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    async with await trio.open_file(
        out, "wb", strength=trio.WriteStrength.ATOMIC
    ):
        with pytest.raises(trio.WriteConflictError):
            await trio.run_process(
                [sys.executable, "-c", "print('x')"],
                stdout=out,
                redirect_strength=trio.WriteStrength.ATOMIC,
                redirect_conflict=trio.WriteConflict.FAIL,
            )
    # The redirect never spawned: target still missing (no commit happened).
    assert not out.exists() or out.read_bytes() == b""


async def test_subprocess_cancelled_redirect_discarded(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "out"
    out.write_text("KEPT")
    with trio.move_on_after(0.5):
        await trio.run_process(
            [
                sys.executable,
                "-c",
                "import sys,time; sys.stdout.write('x'*4096); "
                "sys.stdout.flush(); time.sleep(5)",
            ],
            stdout=out,
            redirect_strength=trio.WriteStrength.ATOMIC,
        )
    assert out.read_text() == "KEPT"
    assert _staging_files(tmp_path) == []
