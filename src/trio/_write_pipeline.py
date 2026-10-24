"""Unified write pipeline for Trio's filesystem write surface.

Every write that Trio performs on behalf of its users funnels through this
module.  Callers declare how strong they want the write to be
(:class:`WriteStrength`) and what should happen when another writer is
already working on the same target (:class:`WriteConflict`).

The pipeline knows four distinct stages, each of which can fail
independently:

``create``
    Creating the staging (temporary) file in the target's directory.

``write``
    Writing the payload into the staging file.

``flush``
    Flushing userspace buffers (and, for :attr:`WriteStrength.DURABLE`,
    calling ``fsync`` on the file descriptor).

``replace``
    The one-shot ``os.replace`` that publishes the staging file as the
    target (plus a directory ``fsync`` for :attr:`WriteStrength.DURABLE`).

Failures at each stage raise a dedicated :class:`WriteError` subclass, so
callers can tell the stages apart while still being able to catch plain
:exc:`OSError`.
"""

from __future__ import annotations

import atexit
import contextlib
import enum
import functools
import io
import itertools
import os
import pathlib
import sys
import uuid
from typing import TYPE_CHECKING, Final, Protocol, TypeVar, cast

import trio

from ._core import BusyResourceError, WouldBlock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ._file_io import AsyncIOWrapper

    # A custom opener passed through to io.open, mirroring builtins.open.
    _OpenerLike = Callable[[str, int], int]

T = TypeVar("T")


class _AsyncStage(Protocol):
    def __call__(self, *args: object) -> Awaitable[object]: ...


# Prefix used for every staging file / sidecar lock, so leftovers are
# recognisable and can be ignored by directory scanners.
_STAGING_PREFIX: Final = ".trio-write-"
_LOCK_SUFFIX: Final = ".lock"

_temp_counter = itertools.count()

# Resolved-target-keyed in-process locks.  Trio tasks are all scheduled on a
# single thread, so dict get/create needs no extra synchronisation.
_target_locks: dict[str, trio.Lock] = {}

# Which task currently holds the in-process lock for a target, so nested
# attempts from the same task fail explicitly instead of deadlocking.
_lock_owners: dict[str, object] = {}

# Staging files that currently exist on disk.  On a normal interpreter exit
# we sweep whatever is left over, so a cancelled run never leaves debris.
_live_staging: set[str] = set()


# ---------------------------------------------------------------------------
# Public declarative types
# ---------------------------------------------------------------------------


class WriteStrength(enum.Enum):
    """How strongly a write should be persisted.

    * :attr:`DIRECT` is plain in-place truncating writing, identical to the
      default behaviour of :func:`trio.open_file` and :class:`trio.Path`.
    * :attr:`ATOMIC` writes to a same-directory staging file and publishes it
      with a single atomic ``os.replace``.  Readers either see the complete
      old file or the complete new one, but no extra disk flushes are forced.
    * :attr:`DURABLE` is like :attr:`ATOMIC`, but additionally flushes the
      file contents (``fsync``) and the containing directory entry, so the
      replacement is still visible after a power loss.
    """

    DIRECT = "direct"
    ATOMIC = "atomic"
    DURABLE = "durable"


class WriteConflict(enum.Enum):
    """What happens when a target already has an atomic writer registered.

    * :attr:`WAIT` serialises the two writers (the default).
    * :attr:`FAIL` makes the late writer raise
      :class:`WriteConflictError` immediately, instead of silently
      overwriting the other writer's result.
    """

    WAIT = "wait"
    FAIL = "fail"


class WriteError(OSError):
    """Base class for failures of the declared write pipeline.

    All subclasses are also :exc:`OSError` subclasses, so existing
    ``except OSError`` handlers keep working.  The :attr:`stage` attribute
    identifies which stage of the pipeline failed; it is one of
    ``"create"``, ``"write"``, ``"flush"`` or ``"replace"``.
    """

    stage: str = ""

    def __str__(self) -> str:
        message = super().__str__()
        return f"write pipeline failed at the {self.stage!r} stage: {message}"


class TempFileCreateError(WriteError):
    """Creating the same-directory staging file failed."""

    stage = "create"


class ContentWriteError(WriteError):
    """Writing the payload into the staging file failed."""

    stage = "write"


class FlushError(WriteError):
    """Flushing (or ``fsync``-ing) the staging file failed."""

    stage = "flush"


class ReplaceError(WriteError):
    """The atomic ``os.replace`` (or directory ``fsync``) failed."""

    stage = "replace"


class WriteConflictError(BusyResourceError):
    """Raised for a late writer when :attr:`WriteConflict.FAIL` is declared
    and another writer is already committed to the same target."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_strength(strength: WriteStrength | str | None) -> WriteStrength | None:
    if strength is None or isinstance(strength, WriteStrength):
        return strength
    if isinstance(strength, str):
        try:
            return WriteStrength(strength)
        except ValueError:
            pass
    raise ValueError(
        "strength must be a trio.WriteStrength member or one of "
        f"'direct', 'atomic', 'durable', not {strength!r}",
    )


def _coerce_conflict(conflict: WriteConflict | str) -> WriteConflict:
    if isinstance(conflict, WriteConflict):
        return conflict
    if isinstance(conflict, str):
        try:
            return WriteConflict(conflict)
        except ValueError:
            pass
    raise ValueError(
        "conflict must be a trio.WriteConflict member or 'wait' or 'fail', "
        f"not {conflict!r}",
    )


def _target_path(
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
) -> pathlib.Path:
    return pathlib.Path(os.fsdecode(os.fspath(target)))


def _lock_key(target: pathlib.Path) -> str:
    return os.path.realpath(os.fspath(target))


def _get_target_lock(key: str) -> trio.Lock:
    lock = _target_locks.get(key)
    if lock is None:
        lock = trio.Lock()
        _target_locks[key] = lock
    return lock


def _conflict_error(target: pathlib.Path) -> WriteConflictError:
    return WriteConflictError(
        f"another writer is already writing to {target!s}",
    )


async def acquire_target_lock(
    target: pathlib.Path, conflict: WriteConflict
) -> trio.Lock:
    """Acquire the per-target in-process lock.

    Re-entry from the task that already holds the lock always raises
    :class:`WriteConflictError` -- the underlying lock is not reentrant and
    waiting would deadlock.
    """

    key = _lock_key(target)
    lock = _get_target_lock(key)
    current_task = trio.lowlevel.current_task()
    if _lock_owners.get(key) is current_task:
        raise _conflict_error(target)
    if conflict is WriteConflict.FAIL:
        try:
            lock.acquire_nowait()
        except WouldBlock:
            raise _conflict_error(target) from None
    else:
        await lock.acquire()
    _lock_owners[key] = current_task
    return lock


def release_target_lock(target: pathlib.Path, lock: trio.Lock) -> None:
    _lock_owners.pop(_lock_key(target), None)
    lock.release()


def _staging_path(target: pathlib.Path) -> pathlib.Path:
    name = (
        f"{_STAGING_PREFIX}{target.name}-{os.getpid()}-"
        f"{next(_temp_counter)}-{uuid.uuid4().hex[:8]}.tmp"
    )
    return target.parent / name


def _lock_path(target: pathlib.Path) -> pathlib.Path:
    return target.parent / f"{_STAGING_PREFIX}{target.name}{_LOCK_SUFFIX}"


def _register_staging(path: pathlib.Path) -> None:
    _live_staging.add(os.fspath(path))


def _unregister_staging(path: pathlib.Path) -> None:
    _live_staging.discard(os.fspath(path))


def _cleanup_on_exit() -> None:
    # Normal interpreter exit only -- nothing here can survive a hard kill,
    # but unique staging names mean a leftover staging file is harmless and
    # the target itself is always left untouched.
    for raw_path in list(_live_staging):
        with contextlib.suppress(OSError):
            os.unlink(raw_path)


atexit.register(_cleanup_on_exit)


def _wrap_os_error(error: OSError, wrapper: type[WriteError]) -> WriteError:
    # Reconstruct the same errno/strerror/filename triple so the result
    # quacks exactly like the original OSError.
    try:
        return wrapper(*error.args)
    except TypeError:
        return wrapper(str(error))


# ---------------------------------------------------------------------------
# Cross-process advisory locks
# ---------------------------------------------------------------------------


class _LockUnavailable(Exception):
    """Internal: a non-blocking cross-process lock acquire failed."""


if sys.platform == "win32":  # pragma: no branch - branch selected at import time

    def _xlock_acquire(sidecar: str, blocking: bool) -> int:
        import msvcrt

        fd = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(fd, mode, 1)
            except OSError:
                raise _LockUnavailable from None
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _xlock_unlock(fd: int) -> None:
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            os.close(fd)

else:  # pragma: no branch - branch selected at import time

    def _xlock_acquire(sidecar: str, blocking: bool) -> int:
        import fcntl

        fd = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o644)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            raise _LockUnavailable from None
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _xlock_unlock(fd: int) -> None:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)


async def _acquire_xlock(sidecar: pathlib.Path, conflict: WriteConflict) -> int:
    raw_sidecar = os.fspath(sidecar)
    while True:
        try:
            return await trio.to_thread.run_sync(_xlock_acquire, raw_sidecar, False)
        except _LockUnavailable:
            if conflict is WriteConflict.FAIL:
                raise WriteConflictError(
                    f"another writer is already writing to {sidecar!s}",
                ) from None
            await trio.sleep(0.02)
        except OSError as error:
            # Creating the sidecar lock is part of staging preparation.
            raise _wrap_os_error(error, TempFileCreateError) from error


def _release_xlock(fd: int, sidecar: pathlib.Path) -> None:
    try:
        _xlock_unlock(fd)
    finally:
        # Best effort; the sidecar is harmless if another waiter still holds it.
        with contextlib.suppress(OSError):
            os.unlink(os.fspath(sidecar))


# ---------------------------------------------------------------------------
# Filesystem stage primitives (blocking; run in worker threads)
# ---------------------------------------------------------------------------


def _fsync_parent_dir(target: pathlib.Path) -> None:
    if os.name != "posix":
        # os.replace is atomic on NTFS and metadata is journalled; there is no
        # portable directory-fsync equivalent on Windows.
        return
    directory = os.fspath(target.parent)
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is not None:
        flags |= directory_flag
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_staging(
    staging: pathlib.Path,
    mode: str,
    buffering: int,
    encoding: str | None,
    errors: str | None,
    newline: str | None,
    opener: _OpenerLike | None,
) -> io.IOBase:
    return cast(
        "io.IOBase",
        open(
            os.fspath(staging),
            mode,
            buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            opener=opener,
        ),
    )


def _open_staging_raw(staging: pathlib.Path, mode: int, perms: int = 0o666) -> int:
    return os.open(os.fspath(staging), mode, perms)


def _write_all(file: io.IOBase, data: bytes | str) -> None:
    if isinstance(file, io.RawIOBase):
        view = memoryview(data)  # type: ignore[arg-type]
        while view:
            written = file.write(view)
            if written is None or written == 0:
                raise OSError("short write to staging file")
            view = view[written:]
    else:
        file.write(data)


def _finish_file(file: io.IOBase, durable: bool) -> None:
    file.flush()
    if durable:
        os.fsync(file.fileno())
    file.close()


def _fsync_fd(fd: int, durable: bool) -> None:
    if durable:
        os.fsync(fd)


def _close_quietly(file: io.IOBase | int | None) -> None:
    if file is None:
        return
    try:
        if isinstance(file, int):
            os.close(file)
        else:
            file.close()
    except OSError:
        pass


def _replace_only(staging: pathlib.Path, target: pathlib.Path, durable: bool) -> None:
    os.replace(os.fspath(staging), os.fspath(target))
    if durable:
        _fsync_parent_dir(target)


def _discard_staging(
    staging: pathlib.Path | None,
    file: io.IOBase | int | None,
    xlock_fd: int | None,
    sidecar: pathlib.Path | None,
) -> None:
    _close_quietly(file)
    if staging is not None:
        with contextlib.suppress(OSError):
            os.unlink(os.fspath(staging))
        _unregister_staging(staging)
    if xlock_fd is not None and sidecar is not None:
        _release_xlock(xlock_fd, sidecar)


async def _run_stage(
    error_cls: type[WriteError],
    func: object,
    *args: object,
) -> T:
    try:
        return await trio.to_thread.run_sync(
            cast(
                "Callable[[], T]",
                functools.partial(func, *args),  # type: ignore[operator,arg-type,misc]
            ),
        )
    except OSError as error:
        raise _wrap_os_error(error, error_cls) from error


# ---------------------------------------------------------------------------
# Whole-content writes (trio.Path)
# ---------------------------------------------------------------------------


async def write_whole(
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    data: bytes | str,
    *,
    text: bool,
    encoding: str | None,
    errors: str | None,
    newline: str | None,
    strength: WriteStrength | str | None,
    conflict: WriteConflict | str,
) -> None:
    strength = _coerce_strength(strength)
    conflict = _coerce_conflict(conflict)
    target_path = _target_path(target)

    # Declared default / explicit direct: byte-for-byte legacy behaviour.
    if strength is None or strength is WriteStrength.DIRECT:
        if text:

            def _direct_write_text() -> None:
                target_path.write_text(
                    data,  # type: ignore[arg-type]
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                )

            await trio.to_thread.run_sync(_direct_write_text)
        else:
            await trio.to_thread.run_sync(
                lambda: target_path.write_bytes(data)  # type: ignore[arg-type]
            )
        return

    durable = strength is WriteStrength.DURABLE
    lock = await acquire_target_lock(target_path, conflict)

    staging = _staging_path(target_path)
    sidecar = _lock_path(target_path)
    file: io.IOBase | None = None
    xlock_fd: int | None = None
    try:
        xlock_fd = await _acquire_xlock(sidecar, conflict)
        mode = "w" if text else "wb"
        file = await _run_stage(
            TempFileCreateError,
            _open_staging,
            staging,
            mode,
            -1,
            encoding if text else None,
            errors if text else None,
            newline if text else None,
            None,
        )
        _register_staging(staging)
        await _run_stage(ContentWriteError, _write_all, file, data)
        await _run_stage(FlushError, _finish_file, file, durable)
        file = None
        await _run_stage(ReplaceError, _replace_only, staging, target_path, durable)
        _unregister_staging(staging)
        _release_xlock(xlock_fd, sidecar)
        xlock_fd = None
    except BaseException:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(
                _discard_staging, staging, file, xlock_fd, sidecar
            )
        raise
    finally:
        release_target_lock(target_path, lock)


async def touch(
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mode: int = 0o666,
    exist_ok: bool = True,
    *,
    strength: WriteStrength | str | None = None,
    conflict: WriteConflict | str = WriteConflict.WAIT,
) -> None:
    strength = _coerce_strength(strength)
    conflict = _coerce_conflict(conflict)
    target_path = _target_path(target)

    if strength is None or strength is WriteStrength.DIRECT:
        await trio.to_thread.run_sync(
            lambda: target_path.touch(mode=mode, exist_ok=exist_ok)
        )
        return

    durable = strength is WriteStrength.DURABLE
    lock = await acquire_target_lock(target_path, conflict)

    staging = _staging_path(target_path)
    sidecar = _lock_path(target_path)
    fd: int | None = None
    xlock_fd: int | None = None
    try:
        xlock_fd = await _acquire_xlock(sidecar, conflict)

        def _create_empty() -> int:
            if not exist_ok and os.path.exists(os.fspath(target_path)):
                # Native type, matching pathlib.Path.touch(exist_ok=False).
                raise FileExistsError(17, "File exists", os.fspath(target_path))
            return os.open(
                os.fspath(staging),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode,
            )

        try:
            fd = await trio.to_thread.run_sync(_create_empty)
        except OSError as error:
            # FileExistsError from exist_ok=False must keep its native type.
            if isinstance(error, FileExistsError):
                raise
            raise _wrap_os_error(error, TempFileCreateError) from error
        _register_staging(staging)
        await _run_stage(FlushError, _fsync_fd, fd, durable)
        await trio.to_thread.run_sync(os.close, fd)
        fd = None
        await _run_stage(ReplaceError, _replace_only, staging, target_path, durable)
        _unregister_staging(staging)
        _release_xlock(xlock_fd, sidecar)
        xlock_fd = None
    except BaseException:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(
                _discard_staging, staging, fd, xlock_fd, sidecar
            )
        raise
    finally:
        release_target_lock(target_path, lock)


async def move(
    source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    *,
    overwrite: bool,
    strength: WriteStrength | str | None,
    conflict: WriteConflict | str,
) -> pathlib.Path:
    strength = _coerce_strength(strength)
    conflict = _coerce_conflict(conflict)
    source_path = _target_path(source)
    target_path = _target_path(target)

    if strength is None or strength is WriteStrength.DIRECT:
        operation = os.replace if overwrite else os.rename
        await trio.to_thread.run_sync(
            operation, os.fspath(source_path), os.fspath(target_path)
        )
        return target_path

    durable = strength is WriteStrength.DURABLE

    # Rename is itself the commit step, so serialise against writers of both
    # the source name and the destination name (sorted to avoid deadlocks).
    paths_by_key = {
        _lock_key(source_path): source_path,
        _lock_key(target_path): target_path,
    }
    ordered_keys = sorted(paths_by_key)
    acquired_keys: list[str] = []
    sidecar = _lock_path(target_path)
    xlock_fd: int | None = None
    try:
        for key in ordered_keys:
            await acquire_target_lock(paths_by_key[key], conflict)
            acquired_keys.append(key)

        xlock_fd = await _acquire_xlock(sidecar, conflict)

        def _do_move() -> None:
            try:
                if overwrite:
                    os.replace(os.fspath(source_path), os.fspath(target_path))
                else:
                    os.rename(os.fspath(source_path), os.fspath(target_path))
            except OSError as error:
                raise _wrap_os_error(error, ReplaceError) from error
            if durable:
                _fsync_parent_dir(target_path)
                if os.path.dirname(os.fspath(source_path)) != os.path.dirname(
                    os.fspath(target_path),
                ):
                    _fsync_parent_dir(source_path)

        await trio.to_thread.run_sync(_do_move)
    finally:
        if xlock_fd is not None:
            _release_xlock(xlock_fd, sidecar)
        for key in reversed(acquired_keys):
            release_target_lock(paths_by_key[key], _get_target_lock(key))

    return target_path


# ---------------------------------------------------------------------------
# Streaming writes (trio.open_file / trio.Path.open)
# ---------------------------------------------------------------------------


def parse_atomic_mode(mode: str) -> tuple[str, str]:
    """Validate that *mode* describes a truncating/exclusive whole-file write."""

    if not isinstance(mode, str):
        raise TypeError("open mode must be a string")
    create: str | None = None
    kind = "t"
    for char in mode:
        if char in ("w", "x"):
            if create is not None:
                raise ValueError(
                    f"mode {mode!r} cannot be combined with a write strength",
                )
            create = char
        elif char in ("b", "t"):
            kind = char
        else:
            raise ValueError(
                f"mode {mode!r} cannot be combined with a write strength: "
                "only pure write modes ('w'/'x', text or binary) are "
                "supported because the target is replaced atomically on close",
            )
    if create is None:
        raise ValueError(
            "a write strength requires a truncating ('w') or exclusive "
            f"('x') mode, not {mode!r}; open without declaring a strength "
            "for reading, appending or updating in place",
        )
    return create, kind


class AtomicFileState:
    """Bookkeeping attached to an :class:`AsyncIOWrapper` opened with a
    declared write strength."""

    def __init__(
        self,
        file: io.IOBase,
        target: pathlib.Path,
        staging: pathlib.Path,
        sidecar: pathlib.Path,
        strength: WriteStrength,
        lock: trio.Lock,
        xlock_fd: int,
    ) -> None:
        self.file = file
        self.target = target
        self.staging = staging
        self.sidecar = sidecar
        self.strength = strength
        self.lock = lock
        self.xlock_fd = xlock_fd
        self.failed = False
        self.finalized = False

    @property
    def durable(self) -> bool:
        return self.strength is WriteStrength.DURABLE

    def close_and_fsync(self) -> None:
        if not self.file.closed:
            self.file.flush()
            if self.durable:
                os.fsync(self.file.fileno())
            self.file.close()

    def commit(self) -> None:
        os.replace(os.fspath(self.staging), os.fspath(self.target))
        if self.durable:
            _fsync_parent_dir(self.target)

    def publish(self) -> None:
        _unregister_staging(self.staging)
        _release_xlock(self.xlock_fd, self.sidecar)
        self.finalized = True

    def cleanup(self) -> None:
        try:
            _discard_staging(
                self.staging, self.file, self.xlock_fd, self.sidecar
            )
        finally:
            self.finalized = True


def install_write_guards(
    wrapper: AsyncIOWrapper[io.IOBase], state: AtomicFileState
) -> None:
    """Classify write/flush failures on *wrapper* and mark it for discard."""

    for method_name, error_cls in (
        ("write", ContentWriteError),
        ("writelines", ContentWriteError),
        ("flush", FlushError),
    ):
        if not hasattr(wrapper.wrapped, method_name):
            continue
        original = cast("_AsyncStage", getattr(wrapper, method_name))

        async def guarded(
            *args: object,
            _original: _AsyncStage = original,
            _error_cls: type[WriteError] = error_cls,
        ) -> object:
            try:
                return await _original(*args)
            except OSError as error:
                state.failed = True
                raise _wrap_os_error(error, _error_cls) from error

        setattr(wrapper, method_name, guarded)


async def finalize_atomic_file(state: AtomicFileState) -> None:
    """Close-and-commit (or discard) an atomically opened file."""

    if state.finalized:
        await trio.lowlevel.checkpoint_if_cancelled()
        return

    # If cancellation is already in flight when close is requested, discard
    # the staging file instead of publishing a partial write.
    try:
        await trio.lowlevel.checkpoint_if_cancelled()
    except BaseException:
        try:
            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(state.cleanup)
        finally:
            release_target_lock(state.target, state.lock)
        raise

    try:
        if state.failed:
            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(state.cleanup)
            return
        try:
            with trio.CancelScope(shield=True):
                await _run_stage(FlushError, state.close_and_fsync)
                await _run_stage(ReplaceError, state.commit)
        except BaseException:
            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(state.cleanup)
            raise
        state.publish()
    finally:
        release_target_lock(state.target, state.lock)

    await trio.lowlevel.checkpoint_if_cancelled()


async def open_atomic(
    file: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    mode: str,
    buffering: int,
    encoding: str | None,
    errors: str | None,
    newline: str | None,
    opener: _OpenerLike | None,
    *,
    strength: WriteStrength,
    conflict: WriteConflict,
) -> tuple[io.IOBase, AtomicFileState]:
    """Open the staging file for an atomic whole-file write."""

    create, _kind = parse_atomic_mode(mode)
    target_path = _target_path(file)

    lock = await acquire_target_lock(target_path, conflict)

    staging = _staging_path(target_path)
    sidecar = _lock_path(target_path)
    xlock_fd: int | None = None
    opened: io.IOBase | None = None
    try:
        xlock_fd = await _acquire_xlock(sidecar, conflict)

        def _check_exclusive_and_open() -> io.IOBase:
            if create == "x" and os.path.exists(os.fspath(target_path)):
                # Native type, matching a regular exclusive open.
                raise FileExistsError(17, "File exists", os.fspath(target_path))
            return _open_staging(
                staging,
                mode,
                buffering,
                encoding,
                errors,
                newline,
                opener,
            )

        try:
            opened = await trio.to_thread.run_sync(_check_exclusive_and_open)
        except OSError as error:
            # FileExistsError must keep its native type.
            if isinstance(error, FileExistsError):
                raise
            raise _wrap_os_error(error, TempFileCreateError) from error
        _register_staging(staging)
    except BaseException:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(
                _discard_staging, staging, opened, xlock_fd, sidecar
            )
        release_target_lock(target_path, lock)
        raise

    return opened, AtomicFileState(
        opened,
        target_path,
        staging,
        sidecar,
        strength,
        lock,
        xlock_fd,
    )


# ---------------------------------------------------------------------------
# Subprocess output redirection
# ---------------------------------------------------------------------------


def is_path_like(value: object) -> bool:
    return isinstance(value, (str, bytes, os.PathLike))


class OutputRedirect:
    """A child output stream that lands in a same-directory staging file."""

    def __init__(
        self,
        target: pathlib.Path,
        staging: pathlib.Path,
        sidecar: pathlib.Path,
        fd: int,
        xlock_fd: int,
        lock: trio.Lock,
        durable: bool,
    ) -> None:
        self.target = target
        self.staging = staging
        self.sidecar = sidecar
        self.fd = fd
        self.xlock_fd = xlock_fd
        self.lock = lock
        self.durable = durable
        self.finalized = False

    def finalize(self, abandon: bool) -> None:
        if self.finalized:
            return
        self.finalized = True
        try:
            if abandon:
                _discard_staging(self.staging, self.fd, self.xlock_fd, self.sidecar)
                return
            try:
                try:
                    if self.durable:
                        os.fsync(self.fd)
                except OSError as error:
                    raise _wrap_os_error(error, FlushError) from error
                finally:
                    os.close(self.fd)
                try:
                    os.replace(os.fspath(self.staging), os.fspath(self.target))
                    if self.durable:
                        _fsync_parent_dir(self.target)
                except OSError as error:
                    raise _wrap_os_error(error, ReplaceError) from error
            except BaseException:
                # Staging still exists (or fd is already closed); sweep it and
                # release the locks so the target stays untouched.
                _discard_staging(self.staging, None, self.xlock_fd, self.sidecar)
                raise
            _unregister_staging(self.staging)
            _release_xlock(self.xlock_fd, self.sidecar)
        finally:
            release_target_lock(self.target, self.lock)


async def redirect_output(
    destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    *,
    strength: WriteStrength | str | None,
    conflict: WriteConflict | str,
) -> tuple[int, OutputRedirect | None]:
    """Prepare an fd for a child process to write its output into.

    Returns the fd to hand to :class:`subprocess.Popen` and an
    :class:`OutputRedirect` to finalise once the child has exited (or
    ``None`` for direct in-place redirection).
    """

    strength = _coerce_strength(strength)
    conflict = _coerce_conflict(conflict)
    target_path = _target_path(destination)

    if strength is None or strength is WriteStrength.DIRECT:
        direct_fd = await trio.to_thread.run_sync(
            lambda: os.open(
                os.fspath(target_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o666,
            ),
        )
        return direct_fd, None

    durable = strength is WriteStrength.DURABLE
    lock = await acquire_target_lock(target_path, conflict)

    staging = _staging_path(target_path)
    sidecar = _lock_path(target_path)
    xlock_fd: int | None = None
    fd: int | None = None
    created_fd: int
    try:
        xlock_fd = await _acquire_xlock(sidecar, conflict)
        created_fd = await _run_stage(
            TempFileCreateError,
            _open_staging_raw,
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        fd = created_fd
        _register_staging(staging)
    except BaseException:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(
                _discard_staging, staging, fd, xlock_fd, sidecar
            )
        release_target_lock(target_path, lock)
        raise

    return created_fd, OutputRedirect(
        target_path,
        staging,
        sidecar,
        created_fd,
        xlock_fd,
        lock,
        durable,
    )


def finalize_redirects(redirects: list[OutputRedirect], abandon: bool) -> None:
    error: BaseException | None = None
    for redirect in redirects:
        try:
            redirect.finalize(abandon)
        except BaseException as exc:
            if error is None:
                error = exc
    if error is not None:
        raise error
