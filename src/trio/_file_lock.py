"""Asynchronous, cross-process file locks.

This module implements advisory file locks that work both between different
processes and between different tasks in the same process.  The lock is an
ordinary byte-range/``flock`` lock taken out on a small *sidecar* file
(``"<target>.trio-lock"``), which also carries a fixed-size metadata record
describing the current holder (process id, acquisition time, ...).

Design properties:

* The actual mutual exclusion is always enforced by the operating system
  (``fcntl.flock`` on POSIX, ``LockFileEx`` on Windows).  We never pretend a
  lock was acquired when the platform could not grant one; if the platform
  lacks locking support for a filesystem, :class:`FileLockUnsupported` is
  raised.
* OS locks are bound to the open file description / handle, so the kernel
  releases them automatically when the descriptor is closed -- including when
  the holding process crashes or is killed.  A sidecar left behind by a dead
  holder is detected via its metadata and taken over according to the
  configured :class:`StaleLockPolicy`, with the evidence recorded in
  :attr:`FileLock.last_takeover`.
* Waiting happens in worker threads, but every wait is fully cancellable and
  deadline aware: on cancellation/timeout the lock is either not held, or is
  released and its descriptor closed by the cleanup path -- no descriptor or
  half-open lock is ever leaked.
"""

from __future__ import annotations

import contextlib
import enum
import errno
import json
import math
import os
import platform
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias, cast

import trio

from ._abc import AsyncResource
from ._core import BusyResourceError, ClosedResourceError
from ._util import final

if TYPE_CHECKING:
    import io
    from collections.abc import AsyncIterator
    from types import TracebackType

    from typing_extensions import Self

    from ._file_io import AsyncIOWrapper

    StrOrBytesPath: TypeAlias = str | bytes | "os.PathLike[str]" | "os.PathLike[bytes]"

#: Suffix appended to the locked target to form the sidecar (metadata +
#: OS lock) file name.
SIDECAR_SUFFIX = ".trio-lock"

#: Size of the fixed-width metadata record at the start of the sidecar.
_METADATA_SIZE = 512

#: Byte offset of the single locked byte on Windows.  It lives well past the
#: metadata region so that contenders can always read holder information out
#: of the (unlocked) beginning of the file.  Windows allows locking ranges
#: that extend past end of file.
_LOCK_BYTE_OFFSET = 0x10000

_HOST = platform.node()


###############################################################################
# Exceptions and policy types
###############################################################################


class FileLockError(Exception):
    """Base class for errors from the file-locking machinery."""


class FileLockUnsupported(FileLockError, RuntimeError):
    """Raised when the platform (or the filesystem the sidecar lives on)
    cannot provide the locking primitives this module relies on.

    When this is raised, no lock has been acquired.  Trio never silently
    pretends that locking succeeded.
    """


class FileLockTimeout(FileLockError):
    """Raised by :meth:`FileLock.acquire` when ``timeout`` elapses before the
    lock becomes available.

    This deliberately inherits only from :class:`FileLockError`, *not* from
    :class:`TimeoutError` (which is an :class:`OSError` subclass): a timeout
    is not an OS-level locking failure, and it must not be swallowed by code
    that retries on ``except OSError``.  It is therefore trivially
    distinguishable from every other failure (which raises :class:`OSError`,
    :class:`StaleLockError`, ...).  The last readable holder information is
    attached as :attr:`holder`.
    """

    def __init__(self, holder: LockHolder | None = None) -> None:
        self.holder = holder
        if holder is None:
            msg = "timed out waiting for file lock"
        else:
            msg = f"timed out waiting for file lock; current holder: {holder}"
        super().__init__(msg)


class StaleLockError(FileLockError):
    """Raised when a lock appears to be stale and the configured policy does
    not allow taking it over.

    Attributes:
      holder: The metadata record found in the sidecar.
      evidence: A human-readable explanation of why the lock was judged
          stale and why it could not be taken over.
    """

    def __init__(self, holder: LockHolder | None, evidence: str) -> None:
        self.holder = holder
        self.evidence = evidence
        super().__init__(f"refusing to take over a stale file lock: {evidence}")


class StaleLockPolicy(enum.Enum):
    """What to do when the sidecar names a holder whose process no longer
    exists while the OS lock is still held (typically because a child
    inherited the lock descriptor).

    * ``WAIT`` (the default): keep waiting for the OS lock.  This is always
      safe -- the kernel releases the lock as soon as whatever process holds
      it exits.
    * ``ERROR``: raise :class:`StaleLockError` immediately instead of
      waiting.
    * ``TAKE_OVER``: make one immediate non-blocking attempt to grab the OS
      lock.  If it is actually free (the holder process exited without
      zeroing its metadata), the lock is taken over and the decision is
      recorded on :attr:`FileLock.last_takeover`.  If some live process is
      still holding it (e.g. an inheriting child), Trio cannot forcibly
      steal an OS lock, so :class:`StaleLockError` is raised rather than
      pretending success.
    """

    WAIT = "wait"
    ERROR = "error"
    TAKE_OVER = "take_over"


@final
class LockHolder:
    """Information about a lock holder, as recorded in the sidecar.

    Attributes:
      pid: Process id of the holder.
      acquired_at: Wall-clock time (:func:`time.time`) at which the holder
          claims to have acquired the lock.
      host: Host name recorded by the holder.
      exclusive: Whether the holder took an exclusive lock (as opposed to a
          shared one).
      instance_id: Unique id generated for each individual acquisition; use
          it to tell acquisitions apart.
      unreadable: ``True`` if the metadata region could not be parsed (for
          example a torn write); all other fields are then unset.
    """

    __slots__ = ("acquired_at", "exclusive", "host", "instance_id", "pid", "unreadable")

    def __init__(
        self,
        pid: int | None = None,
        acquired_at: float | None = None,
        host: str | None = None,
        exclusive: bool | None = None,
        instance_id: str | None = None,
        unreadable: bool = False,
    ) -> None:
        self.pid = pid
        self.acquired_at = acquired_at
        self.host = host
        self.exclusive = exclusive
        self.instance_id = instance_id
        self.unreadable = unreadable

    @property
    def age(self) -> float | None:
        """Seconds since :attr:`acquired_at`, or ``None`` if unknown."""
        if self.acquired_at is None:
            return None
        return time.time() - self.acquired_at

    @property
    def is_local(self) -> bool:
        """Whether the record names this host (liveness checks only work
        locally)."""
        return self.host is not None and self.host == _HOST

    def __repr__(self) -> str:
        if self.unreadable:
            return "<LockHolder: unreadable metadata>"
        return (
            f"<LockHolder pid={self.pid!r} host={self.host!r} "
            f"acquired_at={self.acquired_at!r} exclusive={self.exclusive!r}>"
        )


@final
class LockTakeover(NamedTuple):
    """Diagnostic record attached to :attr:`FileLock.last_takeover` whenever a
    lock was acquired over a sidecar left by a previous holder."""

    previous: LockHolder | None
    evidence: str
    at: float


###############################################################################
# Platform backends
###############################################################################


class _Backend:
    """The handful of OS operations the lock protocol needs."""

    __slots__ = ()

    def open_sidecar(self, path: str | bytes) -> int:
        raise FileLockUnsupported  # pragma: no cover

    def try_lock(self, fd: int, exclusive: bool) -> bool:
        raise FileLockUnsupported  # pragma: no cover

    def lock(self, fd: int, exclusive: bool) -> None:
        raise FileLockUnsupported  # pragma: no cover

    def unlock(self, fd: int) -> None:
        raise FileLockUnsupported  # pragma: no cover

    def close(self, fd: int) -> None:
        raise FileLockUnsupported  # pragma: no cover

    def pid_alive(self, pid: int) -> bool | None:
        """Return True/False if liveness can be determined, else None."""
        raise FileLockUnsupported  # pragma: no cover


if sys.platform != "win32":
    # POSIX (flock is available on every Unix-like platform Trio supports)
    import fcntl

    _UNSUPPORTED_ERRNOS = {errno.ENOSYS, errno.ENOLCK}
    if hasattr(errno, "EOPNOTSUPP"):
        _UNSUPPORTED_ERRNOS.add(errno.EOPNOTSUPP)

    @final
    class _PosixBackend(_Backend):
        __slots__ = ()

        def open_sidecar(self, path: str | bytes) -> int:
            return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)

        def try_lock(self, fd: int, exclusive: bool) -> bool:
            flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(fd, flags | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    return False
                if exc.errno in _UNSUPPORTED_ERRNOS:
                    raise FileLockUnsupported(
                        f"flock() is not supported on this filesystem (errno "
                        f"{exc.errno}: {os.strerror(exc.errno)})",
                    ) from exc
                raise
            return True

        def lock(self, fd: int, exclusive: bool) -> None:
            flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(fd, flags)
            except OSError as exc:
                if exc.errno in _UNSUPPORTED_ERRNOS:
                    raise FileLockUnsupported(
                        f"flock() is not supported on this filesystem (errno "
                        f"{exc.errno}: {os.strerror(exc.errno)})",
                    ) from exc
                raise

        def unlock(self, fd: int) -> None:
            fcntl.flock(fd, fcntl.LOCK_UN)

        def close(self, fd: int) -> None:
            os.close(fd)

        def pid_alive(self, pid: int) -> bool | None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                # Process exists, we just may not signal it.
                return True
            except OSError:
                return None
            return True

    _backend: _Backend = _PosixBackend()

elif sys.platform == "win32":  # pragma: no branch
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _ERROR_LOCK_VIOLATION = 33
    _ERROR_NOT_SUPPORTED = 50
    _ERROR_INVALID_FUNCTION = 1
    _ERROR_BAD_COMMAND = 22
    _ERROR_INVALID_PARAMETER = 87
    _ERROR_ACCESS_DENIED = 5
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259

    _LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    _LOCKFILE_FAIL_IMMEDIATELY = 0x00000001

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.CloseHandle.restype = wintypes.BOOL

    class _Overlapped(ctypes.Structure):
        _fields_ = (  # type: ignore[mutable-override]
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("Pointer", ctypes.c_void_p),
            ("hEvent", wintypes.HANDLE),
        )

    def _win_error(msg: str) -> OSError:
        err = ctypes.get_last_error()
        return OSError(err, f"{msg} (Windows error {err})")

    @final
    class _WindowsBackend(_Backend):
        __slots__ = ()

        def _overlapped(self) -> _Overlapped:
            ov = _Overlapped()
            ov.Offset = _LOCK_BYTE_OFFSET & 0xFFFFFFFF
            ov.OffsetHigh = _LOCK_BYTE_OFFSET >> 32
            return ov

        def open_sidecar(self, path: str | bytes) -> int:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
            return os.open(path, flags, 0o644)

        def _lock(self, fd: int, exclusive: bool, block: bool) -> bool:
            handle = msvcrt.get_osfhandle(fd)
            flags = _LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0
            if not block:
                flags |= _LOCKFILE_FAIL_IMMEDIATELY
            ov = self._overlapped()
            ok = _kernel32.LockFileEx(
                wintypes.HANDLE(handle),
                wintypes.DWORD(flags),
                wintypes.DWORD(0),
                wintypes.DWORD(1),
                wintypes.DWORD(0),
                ctypes.byref(ov),
            )
            if ok:
                return True
            err = ctypes.get_last_error()
            if err == _ERROR_LOCK_VIOLATION:
                return False
            if err in (
                _ERROR_NOT_SUPPORTED,
                _ERROR_INVALID_FUNCTION,
                _ERROR_BAD_COMMAND,
            ):
                raise FileLockUnsupported(
                    f"LockFileEx is not supported on this filesystem "
                    f"(Windows error {err})",
                )
            raise _win_error("LockFileEx failed")

        def try_lock(self, fd: int, exclusive: bool) -> bool:
            return self._lock(fd, exclusive, block=False)

        def lock(self, fd: int, exclusive: bool) -> None:
            self._lock(fd, exclusive, block=True)

        def unlock(self, fd: int) -> None:
            handle = msvcrt.get_osfhandle(fd)
            ov = self._overlapped()
            if not _kernel32.UnlockFileEx(
                wintypes.HANDLE(handle),
                wintypes.DWORD(0),
                wintypes.DWORD(1),
                wintypes.DWORD(0),
                ctypes.byref(ov),
            ):
                raise _win_error("UnlockFileEx failed")

        def close(self, fd: int) -> None:
            os.close(fd)

        def pid_alive(self, pid: int) -> bool | None:
            handle = _kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )
            if not handle:
                err = ctypes.get_last_error()
                if err == _ERROR_INVALID_PARAMETER:
                    return False
                # Access denied or anything else: we cannot prove liveness.
                return None
            try:
                code = wintypes.DWORD()
                if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == _STILL_ACTIVE
            finally:
                _kernel32.CloseHandle(handle)

    _backend = _WindowsBackend()

else:  # pragma: no cover
    _backend = _Backend()


# Waiting threads only sleep inside a kernel lock call; don't charge them
# against the default worker-thread limiter. Created lazily to avoid import
# cycles (the module is imported before trio finishes initializing).
_wait_limiter: trio.CapacityLimiter | None = None


def _get_wait_limiter() -> trio.CapacityLimiter:
    global _wait_limiter
    if _wait_limiter is None:
        _wait_limiter = trio.CapacityLimiter(math.inf)
    return _wait_limiter


###############################################################################
# Sidecar metadata protocol
###############################################################################


def _sidecar_path(path: StrOrBytesPath) -> str | bytes:
    fspath = os.fspath(path)
    if isinstance(fspath, bytes):
        return fspath + b".trio-lock"
    return fspath + SIDECAR_SUFFIX


def _encode_record(
    pid: int,
    instance_id: str,
    exclusive: bool,
) -> bytes:
    record = {
        "v": 1,
        "pid": pid,
        "time": time.time(),
        "host": _HOST,
        "id": instance_id,
        "ex": bool(exclusive),
    }
    payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
    if len(payload) >= _METADATA_SIZE:  # pragma: no cover
        raise RuntimeError("file lock metadata record is larger than the slot")
    return payload + b" " * (_METADATA_SIZE - len(payload))


def _write_all_at(fd: int, offset: int, data: bytes) -> None:
    os.lseek(fd, offset, os.SEEK_SET)
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        remaining = remaining[written:]


def _write_record(fd: int, instance_id: str, exclusive: bool) -> None:
    _write_all_at(
        fd,
        0,
        _encode_record(os.getpid(), instance_id, exclusive),
    )


def _clear_record(fd: int) -> None:
    # Best-effort: mark a clean release so that later contenders do not
    # confuse us with a crashed holder.
    with contextlib.suppress(OSError):
        _write_all_at(fd, 0, b" " * _METADATA_SIZE)


def _read_all(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_holder_sync(sidecar: str | bytes) -> LockHolder | None:
    try:
        fd = os.open(sidecar, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return None
    try:
        raw = _read_all(fd, _METADATA_SIZE)
    finally:
        os.close(fd)
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return LockHolder(unreadable=True)
    if not isinstance(data, dict) or data.get("v") != 1:
        return LockHolder(unreadable=True)
    return LockHolder(
        pid=data.get("pid"),
        acquired_at=data.get("time"),
        host=data.get("host"),
        exclusive=data.get("ex"),
        instance_id=data.get("id"),
    )


def _liveness_sync(holder: LockHolder | None) -> tuple[bool, str | None]:
    """Return (definitely_dead, evidence)."""
    if holder is None or holder.unreadable or holder.pid is None:
        return False, None
    if not holder.is_local:
        return False, f"holder pid {holder.pid} lives on host {holder.host!r}"
    alive = _backend.pid_alive(holder.pid)
    age = holder.age
    if alive is False:
        return True, (
            f"sidecar names pid {holder.pid} (acquired {age:.1f}s ago) but no "
            f"process with that pid exists"
        )
    if alive is None:
        return False, (f"liveness of holder pid {holder.pid} could not be determined")
    return False, None


###############################################################################
# Internal wait state
###############################################################################


class _WaitState:
    """Coordination between the async side and the worker thread that performs
    the blocking lock operation.

    The worker blocks inside the OS lock call without holding ``guard``.  It
    only takes ``guard`` for the tiny handoff after it acquired the lock.
    The cleanup path (timeout or cancellation) can therefore always run
    promptly: either the worker is still blocked (in which case it will see
    ``cancelled`` after it wakes and dispose of the descriptor itself) or the
    handoff already happened (in which case cleanup disposes of the lock).
    """

    __slots__ = ("cancelled", "fd_disposed", "guard", "kept")

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.cancelled = False
        self.kept = False
        self.fd_disposed = False


###############################################################################
# FileLock
###############################################################################


_USE_DEFAULT: Final = object()


@final
class FileLock(AsyncResource):
    """An asynchronous, cross-process advisory file lock.

    The lock protects ``path`` (which need not exist yet) by locking a
    sidecar file next to it, named ``str(path) + ".trio-lock"``.  Use it
    either directly::

        lock = trio.FileLock(path)
        await lock.acquire(timeout=10)
        try:
            ...
        finally:
            await lock.aclose()

    or, more commonly, by passing it to :func:`trio.open_file`::

        async with await trio.open_file(path, "w", lock=True) as f:
            ...

    Args:
      path: The file (or directory entry) to protect.
      exclusive: If True (the default), take an exclusive (write) lock;
          otherwise a shared (read) lock.
      stale_policy: What to do when the sidecar metadata names a dead
          process; see :class:`StaleLockPolicy`.
      default_timeout: Default for the ``timeout`` argument of
          :meth:`acquire` (``None`` means wait forever).  This is mainly
          useful when the lock is driven by another entry point, e.g.
          ``trio.open_file(path, lock=trio.FileLock(path,
          default_timeout=5))``.

    The lock is released by :meth:`release`, :meth:`aclose`, or leaving an
    ``async with`` block that acquired it.  Cancellation and exceptions
    still release it, and the OS releases it automatically if the process
    is killed -- the descriptor and the OS lock never outlive the handle.
    """

    def __init__(
        self,
        path: StrOrBytesPath,
        *,
        exclusive: bool = True,
        stale_policy: StaleLockPolicy = StaleLockPolicy.WAIT,
        default_timeout: float | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._sidecar = _sidecar_path(self._path)
        self._exclusive = exclusive
        self._stale_policy = stale_policy
        self._default_timeout = default_timeout
        self._fd: int | None = None
        self._closed = False
        self._instance_id = uuid.uuid4().hex
        self.last_takeover: LockTakeover | None = None

    def __repr__(self) -> str:
        state = "locked" if self.locked else ("closed" if self._closed else "unlocked")
        return f"<trio.FileLock {self._path!r}: {state}>"

    @property
    def path(self) -> str | bytes:
        """The protected path."""
        return self._path

    @property
    def sidecar_path(self) -> str | bytes:
        """Path of the sidecar file carrying the OS lock and metadata."""
        return self._sidecar

    @property
    def exclusive(self) -> bool:
        """Whether this :class:`FileLock` takes exclusive locks."""
        return self._exclusive

    @property
    def locked(self) -> bool:
        """Whether this object currently holds the lock."""
        return self._fd is not None

    async def current_holder(self) -> LockHolder | None:
        """Read the current holder record from the sidecar.

        Returns a :class:`LockHolder` describing the holder (including its
        process id and acquisition time), ``None`` if the sidecar is absent
        or empty (lock is free or was released cleanly), or a
        :class:`LockHolder` with ``unreadable=True`` if the record could not
        be parsed.
        """
        return await trio.to_thread.run_sync(_read_holder_sync, self._sidecar)

    async def acquire_nowait(self) -> bool:
        """Attempt to acquire the lock without blocking.

        Returns ``True`` if the lock was acquired, ``False`` if someone else
        currently holds it.  Raises :class:`FileLockUnsupported` if the
        platform cannot lock the sidecar's filesystem.
        """
        await trio.lowlevel.checkpoint()
        return await self._acquire(deadline=-math.inf, nowait=True)

    # A bounded wait is an explicit part of the cross-process lock API;
    # internally it still uses an ordinary deadline scope.
    async def acquire(
        self,
        *,
        timeout: float | object | None = _USE_DEFAULT,  # noqa: ASYNC109
    ) -> None:
        """Acquire the lock.

        Args:
          timeout: Maximum time to wait, in seconds.  ``None`` waits
              indefinitely; ``0`` only makes a single non-blocking attempt.
              If omitted, the lock's ``default_timeout`` is used.

        Raises:
          FileLockTimeout: if ``timeout`` elapses first.  This is a
              :class:`FileLockError`, distinct from every other failure.
          StaleLockError: if the stale-lock policy rejects taking over.
          FileLockUnsupported: if the platform cannot provide locking.
          trio.BusyResourceError: if this object already holds the lock.
          trio.ClosedResourceError: if it has been closed.
        """
        if timeout is _USE_DEFAULT:
            timeout = self._default_timeout
        if timeout is None:
            deadline = math.inf
        else:
            assert isinstance(timeout, (int, float))
            deadline = trio.current_time() + timeout
        await self._acquire(deadline=deadline, nowait=False)

    async def _acquire(self, *, deadline: float, nowait: bool) -> bool:
        if self._closed:
            raise ClosedResourceError("this FileLock has been closed")
        if self._fd is not None:
            raise BusyResourceError("this FileLock is already acquired")

        fd: int | None = None
        # True while a blocking wait owns (or may still own, via the worker
        # thread) the descriptor; the async side must not close it then.
        awaiting = False
        try:
            # Open the sidecar and make one immediate non-blocking attempt.
            # This is also the platform capability probe.
            fd, got = await self._open_and_try_async()

            takeover: LockTakeover | None = None
            holder: LockHolder | None = None

            if not got:
                holder = await trio.to_thread.run_sync(
                    _read_holder_sync,
                    self._sidecar,
                )

                definitely_dead, evidence = await trio.to_thread.run_sync(
                    _liveness_sync,
                    holder,
                )
                if definitely_dead:
                    if self._stale_policy is StaleLockPolicy.ERROR:
                        raise StaleLockError(holder, evidence or "holder is dead")
                    if self._stale_policy is StaleLockPolicy.TAKE_OVER:
                        got = await trio.to_thread.run_sync(
                            _backend.try_lock,
                            fd,
                            self._exclusive,
                        )
                        if got:
                            takeover = LockTakeover(
                                holder,
                                f"{evidence}; OS lock was free, took it over",
                                time.time(),
                            )
                        else:
                            raise StaleLockError(
                                holder,
                                f"{evidence}; but the OS lock is still held "
                                f"by a live process (e.g. a child that "
                                f"inherited the descriptor), and an OS lock "
                                f"cannot be stolen",
                            )

            if not got:
                if nowait:
                    # The blocking attempt never started, so the descriptor
                    # is ours to dispose of.
                    await trio.to_thread.run_sync(self._dispose, fd)
                    fd = None
                    return False
                if deadline != math.inf and trio.current_time() >= deadline:
                    raise FileLockTimeout(holder)
                awaiting = True
                # On success the descriptor (now locked) is handed back to
                # us; on timeout/cancellation the wait machinery (the worker
                # thread) disposes of it, so we leave ``awaiting`` set and do
                # not touch it.
                fd = await self._wait_lock(fd, deadline, holder)
                awaiting = False

            # We hold the OS lock now (immediately, via stale takeover, or
            # after blocking).  Whatever the previous holder left in the
            # metadata slot is still there; inspect it before overwriting so
            # a crashed holder (OS lock released by the kernel, record left
            # behind) is diagnosed regardless of which path got us here.
            if takeover is None:
                prior = await trio.to_thread.run_sync(
                    _read_holder_sync,
                    self._sidecar,
                )
                if (
                    prior is not None
                    and not prior.unreadable
                    and prior.instance_id != self._instance_id
                ):
                    dead, why = await trio.to_thread.run_sync(_liveness_sync, prior)
                    if dead:
                        takeover = LockTakeover(
                            prior,
                            f"{why}; the OS lock had already been released "
                            f"by the kernel and is now held by us",
                            time.time(),
                        )

            await trio.to_thread.run_sync(
                _write_record,
                fd,
                self._instance_id,
                self._exclusive,
            )
            self._fd = fd
            self.last_takeover = takeover
            return True
        except BaseException:
            if fd is not None and self._fd is None and not awaiting:
                with trio.CancelScope(shield=True):
                    await trio.to_thread.run_sync(self._dispose, fd)
            raise

    async def _open_and_try_async(self) -> tuple[int, bool]:
        def sync() -> tuple[int, bool]:
            opened = _backend.open_sidecar(self._sidecar)
            try:
                return opened, _backend.try_lock(opened, self._exclusive)
            except BaseException:
                _backend.close(opened)
                raise

        return await trio.to_thread.run_sync(sync)

    def _dispose(self, fd: int) -> None:
        with contextlib.suppress(OSError):
            _backend.unlock(fd)
        _backend.close(fd)

    async def _wait_lock(
        self,
        fd: int,
        deadline: float,
        holder: LockHolder | None,
    ) -> int:
        """Block until the OS lock is acquired.

        Returns ``fd`` on success.  On timeout or cancellation the
        descriptor is disposed of either by the (still blocked) worker when
        it wakes or by ``reap`` if the worker's handoff raced
        cancellation -- the caller must not close it itself in that case.
        """
        state = _WaitState()

        def worker() -> None:
            _backend.lock(fd, self._exclusive)
            with state.guard:
                if state.cancelled:
                    # The async side has already given up on us; release and
                    # close the descriptor ourselves immediately.
                    try:
                        _backend.unlock(fd)
                    finally:
                        _backend.close(fd)
                    state.fd_disposed = True
                else:
                    state.kept = True

        def reap() -> None:
            with state.guard:
                state.cancelled = True
                if state.kept:
                    # The worker finished the handoff just as cancellation
                    # landed; its return value was discarded, so we must
                    # dispose of the lock it acquired.
                    try:
                        _backend.unlock(fd)
                    finally:
                        _backend.close(fd)
                    state.kept = False
                    state.fd_disposed = True

        try:
            if deadline == math.inf:
                await trio.to_thread.run_sync(
                    worker,
                    abandon_on_cancel=True,
                    limiter=_get_wait_limiter(),
                )
            else:
                with trio.move_on_at(deadline) as cancel_scope:
                    await trio.to_thread.run_sync(
                        worker,
                        abandon_on_cancel=True,
                        limiter=_get_wait_limiter(),
                    )
                if cancel_scope.cancelled_caught:
                    with trio.CancelScope(shield=True):
                        await trio.to_thread.run_sync(
                            reap,
                            limiter=_get_wait_limiter(),
                        )
                    raise FileLockTimeout(holder)
        except trio.Cancelled:
            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(reap, limiter=_get_wait_limiter())
            raise

        if state.fd_disposed:  # pragma: no cover
            # Defensive: should be unreachable because every non-success path
            # raises, but never hand out a disposed descriptor.
            raise FileLockError("lock wait ended without a usable descriptor")
        return fd

    async def release(self) -> None:
        """Release the lock and close the sidecar descriptor.

        Raises:
          RuntimeError: if the lock is not currently held.
        """
        if self._fd is None:
            raise RuntimeError("release() called on a FileLock that is not held")
        fd, self._fd = self._fd, None

        def sync() -> None:
            # A clean exclusive release wipes the metadata slot, so the
            # sidecar cannot be mistaken for a stale lock later.  Shared
            # locks may have other active holders, so leave metadata alone in
            # that case.
            if self._exclusive:
                _clear_record(fd)
            try:
                _backend.unlock(fd)
            finally:
                _backend.close(fd)

        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(sync)
        await trio.lowlevel.checkpoint_if_cancelled()

    async def aclose(self) -> None:
        """Release the lock if held.  Idempotent and safe under
        cancellation: the OS descriptor is always closed before this
        returns, and if the process dies the kernel releases the lock on its
        own."""
        if self._fd is not None:
            await self.release()
        self._closed = True
        await trio.lowlevel.checkpoint_if_cancelled()

    async def __aenter__(self) -> Self:
        if not self.locked:
            await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


@asynccontextmanager
async def open_file_lock(
    path: StrOrBytesPath,
    *,
    exclusive: bool = True,
    timeout: float | None = None,  # noqa: ASYNC109
    stale_policy: StaleLockPolicy = StaleLockPolicy.WAIT,
) -> AsyncIterator[FileLock]:
    """Acquire a :class:`FileLock` for the duration of an ``async with``
    block::

        async with trio.open_file_lock("state.db", timeout=5):
            ...

    On normal exit, exception, or cancellation the lock is released and the
    sidecar descriptor is closed.
    """
    lock = FileLock(path, exclusive=exclusive, stale_policy=stale_policy)
    try:
        await lock.acquire(timeout=timeout)
        yield lock
    finally:
        await lock.aclose()


###############################################################################
# Locked temporary and state files
###############################################################################


@asynccontextmanager
async def open_temp_file(
    *,
    mode: str = "w+b",
    suffix: str = ".tmp",
    prefix: str = "trio-",
    dir: str | os.PathLike[str] | None = None,
    delete: bool = True,
    lock: bool = True,
    timeout: float | None = None,  # noqa: ASYNC109
    stale_policy: StaleLockPolicy = StaleLockPolicy.WAIT,
) -> AsyncIterator[tuple[AsyncIOWrapper[io.IOBase], str]]:
    """Create a temporary file, optionally locked, like an async
    :func:`tempfile.NamedTemporaryFile`.

    The file is created with :func:`tempfile.mkstemp` (so creation itself is
    race-free), opened with the requested ``mode``, and -- when ``lock`` is
    true -- protected by an exclusive :class:`FileLock` for the lifetime of
    the returned file object.  Exiting the ``async with`` block flushes and
    closes the file, releases the lock, and (when ``delete`` is true) removes
    both the file and its sidecar.

    Yields:
      A tuple ``(async_file, name)``.
    """
    from ._file_io import _LockedAsyncIOWrapper, wrap_file

    def create() -> tuple[int, str]:
        return tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)

    def open_fd(fd: int) -> io.IOBase:
        # ``mode`` is not a literal, so static types say ``open`` may return
        # any IO flavour; they all derive from io.IOBase.
        return cast("io.IOBase", open(fd, mode))  # owns fd

    fd, name = await trio.to_thread.run_sync(create)
    sync_file = await trio.to_thread.run_sync(open_fd, fd)

    file_lock: FileLock | None = None
    try:
        wrapper: AsyncIOWrapper[io.IOBase]
        if lock:
            file_lock = FileLock(
                name,
                stale_policy=stale_policy,
                default_timeout=timeout,
            )
            await file_lock.acquire()
            wrapper = _LockedAsyncIOWrapper(
                sync_file,
                file_lock,
                owns_lock=True,
            )
        else:
            wrapper = wrap_file(sync_file)
        try:
            yield wrapper, name
        finally:
            await wrapper.aclose()
    finally:
        if delete:

            def unlink() -> None:
                with contextlib.suppress(FileNotFoundError, OSError):
                    os.unlink(name)
                if lock:
                    with contextlib.suppress(FileNotFoundError, OSError):
                        os.unlink(_sidecar_path(name))

            with trio.CancelScope(shield=True):
                await trio.to_thread.run_sync(unlink)


@asynccontextmanager
async def open_state_file(
    path: StrOrBytesPath,
    mode: str = "r+b",
    *,
    create: bool = True,
    exclusive: bool = True,
    timeout: float | None = None,  # noqa: ASYNC109
    stale_policy: StaleLockPolicy = StaleLockPolicy.WAIT,
) -> AsyncIterator[AsyncIOWrapper[io.IOBase]]:
    """Open a *state file* -- something multiple tasks or processes read and
    update cooperatively -- under a file lock.

    This is :func:`open_file` with an exclusive :class:`FileLock` attached:
    the lock is acquired before the file is handed out and released when the
    ``async with`` block exits (including on cancellation or exceptions).
    When ``create`` is true (the default), the file is first created if it
    does not exist, without truncating an existing one.

    Example::

        async with trio.open_state_file(path) as f:
            data = await f.read()
            ...
            await f.seek(0)
            await f.write(new_data)
    """
    if create:

        def ensure() -> None:
            fd = os.open(
                os.fspath(path),
                os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0),
                0o644,
            )
            os.close(fd)

        await trio.to_thread.run_sync(ensure)

    file_lock = FileLock(
        path,
        exclusive=exclusive,
        stale_policy=stale_policy,
        default_timeout=timeout,
    )
    # Pass an unacquired lock: open_file performs the acquisition and the
    # returned file object owns the lock, so closing it releases it.
    from ._file_io import open_file

    f = cast(
        "AsyncIOWrapper[io.IOBase]",
        await open_file(path, mode, lock=file_lock),
    )
    try:
        yield f
    finally:
        await f.aclose()
