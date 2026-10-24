import subprocess
import sys

import trio
from typing_extensions import assert_type


async def test() -> None:
    # this could test more by using platform checks, but currently this
    # is just regression tests + sanity checks.
    await trio.run_process("python", executable="ls")
    await trio.lowlevel.open_process("python", executable="ls")

    # note: there's no error code on the type ignore as it varies
    # between platforms.
    await trio.run_process("python", capture_stdout=True)
    await trio.lowlevel.open_process("python", capture_stdout=True)  # type: ignore

    # output limits can be declared as ints or OutputLimits
    completed = await trio.run_process(
        "python",
        capture_stdout=True,
        stdout_limit=100,
        stderr_limit=trio.OutputLimit(
            200,
            on_overflow="keep_head_tail",
            tail_bytes=50,
        ),
    )
    # *capture attributes are attached at runtime to the stdlib
    # CompletedProcess/CalledProcessError/Cancelled objects, whose
    # stubs can't declare them.
    stdout_capture: trio.CapturedOutput | None = completed.stdout_capture  # type: ignore[attr-defined]
    if stdout_capture is not None:
        assert_type(stdout_capture.limit, trio.OutputLimit | None)
        assert_type(stdout_capture.data, bytes)
        assert_type(stdout_capture.discarded_bytes, int)
        assert_type(stdout_capture.truncated, bool)

    # the same declaration works on a stream held directly
    proc = await trio.lowlevel.open_process("python", stdout=subprocess.PIPE)
    if proc.stdout is not None:
        capturing = trio.CapturingReceiveStream(proc.stdout, limit=trio.OutputLimit(64))
        assert_type(capturing.capture, trio.CapturedOutput)
        assert_type(await capturing.receive_some(), bytes | bytearray)

    if sys.platform != "win32" and sys.version_info >= (3, 9):
        await trio.run_process("python", extra_groups=[5])
        await trio.lowlevel.open_process("python", extra_groups=[5])

        # 3.11+:
        await trio.run_process("python", process_group=5)  # type: ignore
        await trio.lowlevel.open_process("python", process_group=5)  # type: ignore
