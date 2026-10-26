"""Portable wall-clock timeout for tool calls.

signal.alarm is POSIX-only, so use a thread pool: run the callable in a
worker thread and give up after `seconds`. On timeout, the worker thread
keeps running in the background (daemonless pool shutdown waits for the
burning request only until the context closes), but the CALLER returns
immediately with a TimeoutError the tool converts into a normal
{"error": ...} result — so the existing failure/retry machinery treats
timeouts like any other tool failure.
"""

from __future__ import annotations

import concurrent.futures


def run_with_timeout(fn, seconds: int, *args, **kwargs):
    """Run `fn(*args, **kwargs)` with a hard wall-clock limit.

    Raises concurrent.futures.TimeoutError if exceeded.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError:
            raise
