#!/usr/bin/env python3
"""Single-writer lock for results/aggregate/html/results_tables.html.

The shared report is written by MANY scripts (render_final_tables.py, the
NIAH/memory section renderers, ad-hoc patch_*.py). With no lock, two concurrent
writers race -> the file gets duplicated or one writer carries a STALE copy of
another section forward (this is how the Hidden-persona Accuracy row kept
reverting). Every script that opens results_tables.html for WRITING must hold
this lock for the whole read-modify-write.

Usage (Python):
    from _htmllock import html_lock          # _scripts/ is on sys.path for scripts run from there
    with html_lock():
        html = open(HTML).read()
        ...                                   # modify
        open(HTML, "w").write(html)

Usage (bash, ad-hoc one-off edits):
    exec 9>results/aggregate/html/.results_tables.lock
    flock -w 180 9 || { echo "results_tables.html is locked by another session"; exit 1; }
    python3 results/_scripts/some_patch.py
    flock -u 9
"""
import contextlib
import fcntl
import os
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HTML = os.path.join(_REPO, "results", "aggregate", "html", "results_tables.html")
LOCKFILE = os.path.join(_REPO, "results", "aggregate", "html", ".results_tables.lock")


@contextlib.contextmanager
def html_lock(timeout=180, poll=0.5):
    """Exclusive advisory lock around any write to results_tables.html.

    Blocks (polling) up to `timeout` seconds, then raises TimeoutError rather
    than risk a concurrent overwrite. Released automatically on exit.
    """
    os.makedirs(os.path.dirname(LOCKFILE), exist_ok=True)
    f = open(LOCKFILE, "w")
    start = time.time()
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - start > timeout:
                f.close()
                raise TimeoutError(
                    f"could not acquire results_tables.html lock within {timeout}s "
                    f"({LOCKFILE}); another session is editing it"
                )
            time.sleep(poll)
    try:
        yield f
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()
