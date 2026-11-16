import os
import signal
import threading
import traceback

from poller.Poller import Poller


def main() -> None:
    interval = float(os.environ.get("POLL_INTERVAL", "900"))
    stop = threading.Event()

    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    with Poller() as poller:
        while not stop.is_set():
            try:
                poller.run_once()
            except Exception:
                traceback.print_exc()
            stop.wait(interval)
