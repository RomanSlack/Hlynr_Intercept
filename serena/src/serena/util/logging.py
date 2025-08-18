import queue
import threading
from collections.abc import Callable

from sensai.util import logging

from serena.constants import SERENA_LOG_FORMAT


class MemoryLogHandler(logging.Handler):
    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self.setFormatter(logging.Formatter(SERENA_LOG_FORMAT))
        self._log_buffer = LogBuffer()
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._emit_callbacks: list[Callable[[str], None]] = []

        # start background thread to process logs
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def add_emit_callback(self, callback: Callable[[str], None]) -> None:
        """
        Adds a callback that will be called with each log message.
        The callback should accept a single string argument (the log message).
        """
        self._emit_callbacks.append(callback)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._log_queue.put_nowait(msg)

    def _process_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._log_queue.get(timeout=1)
                self._log_buffer.append(msg)
                for callback in self._emit_callbacks:
                    try:
                        callback(msg)
                    except:
                        pass
                self._log_queue.task_done()
            except queue.Empty:
                continue

    def get_log_messages(self) -> list[str]:
        return self._log_buffer.get_log_messages()


class LogBuffer:
    """
    A thread-safe buffer for storing log messages.
    """

    def __init__(self) -> None:
        self._log_messages: list[str] = []
        self._lock = threading.Lock()

    def append(self, msg: str) -> None:
        with self._lock:
            self._log_messages.append(msg)

    def get_log_messages(self) -> list[str]:
        with self._lock:
            return self._log_messages.copy()
