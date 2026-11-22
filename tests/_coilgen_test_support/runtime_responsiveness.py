"""Shared fixtures for runtime responsiveness."""

import queue


class _Events(queue.Queue):
    def close(self):
        pass


class _Process:
    alive = True
    exitcode = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    kill = terminate

    def join(self, timeout=None):
        pass
