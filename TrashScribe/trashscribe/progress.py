"""Live progress of a run: runs/<run>/progress.json, read by scripts/progress.py.

Written atomically and at most every `every` seconds (phase changes are always
written), so it costs nothing next to a training step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Progress:
    def __init__(self, run_dir, every: float = 10.0):
        self.path = Path(run_dir) / "progress.json"
        self.every = every
        self.last = 0.0
        try:
            self.state = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.state = {}

    def __call__(self, force: bool = False, **kw):
        if "phase" in kw and kw["phase"] != self.state.get("phase"):
            force = True
            kw.setdefault("phase_started", time.time())
        self.state.update(kw)
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        self.state["updated"] = now
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
        except OSError:
            pass
