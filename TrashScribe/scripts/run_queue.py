"""Run the experiment queue (configs/gpu_queue.yaml) one job at a time on this GPU.

Unlike run_grid.py it is meant to run unattended for days on a shared machine:
* the queue file is read again before every job, so jobs can be added or
  reordered while it runs;
* a job is done when its runs/<name>/test_metrics.json exists;
* a failed training job is retried with --resume (continues from last.pt);
* after every job RESULTS.md and paper/tables/*.tex are regenerated;
* runs/_queue_status.json says what is running and what happened so far.

    python scripts/run_queue.py [configs/gpu_queue.yaml] [--dry]
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PY = sys.executable
SCRIPTS = {"train": "train.py", "segformer": "baseline_segformer.py", "zeroshot": "baseline_zeroshot.py"}


def load_queue(path):
    q = yaml.safe_load(Path(path).read_text())
    return q.get("common", {}), q["jobs"]


def command(job, common):
    kind = job.get("kind", "train")
    ov = list(common.get(kind, [])) + list(job.get("overrides", []))
    cmd = [PY, str(ROOT / "scripts" / SCRIPTS[kind]), str(ROOT / job["config"])]
    cmd += ov + list(job.get("args", []))
    return cmd + (["--resume"] if kind == "train" else [])


def _oom_tail(log: Path, n=4000) -> bool:
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - n))
            return b"OutOfMemoryError" in f.read()
    except OSError:
        return False


def status_write(st):
    tmp = RUNS / "_queue_status.json.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(RUNS / "_queue_status.json")


def report():
    for s in ("report_results.py", "make_tables.py"):
        subprocess.call([PY, str(ROOT / "scripts" / s)], cwd=ROOT, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", nargs="?", default=str(ROOT / "configs" / "gpu_queue.yaml"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"}
    st = json.loads((RUNS / "_queue_status.json").read_text()) if (RUNS / "_queue_status.json").exists() else {}
    st.setdefault("history", [])
    attempts, ooms = {}, {}
    while True:
        common, jobs = load_queue(a.queue)
        todo = [j for j in jobs if not (RUNS / j["name"] / "test_metrics.json").exists()
                and attempts.get(j["name"], 0) < int(j.get("max_attempts", 3))]
        if a.dry:
            for j in jobs:
                done = (RUNS / j["name"] / "test_metrics.json").exists()
                print("done " if done else "todo ", j["name"], " ".join(command(j, common)[2:]))
            return
        if not todo:
            break
        job = todo[0]
        name = job["name"]
        attempts[name] = attempts.get(name, 0) + 1
        cmd = command(job, common)
        t0 = time.time()
        st["current"] = {"name": name, "attempt": attempts[name], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "cmd": " ".join(cmd[1:])}
        status_write(st)
        print(time.strftime("%m-%d %H:%M"), "run", name, "attempt", attempts[name], flush=True)
        with open(RUNS / f"{name}.log", "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempts[name]}: {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        ok = (RUNS / name / "test_metrics.json").exists()
        if not ok and _oom_tail(RUNS / f"{name}.log"):
            # 15 Nov: 5 SegFormer jobs OOMed identically 3 times each (3.3 h lost). Two CUDA OOMs in
            # a row with the same command are deterministic: stop retrying, go to the next job.
            ooms[name] = ooms.get(name, 0) + 1
            if ooms[name] >= 2:
                attempts[name] = 10 ** 6
                print(time.strftime("%m-%d %H:%M"), "give up", name, "(CUDA OOM twice)", flush=True)
        st["history"].append({"name": name, "attempt": attempts[name], "rc": rc, "ok": ok,
                              "minutes": round((time.time() - t0) / 60, 1),
                              "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        st["current"] = None
        status_write(st)
        print(time.strftime("%m-%d %H:%M"), "end", name, "rc", rc, "ok", ok,
              f"{(time.time() - t0) / 60:.0f} min", flush=True)
        report()
    st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_write(st)
    report()
    print("queue finished")


if __name__ == "__main__":
    main()
