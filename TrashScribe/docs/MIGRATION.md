# Moving TrashScribe out of the LENTE repository

Today TrashScribe lives in `LENTE/TrashScribe/` (git remote of the parent: `SalvatoreCapuozzo/LENTE`). It is an
independent work and is meant to become its own repository. The owner will decide when; until then it stays.

## What makes the move easy

* **No code dependency on LENTE.** Every path in code and configs resolves against the TrashScribe folder
  (`PKG_ROOT` in `trashscribe/engine.py`) or against `data_root` (`~/data/waste`). Mentions of LENTE in the code
  are comments and the paper's citation of the earlier work. No file outside the folder refers to TrashScribe.
* `.gitignore` is already inside the folder and works at the new repository root.

## What changes with the path

| Item | Action |
|---|---|
| `stop_and_restart.txt`, RESULTS.md header | replace `~/LENTE/TrashScribe` with the new path |
| systemd units (`trashscribe-queue`, `trashscribe-report`) | **stop the queue first** (between jobs or after the grid; `stop_and_restart.txt` "SAFE STOP"), move, restart with the new `--working-directory` |
| `runs/` (3.7 GB, git-ignored: checkpoints, exports, post-hoc dumps) | moves with the folder; copy it if the repository is cloned elsewhere (`init_from` paths in E9/E10 configs are relative, `runs/e3_lodo_<site>/best.pt`) |
| `runs/_queue_status.json` | contains absolute command lines: informational only, can stay |
| `~/data/waste` | outside the repository, unchanged |
| Hugging Face cache (`~/.cache/huggingface/hub`) | unchanged |
| conda env `trashscribe` | unchanged |
| assistant memory | Claude Code keys its memory by project path: notes kept for `.../LENTE` are not visible from the new path. These docs replace them. |

## Keeping the git history

Commits touching TrashScribe so far: `8cff538 Addition of TrashScribe Project` plus later work. To keep history:

```bash
# from a fresh clone of LENTE
git subtree split -P TrashScribe -b trashscribe-only          # or: git filter-repo --subdirectory-filter TrashScribe
mkdir ../TrashScribe-repo && cd ../TrashScribe-repo && git init
git pull ../LENTE trashscribe-only
git remote add origin <new remote> && git push -u origin main
```

Then remove `TrashScribe/` from LENTE in a separate commit, if the owner wants it gone there. Do not do any of
this without the owner's go-ahead.

## Checklist after the move

1. `python -m pytest tests -q` (14 pass).
2. `python scripts/run_queue.py --dry` lists the remaining jobs.
3. `python scripts/progress.py` renders.
4. `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` compiles.
5. Update the paths in `stop_and_restart.txt` and the RESULTS.md header.
