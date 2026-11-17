# TrashScribe: start here (for AI agents and new collaborators)

TrashScribe is a research project with a working codebase, a running experiment grid and a manuscript in
preparation. It currently lives inside the LENTE repository (`LENTE/TrashScribe/`) but is an
**independent work**: nothing in the code depends on LENTE files, and the folder is meant to be moved
into its own repository (see [docs/MIGRATION.md](docs/MIGRATION.md)).

**One-sentence summary.** A 34 M-parameter network (frozen SAM2.1-Hiera-Tiny + zero-init adapters) that,
in one forward pass suitable for a Jetson Orin Nano, segments waste in any environment, describes each
object with a controlled vocabulary of visual properties (material, rigidity, optics, shape, ...), names
it by reading a text atlas, and is benchmarked across five public waste datasets / environments against
SAM 3 and other baselines.

## Read in this order

| # | File | What it gives you |
|---|---|---|
| 1 | [docs/CONTEXT.md](docs/CONTEXT.md) | why the project exists, venue and deadline, the owner's preferences and rules, the history of decisions, what the paper can and cannot claim |
| 2 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | model, data pipeline, losses, naming, distilled CLIP head, edge runtime, code map |
| 3 | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | every protocol (E1 to E10), its config, its result, and the reading of the results |
| 4 | [docs/OPERATIONS.md](docs/OPERATIONS.md) | environment, data, host memory constraints, the GPU queue service, restarts, monitoring, post-hoc tools, building the paper |
| 5 | [docs/JETSON.md](docs/JETSON.md) | deployment on the Jetson Orin Nano and **the files that are not in git** |
| 6 | [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | prioritised to-do list and open decisions for the owner |
| 7 | [docs/MIGRATION.md](docs/MIGRATION.md) | how to move this folder out of LENTE into its own repository |

Live sources of truth (always newer than these docs): [RESULTS.md](RESULTS.md) (journal + tables regenerated
every 15 min), [attempts/ATTEMPTS.md](attempts/ATTEMPTS.md) (every change with hypothesis and decision),
[configs/gpu_queue.yaml](configs/gpu_queue.yaml) (what runs next), `runs/<run>/test_metrics.json`.
Status of these docs: written 17 Nov 2026, 22:30 CET.

## Warnings an agent must raise with the owner

1. **Jetson files are not in git.** Deploying needs files that `.gitignore` excludes (ONNX graph, atlas
   cache, checkpoints, model weights, test images). Before any Jetson step, tell the owner which files to
   copy, using the checklist in [docs/JETSON.md](docs/JETSON.md#files-not-in-git). Do not assume they are
   on the device.
2. **No on-device numbers exist yet.** Every latency/power/memory figure in the paper is a red `\tbd`. The
   edge claim and the "SAM 3 cannot run at frame rate on the device" argument need measurements made on the
   Jetson by the owner (SAM 3 included). Never invent them; laptop numbers are not edge results.
3. **Host RAM kills VS Code.** On the training workstation, `systemd-oomd` kills the whole user session
   when memory pressure stays above 50 % for 20 s. Never launch heavy jobs from the VS Code terminal; use
   capped `systemd-run --user` units ([docs/OPERATIONS.md](docs/OPERATIONS.md#host-memory)).
4. **The GPU belongs to the queue.** A training run uses about 6 GB of the 8 GB card. Extra GPU jobs must
   cap themselves (`torch.cuda.set_per_process_memory_fraction`, about 0.17) or run on CPU.
5. **Do not commit or push unless the owner asks.** Staging is fine when asked.
6. **Paper style rules (owner's, binding):** no em dashes; no ", and" constructs; few subsections;
   section/subsection titles without parentheses and not whole sentences; the introduction ends with a
   roadmap of the sections; every section with subsections starts with a short lead-in.
7. **Honesty rules learnt the hard way:** fix an evaluation rule before seeing results, report both
   variants when a rule changes, never choose per-dataset rules on test numbers, single-seed differences
   below about 3 mAP points or 0.1 H are noise.

## Current state in five lines (17 Nov 2026)

* Grid: 40+ runs done, queue still running (expected end about Fri 20 Nov, see `python scripts/progress.py`).
* Strongest results: (a) 243 waste-free indoor photos fix the indoor failure (MJU 0.055 → 0.363, above
  SAM 3 0.221); (b) with the site in training the edge model beats SAM 3 on all five datasets; (c) at a new
  site SAM 3 labels or 10 labelled images put the edge model above SAM 3 on TACO, MJU, BePLi (UAVVaste pending).
* Not supported: open-vocabulary naming better than CLIP; environment removal by the adversary.
* Paper: `paper/main.tex`, compiles (26 pp), marked PRELIMINARY, red `\tbd` where numbers are pending.
  The title/abstract/introduction still describe the old framing; a rewrite around the new findings is
  pending the owner's go-ahead ([docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)).
* Deadline: Neural Computing and Applications topical collection, **30 Nov 2026**.
