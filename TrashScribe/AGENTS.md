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
2. **On-device numbers come only from the Jetson.** Table `tab:edge` holds the JetPack 6.2 runs of 24-25 Nov
   (`runs/e6_edge_640/jetson_jp6/summary.md`, SAM 3 in PyTorch and in TensorRT included). Never invent device numbers; laptop numbers are not
   edge results. Clock locking (`sudo jetson_clocks`) needs the owner (no sudo for the agent).
3. **Host RAM kills VS Code.** On the training workstation, `systemd-oomd` kills the whole user session
   when memory pressure stays above 50 % for 20 s. Never launch heavy jobs from the VS Code terminal; use
   capped `systemd-run --user` units ([docs/OPERATIONS.md](docs/OPERATIONS.md#host-memory)).
4. **The GPU belongs to the queue.** A training run uses about 6 GB of the 8 GB card. Extra GPU jobs must
   cap themselves (`torch.cuda.set_per_process_memory_fraction`, about 0.17) or run on CPU.
5. **Do not commit or push unless the owner asks.** Staging is fine when asked.
6. **Paper style rules (owner's, binding):** no em dashes; no ", and" constructs; few subsections;
   section/subsection titles without parentheses and not whole sentences; the introduction ends with a
   roadmap of the sections; every section with subsections starts with a short lead-in.
   American English spelling (labeled, generalization, anonymized, license, color, toward); titles of
   cited works keep their published spelling.
7. **Honesty rules learnt the hard way:** fix an evaluation rule before seeing results, report both
   variants when a rule changes, never choose per-dataset rules on test numbers, single-seed differences
   below about 3 mAP points or 0.1 H are noise.

## Current state (24 Nov 2026)

* Grid: **finished** (queue ended 20 Nov 15:45). The report timer still refreshes RESULTS.md every 15 min.
* Main results: with the site in training the 34 M edge model beats SAM 3 (848 M) on all five datasets; with no
  site data (LODO) SAM 3 (fair rule) is ahead on all five, not four; on a new site, 10 labelled images or SAM 3 labels on unlabelled site images bring it to or above SAM 3 on four of five;
  243 generic indoor photos without waste fix the indoor failure (MJU 0.055 → 0.363); aerial litter is the
  exception. A fully fine-tuned SegFormer-B2 transfers as well or better (no transfer advantage of the frozen
  encoder). Description at 3× chance; the CLIP-distilled head reads new classes/properties like CLIP in one pass.
* Real-world use (owner): TrashScribe runs on a remotely guided wheeled land rover with a Jetson Orin Nano that
  surveys beaches and natural habitats; autonomous driving is future work. The paper's focus stays on the
  segmentation/description system.
* Jetson (board reflashed to JetPack 6.2, 24 Nov; clocks locked, 15 W): TensorRT FP16 network 47.9 ms, full pipeline
  97-126 ms (8.0-10.3 FPS), 0.5 GB, 13.8 W (0.72 J/frame). SAM 3 on the same board, 16 prompts: PyTorch bf16
  5.2 s/frame (0.19 FPS), 3.2 GB, 66 J/frame; TensorRT FP16 (25 Nov, A22) 3.2 s/frame (0.32 FPS), 1.6 GB, 45 J/frame,
  masks agree with PyTorch bf16 at IoU 0.991 (FP32: 0.992); TrashScribe 25-33x faster, SAM 3's image encoder alone
  1.12 s. TensorRT 10.3 needed workarounds in both mask decoders (TrashScribe's hypernetwork, SAM 3's presence head:
  docs/JETSON.md). The JetPack 5.1.1 run (unlocked clocks, ~5 FPS) is kept and cited once.
* Paper: 25 pages, final numbers in (Jetson included, 24-25 Nov); red `\tbd` only for the repository link (the owner
  inserts it after creating the new TrashScribe repository). Abstract 242 words (limit 250), LLM-use statement and
  faces/licence plates statement written on 25 Nov.
* Deadline: Neural Computing and Applications topical collection, **30 Nov 2026**.
