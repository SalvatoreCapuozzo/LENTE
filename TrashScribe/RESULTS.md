# TrashScribe — GPU experiment results

Live report of the experiment grid (E1–E6 and baselines) on the NVIDIA machine. It is updated
progressively: the tables inside the `AUTO` block are regenerated from `runs/` by
`scripts/report_results.py` after every job of the queue; the sections around them are written
by hand as results come in. Paper tables (`paper/tables/*.tex`) are regenerated at the same time
by `scripts/make_tables.py`.

The grid runs as its own systemd user service, outside VS Code, with a RAM cap (see "Host memory"
below for why):

```bash
cd ~/LENTE/TrashScribe
systemd-run --user --unit=trashscribe-queue --collect \
  -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
  --working-directory=$PWD --setenv=HF_HUB_OFFLINE=1 \
  -p StandardOutput=append:$PWD/runs/_queue.log -p StandardError=append:$PWD/runs/_queue.log \
  ~/anaconda3/envs/trashscribe/bin/python scripts/run_queue.py
systemctl --user status trashscribe-queue      # state, memory in use
systemctl --user stop trashscribe-queue        # stop; starting again resumes from runs/<run>/last.pt
```
**Live progress** (done / now / next, with ETAs):
```bash
conda activate trashscribe
cd ~/LENTE/TrashScribe
python scripts/progress.py --watch 30      # refresh every 30 s, Ctrl-C to quit; without --watch: one snapshot
```

Queue and order: [configs/gpu_queue.yaml](configs/gpu_queue.yaml) (edit freely while it runs);
live status: `runs/_queue_status.json`; per-run logs `runs/<run>.log`; per-epoch numbers
`runs/<run>/history.json`; memory trace `runs/_queue_mem.log`.

---

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Laptop, 8 GB, 80 W cap (driver 580.178, CUDA 13.0) — thermally limited under sustained load (83–89 °C, clocks 0.8–1.8 GHz) |
| Host | 16 CPU threads, 14 GB RAM, Linux 6.8; shared with other jobs |
| Python env | conda env `trashscribe` from [environment.yml](environment.yml) (since 5 Nov 17:50; the first 7 E4 epochs ran in a venv on `dl_env` with the same pinned packages — data-pipeline outputs verified bit-identical): Python 3.12.3, torch 2.9.1+cu130 (cuDNN 9.13), transformers 5.2.0, huggingface-hub 1.4.1, numpy 1.26.4, scikit-learn 1.7.2, OpenCV 4.10.0.84 |
| Weights | `facebook/sam2.1-hiera-tiny`, `openai/clip-vit-base-patch16`, `nvidia/mit-b2`, `depth-anything/Depth-Anything-V2-Small-hf`, OWLv2 (zero-shot baseline) |
| Throughput (E4, 1024 px, real data) | 8.6 img/s training (11.2 img/s for the network alone); one epoch = 7.7 min train + ~1.5 min validation, so ≤ 6.3 h per 40-epoch run |

### Host memory (the constraint that shaped the setup)

VS Code with its extensions already holds ~7 GB of the 15 GB, and `systemd-oomd` kills the
largest group of the user session when memory pressure stays above 50 % for 20 s. On 5 Nov at
13:13 the first E4 launch (started from the VS Code terminal, hence inside the VS Code cgroup)
pushed the session into swap and oomd killed **the whole VS Code scope, training included**.
What changed:

* the queue runs in its own service `trashscribe-queue` capped at 8 GB with no swap: if it ever
  overshoots, the kernel kills a training process inside it (the queue resumes the run from
  `last.pt`), never the editor; `OOMPolicy=continue` keeps the queue alive after such a kill;
* BePLi images carry up to 363 objects: the canonicaliser stacked every object mask as float32
  at 1024² (> 1 GB for one sample) and augmentation copied every mask; objects are now packed into
  a few uint16 label layers and downsampled one at a time — **outputs verified identical** on 76
  calls including the 4 heaviest images, and 2× faster;
* COCO annotations are kept as packed bytes in one numpy buffer instead of pycocotools dicts, so
  forked data workers stop growing private copies of 70k+ annotations epoch after epoch —
  **samples verified identical** on 5 290 loads;
* 6 training workers (8.8 img/s, vs 9.2 with 8), 3 validation workers with batch 4 and prefetch 1
  (evaluation is per image and per object, so validation batch size does not change any metric);
* `pin_memory` off (`train.pin_memory=false`): CUDA's caching host allocator kept 1.38 GB of
  pinned blocks in the main process (batch sizes vary with the object count). Cost: the
  host→GPU copy is synchronous, ≈ 28 ms instead of ≈ 18 ms per micro-batch, i.e. a few % per
  epoch (epoch 5: 542 s vs 460–471 s for epochs 1–4, part of which is thermal).

Measured during E4 with the final setup: ≤ 6.1 GB process memory + ≤ 1.0 GB shared memory,
**total ≤ 6.6 GB** under the 8 GB cap, system never below 6.2 GB available (with pinned memory the
peak was 7.45 GB). Two capped attempts were killed before these settings (logs kept as
`runs/e4_heldout_classes.oom*.log`). The run that counts started at 14:54; it was stopped and
resumed from `last.pt` twice at epoch boundaries (after epoch 2 for unbuffered logs, after epoch 4
for `pin_memory`), which also tested the resume path on a real run.

## Protocol, and what differs from the manuscript

Unchanged (Sect. "Implementation" of `paper/main.tex`): 1024 px letterboxed (640 px for E6),
AdamW 3e-4 / 1e-4 / 5e-5, weight decay 1e-4, 300 warm-up steps + cosine, **40 epochs of 4000
samples, batch 8**, bf16 autocast, early stopping on val IoU + desc mAP with patience 8, seed 42.

Execution details forced by the 8 GB GPU / 14 GB RAM (do not change the optimisation problem, but
worth one sentence in the paper):

1. **Batch 8 = micro-batch 4 × 2 accumulated steps** (E6: 6 × 2; SegFormer: 2 × 4). Batch 8 at
   1024 px needs > 8 GB. One consequence: MixStyle mixes feature statistics inside the micro-batch
   (4 images instead of 8).
2. **SegFormer-B2 attention through `F.scaled_dot_product_attention`**: same maths as the eager
   implementation (max |Δ| = 3e-5 on logits), but the eager one stores the 65k × 1k attention of
   stage 1 and does not fit even batch 2 at 1024 px.
3. **Images stored at longest side ≤ 1024** (`scripts/prepare_coco.py`, annotations rescaled): the
   model letterboxes to ≤ 1024 anyway, so no information reaches the network differently; it cut
   the data from ~23 GB to 2.9 GB and the decode time.
4. **Official splits wherever they exist**: ZeroWaste-f, MJU-Waste, UAVVaste, BePLi v2. TACO has
   none: seeded random 70/10/20 (seed 42) as in `datasets.example.yaml`. (The example config used
   random splits for UAVVaste and BePLi; their official splits were found in the archives.)
5. **Geometry ablation: Depth-Anything-V2-Small depth for all five datasets** (512 px), including
   MJU-Waste instead of its sensor depth (opposite polarity, holes; `geometry_depth_dir` in
   `configs/datasets.yaml`, read only when `model.geometry` is on).
6. Host-RAM measures (see "Host memory"): 6 training / 3 validation workers, validation batch 4
   (prefetch 1), validation workers not persistent; checkpoints carry the optimizer, scheduler and
   early-stopping state so an interrupted run resumes (`train.py --resume`).
7. **Paper is waste** (decision of 5 Nov). ZeroWaste-f annotates only cardboard, soft plastic, rigid
   plastic and metal on a paper sorting line, so its unannotated pixels are either belt or
   unlabelled waste. Negatives there are the **conveyor belt only**, estimated by a fixed colour
   rule (dark and unsaturated, `scripts/make_belt_masks.py`, checked visually); every other
   unannotated pixel is ignored by the loss and by all metrics, so ZeroWaste IoU measures waste vs
   belt. Boundary F1 is now scored only on edges inside the supervised region (masking first drew
   fake edges along ignored areas). Belt / annotated / ignored share of the images: train
   37 / 16 / 47 %, val 38 / 19 / 43 %, test 42 / 18 / 40 %.
8. **BePLi v2 classes are described** (decision of 5 Nov): aliases of TACO entries where the
   object is the same, five new entries in `atlas/bepli.yaml` otherwise (table below).
9. **Fishing net, a novel class, is now measured.** It is written only as text
   (`atlas/novel_field_classes.yaml`); BePLi's fishing nets train the waste mask only (never
   described, never counted as seen) and its 323 test nets (165 images) are named and retrieved
   from the text entry alone (`eval_only_entries` in the registry). The other three novel classes
   (face mask, vape, tyre) have no test data.
10. Held-out classes evaluated: 6 of the 7 (Egg carton has no object in the seeded TACO test
   split; accepted on 5 Nov), plus Fishing net.

## Data

| dataset | source | train / val / test images | objects (train) | labels used | surroundings |
|---|---|---|---|---|---|
| TACO v1.0 | Zenodo 3587843 | 1050 / 150 / 300 (random, seed 42) | 3215 | 60 classes → `atlas/taco.yaml` (all covered) | per-image scene labels |
| ZeroWaste-f v1.2.1 | Zenodo 6412647 | 3002 / 572 / 929 (official) | 18002 | 4 classes → `atlas/zerowaste.yaml` | conveyor belt |
| UAVVaste | Zenodo 8214061 | 540 / 116 / 116 (official) | 2595 | binary | aerial view |
| MJU-Waste v1.0 | Google Drive (authors) | 1485 / 248 / 742 (official) | 1513 | binary; no instance masks (connected components, no contour loss) | indoor floor |
| BePLi v2 | SEANOE 00858/96963 | 2234 / 744 / 744 (official) | 71889 | 13 classes → 7 TACO aliases + 5 `atlas/bepli.yaml` entries + Fishing net (eval-only) | sand |

Every image of every split was loaded once and checked (12 972 images, 0 failures); mask overlays
were checked visually on each dataset. E4 removes the 7 held-out TACO classes from train and val
(36 of 150 TACO val images dropped; their counts in the full set: Glass bottle 104, Pop tab 99,
Plastic straw 157, Crisp packet 39, Paper bag 27, Foam cup 13, Egg carton 11 objects).

BePLi v2 class → atlas entry (objects in train / test):

| BePLi class | atlas entry | train | test |
|---|---|---|---|
| pet_bottle | Clear plastic bottle (TACO) | 2740 | 997 |
| other_bottle | Other plastic bottle (TACO) | 1488 | 514 |
| rope, other_string | Rope & strings (TACO) | 2273 | 585 |
| styrene_foam | Styrofoam piece (TACO) | 5924 | 2610 |
| other_container | Other plastic container (TACO) | 740 | 327 |
| others | Other plastic (TACO) | 13932 | 4693 |
| plastic_bag | Plastic bag (new) | 1848 | 242 |
| box_shaped_case | Plastic box or case (new) | 630 | 262 |
| buoy | Buoy or float (new) | 1503 | 887 |
| other_fishing_gear | Other fishing gear (new) | 631 | 185 |
| fragment | Plastic fragment (new) | 38938 | 10415 |
| fishing_net | Fishing net (novel atlas, eval-only) | 1242 (mask only) | 323 |

Label conventions to keep in mind when reading the waste-IoU numbers:
* **ZeroWaste-f**: waste vs belt only (item 7 above); its IoU is not comparable with ZeroWaste-f
  papers, which score the 4 annotated classes against everything else.
* MJU-Waste images show one object held by a person in an office (≈ 2 % of the pixels); its
  "surroundings" value in the registry is *an indoor floor*.
* BePLi v2 has ~32 small objects per image, so it dominates object-level pools (leakage probe,
  pooled IoU) whenever it is in a test set.

## Bugs found and fixed before the grid (all would have biased numbers silently)

1. **MJU-Waste masks were read inverted.** Its class PNGs store waste as palette index 0 (drawn
   (128,0,0)) and background as index 1 (drawn black); the VOC adapter read raw indices with
   `waste_values=(1,)`, so "waste" was 98 % of every image and the pilot reported MJU IoU 0.974
   after 800 samples. Fix: masks decoded by palette colour (`waste_colors: [[128,0,0]]`,
   `VocAdapter`); 8 images with an all-black palette are now correctly empty.
2. **TACO repeats annotation ids** (e.g. id 309 on images 93 and 101). `loadAnns(getAnnIds(img))`
   then returns another image's object: wrong mask and wrong class, and a crash only when the two
   images differ in size. Fix: `CocoAdapter` reads `coco.imgToAnns[img]`; TACO ids renumbered too.
3. `np.trapezoid` (AUSUC) exists only in numpy ≥ 2: evaluation crashed at the end of every run.
4. The environment-leakage probe returned nothing when any environment had < 5 objects; now
   those environments are left out and the number used is reported (`n_env`).
5. 16 persistent data-loader workers (train + val) exhausted the 14 GB of RAM (swap, 2× slower).

## Preliminary check — `runs/_prelim_e4_bepli_binary` (not a paper result)

E4 stopped after **epoch 7 of 40** (learning rate still near its peak), trained with the *old*
labels (BePLi binary, ZeroWaste paper = background), evaluated on the full test split
(27 853 objects; 500 permutations instead of 2000).

| | value | reading |
|---|---|---|
| waste IoU (test) | TACO 0.831 · MJU 0.868 · ZeroWaste 0.564 · BePLi 0.549 · UAVVaste 0.450 · pooled 0.603 | in-domain segmentation already solid on TACO / MJU; small aerial and beach objects are the hard part |
| descriptor mAP (intrinsic macro) | 0.530 vs floor 0.226 (2.3×) | every field well above its floor: rigidity 0.74/0.33, optics 0.65/0.33, surface 0.55/0.18, color 0.55/0.17, shape 0.34/0.09, material 0.32/0.12 |
| naming, seen classes | top-1 0.129 (default) / 0.121 (val-calibrated) | 8× chance on a 62-class long tail |
| naming, held-out classes | **0.000** at both points, H = 0 | the seen-class bias already seen on synthetic data (A4) |
| AUSUC (calibration-free) | 0.013; best H along the curve 0.101 (seen 0.097, unseen 0.104) | the signal exists, the validation rule (γ = 0.2) picks a point far from it |
| held-out retrieval (lift, p) | Foam cup 139× (0.004) · Glass bottle 15.8× (0.002) · Pop tab 7.8× (0.024) · Plastic straw 7.2× (0.008) · Crisp packet 3.7× (0.08) · Paper bag 0.65× (n.s.) | 4 of 6 classes found significantly above chance from their text entry alone |
| environment probe | 0.987 vs majority 0.748 (context 0.992) | surroundings still fully decodable from object embeddings; the adversary's weight ramps with training progress and is still small at epoch 7 |

<!-- assessment written 2026-11-05 -->
**Assessment.** Segmentation and description are on track for the paper. The held-out *naming*
claim is not: top-1 on unseen classes is 0 at both operating points, although retrieval and the
AUSUC curve show the information is in the embeddings. The bias toward seen classes is a decision
problem at evaluation time, so it can be worked on post hoc on trained checkpoints without
retraining (ATTEMPTS A4 lists the options: leave-class-out calibration on validation, a lower
`calib_keep`, more weight on the descriptor-agreement stream). The environment-leakage claim
cannot be judged before the adversary is at full strength. Comparative claims (LODO vs
SegFormer-B2 and OWLv2→SAM2) need the E3 and baseline runs.

## Journal

**2026-11-05, morning** — Environment, data download and preparation (≈ 23 GB of archives, 2.9 GB
kept), GPU memory/throughput profiling, a 2-epoch pilot (`runs/pilot_e4`, not a result: it ran
before fixes 1–2), the fixes above.

**2026-11-05, 13:03–14:54** — First E4 launch killed by systemd-oomd together with VS Code (host
RAM, not GPU). Memory work described under "Host memory"; queue moved to a capped systemd service.

**2026-11-05, 14:54** — E4 running. Epoch 1 (validation, pooled over datasets): IoU 0.484, desc mAP
0.447 (floor 0.211); per dataset IoU TACO 0.82, MJU 0.80, BePLi 0.50, ZeroWaste 0.45, UAVVaste 0.44.
Epochs 2–7: IoU 0.540 / 0.542 / 0.556 / 0.566 / 0.572 / 0.577, mAP 0.505 / 0.516 / 0.549 / 0.554 /
0.599 / 0.588.

**2026-11-05, 16:10** — Decisions: paper is waste (belt-only negatives on ZeroWaste-f), BePLi classes
described, Fishing net measured as a novel class, 6 held-out classes accepted. These change what
every run learns, so E4 was stopped after epoch 7 and **the whole grid restarts from scratch**. The
stopped E4 is kept as `runs/_prelim_e4_bepli_binary` (old labels: BePLi binary, ZeroWaste paper =
background) and was evaluated on test as a preliminary check — see "Preliminary check" below.

**2026-11-06 → 11-11** — E4 (40/40 epochs, 6.2 h), E6 (38/40, early stop, 3.1 h) and E3-TACO (40/40,
6.9 h) finished on 6 Nov. The queue was stopped twice (6 Nov ≈ 10:30 → 9 Nov 08:12, and again until
11 Nov 23:18) and resumed from `last.pt` both times (E3-ZeroWaste "resumed after epoch 2", then
"after epoch 19"), as designed.

**2026-11-13, 16:00 — first reading of the 10 finished runs** (E4, E6, E3 × 5, OWLv2→SAM2, E5
−transplant, −adversary). Numbers from the tables below.

1. *Segmentation with every environment in training is strong*: E4 pooled IoU 0.870 (TACO 0.885,
   ZeroWaste-f 0.912 waste-vs-belt, MJU 0.895, BePLi 0.616, UAVVaste 0.490).
2. *Leave-one-dataset-out splits the datasets in two groups.* TACO 0.630 and ZeroWaste-f 0.747
   transfer (71 % and 82 % of E4; 5× and 3× OWLv2→SAM2). UAVVaste 0.262 (precision 0.33) and BePLi
   0.282 (recall 0.31) lose about half.
3. **MJU LODO fails: IoU 0.055 with recall 0.989 and precision 0.055.** Not a label bug: E4 scores
   0.895 on the same images and labels. The qualitative figure (`paper/figures/qualitative.png`)
   shows the LODO model marking the person (face, hands, clothes) and office furniture as waste: no
   training environment has people or furniture among its negatives. Remedy (outside the grid):
   waste-free images of indoor scenes with people as extra negatives.
4. *Description*: macro mAP 0.691 vs floor 0.226 (3.1×) and raw CLIP 0.305; every field above both.
   On unseen datasets the margin over the floor shrinks to 0.09–0.15 (TACO 0.358/0.248, ZeroWaste-f
   0.480/0.333, BePLi 0.457/0.370).
5. *Retrieval from text*: 6 of 7 unseen classes significant (p ≤ 0.009); the text-only Fishing net
   has AP 0.094 against a prevalence of 0.011 (n = 319). Paper bag n.s. (5 objects).
6. *Naming of unseen classes is still the weak point* (as in the preliminary check): top-1 2.9 %,
   chance 1.8 %, γ chosen on validation = 0, AUSUC 0.031, oracle H 0.126. Raw CLIP crops name them at
   29.1 % (H 0.224), 10× better: training biases the shared text space toward the 61 seen classes.
   The A4 calibration options can be tested post hoc on the saved checkpoints, on CPU, without
   touching the grid.
7. *Leakage*: probe 0.985 vs majority 0.748 with the adversary, 0.992 without. The adversary barely
   changes it, but the probe is confounded: here the class predicts the place (BePLi classes only on
   sand, the ZeroWaste-f classes only on the belt). A within-class probe on TACO scenes is needed
   before any claim either way.
8. *Ablations so far*: −transplant costs 3.0 desc-mAP points with no IoU change; −adversary changes
   neither (calibrated H 0.084 vs 0.049, within noise). E6 at 640 px: −1.4 IoU (UAVVaste −3.9,
   BePLi −3.7), −5.9 mAP.

*Promotion assessment.* Segmentation (seen environments, and street/conveyor transfer) and
description are supported. Open-vocabulary naming and "environment removed" are not supported by
the current numbers. The MJU failure is a result worth reporting, not a bug to hide.

**2026-11-13, 17:00 — manuscript updated with these numbers** (`paper/main.tex`, marked
PRELIMINARY): results and discussion for the finished runs; new Fig. 1 (data flow, TikZ,
`paper/figures/pipeline.tex`); qualitative figure + table (`scripts/make_figures.py` → 
`paper/figures/qualitative.{pdf,json}`, `qualitative_objects.tex`; images by a seeded rule, CPU
only); datasets table filled; `tab_lodo` gains an E4 column and `tab_retrieval` prints p < 0.001.
Still `\tbd`: SegFormer, E1/E2, the other E5 rows, naming calibration, Jetson, atlas re-reading.

**2026-11-13, 17:00–18:30 — post-hoc analyses on saved checkpoints** (`scripts/posthoc.py`;
per-object embeddings dumped on the GPU with a 1.35 GB per-process cap next to the training, 6–12
min per split; analyses on CPU). Outputs: `runs/<run>/posthoc/*.json`, `runs/_posthoc_*.json`.

* *Naming calibration (E4, test, 48 seen / 7 unseen classes).* The current rule (γ keeping 90 % of
  validation seen accuracy) gives H 0.049. Keeping 70 %: seen 0.141 / unseen 0.172 / **H 0.155**;
  descriptor agreement alone at 90 %: H 0.147. Entry-wise standardisation of scores fails (H ≤ 0.06).
* *Raw CLIP is the stronger baseline for unseen classes.* Naming H 0.217 (crop vs atlas sentence).
  Retrieval mean AP over the 7 unseen classes: CLIP crop vs class name 0.078 (7/7 p < 0.01), vs
  sentence 0.057 (7/7); TrashScribe fused 0.034 (6/7), descriptor agreement alone 0.054 (7/7).
  Complementary: TrashScribe + CLIP sentence 0.108 mean AP, naming H 0.188.
* *Within-class environment probe* (15 classes present in ≥ 2 environments, 9 191 objects): intrinsic
  embedding +0.069 over the within-class majority (7 classes with n ≥ 20 and majority < 0.9), context
  +0.207. Without adversary +0.097, without transplant +0.068. The pooled 98.5 % probe was the class
  predicting the place; intrinsic pooling does keep most of the place out; the adversary adds little.
* *Metric bug fixed* (`metrics.ausuc`): the γ sweep penalised only the seen classes present in the
  test set. Corrected AUSUC / oracle H: E4 0.055 / 0.172 (was 0.031 / 0.126), −adversary 0.050 /
  0.170, −transplant 0.048 / 0.158. Runs evaluated from now on use the corrected definition; the
  AUSUC column of earlier runs is underestimated.

*Assessment.* Literature search (13 Nov): no prior dense, attribute-level description of waste and no
leave-one-dataset-out waste segmentation benchmark found; zero-shot waste classification by class
names with CLIP/VLMs exists (image level), as do open-vocabulary waste detectors on names and a
multi-dataset single-class litter *detection* benchmark (detect-waste). What survives as new is
**dense description across environments** and **the cross-environment protocol**; naming and
retrieval of new classes do not beat raw CLIP, and environment removal is not an adversary effect.
Whether description is open-vocabulary (not just supervised attribute prediction) is decided by two
runs now at the head of the queue: the closed-vocabulary probe and **E7** (three descriptor values
never supervised, `configs/exp/e7_heldout_values.yaml`, new opt-in `train.loss.unsupervised_values`,
verified: unchanged loss when off, zero gradient on dropped values, unit tests, CPU smoke run).

*Queue changes (13 Nov).* Removed (configs kept, re-add the line to restore): `e5_ablate_lora8`,
`e5_ablate_adapters_both`, `e5_ablate_geometry`, `e2_multi_domain` (≈ 31 GPU-hours). Added
`e7_heldout_values`. New order: probe → E7 → no_dense → no_debias → SegFormer LODO × 5 → E1 × 5 →
lente_rgb.

**2026-11-15, 17:00 — runs finished 13–15 Nov, why SegFormer failed, E7 verdict, improvements.**
Finished: closed probe, E7, −dense, −debias, E1 TACO / ZeroWaste-f / UAVVaste (E1-MJU running).

1. *Closed probe vs open descriptor head (E5).* Description mAP **0.712 vs 0.691**: the closed
   per-value classifier is as good or better on 7 of 9 fields (colour 0.816 vs 0.738, state 0.924
   vs 0.875); IoU identical. On supervised attributes the text-space head brings nothing.
2. **E7 (3 descriptor values never supervised, `runs/_posthoc_values_e7.json`).** AP on test objects
   (floor = prevalence):

   | value | E7 (never supervised) | E4 (supervised) | raw CLIP crops | floor |
   |---|---|---|---|---|
   | material: metal | **0.140** | 0.287 | 0.033 | 0.006 |
   | material: expanded polystyrene foam | 0.096 | 0.432 | 0.114 | 0.097 |
   | surface: crumpled or wrinkled | 0.062 | 0.836 | 0.276 | 0.083 |

   One value of three transfers (metal: 4× raw CLIP, 22× the floor); foam is at the floor and
   crumpled *below* it, where raw CLIP reads both. **The open-vocabulary claim for description is
   not supported**: the head interpolates between phrases near supervised ones, it does not read new
   ones. Together with point 1 the novelty of the description part is the *task and data* (dense
   attribute description of waste across five environments, atlas profiles), not the mechanism.
3. *−dense*: mAP 0.649 (−4.2 points), IoU unchanged: pixel-level descriptor supervision matters.
4. *−debias* (no adversary, no MixStyle, no transplant): IoU 0.874, mAP 0.694, AUSUC 0.048 (E4
   0.055 corrected) — nothing measurable in seen environments. *Run-to-run noise:* −transplant alone
   gave 0.661 (−3.0) while −debias, which also drops the transplant, gives 0.694; naming H moves
   0.03–0.12 across variants that should not change it. All runs are single-seed: differences below
   ≈ 3 mAP points or 0.1 H are not interpretable.
5. *E1 in-domain vs one model for all (E4)*: TACO 0.862 vs 0.885 (+2.3 for E4), ZeroWaste-f 0.922 vs
   0.912 (−1.0), **UAVVaste 0.616 vs 0.490 (−12.6 for E4)**. Joint training costs the aerial view a
   lot (tiny objects, one dataset in five). E1 runs early-stop (17–25 epochs).
6. **Why the 5 SegFormer-B2 LODO runs failed** (15 Nov 03:16–06:36, 15 attempts × 13 min): CUDA out of
   memory at the *first validation*, identical in every attempt. Training ran at batch 2; validation
   at batch 4 and 1024 px, where the SegFormer decode head concatenates four 768-channel maps at
   256 × 256 (1.5 GiB per tensor) on top of the blocks the allocator still cached from training (6.6
   GiB allocated, 7.9 GiB card). Deterministic, so the 3 retries per job were wasted (≈ 3.3 h).
   Fixed in `scripts/baseline_segformer.py`: cache emptied before evaluation, the queue validates at
   batch 1, `last.pt` every epoch (a stopped queue now resumes SegFormer too), and the same
   patience-8 early stop as TrashScribe (same protocol for both). Re-queued with `max_attempts: 6`
   (the running queue keeps its count of 3), MJU first.

*Improvements implemented (15 Nov)* — each targets a measured weakness:

* **E8 `e8_clip_distill`** (raw CLIP names / retrieves unseen classes better and reads values E7
  cannot): a second dense head (`ClipField`) trained so that its mask-pooled vector matches frozen
  CLIP ViT-B/16 on the object's crop (`trashscribe/clip_teacher.py`: square crop 1.35× the longest
  side, cut on the GPU with `roi_align` from the *augmented* batch, ≤ 24 objects per micro-batch,
  longest side ≥ 48 px; loss 1 − cos). Built on E7, so one run tests unseen classes *and* unseen
  values; the distilled field never sees a class name or a descriptor label; selection unchanged.
  CPU smoke run: distillation loss 0.80 → 0.43 in 2 tiny epochs, other terms unaffected.
  `posthoc.py dump` stores the pooled field (`zclip`); `clipret --visual zclip` and `values` score it
  exactly like raw CLIP crops. Queued next (starts when E1-MJU ends, ≈ 21:00).
* **E3b `e3b_lodo_mju_negatives`** (MJU LODO: people and furniture segmented as waste): E3-MJU plus
  243 waste-free COCO val2017 indoor scenes with a person ≥ 5 % of the image and no container,
  tableware, food, bag, book or glass category (`scripts/make_negatives.py`; rule fixed before any
  result; 194 train / 49 val, all-background masks, nothing from MJU-Waste). New opt-in
  `extra_datasets` key: the negatives exist only in this config (the registry is untouched, so
  `train_datasets: null` runs cannot pick them up). Data path checked (augmentation, transplant
  onto the negatives, collate). Queued after SegFormer-MJU: together they say whether the failure is
  the data's, the model's, and whether generic negatives fix it.
* **Zoom-in tiling at test time** (small objects: UAVVaste, BePLi; `posthoc.py tiles`): 2 × 2
  overlapping tiles of the canonical frame, each enlarged to 1024 px, pasted back and averaged with
  the full-frame prediction. On saved checkpoints, no retraining (`runs/<run>/posthoc/tiles2.json`).
  The full-frame column reproduces the official test IoUs exactly (E4 pooled 0.870, UAVVaste 0.489).

  | model | UAVVaste full / tiles / mean | BePLi full / tiles / mean | others (tiles) |
  |---|---|---|---|
  | E4 (all five) | 0.489 / **0.529** / 0.512 | 0.616 / 0.589 / 0.610 | TACO 0.885→0.790, MJU 0.895→0.426, ZeroWaste-f 0.912→0.905 |
  | E3 LODO | 0.262 / **0.295** / 0.276 | 0.282 / 0.268 / 0.274 | – |
  | E1 UAVVaste only | 0.616 / 0.538 / 0.596 | – | – |

  Zoom helps only aerial images, and only for models trained mostly on ground-level views (+4.0
  E4, +3.3 LODO); the aerial-only model gets worse (−7.8), and large close-up objects are cut into
  pieces (MJU −47). Reading: part of the 12.6-point aerial cost of joint training (E1 vs E4) is a
  **scale mismatch** — the joint model learned ground-level object sizes. BePLi does not gain
  because its images are stored at ≤ 1024 px: enlarging adds no information. Not a general
  test-time fix; the training-side remedy is scale augmentation (random zoom-in crops of aerial
  images) in multi-domain training — proposed, not run. A deployment that knows it is on a drone
  can tile: +4.0 IoU at 5× the compute.
* **Literature comparison figure** (`scripts/make_comparison.py` → `paper/figures/comparison.*`):
  the five qualitative-figure images, run through one representative of each family of prior work
  (image-level CLIP zero-shot classification; Grounding DINO → SAM 2 open-vocabulary detection;
  SAM 3 text-prompted concept segmentation; SegFormer-B2 once its LODO runs exist), next to
  TrashScribe with the image's dataset unseen (E3) and with all datasets (E4); a table compares the
  description of one object per image: annotation profile vs raw-CLIP attribute prompting vs
  TrashScribe. CPU only, each method in its own memory-capped process.

  Per-image IoU (TACO / ZeroWaste-f / UAVVaste / MJU / BePLi): Grounding DINO→SAM 2 0.97 / 0.29 /
  0.49 / 0.79 / 0.00 (no box on the beach); **SAM 3 0.97 / 0.94 / 0.65 / 0.52 / 0.80**; TrashScribe
  LODO 0.97 / 0.92 / 0.51 / 0.15 / 0.63; TrashScribe E4 0.97 / 0.97 / 0.59 / 0.93 / 0.68. Image-level
  CLIP says "paper" for the aerial and office scenes (the background). Descriptions of the three
  objects with an atlas profile: TrashScribe agrees on 9 of 11 fields, raw CLIP on 3.
  **Warning for the paper:** SAM 3 zero-shot (848 M parameters) beats TrashScribe-LODO on 4 of 5
  images. Five images are not a measurement, so SAM 3 and Grounding DINO→SAM 2 are queued on the
  full test split (GPU, ~30 min each, `baseline_zeroshot.py --detector sam3|gdino`; CPU check on 2
  images passed). If SAM 3 matches TrashScribe-LODO there, the segmentation-transfer claim becomes
  "transfer at edge cost" (34 M parameters, one pass, description included), not "better transfer".

*Queue (15 Nov 17:45):* E1-MJU → **SAM 3 zero-shot** → **Grounding DINO→SAM 2 zero-shot** → **E8** →
SegFormer-MJU → **E3b** → E1-BePLi → SegFormer UAVVaste, BePLi, TACO, ZeroWaste-f → lente_rgb.

**2026-11-15, 18:30 — manuscript style pass (user request).** No em dashes, no ", and" constructs,
headings without parentheses or whole phrases, fewer subsections (Related work 3, Method 4, Setup 2,
Results 5), a roadmap at the end of the introduction and a lead-in before the subsections of each
section. Title without its em-dash clause (also in `title_page.tex` and `cover_letter.tex`). Three
statements made stale by the 13 Nov queue pruning were corrected: the depth gate and LoRA are no
longer "evaluated as ablations", and E2 is no longer listed (E4 is the multi-domain reference).
Backup of the previous text: scratchpad `main.tex.before-style`. Compiles cleanly (24 pages).

**2026-11-15, 19:00 — queue reordered by expected significance (user request); E9 added.**
*SAM 3 prompts.* With the four waste words SAM 3 finds nothing in MJU-Waste training images: its
presence head judges "litter"/"trash"/"garbage" absent from an office where a person holds a bottle
(presence 0.02–0.26; detection scores 0.6–0.7 are multiplied by it), while "bottle" reaches 0.995.
Fixed before any result: a second prompt set, `objects16` = the 4 waste words + 12 waste-object nouns
after the most frequent TACO supercategories (bottle, can, cup, plastic bag, food wrapper, carton,
cardboard box, paper, bottle cap, plastic fragment, styrofoam piece, cigarette butt), the same for every
site. Text embeddings are computed once (`get_text_features` → `text_embeds`). Both sets are measured;
the paper's SAM 3 column uses the 16 prompts (stronger reference).
*E9, SAM 3 as an offline teacher* (`configs/exp/e9_sam3_{mju,uavvaste,bepli}.yaml`): SAM 3 (16 prompts)
labels the target's train/val images (`baseline_zeroshot.py --tasks pseudo`; score ≥ 0.5 waste,
0.3–0.5 or > half the image ignored, rest background; written to `~/data/waste/_pseudo/sam3/<ds>/`);
the E3 LODO model is fine-tuned (`train.init_from`, 12 epochs, patience 4) on the four labelled
datasets plus these images; selection on the four val sets plus the SAM 3-labelled target val; test
on the real target labels. Real target labels are never used (pseudo-label quality against them is
recorded only). CPU smoke checks: on 3 MJU train images the 16-prompt labels have recall 0.94,
precision 0.32 (other desk objects labelled too: MJU annotates only the hand-held object); the E9
config initialises from `runs/e3_lodo_mju/best.pt`, trains, validates and tests.
*Order now:* E1-MJU (running) → SAM 3 zero-shot 16 prompts (full test) → [SAM 3 labels → E9] for
MJU, UAVVaste, BePLi → E8 → SegFormer-MJU → E3b → SAM 3 4-word and Grounding DINO zero-shot →
SegFormer UAVVaste, BePLi → E1-BePLi → SegFormer TACO, ZeroWaste-f → lente_rgb. Grid end ≈ Thu 19 Nov
early morning (upper bound).

**2026-11-16, morning — overnight results (all jobs ok; E8 running, epoch 13, val IoU 0.81).**
*"nan" in the monitor was a display problem, not failures:* (a) `progress.py` read only OWLv2's result key, so the
SAM 3 and pseudo-label jobs printed "zero-shot IoU nan"; (b) runs tested only on UAVVaste or MJU-Waste have no
described objects, so description / naming are undefined (NaN). Fixed: "n/a" with the reason, SAM 3 and
pseudo-label jobs show their own numbers; legend added under the results table.

| target | TrashScribe LODO | SAM 3 zero-shot, 16 prompts (first run) | **E9: LODO + SAM 3 teacher** | SAM 3 label quality on target train / val (record only) | E4 all five | E1 |
|---|---|---|---|---|---|---|
| UAVVaste | 0.262 | 0.424 | **0.375** (+11.3) | 0.485 / 0.391 | 0.490 | 0.616 |
| MJU-Waste | 0.055 | 0.221 | **0.231** (+17.6) | 0.236 / 0.214 | 0.895 | 0.914 |
| BePLi v2 | 0.282 | 0.272 | **0.398** (+11.6) | 0.492 / 0.514 | 0.616 | – |

SAM 3 zero-shot (16 prompts) elsewhere: TACO 0.759 (LODO 0.630), ZeroWaste-f 0.793 (0.747); pooled 0.579.
Readings: (1) SAM 3 beats TrashScribe-LODO on 4 of 5 unseen environments, by 5–17 points; TrashScribe trained
on an environment beats SAM 3 everywhere (E4). (2) The SAM 3 teacher lifts the edge model by 11–18 points on
the three hardest sites; the student ends close to the quality of its labels (MJU 0.231 vs labels 0.236).
MJU stays low for everyone: its annotation counts only the hand-held object, while "bottle", "cup", "paper"
also fire on desk objects (label precision 0.24). (3) All three E9 runs were still improving at their last
epoch (best = epoch 12 of 12): under-trained. (4) **The SAM 3 test numbers are not yet comparable with E9:**
the first SAM 3 evaluation counted every mask as waste, while the Grounding DINO baseline and the E9 labels
drop detections larger than half the image. On BePLi the labels of SAM 3 reach 0.49–0.51 against the real
annotations but SAM 3 scores 0.272 on test, which points at whole-scene masks. `baseline_zeroshot.py` now writes
both variants (`where_sam3`, `where_sam3_area`); `e4_heldout_classes__zeroshot_sam3obj_area` is queued right
after E8 (≈ 1 h; the first result is kept). Until then "E9 beats SAM 3 on BePLi and MJU" is not a claim.

*Longer E9 training: not queued (16 Nov, 11:00).* "Best at epoch 12 of 12" is the end of the cosine schedule,
not a trend: pooled val IoU is flat from epoch ~6 in all three runs (MJU 0.833–0.836, UAVVaste 0.835–0.836,
BePLi 0.834–0.839), and the rising val score comes from description mAP on the source datasets. On the
target's own val images the student already agrees with the SAM 3 labels at 0.857 (MJU), 0.638 (UAVVaste),
0.687 (BePLi), while those labels agree with the real annotations at only 0.21 / 0.39 / 0.51 and the test
IoUs sit close to them. The ceiling is label quality, so more epochs would only copy the teacher's errors
more faithfully. Cheap check instead: zoom-in tiling at test time on `e9_sam3_uavvaste` (CPU, `posthoc.py tiles`).
Result (`runs/e9_sam3_uavvaste/posthoc/tiles2.json`): full frame 0.375 (= official test IoU), 2 × 2 tiles alone
0.304 (−7.1), mean of both **0.391 (+1.6**, precision 0.453 → 0.479). Like the aerial-only E1 model (−7.8 with
tiles), E9 has now learnt the aerial object scale from the SAM 3-labelled images, so zooming no longer helps on
its own; the averaged prediction gains a little. Still below SAM 3 zero-shot (0.424, unfiltered); no further
UAVVaste action until the fair SAM 3 number is in.

**2026-11-17, 11:45 — after two reboots; fair SAM 3; SegFormer-MJU; queue reordered.**
*Reboots.* 16 Nov 15:08 (queue restarted 17:39, after E8 had finished at 14:37) and 17 Nov 08:15, a clean
requested reboot (`systemd-reboot.service`; kernel updated to 6.8.0-149), machine back at 11:15. E3b
(started 05:43) had saved 14 epochs; queue and report timer restarted at 11:31, E3b "resumed after epoch 14".
*Fair SAM 3* (`e4_heldout_classes__zeroshot_sam3obj_area`, 16 prompts; masks > half the image dropped, the
same rule as the Grounding DINO boxes and the E9 labels): TACO 0.713 (raw 0.759), ZeroWaste-f 0.783 (0.793),
UAVVaste 0.424 (0.424), MJU 0.221 (0.221), **BePLi 0.424 (raw 0.272)**, pooled 0.613 (raw 0.579). The paper
table uses this rule (stronger pooled, consistent with the other baselines); raw numbers stay here.

| site | TrashScribe LODO | SegFormer-B2 LODO | SAM 3 (fair) | E9 (+ SAM 3 labels) | E9 / SAM 3 | E4 (site seen) |
|---|---|---|---|---|---|---|
| UAVVaste | 0.262 | queued | 0.424 | 0.375 (0.391 tile-mean) | 88 % (92 %) | 0.490 |
| MJU-Waste | 0.055 | **0.104** | 0.221 | 0.231 | 105 % | 0.895 |
| BePLi v2 | 0.282 | queued | 0.424 | 0.398 | 94 % | 0.616 |

Readings: (1) E9 does not beat SAM 3 on BePLi: the earlier "+12.6" came from the unfair SAM 3 number. The
edge model with SAM 3 labels reaches 88–105 % of SAM 3 at a new site with 34 M against 848 M parameters.
(2) Where the edge model has labels (E4) it beats SAM 3 everywhere, by 6.6 (UAVVaste) to 67.4 (MJU) points.
(3) **SegFormer-B2 fails on MJU too** (0.104, precision 0.108, recall 0.707, 40 epochs, 11 h): the MJU failure
belongs to the protocol (no indoor negatives anywhere in training), not to TrashScribe.
*Queue (by significance):* E3b (running) → **E9 on TACO and ZeroWaste-f** (completes the SAM 3-teacher
column; the LODO model is already close to SAM 3 there) → SAM 3 4-word and Grounding DINO columns →
SegFormer UAVVaste, BePLi → lente_rgb → E1-BePLi → SegFormer TACO, ZeroWaste-f.

**2026-11-17, 12:15 — E8: the distilled CLIP field** (`runs/_posthoc_e8.log`, `runs/e8_clip_distill/posthoc/`,
`runs/_posthoc_values_e8.json`). Standard metrics as E7 (IoU 0.870, mAP 0.658 vs 0.664, val-calibrated H 0.135
vs 0.124, AUSUC 0.041 vs 0.043). What the second head adds, scored like raw CLIP crops:

| test (7 unseen classes / 3 unseen values) | raw CLIP crops (E4 dump) | E8 distilled field | E8 descriptor head |
|---|---|---|---|
| retrieval, class name: mean AP (p<0.01) | 0.078 (7/7) | **0.082 (7/7)** | fused streams 0.091 (5/7; Foam cup, n = 5, gives 0.447) |
| retrieval, atlas sentence: mean AP | 0.057 (7/7) | 0.051 (6/7) | – |
| naming, class name: seen / unseen / H | 0.182 / 0.291 / 0.217 | 0.086 / 0.198 / 0.120 | fused 0.145 / 0.114 / 0.128 |
| value AP: metal (floor 0.006) | 0.033 | 0.053 | **0.147** |
| value AP: expanded polystyrene foam (floor 0.097) | 0.114 | **0.152** | 0.091 |
| value AP: crumpled or wrinkled (floor 0.083) | 0.276 | 0.240 | 0.061 |

Fixed fusion (mean of z-scored head posterior and field similarity): metal 0.133, foam 0.113, crumpled
0.147, all above their floors but below the better source on each. Reading: the distilled field brings
CLIP's open reading of new classes and properties into the single forward pass (retrieval on par with raw
CLIP, unseen values at or near raw CLIP), for 0.7 M extra parameters and no loss on supervised description.
Naming of new classes still trails raw CLIP (H 0.120 vs 0.217): the field names seen classes poorly (0.086).
Manuscript updated (description, naming and discussion paragraphs).

**2026-11-17, 17:15 — E3b: generic negatives fix the indoor failure.** MJU-Waste leave-one-out with 243
waste-free COCO val2017 indoor scenes with people as extra negatives (no MJU image, no waste annotation):
**IoU 0.363** (precision 0.572, recall 0.498, BF1 0.192) against 0.055 without them, 0.104 for SegFormer-B2,
0.221 for SAM 3 (fair) and 0.231 for E9 (SAM 3 labels on MJU images). The cheapest fix is the best one, and
it is the one case where the 34 M edge model beats SAM 3 on an environment it never saw. Literature re-check
(17 Nov): still no leave-one-dataset-out waste benchmark; closest attribute work is GSA2Seg (arXiv
2404.18112: placement attributes of garbage, robot dogs, one dataset, cloud), now cited. Manuscript: E3b,
SegFormer-MJU, fair SAM 3 and E9 paragraphs written in the cross-environment section.

**2026-11-17, 18:00 — E10 queued: how many labelled images does the edge model need to match SAM 3?**
User goal: claim that the edge model behaves as well as or better than SAM 3. Today that holds where the site
is in training (E4 > SAM 3 on all five, by 6.6–67.4 points) but not at a site with no data at all (SAM 3 wins
4 of 5; E9 reaches 88–105 %). A claim that holds everywhere needs a stated budget of site data. E10: the
leave-one-out model fine-tuned with **10 labelled images** of the new site (seeded random subset, 25 % of
the training samples, the four other datasets for the rest), 6 epochs, last epoch tested (validation holds
no image of the site). New opt-in keys: `train.subsample`, `train.dataset_share`, `train.select: last`
(unit test added, 14/14; CPU smoke run ok). Rule fixed now: a site where 10 images stay below SAM 3 gets a
50-image run; nothing else is tuned. Queued right after E9-TACO; results ≈ 20:30 (UAVVaste) to ≈ 01:00 (MJU).

**2026-11-17, 22:15 — first answers to "edge model ≥ SAM 3?"** (fair SAM 3 = 16 prompts, masks ≤ half image)

| site | SAM 3 | LODO | + negatives (E3b) | + SAM 3 labels (E9) | + 10 labels (E10) | site seen (E4) |
|---|---|---|---|---|---|---|
| TACO | 0.713 | 0.630 | – | **0.783** | running | **0.885** |
| ZeroWaste-f | 0.783 | 0.747 | – | queued | queued | **0.912** |
| UAVVaste | 0.424 | 0.262 | – | 0.375 | 0.289 → 50 images queued | **0.490** |
| MJU-Waste | 0.221 | 0.055 | **0.363** | **0.231** | queued | **0.895** |
| BePLi v2 | 0.424 | 0.282 | – | 0.398 | **0.454** | **0.616** |

Bold = at or above SAM 3. E9-TACO: SAM 3 labels agree with the TACO annotations at 0.757 (train) / 0.788
(val), the student reaches 0.783 on test (precision 0.829, recall 0.934). E10-UAVVaste adds only 2.7 points
(precision 0.345): per the rule fixed beforehand, `e10_k50_uavvaste` is queued after the 10-image runs.
Paper: new Table `tab:ladder` (generated by `make_tables.py`) with this layout; the LODO table keeps the
benchmark columns. Edge export done: `runs/e6_edge_640/export/` (ONNX 127.5 MB, max |ONNX − PyTorch|
2.4e-5, atlas_cache.pt, export_meta.json; all git-ignored, see docs/JETSON.md).

<!-- AUTO:BEGIN (scripts/report_results.py) -->

_Tables regenerated 2026-11-19 10:11 from `runs/*/test_metrics.json` and `runs/*/history.json`. Numbers are fractions (0–1), test split unless stated._

### Queue status

Snapshot 2026-11-19 10:11 · service active · RAM 1.2/8.0 GB · GPU 37% · 0.1 GB · 53 °C · 420 MHz · grid ends ≤ **Fri 20 Nov 2026, 14:24** (all jobs at full length; early stops make it sooner). Live view: `conda activate trashscribe && python scripts/progress.py --watch 30`.

| # | run | status | epochs | best validation (pooled IoU, desc mAP) | time | ends |
|---|---|---|---|---|---|---|
| 1 | `e4_heldout_classes` | done | 40/40 | ep 35: IoU 0.832, mAP 0.690 | 6.2 h |  |
| 2 | `e6_edge_640` | done | 38/40 | ep 30: IoU 0.816, mAP 0.655 | 3.1 h |  |
| 3 | `e3_lodo_taco` | done | 40/40 | ep 39: IoU 0.839, mAP 0.737 | 6.9 h |  |
| 4 | `e3_lodo_zerowaste` | done | 40/40 | ep 33: IoU 0.713, mAP 0.717 | 3.8 h |  |
| 5 | `e3_lodo_uavvaste` | done | 40/40 | ep 38: IoU 0.835, mAP 0.726 | 7.2 h |  |
| 6 | `e3_lodo_mju` | done | 40/40 | ep 37: IoU 0.834, mAP 0.732 | 7.2 h |  |
| 7 | `e3_lodo_bepli` | done | 40/40 | ep 38: IoU 0.890, mAP 0.771 | 6.3 h |  |
| 8 | `e4_heldout_classes__zeroshot` | done |  |  | 38 min |  |
| 9 | `e5_ablate_no_transplant` | done | 32/40 | ep 24: IoU 0.832, mAP 0.703 | 5.9 h |  |
| 10 | `e5_ablate_no_adversary` | done | 40/40 | ep 40: IoU 0.834, mAP 0.728 | 7.6 h |  |
| 11 | `e5_ablate_no_mixstyle` | done | 40/40 | ep 33: IoU 0.836, mAP 0.674 | 7.9 h |  |
| 12 | `e5_ablate_probe` | done | 40/40 | ep 39: IoU 0.833, mAP 0.723 | 7.5 h |  |
| 13 | `e7_heldout_values` | done | 40/40 | ep 40: IoU 0.832, mAP 0.712 | 7.5 h |  |
| 14 | `e5_ablate_no_dense` | done | 40/40 | ep 40: IoU 0.833, mAP 0.666 | 7.2 h |  |
| 15 | `e5_ablate_no_debias` | done | 40/40 | ep 35: IoU 0.837, mAP 0.718 | 7.2 h |  |
| 16 | `e1_indomain_taco` | done | 17/40 | ep 9: IoU 0.881, mAP 0.740 | 2.3 h |  |
| 17 | `e1_indomain_zerowaste` | done | 25/40 | ep 17: IoU 0.904, mAP 0.782 | 3.5 h |  |
| 18 | `e1_indomain_uavvaste` | done | 25/40 | ep 17: IoU 0.586 | 3.2 h |  |
| 19 | `e1_indomain_mju` | done | 30/40 | ep 22: IoU 0.918 | 4.4 h |  |
| 20 | `e4_heldout_classes__zeroshot_sam3obj` | done |  |  | 54 min |  |
| 21 | `e3_lodo_mju__zeroshot_sam3obj` | done |  |  | 31 min |  |
| 22 | `e9_sam3_mju` | done | 12/12 | ep 12: IoU 0.836, mAP 0.735 | 2.5 h |  |
| 23 | `e3_lodo_uavvaste__zeroshot_sam3obj` | done |  |  | 16 min |  |
| 24 | `e9_sam3_uavvaste` | done | 12/12 | ep 12: IoU 0.836, mAP 0.724 | 2.2 h |  |
| 25 | `e3_lodo_bepli__zeroshot_sam3obj` | done |  |  | 67 min |  |
| 26 | `e9_sam3_bepli` | done | 12/12 | ep 12: IoU 0.839, mAP 0.767 | 2.2 h |  |
| 27 | `e8_clip_distill` | done | 40/40 | ep 36: IoU 0.833, mAP 0.699 | 9.0 h |  |
| 28 | `e4_heldout_classes__zeroshot_sam3obj_area` | done |  |  | 64 min |  |
| 29 | `e3_lodo_mju__segformer_mit-b2` | done | 40/40 | ep 33: IoU 0.861 | 11.7 h |  |
| 30 | `e3b_lodo_mju_negatives` | done | 40/40 | ep 37: IoU 0.825, mAP 0.726 | 5.0 h |  |
| 31 | `e3_lodo_taco__zeroshot_sam3obj` | done |  |  | 25 min |  |
| 32 | `e9_sam3_taco` | done | 12/12 | ep 12: IoU 0.836, mAP 0.741 | 2.4 h |  |
| 33 | `e10_k10_uavvaste` | done | 6/6 | ep 6: IoU 0.834, mAP 0.732 | 73 min |  |
| 34 | `e10_k10_bepli` | done | 6/6 | ep 6: IoU 0.890, mAP 0.749 | 72 min |  |
| 35 | `e10_k10_taco` | done | 6/6 | ep 6: IoU 0.837, mAP 0.728 | 74 min |  |
| 36 | `e10_k10_zerowaste` | done | 6/6 | ep 6: IoU 0.706, mAP 0.711 | 66 min |  |
| 37 | `e10_k10_mju` | done | 6/6 | ep 6: IoU 0.833, mAP 0.726 | 68 min |  |
| 38 | `e10_k50_uavvaste` | done | 6/6 | ep 6: IoU 0.834, mAP 0.725 | 67 min |  |
| 39 | `e3_lodo_zerowaste__zeroshot_sam3obj` | done |  |  | 1.7 h |  |
| 40 | `e9_sam3_zerowaste` | done | 12/12 | ep 10: IoU 0.949, mAP 0.713 | 2.4 h |  |
| 41 | `e4_heldout_classes__zeroshot_sam3` | done |  |  | 39 min |  |
| 42 | `e4_heldout_classes__zeroshot_gdino` | done |  |  | 24 min |  |
| 43 | `e3_lodo_uavvaste__segformer_mit-b2` | done | 40/40 | ep 36: IoU 0.865 | 12.1 h |  |
| 44 | `e3_lodo_bepli__segformer_mit-b2` | done | 30/40 | ep 22: IoU 0.909 | 8.7 h |  |
| 45 | `e5_ablate_lente_rgb` | done | 39/40 | ep 31: IoU 0.841, mAP 0.276 | 7.3 h |  |
| 46 | `e1_indomain_bepli` | queued | 0/40 |  | ~6.1 h | ≤ 16:17 |
| 47 | `e3_lodo_taco__segformer_mit-b2` | queued | 0/40 |  | ~11.1 h | ≤ Fri 20 Nov 03:20 |
| 48 | `e3_lodo_zerowaste__segformer_mit-b2` | queued | 0/40 |  | ~11.1 h | ≤ Fri 20 Nov 14:24 |

### All finished TrashScribe runs (test split, pooled over the run's test datasets)

Naming = generalised zero-shot top-1 over the atlas (seen classes / held-out classes / harmonic mean). "Default" uses no seen-class penalty; "val-calibrated" uses the penalty γ chosen on validation. Env probe = 5-fold linear probe predicting the surroundings from the intrinsic object embedding. "–" = not defined for that run, not a failure: UAVVaste and MJU-Waste have a single "waste" class with no atlas description, so runs tested only on them have no description, naming or probe score; runs without held-out classes in their test set have no unseen / H / AUSUC.

| run | IoU | BF1 | desc mAP (floor) | naming default: seen / unseen / H | naming val-calibrated: seen / unseen / H | AUSUC | γ | env probe (majority) |
|---|---|---|---|---|---|---|---|---|
| `e4_heldout_classes` | 0.870 | 0.559 | 0.691 (0.226) | 0.182 / 0.029 / 0.049 | 0.182 / 0.029 / 0.049 | 0.031 | 0.000 | 0.985 (0.748) |
| `e6_edge_640` | 0.856 | 0.636 | 0.632 (0.226) | 0.147 / 0.057 / 0.082 | 0.144 / 0.057 / 0.082 | 0.029 | 0.100 | 0.983 (0.721) |
| `e3_lodo_taco` | 0.630 | 0.548 | 0.358 (0.248) | 0.218 / 0.009 / 0.018 | 0.218 / 0.015 / 0.027 | 0.006 | 0.300 | 0.578 (0.452) |
| `e3_lodo_zerowaste` | 0.747 | 0.293 | 0.480 (0.333) | – / 0.141 / – | – / 0.178 / – | – | 0.100 | – (–) |
| `e3_lodo_uavvaste` | 0.262 | 0.682 | – (–) | – / – / – | – / – / – | – | 0.400 | – (–) |
| `e3_lodo_mju` | 0.055 | 0.016 | – (–) | – / – / – | – / – / – | – | 1.000 | – (–) |
| `e3_lodo_bepli` | 0.282 | 0.338 | 0.457 (0.370) | 0.091 / 0.000 / 0.000 | 0.085 / 0.063 / 0.072 | 0.011 | 0.500 | – (–) |
| `e5_ablate_no_transplant` | 0.868 | 0.553 | 0.661 (0.226) | 0.168 / 0.000 / 0.000 | 0.157 / 0.036 / 0.058 | 0.033 | 0.200 | 0.984 (0.748) |
| `e5_ablate_no_adversary` | 0.870 | 0.556 | 0.699 (0.226) | 0.174 / 0.029 / 0.049 | 0.161 / 0.057 / 0.084 | 0.035 | 0.200 | 0.992 (0.748) |
| `e5_ablate_no_mixstyle` | 0.872 | 0.559 | 0.691 (0.226) | 0.180 / 0.029 / 0.049 | 0.166 / 0.029 / 0.049 | 0.032 | 0.200 | 0.982 (0.748) |
| `e5_ablate_probe` | 0.869 | 0.556 | 0.712 (0.226) | n/a | n/a | n/a | 3.000 | 0.978 (0.748) |
| `e7_heldout_values` | 0.871 | 0.559 | 0.664 (0.226) | 0.161 / 0.057 / 0.084 | 0.135 / 0.114 / 0.124 | 0.043 | 0.200 | 0.981 (0.748) |
| `e5_ablate_no_dense` | 0.870 | 0.559 | 0.649 (0.226) | 0.155 / 0.029 / 0.048 | 0.145 / 0.046 / 0.070 | 0.044 | 0.300 | 0.989 (0.748) |
| `e5_ablate_no_debias` | 0.874 | 0.567 | 0.694 (0.226) | 0.188 / 0.000 / 0.000 | 0.174 / 0.018 / 0.032 | 0.048 | 0.200 | 0.990 (0.748) |
| `e1_indomain_taco` | 0.862 | 0.722 | 0.697 (0.248) | 0.194 / – / – | 0.177 / – / – | – | 0.200 | 0.754 (0.452) |
| `e1_indomain_zerowaste` | 0.922 | 0.430 | 0.771 (0.333) | 0.459 / – / – | 0.421 / – / – | – | 0.200 | – (–) |
| `e1_indomain_uavvaste` | 0.616 | 0.830 | – (–) | – / – / – | – / – / – | – | 3.000 | – (–) |
| `e1_indomain_mju` | 0.914 | 0.698 | – (–) | – / – / – | – / – / – | – | 3.000 | – (–) |
| `e9_sam3_mju` | 0.231 | 0.214 | – (–) | – / – / – | – / – / – | – | 0.800 | – (–) |
| `e9_sam3_uavvaste` | 0.375 | 0.581 | – (–) | – / – / – | – / – / – | – | 0.600 | – (–) |
| `e9_sam3_bepli` | 0.398 | 0.364 | 0.427 (0.370) | 0.031 / 0.000 / 0.001 | 0.029 / 0.072 / 0.042 | 0.004 | 0.400 | – (–) |
| `e8_clip_distill` | 0.870 | 0.557 | 0.658 (0.226) | 0.146 / 0.114 / 0.128 | 0.129 / 0.143 / 0.135 | 0.041 | 0.200 | 0.983 (0.748) |
| `e3b_lodo_mju_negatives` | 0.363 | 0.192 | – (–) | – / – / – | – / – / – | – | 1.000 | – (–) |
| `e9_sam3_taco` | 0.783 | 0.586 | 0.370 (0.248) | 0.208 / 0.007 / 0.014 | 0.208 / 0.012 / 0.023 | 0.004 | 0.300 | 0.554 (0.452) |
| `e10_k10_uavvaste` | 0.289 | 0.740 | – (–) | – / – / – | – / – / – | – | 0.500 | – (–) |
| `e10_k10_bepli` | 0.454 | 0.446 | 0.468 (0.370) | 0.109 / 0.000 / 0.000 | 0.108 / 0.003 / 0.006 | 0.004 | 1.100 | – (–) |
| `e10_k10_taco` | 0.733 | 0.629 | 0.436 (0.248) | 0.080 / – / – | 0.054 / – / – | – | 0.800 | 0.623 (0.452) |
| `e10_k10_zerowaste` | 0.843 | 0.286 | 0.662 (0.333) | 0.317 / – / – | 0.311 / – / – | – | 0.700 | – (–) |
| `e10_k10_mju` | 0.751 | 0.502 | – (–) | – / – / – | – / – / – | – | 0.900 | – (–) |
| `e10_k50_uavvaste` | 0.294 | 0.768 | – (–) | – / – / – | – / – / – | – | 0.600 | – (–) |
| `e9_sam3_zerowaste` | 0.805 | 0.287 | 0.390 (0.333) | – / 0.004 / – | – / 0.021 / – | – | 0.200 | – (–) |
| `e5_ablate_lente_rgb` | 0.877 | 0.574 | 0.252 (0.226) | 0.007 / 0.074 / 0.013 | 0.007 / 0.074 / 0.013 | 0.001 | 0.000 | 0.974 (0.748) |

### Waste segmentation per test dataset (IoU / boundary F1)

| run | TACO IoU / BF1 | ZeroWaste-f IoU / BF1 | UAVVaste IoU / BF1 | MJU-Waste IoU / BF1 | BePLi v2 IoU / BF1 | pooled IoU |
|---|---|---|---|---|---|---|
| `e4_heldout_classes` | 0.885 / 0.740 | 0.912 / 0.379 | 0.490 / 0.790 | 0.895 / 0.661 | 0.616 / 0.568 | 0.870 |
| `e6_edge_640` | 0.866 / 0.742 | 0.903 / 0.489 | 0.451 / 0.818 | 0.888 / 0.786 | 0.579 / 0.594 | 0.856 |
| `e3_lodo_taco` | 0.630 / 0.548 |  |  |  |  | 0.630 |
| `e3_lodo_zerowaste` |  | 0.747 / 0.293 |  |  |  | 0.747 |
| `e3_lodo_uavvaste` |  |  | 0.262 / 0.682 |  |  | 0.262 |
| `e3_lodo_mju` |  |  |  | 0.055 / 0.016 |  | 0.055 |
| `e3_lodo_bepli` |  |  |  |  | 0.282 / 0.338 | 0.282 |
| `e4_heldout_classes__zeroshot` | 0.121 / 0.399 | 0.243 / 0.157 | 0.019 / 0.574 | 0.125 / 0.130 | 0.137 / 0.266 | 0.191 |
| `e5_ablate_no_transplant` | 0.855 / 0.726 | 0.914 / 0.359 | 0.472 / 0.793 | 0.912 / 0.669 | 0.608 / 0.570 | 0.868 |
| `e5_ablate_no_adversary` | 0.883 / 0.732 | 0.915 / 0.371 | 0.349 / 0.786 | 0.893 / 0.662 | 0.613 / 0.570 | 0.870 |
| `e5_ablate_no_mixstyle` | 0.885 / 0.737 | 0.914 / 0.375 | 0.493 / 0.792 | 0.894 / 0.660 | 0.617 / 0.575 | 0.872 |
| `e5_ablate_probe` | 0.881 / 0.735 | 0.913 / 0.372 | 0.494 / 0.791 | 0.894 / 0.662 | 0.615 / 0.567 | 0.869 |
| `e7_heldout_values` | 0.884 / 0.734 | 0.914 / 0.378 | 0.467 / 0.788 | 0.897 / 0.665 | 0.616 / 0.570 | 0.871 |
| `e5_ablate_no_dense` | 0.875 / 0.737 | 0.914 / 0.376 | 0.510 / 0.799 | 0.896 / 0.664 | 0.616 / 0.571 | 0.870 |
| `e5_ablate_no_debias` | 0.880 / 0.734 | 0.918 / 0.388 | 0.470 / 0.799 | 0.912 / 0.671 | 0.616 / 0.580 | 0.874 |
| `e1_indomain_taco` | 0.862 / 0.722 |  |  |  |  | 0.862 |
| `e1_indomain_zerowaste` |  | 0.922 / 0.430 |  |  |  | 0.922 |
| `e1_indomain_uavvaste` |  |  | 0.616 / 0.830 |  |  | 0.616 |
| `e1_indomain_mju` |  |  |  | 0.914 / 0.698 |  | 0.914 |
| `e4_heldout_classes__zeroshot_sam3obj` | 0.759 / 0.649 | 0.793 / 0.235 | 0.424 / 0.635 | 0.221 / 0.217 | 0.272 / 0.418 | 0.579 |
| `e9_sam3_mju` |  |  |  | 0.231 / 0.214 |  | 0.231 |
| `e9_sam3_uavvaste` |  |  | 0.375 / 0.581 |  |  | 0.375 |
| `e9_sam3_bepli` |  |  |  |  | 0.398 / 0.364 | 0.398 |
| `e8_clip_distill` | 0.886 / 0.738 | 0.914 / 0.373 | 0.497 / 0.789 | 0.892 / 0.660 | 0.616 / 0.570 | 0.870 |
| `e4_heldout_classes__zeroshot_sam3obj_area` | 0.759 / 0.649 | 0.793 / 0.235 | 0.424 / 0.635 | 0.221 / 0.217 | 0.272 / 0.418 | 0.579 |
| `e3_lodo_mju__segformer_mit-b2` |  |  |  | 0.104 / 0.082 |  | 0.104 |
| `e3b_lodo_mju_negatives` |  |  |  | 0.363 / 0.192 |  | 0.363 |
| `e9_sam3_taco` | 0.783 / 0.586 |  |  |  |  | 0.783 |
| `e10_k10_uavvaste` |  |  | 0.289 / 0.740 |  |  | 0.289 |
| `e10_k10_bepli` |  |  |  |  | 0.454 / 0.446 | 0.454 |
| `e10_k10_taco` | 0.733 / 0.629 |  |  |  |  | 0.733 |
| `e10_k10_zerowaste` |  | 0.843 / 0.286 |  |  |  | 0.843 |
| `e10_k10_mju` |  |  |  | 0.751 / 0.502 |  | 0.751 |
| `e10_k50_uavvaste` |  |  | 0.294 / 0.768 |  |  | 0.294 |
| `e9_sam3_zerowaste` |  | 0.805 / 0.287 |  |  |  | 0.805 |
| `e4_heldout_classes__zeroshot_sam3` | 0.473 / 0.573 | 0.791 / 0.229 | 0.417 / 0.627 | 0.119 / 0.077 | 0.271 / 0.416 | 0.592 |
| `e4_heldout_classes__zeroshot_gdino` | 0.510 / 0.532 | 0.273 / 0.135 | 0.183 / 0.482 | 0.154 / 0.170 | 0.180 / 0.132 | 0.266 |
| `e3_lodo_uavvaste__segformer_mit-b2` |  |  | 0.480 / 0.702 |  |  | 0.480 |
| `e3_lodo_bepli__segformer_mit-b2` |  |  |  |  | 0.276 / 0.314 | 0.276 |
| `e5_ablate_lente_rgb` | 0.876 / 0.737 | 0.920 / 0.399 | 0.491 / 0.802 | 0.909 / 0.678 | 0.631 / 0.583 | 0.877 |

### E3 leave-one-dataset-out (waste IoU on the dataset never seen in training or selection)

| held-out dataset | OWLv2→SAM2 zero-shot | Grounding DINO→SAM2 zero-shot | SAM 3 zero-shot (4 waste words) | SAM 3 zero-shot (16 prompts) | SAM 3 zero-shot (16 prompts, masks ≤ half image) | SegFormer-B2 (LODO) | TrashScribe LODO IoU | TrashScribe LODO BF1 | + SAM 3 teacher (E9) IoU | + 10 labelled images (E10) IoU | E4 all five IoU | E1 in-domain IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TACO | 0.121 | 0.510 | 0.473 | 0.759 | 0.713 | – | 0.630 | 0.548 | 0.783 | 0.733 | 0.885 | 0.862 |
| ZeroWaste-f | 0.243 | 0.273 | 0.791 | 0.793 | 0.783 | – | 0.747 | 0.293 | 0.805 | 0.843 | 0.912 | 0.922 |
| UAVVaste | 0.019 | 0.183 | 0.417 | 0.424 | 0.424 | 0.480 | 0.262 | 0.682 | 0.375 | 0.289 | 0.490 | 0.616 |
| MJU-Waste | 0.125 | 0.154 | 0.119 | 0.221 | 0.221 | 0.104 | 0.055 | 0.016 | 0.231 | 0.751 | 0.895 | 0.914 |
| BePLi v2 | 0.137 | 0.180 | 0.271 | 0.272 | 0.424 | 0.276 | 0.282 | 0.338 | 0.398 | 0.454 | 0.616 | – |

### E4 held-out classes: retrieval from the atlas entry alone

| held-out class | n objects | AP | 95% CI | prevalence | lift | perm. p |
|---|---|---|---|---|---|---|
| Crisp packet | 8 | 0.013 | [0.002, 0.083] | 0.000 | 45.325 | 0.003 |
| Fishing net | 319 | 0.094 | [0.078, 0.117] | 0.011 | 8.218 | 0.000 |
| Foam cup | 5 | 0.089 | [0.002, 0.445] | 0.000 | 494.681 | 0.002 |
| Glass bottle | 19 | 0.009 | [0.002, 0.040] | 0.001 | 13.442 | 0.009 |
| Paper bag | 5 | 0.001 | [0.000, 0.001] | 0.000 | 3.377 | 0.119 |
| Plastic straw | 26 | 0.011 | [0.004, 0.032] | 0.001 | 12.041 | 0.002 |
| Pop tab | 10 | 0.019 | [0.000, 0.167] | 0.000 | 53.160 | 0.003 |

Unseen-class top-1 by evidence stream alone (default point): descriptor agreement 0.046, text prototype 0.000, prose 0.021; chance 0.018. AUSUC oracle best H 0.126 (seen 0.099, unseen 0.171).

### E6 (640 px) held-out classes

| held-out class | n objects | AP | 95% CI | prevalence | lift | perm. p |
|---|---|---|---|---|---|---|
| Crisp packet | 8 | 0.005 | [0.001, 0.015] | 0.000 | 14.360 | 0.011 |
| Fishing net | 314 | 0.100 | [0.081, 0.126] | 0.013 | 7.921 | 0.000 |
| Foam cup | 5 | 0.168 | [0.003, 0.629] | 0.000 | 841.624 | 0.000 |
| Glass bottle | 19 | 0.015 | [0.002, 0.072] | 0.001 | 20.201 | 0.003 |
| Paper bag | 5 | 0.001 | [0.000, 0.002] | 0.000 | 3.420 | 0.125 |
| Plastic straw | 22 | 0.020 | [0.002, 0.122] | 0.001 | 23.281 | 0.001 |
| Pop tab | 9 | 0.015 | [0.000, 0.135] | 0.000 | 41.232 | 0.002 |

Unseen-class top-1 by evidence stream alone (default point): descriptor agreement 0.018, text prototype 0.029, prose 0.025; chance 0.018. AUSUC oracle best H 0.130 (seen 0.105, unseen 0.168).

### Description: descriptor mAP per field (E4 test objects)

| field | n objects | prevalence floor | raw CLIP crops | closed probe (E5) | TrashScribe (E4) |
|---|---|---|---|---|---|
| material | 26145 | 0.114 | 0.204 | 0.466 | 0.442 |
| rigidity | 21401 | 0.333 | 0.420 | 0.833 | 0.833 |
| optics | 19570 | 0.333 | 0.374 | 0.716 | 0.720 |
| shape | 21272 | 0.091 | 0.251 | 0.505 | 0.496 |
| surface | 20047 | 0.179 | 0.161 | 0.687 | 0.677 |
| color | 7311 | 0.170 | 0.298 | 0.816 | 0.738 |
| state | 9434 | 0.333 | 0.369 | 0.924 | 0.875 |
| size | 12451 | 0.250 | 0.359 | 0.748 | 0.744 |
| surroundings | 27733 | 0.125 | 0.215 | 0.814 | 0.823 |
| _macro_intrinsic |  | 0.226 | 0.305 | 0.712 | 0.691 |

### Training-free references

OWLv2→SAM2 pooled IoU 0.191; Grounding DINO→SAM2 (boxes ≤ half the image) 0.266; SAM 3 (score ≥ 0.5) with 4 waste words 0.592, with 16 prompts 0.579; raw CLIP description macro mAP 0.305; raw CLIP prose naming seen / unseen / H 0.182 / 0.291 / 0.224.

<!-- AUTO:END -->
