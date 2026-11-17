# Context: why TrashScribe exists, what it claims, how it got here

## Origin and goal

TrashScribe was started on 5 Nov 2026 as the owner's follow-up of two earlier works of theirs, both cited
anonymously in the paper:

* **LENTE** (`anon_lente`): landslide segmentation with a frozen SAM2.1-Hiera-Tiny, zero-initialised
  adapters and an early multiplicative gate driven by terrain. TrashScribe reuses the backbone recipe
  (`trashscribe/models/backbone.py`, `peft.py`). The LENTE repository is the parent folder today.
* **MIRF / "Describing, Not Naming"** (`anon_mirf`): descriptor heads scored in a text space with no
  per-descriptor parameter, read against a knowledge base to retrieve classes never seen in training, in
  diagnostic microscopy. TrashScribe moves this from object crops to dense segmentation.

The owner asked for four ideas in one system (README, "The four requested ideas"): the LENTE architecture
in the waste domain; a dataset- and class-independent model working on visual descriptors; independence
from the environment (no background shortcut); offline operation on a Jetson Orin Nano 8 GB.

**Venue.** *Neural Computing and Applications* (Springer), Topical Collection "Emerging Trends in Smart Waste
Monitoring and Environmental Surveillance". **Deadline 30 Nov 2026.** Double-blind: author data only in
`paper/title_page.tex`. Two guest editors share the authors' university: declare it (see `paper/README.md`).

**Hardware.** Training: one NVIDIA GeForce RTX 3080 Laptop GPU (8 GB) in an Acer Nitro AN515-45 with
15 GB RAM, Ubuntu, kernel 6.8. Deployment target: Jetson Orin Nano 8 GB (JetPack 6.x), not yet used.

## The owner's working preferences (observed and stated)

* Results are reported progressively in [RESULTS.md](../RESULTS.md) (hand-written journal on top, tables
  regenerated automatically at the bottom). Every code/protocol change goes in
  [attempts/ATTEMPTS.md](../attempts/ATTEMPTS.md).
* The GPU queue is ordered by **expected significance of the result** (stated 15 Nov): put first the run
  whose result can change the paper's claims most. Keep doing so when adding runs.
* The owner wants clear, significant claims and plain explanations. On 17 Nov the owner said the
  leave-one-out protocol alone did not convey weight and asked to be able to claim that **the edge model
  behaves equally or better than SAM 3**. The honest form of that claim is "with N labelled images (or
  SAM 3 labels / generic negatives) of a new site" (see below and [EXPERIMENTS.md](EXPERIMENTS.md)).
* "If you see a possibility to improve, go ahead": the owner accepts proactive runs within an approved
  direction, but a claim must never be stated before the evidence exists.
* Paper style (binding, 15 Nov): no em dashes, no ", and" constructs, few subsections, headings without
  parentheses and not whole phrases, roadmap paragraph at the end of the introduction, a lead-in paragraph
  before the subsections of every section.
* Do not commit or push unless asked; staging on request is fine.

## History of decisions (short)

| Date | Decision / event |
|---|---|
| 5 Nov | Code base built on a Mac; moved to the RTX 3080 workstation; first E4 launch killed by `systemd-oomd` together with VS Code → every heavy job runs as a capped `systemd-run --user` unit. Dataset decisions: ZeroWaste-f negatives = conveyor belt only (paper is waste but unlabelled); BePLi classes aliased to TACO entries where the object type is the same; fishing net is a text-only class. Grid restarted from scratch. |
| 6–11 Nov | E4, E6, E3-TACO finished; queue paused twice by the owner and resumed from `last.pt`. |
| 13 Nov | First reading of 10 runs; qualitative figure, data-flow figure, results text. Post-hoc analyses (`scripts/posthoc.py`): raw CLIP beats TrashScribe at naming/retrieving unseen classes; the pooled environment probe was confounded by class, a within-class probe shows intrinsic pooling keeps most of the place out; AUSUC metric bug fixed. Queue pruned (LoRA, adapters-both, depth gate, E2 removed) and E7 added. |
| 15 Nov | E7: the text-space descriptor head does **not** read unsupervised values (1 of 3) → no open-vocabulary description claim. SegFormer runs had all failed with a deterministic CUDA OOM at validation → fixed. Improvements implemented: E8 (CLIP-distilled dense head), E3b (COCO indoor negatives), test-time tiling, literature comparison figure (Grounding DINO, SAM 3). Paper restyled to the owner's rules. Queue ordered by significance. E9 (SAM 3 as offline teacher) added. |
| 16 Nov | Monitor "nan" explained (undefined metrics + a display bug). SAM 3 evaluation found unfair (whole-image masks counted) → re-run with the same half-image rule as the other baselines. Longer E9 training rejected (label-quality ceiling). |
| 17 Nov | Two clean reboots (queue restarted from `last.pt`). Fair SAM 3 beats every leave-one-out model; SegFormer also fails indoors (0.104); E8 restores CLIP-level open reading in one pass; **E3b lifts MJU from 0.055 to 0.363, above SAM 3**; E10 (10 labelled site images) added to test the owner's claim; first E9/E10 results above SAM 3 on TACO and BePLi. Handover docs written. |

## What the paper can claim today (evidence in [EXPERIMENTS.md](EXPERIMENTS.md))

Supported:
1. **A cross-environment benchmark for waste segmentation** (five public datasets, one canonical waste
   mask, one attribute atlas, leave-one-dataset-out). No prior benchmark of this kind was found (searches
   13 and 17 Nov); the closest works train/test per dataset ("Don't waste SAM", arXiv 2606.10696) or stay in
   one plant (ZeroWaste VisDA challenge).
2. **"Not waste" is environment-specific, and the fix is cheap.** Without indoor scenes in training, both
   TrashScribe (0.055) and SegFormer-B2 (0.104) mark people and furniture as waste in an office; 243 generic
   waste-free COCO indoor photos (no target image, no waste label) raise TrashScribe to 0.363, above SAM 3.
3. **Edge model vs SAM 3, quantified.** With the site in training the 34 M model beats SAM 3 (848 M) on all
   five datasets (by 6.6 to 67.4 IoU points). At a new site SAM 3 wins on 4 of 5 with no site data; with SAM 3
   labels of unlabelled site images (E9) or 10 labelled images (E10) the edge model reaches or passes it on
   TACO, MJU and BePLi so far; UAVVaste still below (50-image run queued).
4. **One pass: segmentation + attribute description + open reading.** Description macro mAP 0.691 (3× the
   prevalence floor, 2.3× raw CLIP 0.305); the CLIP-distilled head (E8) retrieves unseen classes like raw
   CLIP (mean AP 0.082 vs 0.078) and reads untrained properties at CLIP's level, inside the same network.

Not supported (do not claim): a text-space head better than a closed classifier (0.691 vs 0.712); naming
unseen classes better than raw CLIP (H 0.12–0.16 vs 0.217); environment removal by adversary / transplant /
MixStyle (no measurable effect, single seed); beating SAM 3 at a new site with no site data (except MJU with
negatives).

## Literature anchors (details in [SOTA.md](SOTA.md), written 5 Nov, and in the paper)

* Waste segmentation per dataset: TACO, ZeroWaste (+ VisDA challenge), MJU-Waste, UAVVaste, BePLi,
  "Don't waste SAM" (fine-tuned SAM per dataset), COSNet, cluttered-background networks.
* Open vocabulary: CLIP, OWLv2, Grounding DINO, YOLO-World, **SAM 3** (concept segmentation; the reference
  to beat), EfficientSAM3 (distilled SAM 3 students, 89–95 M parameters, weights released June 2026, no
  Jetson numbers published), Abid et al. BMVC 2025 (open-vocabulary waste detection needs engineered prompts).
* Attributes of waste: GSA2Seg (arXiv 2404.18112; placement attributes for robot dogs, one dataset, cloud).
* SAM 3 as a teacher for small students: pig monitoring (alphaXiv 2605.25860, student within 1.3 mAP of
  SAM 3 at 6 ms); semi-supervised foundation-model distillation where an 11× smaller student beats its
  adapted teacher (arXiv 2604.03841).
* Zero-shot waste image classification with CLIP/VLMs exists (image level only); the paper still has a red
  `\tbd` to add those citations.
