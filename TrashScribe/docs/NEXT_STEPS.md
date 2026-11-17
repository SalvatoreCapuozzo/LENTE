# Next steps (prioritised) and open decisions

Written 17 Nov 2026, 22:30. Check [RESULTS.md](../RESULTS.md) and `python scripts/progress.py` first: some items
below may already be done. Deadline: **30 Nov 2026** (NCAA topical collection).

## 0. Runs still in the queue (automatic; expected end about Fri 20 Nov evening)

In order (`configs/gpu_queue.yaml`, ordered by expected significance): E10 10-image runs on ZeroWaste-f and MJU
(TACO was running at writing time), E10 50 images on UAVVaste, SAM 3 labels + E9 on ZeroWaste-f, SAM 3 4-word
and Grounding DINO zero-shot columns, SegFormer-B2 LODO on UAVVaste and BePLi (about 11 h each), `e5_ablate_lente_rgb`,
E1-BePLi, SegFormer on TACO and ZeroWaste-f.

When a run finishes: the queue regenerates `paper/tables/*.tex` and RESULTS.md tables automatically; an agent
must (a) add a dated journal entry to RESULTS.md with the reading, (b) update the paragraph text and the red
`\tbd` notes in `paper/main.tex` (Results, "Cross-environment segmentation"), (c) recompile the paper and check
the owner's style rules (no em dashes, no ", and").

## 1. Decisions for the owner (ask, do not assume)

1. **Reframe the paper** around what the evidence supports (see [CONTEXT.md](CONTEXT.md#what-the-paper-can-claim-today)).
   Proposed title: "TrashScribe: Segmenting and Describing Waste across Environments on the Edge". Proposed
   contributions: (i) the cross-environment benchmark; (ii) environment-specific negatives and the 243-photo fix;
   (iii) the edge model against SAM 3 by amount of site data (Table `tab:ladder`); (iv) one-pass segmentation,
   description and open reading (E8). Requires rewriting title, abstract, introduction, contributions,
   discussion and conclusions; the Results section already follows the new findings. The owner asked on 17 Nov
   whether to do it now or after the remaining runs: still open.
2. **The claim "edge model ≥ SAM 3"** must carry its condition (site data used). If UAVVaste stays below SAM 3
   with 50 labelled images, options: 100+ images; combine SAM 3 labels + labelled images + scale augmentation;
   or state UAVVaste as the exception. Decide before writing the abstract.
3. **Jetson measurements** (owner on the device): see [JETSON.md](JETSON.md). Include SAM 3 on the device if possible.

## 2. Engineering tasks (agent can do)

1. Export the E8 distilled CLIP head: add a 4th output to `TrashScribeEdge` (`trashscribe/models/trashscribe.py`),
   score it against CLIP text embeddings precomputed in the atlas cache, fuse with the descriptor scores in
   `edge_runtime.py`; optionally train E8 at 640 px for deployment.
2. If needed for the SAM 3 claim: a combined site-adaptation run (SAM 3 labels + 10 labelled images (+ negatives
   indoors)) from the LODO checkpoint, with the same fixed schedule as E10.
3. Scale augmentation for aerial images in multi-domain training (random zoom-in crops of UAVVaste), to recover
   the 12.6-point aerial cost of joint training (E1 0.616 vs E4 0.490).
4. Regenerate the literature comparison figure with a SegFormer column once its LODO runs exist
   (`python scripts/make_comparison.py run segformer` then `draw`, CPU, capped).

## 3. Paper tasks

* Abstract, introduction, contributions, conclusions: rewrite after decision 1 (currently old framing plus a
  red note summarising the new findings).
* Missing citations: zero-shot waste classification with CLIP/VLMs (red `\tbd` in Related work and in the
  comparison section); `references.bib` entries flagged `VERIFY` (SAM 3 arXiv id and authors; "Don't waste SAM"
  venue/year: an arXiv listing 2606.10696 is dated 2026, the bib says ESANN 2023).
* Edge section and `tab:edge` from the Jetson measurements; JetPack/TensorRT versions in implementation details.
* "A monitoring case" paragraph: re-read the same frames with a second municipality's atlas (no retraining).
* Declarations: LLM-use statement, handling of faces and licence plates in street images, anonymised code URL.
* `docs/SOTA.md` dates from 5 Nov: update it with SAM 3, EfficientSAM3, GSA2Seg and the SAM 3-teacher papers.
* Submission checklist and anonymisation: `paper/README.md`.

## 4. Housekeeping

* README "Status" section points here; keep this file and `AGENTS.md` current when the state changes.
* Moving the folder out of LENTE: [MIGRATION.md](MIGRATION.md).
