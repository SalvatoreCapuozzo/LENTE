# Next steps (prioritised) and open decisions

Written 17 Nov 2026, 22:30. Check [RESULTS.md](../RESULTS.md) and `python scripts/progress.py` first: some items
below may already be done. Deadline: **30 Nov 2026** (NCAA topical collection).

## 0. Runs still in the queue (19 Nov; expected end about Fri 20 Nov evening)

E1-BePLi (running), SegFormer-B2 LODO on TACO and ZeroWaste-f. All three fill empty cells of Table `tab:lodo`
(regenerated automatically). When they finish: update the "Segmentation across environments" paragraph and remove
its red `\tbd`; if SegFormer beats TrashScribe-LODO on TACO or ZeroWaste-f too, say so plainly in the paragraph and
in the discussion. Re-run `python scripts/make_ladder_figure.py` only if a ladder run changes (none queued).

## 1. Decisions for the owner (ask, do not assume)

1. The paper was reframed on 19 Nov (title "TrashScribe: Segmenting and Describing Waste across Environments on an
   Edge Device"). Ask the owner to read the abstract, introduction and conclusions and confirm the framing.
2. Jetson: re-run the suite with `sudo jetson_clocks` and `sudo tegrastats` (power, peak memory) and ≥ 3 GB free;
   update Table `tab:edge` (hand-written in `paper/main.tex`) and the "On-device performance" paragraph.
   SAM 3 on the device needs JetPack 6 (or a container with Python ≥ 3.10).
3. Aerial litter is the one environment where the edge model does not reach SAM 3 with few labels. Options:
   scale augmentation / tiling during training, unfreezing more of the encoder, or stating it as the limitation it
   is now in the paper.

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

## 3. Paper tasks (after the 19 Nov revision)

* Done on 19 Nov: abstract, introduction, related work, method, results, discussion, conclusions rewritten; all
  VERIFY bibliography notes resolved; Jetson numbers in `tab:edge`; ladder figure; comparison figure aligned.
* Still open: Jetson power / peak memory; LLM-use statement; handling of faces and licence plates in street
  images (Declarations); anonymised code URL; the three remaining runs; a final read for the style rules
  (no em dashes, no ", and": `grep -c ', and' paper/main.tex` must only hit the comment on line 6).
* `docs/SOTA.md` dates from 5 Nov: update it with SAM 3, EfficientSAM3, GSA2Seg and the SAM 3-teacher papers.
* Submission checklist and anonymisation: `paper/README.md`.

## 4. Housekeeping

* README "Status" section points here; keep this file and `AGENTS.md` current when the state changes.
* Moving the folder out of LENTE: [MIGRATION.md](MIGRATION.md).
