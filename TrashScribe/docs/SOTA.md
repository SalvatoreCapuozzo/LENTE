# State of the art — notes for TrashScribe (compiled 5 Nov 2026)

Every claim below carries its source. Items marked **verify** were seen only in secondary
listings and must be checked against the primary paper before they go into the manuscript.

## 1. Target venue

* *Neural Computing and Applications* (Springer), Topical Collection **"Emerging Trends in Smart
  Waste Monitoring and Environmental Surveillance"**. Guest editors: A. Greco (Univ. Salerno, lead),
  C. Sansone and B. Vento (Univ. Naples Federico II). Deadline **30 Nov 2026**. Topics include image
  analysis for waste detection/classification, multimodal systems, datasets and benchmarks,
  real-world deployments with resource constraints. Submission: select "TC: Emerging Trends in Smart
  Waste Monitoring and Environmental Surveillance" in *Additional Information*.
  <https://link.springer.com/journal/521/updates/27851508>
* Author guidelines (read 5 Nov 2026): double-blind; abstract 150–250 words; 4–6 keywords; Springer
  Nature LaTeX template with `sn-basic.bst` (numbered citations in square brackets); max three heading
  levels; mandatory "Statements and Declarations" (competing interests at least) and data
  availability statement; LLM use documented in Methods; editable sources required.
  <https://link.springer.com/journal/521/submission-guidelines>
* The lead guest editor's group (MIVIA, Salerno) runs the **WasteVision** workshop at WACV and the
  **Illegal Waste Dumping Detection (IWDD)** contest with the MIVIA-IWDD dataset; WasteVision 2026
  papers are video/illegal-dumping oriented, not segmentation-oriented.
  <https://openaccess.thecvf.com/WACV2026_workshops/WasteVision>, <https://mivia.unisa.it/wastevision2027/>

## 2. Waste segmentation and detection

| Work | What it does | Gap TrashScribe addresses |
|---|---|---|
| TACO (Proença & Simões, 2020, arXiv:2003.06975) | 1,500 images, 4,784 instances, 60 classes / 28 supercategories, scene labels | rough annotation, tiny; used in-domain only |
| ZeroWaste (Bashkirova et al., CVPR 2022, arXiv:2106.02740) | recycling-plant conveyor, 4 classes, semi-supervised splits; DeepLabV3+ best of tested baselines | industrial domain only |
| VisDA 2022 (Bashkirova et al.) | ZeroWaste as domain adaptation across two plants/seasons; augmentation and ensembling help DG | same industrial domain on both sides <https://par.nsf.gov/biblio/10464390> |
| MJU-Waste (Wang et al., Sensors 2020) | RGB-D, multi-level segmentation, also evaluated on TACO <https://pmc.ncbi.nlm.nih.gov/articles/PMC7411711> | depth only in controlled indoor scenes |
| "Don't waste SAM" (Abou Baker & Handmann, ESANN 2023; arXiv:2606.10696) | fine-tuned SAM-ViT-H on ZeroWaste, TACO, TrashCan; large IoU gains per dataset **verify numbers** <https://arxiv.org/abs/2606.10696> | heavy (ViT-H), per-dataset, no cross-dataset test |
| COSNet (WACV 2025) | boundary-enhanced segmentation in cluttered scenes; +1.8 mIoU ZeroWaste-f, +2.1 SpectralWaste **verify** <https://irep.mbzuai.ac.ae/items/6df80e50-fae7-4fb3-bb5b-4b5a8d05cfcb> | closed-set, in-domain |
| Javaid et al., ICML 2026 (arXiv:2606.13587) | spatial+spectral lightweight segmentation, AFEM boundary/blob module; ZeroWaste-aug/-f, SpectralWaste <https://arxiv.org/abs/2606.13587> | closed-set, in-domain |
| StreetView-Waste (Paulo et al., WACV 2026, arXiv:2511.16440) | 36,478 fisheye street images; container detection/tracking + overflow segmentation; geometry-aware 7-channel input (Metric3Dv2 depth + normals) helps lightweight models <https://arxiv.org/html/2511.16440v1> | supports TrashScribe's depth-gated guidance idea; licence-restricted data |
| Robust and Label-efficient Deep Waste Detection (Abid et al., BMVC 2025) | OVOD benchmark on ZeroWaste: class-only prompts poor, LLM-optimised prompts better; fine-tuned detectors 51.6 mAP; ensemble pseudo-labels <https://bmvc2025.bmva.org/proceedings/507/> | prompts are class names, not grounded attributes |
| UAVVaste (Kraft et al., Remote Sensing 2021) | 772 aerial images, 3,718 litter annotations, COCO <https://github.com/PUTvision/UAVVaste> | single class |
| BePLi v1/v2 (beach plastic litter) | v1: 3,709 images, single class; v2: 3,722 images, 13 classes, COCO RLE **class names: verify in the JSON** <https://pmc.ncbi.nlm.nih.gov/articles/PMC10173386/> | beach only |

## 3. Open-vocabulary, attribute and concept recognition

* CLIP, OWL-ViT/OWLv2, Grounding DINO, YOLO-World: open over *object names*; names of waste items
  transfer poorly (see BMVC 2025 above).
* SAM 3 "promptable concept segmentation" and **EfficientSAM3** (arXiv:2511.15833) distil it into
  RepViT/TinyViT/EfficientViT students with MobileCLIP text encoders; Jetson FPS figures circulate
  only in secondary summaries **verify** <https://arxiv.org/abs/2511.15833>. Concepts are noun
  phrases; no structured attribute layer, no naming-by-description.
* Attribute-based zero-shot learning: Direct Attribute Prediction (Lampert et al., TPAMI 2014);
  GZSL protocol and harmonic mean (Xian et al., TPAMI 2019); calibrated stacking and AUSUC (Chao et
  al., ECCV 2016). TrashScribe uses all three.
* *Describing, Not Naming* (the MIRF manuscript in this repository): class-agnostic proposer +
  descriptor head with no per-descriptor parameters over a frozen VLM, atlas read at inference,
  abstention; held-out species retrieved from text alone. TrashScribe transfers this to dense waste
  segmentation, replaces crops with mask pooling of a dense descriptor field, and adds the
  environment-debiasing machinery. **Double-blind: cite it in the third person.**

## 4. Domain generalisation with foundation models

* Rein (Wei et al., CVPR 2024) — frozen VFMs are already strong for DG semantic segmentation;
  ~1% trainable parameters via learnable tokens. Rein++ (arXiv:2508.01667) extends to SAM2-based
  variants. <https://arxiv.org/abs/2312.04265>
* MixStyle (Zhou et al., ICLR 2021); DANN / gradient reversal (Ganin & Lempitsky, ICML 2015);
  copy-paste augmentation (Ghiasi et al., CVPR 2021).
* Gap: no DG benchmark for waste segmentation across litter / plant / aerial / beach / indoor
  domains; TrashScribe's leave-one-dataset-out protocol is that benchmark.

## 5. Edge

* NanoSAM (NVIDIA-AI-IOT): distilled SAM, TensorRT on Orin Nano; reports encoder 27 ms and full
  pipeline 4.2 ms; MobileSAM encoder needed FP32 because FP16 gave wrong outputs in TensorRT —
  hence `bench_edge.py --check`. <https://github.com/NVIDIA-AI-IOT/nanosam>
* NanoOWL: OWL-ViT in real time on Orin with TensorRT <https://github.com/NVIDIA-AI-IOT/nanoowl>.
* EdgeSAM reports >40 FPS at 1024² on Orin Nano **verify**.
* Orin Nano: 1024-core Ampere GPU, 32 Tensor Cores, FP16/INT8 via TensorRT (as described in
  arXiv:2507.12344).
* Benchmark of open-vocabulary instance segmentation pipelines on Jetson AGX Orin (Frontiers in
  Robotics and AI, 2025, doi:10.3389/frobt.2025.1693988).

## 6. Where TrashScribe's claims must be careful

* "First" claims: none of the searches found attribute-level dense grounding for waste or a
  multi-dataset leave-one-dataset-out waste benchmark, but searches are not exhaustive — phrase as
  "to our knowledge".
* All comparisons must be run on the **same splits** (scripts provided); published numbers from
  different splits go in a separate, clearly labelled table.
