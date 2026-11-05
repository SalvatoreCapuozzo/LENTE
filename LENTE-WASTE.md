
# LENTE-Waste: A Dataset-Agnostic and Parameter-Efficient Framework for Cross-Domain Waste Segmentation in Smart Environmental Monitoring

## 1. Research objective

The objective of the proposed research is to develop a robust and computationally efficient computer-vision system for automatic waste segmentation in heterogeneous real-world environments, with particular emphasis on smart environmental and urban waste monitoring.

The system will extend the principles of LENTE toward the waste-monitoring domain by exploiting a foundation segmentation model, parameter-efficient adaptation and a unified training strategy capable of learning from multiple publicly available waste datasets.

The central hypothesis is that waste segmentation systems trained and evaluated on a single dataset tend to become strongly specialized to the acquisition conditions, annotation taxonomy and environmental context of that dataset. A system intended for real-world smart waste monitoring, however, must operate across substantially different conditions, including outdoor litter, urban environments, beaches, recycling facilities and cluttered waste scenes.

Therefore, rather than optimizing a model for a single benchmark, the proposed approach will investigate whether a parameter-efficient segmentation architecture can learn a **domain-general representation of waste** by jointly exploiting heterogeneous datasets while maintaining compatibility with dataset-specific annotation schemes.

The resulting system will be designed with two complementary objectives:

1. **Generalization:** robust segmentation of waste across previously unseen environments and datasets.
2. **Efficiency:** deployment of the resulting model on computationally constrained edge platforms for practical smart-monitoring applications.

---

# 2. Multi-dataset learning strategy

The proposed system will exploit multiple waste datasets rather than relying on a single source.

Candidate datasets include:

- TACO (Trash Annotations in Context);
- ZeroWaste;
- BePLi v2;
- MJU-Waste;
- StreetView-Waste;
- potentially UAVVaste and other specialized datasets.

These datasets are complementary rather than identical.

TACO provides real-world litter images acquired in diverse environments such as roads, woods and beaches, with pixel-level instance annotations organized according to a hierarchical taxonomy. The dataset contains 60 fine-grained categories.

ZeroWaste focuses on waste segmentation in highly cluttered recycling environments and is particularly valuable because it also provides fully labelled, weakly labelled and unlabelled data, enabling future investigation of semi-supervised learning.

BePLi v2 provides a complementary environmental domain focused on plastic litter on beaches, with thousands of images and a large number of instance annotations encoded in COCO format.

MJU-Waste introduces an RGB-D setting and provides pixel-level waste-object masks. The original dataset contains co-registered RGB and depth images acquired using a Kinect sensor.

Consequently, these datasets should not be treated as interchangeable datasets. Instead, they should be regarded as observations of the same high-level problem under different environmental, sensing and annotation conditions.

---

# 3. Dataset harmonization

A fundamental component of the proposed system will therefore be a dataset harmonization layer.

The model should not directly consume the native annotations of every dataset. Instead, all datasets will first be converted into a common internal representation.

The processing pipeline will be:

    Original Dataset
           │
           ▼
    Dataset Adapter
           │
           ├── Image normalization
           ├── Annotation conversion
           ├── Label mapping
           ├── Mask conversion
           └── Metadata extraction
           │
           ▼
    Canonical Waste Representation
           │
           ▼
    LENTE-Waste model

Each dataset will have a dedicated adapter responsible for converting its native format into the common representation.

For example, TACO and BePLi provide COCO-style annotations, whereas MJU-Waste also provides PASCAL VOC and COCO-compatible instance annotations.

The canonical representation should contain at least:

    image
    semantic_mask
    instance_masks
    class_labels
    dataset_id
    environment_id
    modality_id

This representation allows the model to distinguish between the underlying visual content and the annotation conventions used by individual datasets.

---

# 4. Common waste ontology

The most important compatibility problem is not the image format but the label taxonomy.

For example, TACO distinguishes many specific categories such as plastic bottles, plastic film, cans and wrappers, whereas MJU-Waste uses a single waste-object class.

Therefore, the proposed system will use a hierarchical ontology.

At the highest level:

    background
    waste

This binary representation is the common denominator shared by all datasets.

A second level can optionally describe broad material categories:

    plastic
    paper/cardboard
    glass
    metal
    textile/rope
    organic
    other

A third level can retain dataset-specific object categories whenever sufficient annotations are available:

    plastic bottle
    plastic film
    can
    cigarette
    fishing net
    etc.

This produces a hierarchical representation:

                 WASTE
                   │
       ┌───────────┼───────────┐
       │           │           │
    plastic      metal       paper
       │
    ┌──┴────┐
 bottle   film
    │
    └── ...
    
The first experimental objective will therefore be **dataset-independent binary waste segmentation**.

Material and object-level segmentation will constitute additional experiments.

This separation is important because it allows datasets with incompatible fine-grained taxonomies to participate in the same training procedure without artificially forcing incompatible classes to correspond to one another.

---

# 5. Input standardization

The model will operate on a standardized RGB representation.

Input images will undergo a common preprocessing pipeline consisting of:

1. image decoding;
2. RGB conversion;
3. aspect-ratio-preserving resizing;
4. padding/cropping;
5. normalization;
6. geometric augmentation;
7. photometric augmentation.

The model will therefore not assume a fixed native resolution.

For example:

    TACO       3223 × 2825
    MJU-Waste   640 × 480
    BePLi       variable
    ZeroWaste   variable
             ↓
       preprocessing
             ↓
       model resolution

This allows the same architecture to process images originating from different cameras and datasets.

The original image resolution and scale will nevertheless be retained as metadata, since scale variation itself is an important source of domain shift in waste segmentation.

---

# 6. RGB as the common modality

The primary version of the system should be based exclusively on RGB.

This is a deliberate design decision.

Although MJU-Waste provides RGB-D data, the majority of candidate datasets do not provide depth. Requiring RGB-D input would therefore make the proposed system incompatible with a large portion of the benchmark.

Instead, RGB will constitute the universal input:

    RGB ───────────────► LENTE-Waste ───────────────► mask

MJU-Waste depth information can subsequently be exploited in a separate experiment through an optional multimodal branch.

This creates two versions of the system:

### LENTE-Waste-RGB

    RGB → segmentation

This is the principal cross-dataset model.

### LENTE-Waste-RGBD

    RGB ──┐
          ├──► multimodal fusion ──► segmentation
    Depth ┘

This version evaluates whether depth information provides additional benefits in controlled RGB-D environments.

Importantly, RGBD should therefore be considered an **extension of the architecture**, rather than a requirement of the core model.

---

# 7. Foundation-model architecture

The proposed architecture will build upon the LENTE paradigm and use a lightweight version of a modern foundation segmentation model, such as SAM2-Tiny or a comparable efficient segmentation foundation model.

The conceptual architecture is:

    RGB image
        │
        ▼
    Image encoder
        │
        ▼
    Lightweight parameter-efficient adapters
        │
        ▼
    Segmentation representation
        │
        ▼
    Waste-aware mask decoder
        │
        ▼
    Waste segmentation

The foundation-model backbone will preferably remain largely frozen.

Only a small subset of parameters will be optimized through parameter-efficient adaptation modules.

This provides two benefits.

First, it substantially reduces computational and memory requirements during training.

Second, it reduces the risk that training on one waste dataset completely overwrites representations learned from other domains.

---

# 8. Dataset-aware training without dataset-specific models

A critical design principle is that different datasets should not result in different segmentation networks.

Instead, a single model will be trained using samples originating from multiple datasets:

    Dataset A ─┐
    Dataset B ─┤
    Dataset C ─┼──► shared model ──► segmentation
    Dataset D ─┤
    Dataset E ─┘

Each training sample will carry a dataset identifier:

    dataset_id ∈ {TACO, ZeroWaste, BePLi, MJU, ...}

The dataset identifier can initially be used only for analysis and sampling.

A later experiment can investigate whether lightweight dataset/domain embeddings improve adaptation:

    RGB
     │
     ▼
    shared encoder
     │
     ├──── domain embedding
     │
     ▼
    segmentation decoder

The fundamental objective remains that the majority of the learned representation is shared across datasets.

---

# 9. Training objective

The principal training objective will be formulated as a combination of segmentation losses.

For binary waste segmentation:

    L = λ₁ L_BCE + λ₂ L_Dice + λ₃ L_boundary

where:

- L_BCE measures pixel-wise classification error;
- L_Dice handles class imbalance between waste and background;
- L_boundary emphasizes accurate object boundaries.

This is particularly important because waste objects frequently have irregular and amorphous boundaries.

Where instance annotations are available, an additional instance-level objective can be introduced.

The resulting objective could become:

    L_total =
        λ_sem L_semantic
      + λ_inst L_instance
      + λ_boundary L_boundary

Datasets that provide only semantic masks can still participate in the semantic component, while datasets providing instance masks can additionally contribute to the instance objective.

This allows heterogeneous supervision without discarding useful data.

---

# 10. Handling heterogeneous annotations

The model will explicitly support three levels of supervision.

### Full instance supervision

    image
      ↓
    object masks
      ↓
    instance segmentation loss

### Semantic supervision

    image
      ↓
    semantic mask
      ↓
    semantic segmentation loss

### Weak or image-level supervision

    image
      ↓
    image-level label
      ↓
    weak/semi-supervised objective

This is particularly interesting for ZeroWaste because its benchmark includes fully labelled, weakly labelled and unlabelled data.

Consequently, the final framework could evolve from conventional multi-dataset supervised learning toward:

> **multi-dataset semi-supervised waste segmentation**

without changing the fundamental architecture.

---

# 11. Dataset balancing

Simply concatenating all datasets would introduce severe sampling bias.

For example, a dataset with tens of thousands of annotations could dominate a smaller dataset such as MJU-Waste or TACO.

Therefore, training batches should be constructed using dataset-aware sampling.

For example:

    Batch
      ├── TACO samples
      ├── ZeroWaste samples
      ├── BePLi samples
      └── MJU samples

or by sampling datasets according to a controlled probability distribution:

    P(dataset = d)

rather than:

    P(dataset = d) ∝ number of images

This allows small but important domains to contribute meaningfully to the shared representation.

---

# 12. Domain-generalization protocol

The most important scientific experiment should not simply be a random split of the combined datasets.

Instead, the study should evaluate three increasingly difficult scenarios.

## Experiment A — In-domain segmentation

Train and test on the same dataset using independent image splits.

    TACO → TACO
    ZeroWaste → ZeroWaste
    BePLi → BePLi

This establishes conventional benchmark performance.

## Experiment B — Multi-domain training

Train one model jointly on several datasets:

    TACO
    +
    ZeroWaste
    +
    BePLi
    +
    MJU-Waste

and evaluate separately on each domain.

This tests whether joint training improves general robustness.

## Experiment C — Cross-domain generalization

Train on a subset of environments and test on a previously unseen environment.

For example:

    TRAIN
    TACO + ZeroWaste + BePLi

             ↓

    TEST
    StreetView-Waste

or:

    TRAIN
    TACO + ZeroWaste

             ↓

    TEST
    BePLi

This is arguably the most important experiment because it directly measures whether the system has learned a general representation of waste rather than memorizing a particular dataset.

---

# 13. Domain shift analysis

The proposed work should explicitly quantify domain shift.

The datasets differ in:

- camera type;
- image resolution;
- viewpoint;
- lighting;
- background;
- object scale;
- object density;
- object taxonomy;
- environmental context;
- annotation policy.

Therefore, dataset identity should be treated as an experimental variable.

Performance should be reported not only globally but also according to environment.

For example:

    outdoor litter
         │
         ├── roads
         ├── vegetation
         └── beaches

    industrial
         │
         └── recycling facilities

    urban
         │
         └── waste containers

    controlled
         │
         └── RGB-D scenes

The objective is to demonstrate that the proposed model maintains useful segmentation performance across these domains.

---

# 14. Robustness experiments

The system should additionally be evaluated under controlled perturbations.

Potential perturbations include:

- illumination changes;
- motion blur;
- image noise;
- compression;
- scale reduction;
- partial occlusion;
- shadows;
- color shifts;
- background clutter.

This is particularly relevant for a practical monitoring system because the input image cannot be assumed to be captured under laboratory conditions.

The robustness experiment can compare:

    conventional segmentation model
                vs.
    LENTE-Waste

under increasing degradation.

---

# 15. Edge deployment

A second major objective will be deployment on resource-constrained hardware.

The final model should therefore be evaluated not only in terms of segmentation accuracy but also computational efficiency.

The benchmark should report:

    mIoU
    Dice
    precision
    recall
    boundary F1
    FPS
    latency
    GPU/accelerator memory
    model size
    trainable parameters
    power consumption, when measurable

A representative deployment pipeline could be:

    Camera
       │
       ▼
    Edge computer
       │
       ▼
    LENTE-Waste
       │
       ├── waste masks
       ├── waste count
       ├── waste coverage
       └── confidence
       │
       ▼
    Monitoring platform

This transforms the research from a purely offline segmentation experiment into a potential smart-monitoring system.

---

# 16. Smart-monitoring application layer

The segmentation masks can be converted into higher-level monitoring indicators.

For example:

### Waste coverage

    waste_pixels / image_pixels

### Waste object count

    number of detected instances

### Container overflow

    waste pixels outside the expected container region

### Waste density

    waste area / monitored area

### Temporal accumulation

    waste(t₂) - waste(t₁)

Therefore:

    Image
      ↓
    Segmentation
      ↓
    Quantification
      ↓
    Environmental indicator
      ↓
    Smart monitoring system

This provides a direct bridge between computer vision and environmental monitoring.

---

# 17. Final system architecture

The complete proposed system can therefore be represented as:

                  ┌─────────────────┐
                  │   TACO          │
                  ├─────────────────┤
                  │   ZeroWaste     │
                  ├─────────────────┤
                  │   BePLi v2      │
                  ├─────────────────┤
                  │   MJU-Waste     │
                  ├─────────────────┤
                  │ StreetView-Waste│
                  └────────┬────────┘
                           │
                           ▼
                 Dataset harmonization
                           │
                 ┌─────────┴─────────┐
                 │                   │
             RGB images          annotations
                 │                   │
                 └─────────┬─────────┘
                           │
                           ▼
                 Canonical representation
                           │
                           ▼
                    LENTE-Waste
                           │
                 ┌─────────┴─────────┐
                 │                   │
             PEFT adapters      segmentation
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Waste masks
                           │
                ┌──────────┼──────────┐
                ▼          ▼          ▼
             objects     material   coverage
                │          │          │
                └──────────┼──────────┘
                           ▼
                 Smart waste monitoring
                           │
                           ▼
                    Edge deployment

MJU-Waste can additionally activate an optional RGB-D branch:

              RGB ──────────┐
                             ├──► Multimodal fusion
            Depth ──────────┘
                             │
                             ▼
                       segmentation

The RGB-only architecture remains the universal model, ensuring that the principal benchmark remains compatible with all selected datasets.