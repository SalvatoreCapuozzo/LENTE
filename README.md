# LENTE: Landslide Early-fusion Neural Transformer Engine 🏔️
*(Core Diagnostic Module of the ATLANTE Framework)*

## 📖 Abstract
Accurate and rapid segmentation of landslides is critical for disaster response and civil protection. **LENTE** (internally referred to as SAM2TinyLANDSLID) bridges the gap between general-purpose foundation models and the specific needs of Earth Observation by introducing a multimodal, parameter-efficient architecture tailored for topography. 

By adapting the lightweight **SAM2-Tiny (ViT-based)** backbone, the model expands the optical input from 3 to 12 channels. Crucially, it utilizes a **Dual-Stage DEM-Guided Fusion (DSDF)** mechanism, where terrain features (Slope and DEM) are used to physically guide the optical encoder via a spatial gain map. Utilizing zero-delta attention adapters, the SAM2-Tiny encoder remains frozen, significantly reducing computational load while retaining pre-trained visual priors. 

Evaluated rigorously on the **Landslide4Sense 2022** dataset using a 60/20/20 stratified split, our model achieves a state-of-the-art **75.01% Landslide IoU**, significantly outperforming existing SOTA models like TransLandSeg, D2FLS-Net, and TriGEFNet.

---

## 🏗️ System Architecture

LENTE is a Parameter-Efficient Fine-Tuned (PEFT) adaptation. The architecture modifies the vision pipeline through specific engineered components:

### 1. Multimodal Patch Embedding Expansion
Standard foundation models accept only 3-channel (RGB) inputs. Our model expands the patch embedding from 3 to 12 channels. 
* The pre-trained RGB weights are copied over.
* The extra channels are initialized with a small amount of noise (std=0.001) to ensure training stability.

### 2. DSDF Early Guidance Module (Terrain Fusion)
Topography dictates landslide mechanics. Rather than concatenating elevation data, LENTE employs an early-stage guidance module.
* A 2-channel terrain tensor (Slope + DEM) is processed through Conv3x3 + BN + ReLU -> Conv1x1 + Sigmoid to generate a dynamic spatial gain G in [0, 1].
* The optical input is modulated by this gain (optical_mod = optical * G).
* Because this occurs at the early stage, the terrain physically guides the encoder without direct concatenation.

### 3. Parameter-Efficient Attention Adapters
To adapt the network without catastrophic forgetting, the core SAM2-Tiny encoder is kept FROZEN. Instead, bottleneck Adapters are injected into every attention block.
* **Architecture:** Down-projection to a dimension of 64 -> ReLU activation -> Up-projection back to the original dimension with a residual connection.
* **Zero-Delta Initialization:** The adapters use zero-delta initialization.
* **Trainable Parameters:** Only the patch_proj, adapters, DSDF early guidance, and mask_decoder are trained, while the prompt encoder remains untrained.

### 4. Rigorous Training Protocol
* **Split & Augmentation:** Stratified 60/20/20 split (seed=42) with 2x offline augmentation (horizontal/vertical flip, rotate 90) applied only to positive samples.
* **Data Curation:** To focus specifically on segmentation accuracy, only positive patches are included in the training set. Negative patches are reserved strictly for the validation and test sets to ensure a realistic evaluation.
* **Optimization:** BCE Loss (dynamic pos_weight ~2.85) + Dice Loss (lambda=0.3), using the AdamW optimizer (lr=1e-5, wd=1e-4).

---

## 🚀 Advantages Compared to State-of-the-Art (SOTA)

### 1. vs. TransLandSeg (SAM Adaptation)
* **SOTA Architecture:** TransLandSeg utilizes a massive, frozen SAM ViT-L encoder with Adaptive Transfer Learning (ATL) layers.
* **SOTA Limitation:** Its input is restricted strictly to 3-band RGB images, meaning it completely lacks DEM fusion. 
* **LENTE Advantage:** By incorporating early DEM guidance and adapting the much lighter SAM2-Tiny backbone, our model achieves a massive +21.6 percentage point improvement in Landslide IoU over TransLandSeg (75.01% vs 53.41%).

### 2. vs. TriGEFNet (Tri-Stream Architectures)
* **SOTA Architecture:** TriGEFNet processes data using three entirely independent parallel streams (RGB, Slope/DEM, and a Vegetation Index like NDVI/EVI), each with its own backbone encoder. 
* **SOTA Limitation:** Because each stream has its own encoder, there is no premature fusion of low-level features. 
* **LENTE Advantage:** TriGEFNet achieved a Landslide IoU of 62.51%. LENTE's approach of modulating the optical input *before* the single transformer encoder allows it to outperform TriGEFNet's IoU by +12.5 percentage points.

### 3. vs. D2FLS-Net
* **SOTA Architecture:** D2FLS-Net utilizes a Swin-Transformer for its RGB backbone and a ViT-style encoder for DEM data, injecting the DEM data at both an early stage and a late stage.
* **LENTE Advantage:** While D2FLS-Net achieves a 56.69% Landslide IoU, our model leverages the foundational pre-training of SAM2 alongside a more highly optimized early-guidance terrain fusion to vastly surpass its accuracy.

---

## 📊 Performance Summary (Landslide4Sense Test Set)
*(Evaluated on real metrics from the 60/20/20 test split)*

| Model | Landslide IoU | mIoU | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TransLandSeg** | 53.41% | 75.99% | 84.46% | 84.99% | 83.93% |
| **D2FLS-Net** | 56.69% | 77.64% | n.r. | 68.45% | 76.87% |
| **TriGEFNet** | 62.51% | 80.26% | 76.88% | 75.33% | 78.54% |
| **LENTE (Ours)** | **75.01%** | **86.83%** | **84.33%** | **81.15%** | **88.05%** |

*(Note: Our model demonstrated stable convergence during training, dropping from a train loss of 0.31 to 0.12 over 100 epochs with no overfitting, and achieved an IoU background score of 98.65%.)*