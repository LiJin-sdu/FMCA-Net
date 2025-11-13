[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17596909.svg)](https://doi.org/10.5281/zenodo.17596909)

# FMCA-Net: Forgery-aware Multi-scale Cross-Attention Network

This repository provides the **official implementation** of the paper:

**“Multi-Scale Cross-Attention Network for Enhanced Face Forgery Detection (FMCA-Net)”**,  
submitted to *The Visual Computer*.

> 🔔 **Important:** This code is directly related to the manuscript currently submitted to *The Visual Computer*.  
> If you use this repository or find it helpful, please **cite the corresponding paper** once it becomes available.

---

## 1. Overview

FMCA-Net is a **dual-branch spatial–frequency network** for robust face forgery (DeepFake) detection.  
It integrates:

- A **ViT-based spatial branch** for global semantic modeling  
- A **Haar-DWT-based frequency branch** for mid-/high-frequency forgery cues  
- A **Forgery-aware Token Multi-scale Attention (FTMA)** module for token-level multi-scale saliency  
- A **Spatial–Frequency Cross Module (SFCM)** for bidirectional cross-attention between spatial and frequency tokens  
- A **Gated Spatial–Frequency Concatenation (GSFC)** module for adaptive fusion of spatial and frequency representations.
- A robust training strategy leveraging **Focal Loss** and **Exponential Moving Average (EMA)** to stabilize optimization and improve generalization.


The implementation in this repository reproduces the experiments reported in the paper, including training and inference on common DeepFake benchmarks (e.g., FaceForensics++).

---

## 2. Repository Structure

The main implementation is under the `FMCA-Net/` directory:

```text
FMCA-Net/
├── FTMA.py          # Forgery-aware Token Multi-scale Attention (FTMA)
├── SFCM.py          # Spatial–Frequency Cross Module (SFCM, bidirectional cross-attention)
├── config.py        # Global configuration (paths, hyper-parameters)
├── data_utils.py    # Data preprocessing, augmentation, dataset analysis & splitting
├── freq_branch.py   # Haar-DWT and frequency branch for spectral feature extraction
├── inference.py     # Inference & visualization for single / batch images
├── train.py         # Full training pipeline with EMA, Focal Loss, and evaluation
├── README.md        # This file
└── requirements.txt # Python dependencies
```

---

## 3. Environment & Dependencies

### 3.1 Python & Core Libraries

- Python **3.9+**
- PyTorch **2.0+** (CUDA recommended)
- CUDA 11.x / 12.x (for training acceleration)

Main Python dependencies (also provided in `requirements.txt`):

- `torch`, `torchvision`
- `albumentations`
- `opencv-python`
- `numpy`, `scikit-learn`
- `transformers`
- `Pillow`
- `matplotlib`, `seaborn`
- `tqdm`

### 3.2 Installation

```bash
# clone this repository
git clone https://github.com/LiJin-sdu/FMCA-Net.git
cd FMCA-Net/FMCA-Net

# install required packages
pip install -r requirements.txt
```

---

## 4. Datasets & Directory Layout

FMCA-Net is designed for frame-level face forgery detection.  
The default configuration assumes a dataset structure similar to **FaceForensics++ C23**:
```
DATA_ROOT/
├── train/
│ ├── real/ # real face frames
│ └── fake/ # fake face frames (filenames may encode manipulation method)
└── valid/
├── real/ # validation real frames
└── fake/ # validation fake frames
```
You can adapt this layout for other datasets (e.g., **Celeb-DF-v2**, **WildDeepfake**, **DFDCp**).


### 4.1 Set dataset path

Edit `config.py`:

```
python
class Config:
    DATA_ROOT = r'/path/to/your/dataset_root'
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    VAL_DIR   = os.path.join(DATA_ROOT, 'valid')
    ...
```
### 4.2 Dataset preprocessing & splitting

data_utils.py provides utilities for:

- analyzing dataset statistics
- creating a train/validation split
- visualizing class distribution and image quality

---

## 5. Key Components & Algorithms

### 5.1 FTMA — Forgery-aware Token Multi-scale Attention

- Operates on ViT patch tokens to address the tendency of self-attention to overlook subtle local manipulations.  
- Projects tokens into three scales (¼C, ½C, C), each capturing different levels of local–global context.  
- Refines each scale with non-linear transformations and concatenates them into a unified multi-scale representation.  
- Enhances forgery-sensitive token saliency under compression and texture degradation, while preserving global semantics.  



### 5.2 SFCM — Spatial–Frequency Cross Module

- Introduces **bidirectional** cross-attention between spatial tokens (from ViT) and frequency tokens (from the Haar-DWT branch).  
- Spatial → Frequency: spatial tokens query frequency tokens to enrich spatial features with high-frequency artifacts.  
- Frequency → Spatial: frequency tokens query spatial tokens to inject semantic context into spectral representations.  
- Implemented as a single 8-head bidirectional attention layer with dropout, balancing accuracy and efficiency.  



### 5.3 Haar-DWT & Frequency Branch

- Applies 2D Haar-DWT to each RGB channel to obtain LL/LH/HL/HH subbands, yielding a 12-channel frequency map.  
- A lightweight 5-layer CNN (12→64→128→256→256→128, with stride-2 downsampling in the 2nd and 4th layers) encodes frequency maps.  
- Produces compact, noise-suppressed frequency embeddings or tokens that complement spatial features in SFCM and GSFC.  
- Emphasizes subtle high-frequency artifacts that are often indicative of manipulation, especially under compression.  



### 5.4 Spatial–Frequency Interaction & GSFC Fusion

- **Spatial branch:** input image → ViT (DINOv2 ViT-B/14) → FTMA → refined CLS token \(f_s\).  
- **Frequency branch:** input image → Haar-DWT → frequency CNN → global frequency feature \(f_f\) (and/or tokens for SFCM).  
- **Cross-modal interaction:** SFCM performs bidirectional attention between spatial and frequency tokens to achieve explicit spatial–frequency alignment.  
- **Gated fusion (GSFC):**  
  - Linearly compresses \(f_f\) into a reduced dimension and applies a learnable gate \(g = \sigma(\gamma)\).  
  - Concatenates \(f_s\) and gated frequency feature \([f_s, g \tilde{f}_f]\) as the final fused representation.  
  - Suppresses unstable or noisy frequency components while preserving the dominant role of spatial semantics.  
- The fused feature is then fed into a lightweight classifier for binary real/fake prediction.  



### 5.5 Loss & Robust Optimization

- **Focal Loss:**  
  - Used as the main training objective to handle class imbalance (e.g., forged vs. real) and varying sample difficulty.  
  - Parameters follow the paper setup (e.g., α = 0.6, γ = 2.0), focusing more on hard or high-compression samples.  
- **EMA (Exponential Moving Average):**  
  - Maintains a smoothed copy of model parameters during training.  
  - Validation is performed on the EMA model to improve stability and cross-domain robustness.  
- Together with the dual-branch design and GSFC, these strategies enhance the robustness of FMCA-Net across different datasets, manipulations, and compression levels.

---

## 6. Code, Models & Reproducibility

This repository contains all source code required to reproduce the results in our paper, including:

- FTMA, SFCM, and GSFC modules  
- ViT-based spatial branch and Haar-DWT frequency branch  
- Full training pipeline with Focal Loss and EMA  
- Dataset evaluation scripts  

👉 **GitHub repository:** https://github.com/LiJin-sdu/FMCA-Net

👉 **Zenodo archived release (permanent DOI):** https://doi.org/10.5281/zenodo.17596909

Once the paper is accepted, pretrained models and logs will also be released.

> This codebase is directly associated with the manuscript submitted to *The Visual Computer*.  
> Please cite the paper if you find the code helpful.

---

## 7. Citation

If you use FMCA-Net in your research, please cite:

```bibtex
@article{Li2025fmcanet,
  title   = {Multi-Scale Cross-Attention Network for Enhanced Face Forgery Detection},
  author  = {Li, Jin and Zhang, Yupeng and Wang, Chengyou and Zhou, Xiao},
  journal = {The Visual Computer},
  year    = {2025},
  note    = {Manuscript submitted}
}
```

---

## 8. Quick Start

### 8.1 Train FMCA-Net

```bash
python train.py
```

### 8.2 Inference on a single image

```bash
python inference.py --image path/to/image.jpg
```

---



