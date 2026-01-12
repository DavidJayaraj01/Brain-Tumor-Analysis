# Datasets Documentation

## PART 2: DATASETS AND PREPROCESSING

---

## 2.1 Dataset Overview

This project utilizes three complementary datasets to ensure robustness and generalizability:

### Dataset Selection Rationale

| Dataset | Samples | Modality | Target Task | Rationale |
|---------|---------|----------|-------------|-----------|
| BraTS 2020/2021 | 585 | Multi-modal MRI | Segmentation + Grading | Gold standard for tumor segmentation benchmarking |
| Figshare Brain MRI | 3,064 | T1-weighted MRI | Classification | Large-scale dataset for tumor/non-tumor classification |
| Kaggle Brain Tumor | 3,762 | Various MRI | Classification + Segmentation | Diverse tumor types and grades |

**Total Dataset Size: ~7,400 samples**

---

## 2.2 BraTS 2020/2021 Dataset

### Dataset Characteristics

**Source**: Brain Tumor Segmentation Challenge (BraTS)
- Annual competition organized by MICCAI
- Multi-institutional data collection
- Ethical approval and anonymization completed

**Data Format**:
```
BraTS_2021/
├── subject_001/
│   ├── subject_001_t1.nii.gz        # T1-weighted MRI
│   ├── subject_001_t1ce.nii.gz      # T1 contrast-enhanced
│   ├── subject_001_t2.nii.gz        # T2-weighted MRI
│   ├── subject_001_flair.nii.gz     # FLAIR MRI
│   └── subject_001_seg.nii.gz       # Ground truth segmentation
├── subject_002/
│   └── ...
└── ...
```

**Image Specifications**:
- Dimensions: 240 × 240 × 155 voxels
- Voxel spacing: 1mm³
- 4 MRI modalities per subject
- 16-bit grayscale intensity values

### Classes and Segmentation Labels

BraTS uses multi-class segmentation:

```python
# Segmentation Labels
0: Background (non-tumor)
1: Necrotic and Non-enhancing Tumor Core (NCR/NET)
2: Peritumoral Edema (ED)
3: Enhancing Tumor (ET)

# Compound Labels
- Tumor core: Labels 1 + 3
- Whole tumor: Labels 1 + 2 + 3
```

**Tumor Grades (WHO Classification)**:
- Grade I-II (Low-grade glioma - LGG): Favorable prognosis
- Grade III-IV (High-grade glioma - HGG): Unfavorable prognosis

**BraTS Statistics**:
- 210 HGG samples (68%)
- 75 LGG samples (32%)
- Total: 285 subjects

### Preprocessing Pipeline for BraTS

```python
# Step 1: MRI Registration
- Align all 4 modalities to T1 template space
- Register to MNI atlas for normalization

# Step 2: Intensity Normalization
- N4 bias field correction
- Per-modality z-score normalization
- Clipping outliers (±5 std from mean)

# Step 3: Skull Stripping
- Brain extraction using BET algorithm
- Remove extracranial tissues

# Step 4: Cropping
- Crop to 128×128×128 around tumor
- Reduces computation and focuses on ROI

# Step 5: Data Augmentation (Training only)
- Random rotation: ±15 degrees
- Random scaling: 0.9-1.1
- Random flip: left-right
- Elastic deformation: σ=0.03
- Intensity jittering: ±10% Gaussian noise
- Probability per sample: 50%
```

### Train/Validation/Test Split (BraTS)

```
BraTS Training Set (285 subjects):
├── Training:   200 subjects (70%)
│   ├── HGG: 140 subjects
│   └── LGG: 60 subjects
│
├── Validation: 42 subjects (15%)
│   ├── HGG: 30 subjects
│   └── LGG: 12 subjects
│
└── Testing:    43 subjects (15%)
    ├── HGG: 30 subjects
    └── LGG: 13 subjects

Stratification: By grade (HGG/LGG) to ensure balanced classes
```

---

## 2.3 Figshare Brain MRI Dataset

### Dataset Characteristics

**Source**: Figshare (Community-contributed medical imaging dataset)
- Link: https://figshare.com/articles/brain_tumor_dataset/1512427

**Data Format**:
```
Figshare_Brain_MRI/
├── brain_tumor_dataset/
│   ├── yes/           # Tumor present
│   │   ├── Y0.jpg
│   │   ├── Y1.jpg
│   │   └── ...
│   └── no/            # No tumor
│       ├── N0.jpg
│       ├── N1.jpg
│       └── ...
```

**Image Specifications**:
- Format: 2D JPEG slices
- Typical dimensions: 512 × 512 pixels
- Grayscale (single-channel)
- Extracted from various patient MRI scans

### Classes

- **Class 0: No Tumor (Negative)** - 1,500 samples
- **Class 1: Tumor Present (Positive)** - 1,564 samples

**Total**: 3,064 samples

### Preprocessing Pipeline for Figshare

```python
# Step 1: Format Conversion
- Load JPEG images
- Convert to numpy arrays

# Step 2: Resizing
- Resize to 224×224 for model input
- Maintain aspect ratio with padding
- Interpolation: bilinear

# Step 3: Intensity Normalization
- Min-Max normalization to [0, 1]
- Per-image normalization

# Step 4: Outlier Removal
- Remove corrupted/artifact images
- Visual inspection validation

# Step 5: Data Augmentation (Training only)
- Random rotation: ±20 degrees
- Random horizontal flip: 50%
- Random vertical flip: 30%
- Random brightness: ±0.2
- Random contrast: 0.8-1.2
- Gaussian blur: σ=0.5
- Probability: 60%
```

### Train/Validation/Test Split (Figshare)

```
Figshare Dataset (3,064 samples):
├── Training:   2,145 samples (70%)
│   ├── Negative: 1,050 samples
│   └── Positive: 1,095 samples
│
├── Validation: 458 samples (15%)
│   ├── Negative: 225 samples
│   └── Positive: 233 samples
│
└── Testing:    461 samples (15%)
    ├── Negative: 225 samples
    └── Positive: 236 samples

Stratification: By class to ensure balance
```

---

## 2.4 Kaggle Brain Tumor Dataset

### Dataset Characteristics

**Source**: Kaggle - Brain MRI Images for Brain Tumor Detection
- Link: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri-dataset

**Data Format**:
```
Kaggle_Brain_Tumor/
├── Training/
│   ├── glioma/        # Glioma tumors
│   │   ├── image_0.jpg
│   │   └── ...
│   ├── meningioma/    # Meningioma tumors
│   │   ├── image_0.jpg
│   │   └── ...
│   ├── pituitary/     # Pituitary tumors
│   │   ├── image_0.jpg
│   │   └── ...
│   └── notumor/       # No tumor
│       ├── image_0.jpg
│       └── ...
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

**Image Specifications**:
- Format: JPEG images
- Typical dimensions: 512 × 512 pixels
- Grayscale or RGB
- Extracted 2D slices from 3D MRI volumes

### Classes

1. **Glioma** - 926 training samples (grade classification included)
2. **Meningioma** - 937 training samples
3. **Pituitary** - 844 training samples
4. **No Tumor** - 395 training samples

**Total**: 3,102 training + 660 testing samples = 3,762 samples

### Preprocessing Pipeline for Kaggle

```python
# Step 1: Format Conversion
- Load JPEG/PNG images
- Convert to 8-bit grayscale if RGB

# Step 2: Standardization
- Resize to 224×224 for model input
- Maintain square format

# Step 3: Intensity Normalization
- Histogram equalization for contrast enhancement
- Z-score normalization across entire image

# Step 4: Noise Reduction
- Gaussian filter: σ=1.0
- Remove salt-and-pepper noise with median filter

# Step 5: Data Augmentation (Training only)
- Random rotation: ±25 degrees
- Random zoom: 0.8-1.2
- Random shear: 0.2
- Random translation: ±10%
- Random intensity shift: ±0.15
- Probability: 65%
```

### Train/Validation/Test Split (Kaggle)

```
Kaggle Dataset (3,762 samples):
├── Training:   2,632 samples (70%)
│   ├── Glioma:     648 samples
│   ├── Meningioma: 656 samples
│   ├── Pituitary:  591 samples
│   └── No Tumor:   277 samples
│
├── Validation: 565 samples (15%)
│   ├── Glioma:     139 samples
│   ├── Meningioma: 141 samples
│   ├── Pituitary:  127 samples
│   └── No Tumor:   58 samples
│
└── Testing:    565 samples (15%)
    ├── Glioma:     139 samples
    ├── Meningioma: 140 samples
    ├── Pituitary:  126 samples
    └── No Tumor:   60 samples

Pre-existing test set (660 samples): Held for final evaluation
```

---

## 2.5 Unified Dataset Strategy

### Multi-Dataset Integration

```
Unified Training Pipeline:
├── Dataset 1 (BraTS): 
│   ├── Primary task: Segmentation + Grading
│   ├── Weight: 0.4 (high quality, multi-modal)
│   └── Frequency: Every batch contains BraTS samples
│
├── Dataset 2 (Figshare):
│   ├── Primary task: Classification (binary)
│   ├── Weight: 0.3 (large scale, 2D slices)
│   └── Mix: 30% of training batches
│
└── Dataset 3 (Kaggle):
    ├── Primary task: Multi-class classification
    ├── Weight: 0.3 (diverse tumor types)
    └── Mix: 30% of training batches
```

### Cross-Dataset Validation

```
Strategy:
1. Train on all datasets simultaneously (weighted sampling)
2. Validate on separate validation sets from each dataset
3. Test generalization on unseen test sets from all sources
4. Report per-dataset performance metrics
```

### Data Imbalance Handling

```python
# Class weights for imbalanced classes
class_weights = {
    'no_tumor': 1.0,
    'tumor': 1.5,
    'glioma': 1.2,
    'meningioma': 1.2,
    'pituitary': 1.3,
    'LGG': 2.0,  # Low-grade (minority)
    'HGG': 1.0   # High-grade (majority)
}

# Weighted random sampling during training
# Oversampling minority classes
# Undersampling majority classes
```

---

## 2.6 Data Statistics Summary

| Dataset | Samples | Resolution | Modality | Tasks | Classes |
|---------|---------|-----------|----------|-------|---------|
| BraTS 2021 | 285 | 240×240×155 | 4 MRI | Segmentation, Grading | 2 (LGG/HGG) |
| Figshare | 3,064 | 512×512 | T1/T2 | Classification | 2 (Tumor/No) |
| Kaggle | 3,762 | 512×512 | T1/T2 | Classification | 4 (4 types) |
| **Combined** | **7,111** | **Mixed** | **Multi** | **All** | **Up to 8** |

---

## 2.7 Preprocessing Code

Complete preprocessing implementations in `scripts/preprocessing.py`:

```python
# Key functions:
1. load_nifti_volume()          # Load BraTS NIfTI files
2. normalize_intensity()        # Per-modality z-score normalization
3. skull_stripping()            # Brain extraction
4. crop_to_brain()              # Reduce computational burden
5. resize_volume()              # To standard dimensions
6. augment_volume_3d()          # 3D augmentation for BraTS
7. load_2d_image()              # Load Figshare/Kaggle images
8. augment_image_2d()           # 2D augmentation
9. create_dataloaders()         # PyTorch DataLoaders
10. compute_statistics()        # Dataset statistics
```

---

## 2.8 Data Privacy and Ethics

- All datasets are publicly available with ethical approval
- Patient anonymization: All identifying information removed
- Compliance with HIPAA/GDPR standards
- Informed consent documented in original publications

---
