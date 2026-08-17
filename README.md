# Remote Sensing Image Editing Based on Diffusion Models

Image editing for dual-temporal remote sensing imagery using diffusion models to generate balanced datasets for remote sensing change captioning.

---

## Overview

This project investigates remote sensing image editing using diffusion models to address the long-tailed distribution of change categories in remote sensing change captioning datasets.

Instead of collecting additional satellite imagery, the project fine-tunes **InstructPix2Pix** to edit remote sensing images according to natural language instructions. The generated image pairs are then incorporated into an expanded training dataset to improve the performance of downstream change captioning models.

This work demonstrates how diffusion-based image editing can be applied as an effective data augmentation strategy for remote sensing applications.


---


## Dataset

This project uses the **Levir-MCI** remote sensing change captioning dataset.

### Dataset Statistics

- 10,077 image pairs
- 256 × 256 RGB images
- Five captions for each image pair
- Dual-temporal remote sensing imagery

### Dataset Augmentation

For image editing experiments:

- 1,046 original image pairs were selected
- Five editing instructions were designed for each image
- A total of **5,230 edited remote sensing images** were generated

### Dataset Availability

The **Levir-MCI** dataset is **not included** in this repository.

Please obtain the dataset from its official source and place it in the appropriate directory before training or evaluation.

---

## Methodology

The overall workflow is illustrated below.

```text
Original Image
        │
        ▼
Editing Instruction
        │
        ▼
InstructPix2Pix
        │
        ▼
Edited Remote Sensing Image
        │
        ▼
Expanded Dataset
        │
        ▼
RSICCformer
        │
        ▼
Generated Change Caption
```

The project consists of two stages:

### Stage 1 — Remote Sensing Image Editing

The image editing model generates realistic post-change remote sensing images based on textual editing instructions while preserving the unchanged regions of the scene.

### Stage 2 — Change Captioning

The generated image pairs are incorporated into the training dataset and evaluated using **RSICCformer** to assess the impact of dataset augmentation on remote sensing change captioning.

---

## Models

### Image Editing

- Stable Diffusion
- InstructPix2Pix
- DDIM Scheduler
- Classifier-Free Guidance

### Change Captioning

- RSICCformer

---

## Image Editing Examples

The proposed method can generate various realistic scene changes, including:

- Building construction
- Building removal
- Road construction
- Parking lot generation
- Water body generation
- Vegetation changes

while preserving the remaining regions of the original image.

---

## Results

### Qualitative Results

The generated images successfully follow textual editing instructions while maintaining the overall visual consistency of the remote sensing scene.

Typical editing instructions include:

- Build several houses
- Remove existing buildings
- Add a lake
- Construct a parking lot
- Expand roads
- Remove vegetation

Example results are shown below.

<p align="center">
<img src="images/editing_examples.png" width="900">
</p>

---

### Quantitative Results

The augmented dataset is evaluated using **RSICCformer**.

Performance is measured using standard captioning metrics:

- BLEU
- METEOR
- ROUGE-L
- CIDEr

Experimental results demonstrate that the generated dataset improves change captioning performance compared with training using the original dataset alone.

---

## Repository Structure

```text
remote-sensing-image-editing/
│
├── images/                  # Figures used in README
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Training and inference scripts
├── src/                     # Core implementation
├── README.md
├── requirements.txt
└── LICENSE
```

> **Note:**  
> The original dataset and trained checkpoints are **not included** in this repository due to licensing restrictions and file size limitations.

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yaml
conda activate ip2p
---

## Pre-trained Models

The trained checkpoints are **not distributed** with this repository because:

- The checkpoint files are several gigabytes in size.
- They are derived from publicly available diffusion models with separate licenses.
- This repository focuses on demonstrating the implementation and research methodology.

Users should download the required pre-trained models (e.g., Stable Diffusion and InstructPix2Pix) from their official repositories.

---

## Usage

Before running the project:

1. Download the Levir-MCI dataset.
2. Download the required diffusion model checkpoints.
3. Configure the dataset and checkpoint paths.

### Training

```bash
python train.py
```

### Generate Edited Images

```bash
python inference.py
```

### Dataset Augmentation

```bash
python generate_dataset.py
```

### Evaluate Change Captioning

```bash
python evaluate.py
```


---

## Acknowledgements

This project builds upon several excellent open-source projects and publicly available datasets:

- Stable Diffusion
- InstructPix2Pix
- Hugging Face Diffusers
- Levir-MCI Dataset
- RSICCformer

We sincerely thank the authors for making their work publicly available.

---

## License

This project builds upon InstructPix2Pix and Stable Diffusion.

The original InstructPix2Pix license is included in this repository. Portions of the code and models are derived from the Stable Diffusion codebase and may be subject to additional licensing restrictions.

The Levir-MCI dataset and trained model checkpoints are not redistributed in this repository.
