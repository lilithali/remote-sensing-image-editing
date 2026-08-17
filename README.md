# Remote Sensing Image Editing Based on Diffusion Models

Image editing for dual-temporal remote sensing imagery using diffusion models to generate balanced datasets for change captioning.

---

## Overview

This project investigates image editing for dual-temporal remote sensing imagery using diffusion models. Existing remote sensing change captioning datasets suffer from severe data imbalance, where some change types (e.g., buildings) appear much more frequently than others (e.g., lakes, parking lots, demolished structures).

To address this issue, this project fine-tunes **InstructPix2Pix** for remote sensing image editing. Given a pre-change image and a natural language editing instruction, the model generates a realistic post-change image while preserving unchanged regions.

The generated images are then used to construct an augmented dataset, which is further applied to improve the performance of remote sensing change captioning models.

---

## Highlights

- Fine-tuned **InstructPix2Pix** for remote sensing image editing
- Generated over **5,000** new remote sensing images
- Expanded the **Levir-MCI** dataset
- Reduced dataset imbalance for rare change categories
- Improved downstream **change captioning** performance using RSICCformer

---

## Motivation

Remote sensing change captioning relies on paired images captured at different times.

However, existing datasets exhibit several limitations:

- Long-tailed distribution of change categories
- Limited number of rare events
- Expensive manual annotation
- Difficulty collecting new satellite imagery

Instead of collecting additional satellite images, this project generates realistic edited images using diffusion models to enrich existing datasets.

---

## Dataset

### Levir-MCI

The project uses the **Levir-MCI** dataset.

Dataset characteristics:

- 10,077 image pairs
- 256 × 256 RGB images
- Five captions for each image pair
- Dual-temporal remote sensing imagery

For dataset augmentation:

- 1,046 original images were selected
- Five editing instructions were designed for each image
- A total of **5,230 edited remote sensing images** were generated

---

## Methodology

The overall pipeline is shown below.

```
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

---

## Model

### Image Editing

- Stable Diffusion
- InstructPix2Pix
- DDIM Scheduler
- Classifier-Free Guidance

### Change Captioning

- RSICCformer

---

## Image Editing Examples

The model can generate various realistic changes, including

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

The generated images successfully follow textual editing instructions while maintaining the overall scene consistency.

Example editing tasks include:

- Add a lake
- Build several houses
- Construct a parking lot
- Extend an existing building
- Remove vegetation

---

### Quantitative Results

The expanded dataset improves the downstream change captioning model.

Evaluation metrics include

- BLEU
- METEOR
- ROUGE
- CIDEr

The generated dataset demonstrates improved caption quality compared with training using the original dataset alone.

---

## Repository Structure

```
remote-sensing-image-editing/

│
├── dataset/                 # Dataset preprocessing
│
├── checkpoints/             # Trained models
│
├── scripts/                 # Training and inference scripts
│
├── images/                  # Figures used in README
│
├── results/                 # Generated samples
│
├── README.md
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/yourname/remote-sensing-image-editing.git

cd remote-sensing-image-editing

pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py
```

### Generate Edited Images

```bash
python inference.py
```

### Generate Augmented Dataset

```bash
python generate_dataset.py
```

### Evaluate Change Captioning

```bash
python evaluate.py
```

---

## Technologies

- Python
- PyTorch
- HuggingFace Diffusers
- Stable Diffusion
- InstructPix2Pix
- OpenCV
- Transformers
- CUDA

---

## Future Work

Possible future improvements include

- Higher-resolution remote sensing editing
- Multi-object editing
- Automatic instruction generation
- Generalization to additional remote sensing datasets
- Integration with larger vision-language models

---

## Acknowledgements

This project is based on the following works:

- InstructPix2Pix
- Stable Diffusion
- Levir-MCI Dataset
- RSICCformer

---

## License

This repository is intended for academic and research purposes.

