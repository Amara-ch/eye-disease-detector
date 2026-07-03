# Eye Disease Detector

> AI-powered multi-label retinal disease detection using deep learning, with an on-device mobile app.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B.svg)

---

## Overview

**Eye Disease Detector** analyzes retinal fundus images and predicts multiple eye diseases simultaneously (multi-label classification). Built on the RFMiD dataset covering 46 retinal conditions, with explainable AI (Grad-CAM) and a free on-device mobile app.

> Medical Disclaimer: This is an educational / screening aid, NOT a medical diagnostic tool. Always consult a qualified ophthalmologist.

## Features

- Multi-label classification (detects multiple conditions at once)
- Modern backbones (EfficientNetV2 / ConvNeXt via timm)
- Handles class imbalance (Focal / Asymmetric loss)
- Explainability via Grad-CAM heatmaps
- Flutter mobile app, runs fully on-device (offline, private)
- Two input modes: live camera capture + gallery upload
- 100% free and open-source

## Project Structure

    eye-disease-detector/
    ├── ml/              # Machine learning (PyTorch)
    │   ├── configs/     # YAML configs
    │   ├── data/        # Dataset loaders
    │   ├── models/      # Model architectures
    │   ├── losses/      # Focal / asymmetric loss
    │   ├── training/    # Training loop & metrics
    │   ├── export/      # ONNX -> TFLite conversion
    │   └── notebooks/   # Kaggle / Colab notebooks
    ├── app/             # Flutter mobile app
    ├── docs/            # Documentation
    └── requirements.txt

## Dataset

RFMiD - 3,200 fundus images, 46 label columns (45 diseases + 1 binary Disease Risk).
See docs/CONDITIONS.md for the full condition list.

## Getting Started

    git clone https://github.com/Amara-ch/eye-disease-detector.git
    cd eye-disease-detector
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Roadmap

- [x] Project setup
- [ ] Dataset exploration & analysis
- [ ] Data pipeline (loaders + augmentation)
- [ ] Model + training (focal loss, multi-label)
- [ ] Evaluation (macro AUC, per-label F1)
- [ ] Grad-CAM explainability
- [ ] TFLite export
- [ ] Flutter mobile app

## Tech Stack

ML: PyTorch, timm, Albumentations, scikit-learn, Weights & Biases
Mobile: Flutter, TensorFlow Lite
Tools: Kaggle (free GPU), VS Code, Git

## License

MIT License - see LICENSE.
