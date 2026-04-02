# NIRFreq: A Deep Learning Model for Pansharpening with High-Fidelity Downstream Task Execution

This repository contains the official PyTorch implementation of the **NIRFreq** model, a deep learning architecture designed for high-fidelity pansharpening of satellite imagery. Our model not only achieves state-of-the-art visual quality but also significantly improves the accuracy of downstream remote sensing machine vision tasks.

## 🌟 Features

- **Advanced Fusion Architecture**: Implements several key modules for state-of-the-art performance.
- **Frequency-Aware Gated Cross-Fusion Module (GCFM)**: Dynamically fuses panchromatic (PAN) and multispectral (MS) features using a content-aware and frequency-aware mechanism.
- **Hierarchical Feature Aggregation (HFA)**: Adaptively aggregates features from different network depths to preserve both spatial details and spectral fidelity.
- **Downstream-Oriented Evaluation**: Built-in scripts for objective downstream task evaluation (e.g., Water/Forest segmentation) and deep spectral/frequency analysis.
- **Configurable & Reproducible**: All experiments are managed through a centralized YAML configuration system, ensuring full reproducibility.

## 📂 File Structure

The codebase is organized to meet high academic standards, separating logic, scripts, and configurations.

```text
NIRFreq/
├── configs/
│   └── default.yaml        # Main configuration file for all hyperparameters
├── src/
│   ├── data/
│   │   └── dataset.py      # PyTorch Lightning DataModule and Dataset classes
│   ├── losses/
│   │   └── combined_loss.py  # Combined loss function (L1, MSE, SSIM)
│   ├── metrics/
│   │   └── evaluation.py   # Functions for evaluation metrics (CC, etc.)
│   ├── models/
│   │   ├── nir_freq_model.py # Main PyTorch Lightning module
│   │   └── network.py         # Core neural network architecture
│   └── utils/
│       └── common.py        # Utility functions
├── train.py                # Script to run model training
├── test.py                 # Script to run model testing and standard evaluation
├── evaluate_target.py      # Script for downstream tasks (Water/Forest segmentation)
├── evaluate_advanced.py    # Script for Spectral Profile & FFT Spectrum analysis
├── requirements.txt        # Project dependencies
└── README.md               # This file
