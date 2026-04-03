# NIRFreqNet: A Frequency-Aware and Correlation-Guided Network for Remote Sensing Pansharpening

This repository contains the official PyTorch implementation of **NIRFreqNet**, a deep learning framework designed for high-fidelity pansharpening. By integrating frequency-domain modeling with RGB-NIR cross-modal correlation guidance, NIRFreqNet achieves a superior balance between spatial detail injection and spectral fidelity.

## 🚀 Key Modules

- **Shallow Feature Extractor (SFE)**: Decouples RGB and NIR streams and generates pixel-wise spectral correlation maps to guide the fusion process.
- **Frequency-Aware Gated Cross-Fusion Module (GCFM)**: Combines static Laplacian-based high-frequency priors with dynamic convolution to adaptively inject textures while suppressing spectral distortion.
- **Correlation-Guided Hierarchical Feature Aggregation (CHFA)**: Adaptively merges multi-level features across layers using spectral correlation priors as a stable reference.
---
