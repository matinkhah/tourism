# Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting

**A novel deep learning framework combining multi-scale attention and frequency decomposition for accurate long-term tourism demand prediction.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the official implementation of the **Multi-Scale Frequency-Enhanced Model** for tourism demand forecasting. The model integrates hierarchical multi-scale attention with an extended FEDformer-style frequency decomposition to capture both temporal hierarchies and complex periodic patterns in tourism arrival data.

Key innovations include:
- Multi-scale temporal attention mechanism
- Multi-band frequency decomposition (weekly, monthly, seasonal, climate trends)
- Comprehensive ablation studies and extreme weather robustness analysis
- Rigorous multi-run evaluation (30 independent runs per configuration)

## Features

- Unified experimental pipeline with full reproducibility
- Consistent visualization suite (Figures 1–7)
- Support for multiple forecasting horizons (24, 48, 96 steps)
- Detailed ablation study on key components
- Frequency analysis and attention pattern visualization

## Dataset

The models are evaluated on the **Jena Climate Dataset** (public benchmark), adapted for time series forecasting benchmarks. 

- **Source**: Max Planck Institute for Biogeochemistry, Jena, Germany
- **Resolution**: 10-minute intervals
- **Target Variable**: Air temperature (`OT`)
- **Time Range**: January 2020 onward (benchmark subset)

Full dataset details and preprocessing steps are provided in the paper.

## Project Structure

```bash
tourism-forecasting/
├── data/                    # Preprocessed datasets
├── models/                  # Model implementations
├── utils/                   # Data loaders, metrics, visualization
├── notebooks/               # Exploratory analysis
├── results/                 # Saved model checkpoints & logs
├── tourism_2.py             # Main visualization script (Figures 1-7)
├── train.py                 # Training script
├── config.py                # Hyperparameters
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/yourusername/tourism-forecasting.git
cd tourism-forecasting
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.1.0 + CUDA 12.1
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy

## Reproducibility

All results are averaged over **30 independent runs** with seeds `{42 + i for i in range(30)}`.  
Deterministic mode is enabled for PyTorch/CuDNN.

**Hardware used for experiments**:
- GPU: NVIDIA RTX 4090 (24 GB)
- CPU: Intel Core i9-13900K
- RAM: 64 GB

Complete code, configurations, and trained models will be released upon paper acceptance.

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{paper2026,
  title={Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting},
  author={matinkhah et al.},
  journal={Journal Name},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)

## Contact

- Author: [Dr S. Mojtaba Matinkhah]
- Email: [matinkhah@yazd.ac.ir]
- GitHub Issues: Please open an issue for questions or bug reports.

---
