# Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting

Official implementation of the multi-scale deep learning framework described in:

> **Mitigating Weather Anomalies in Tourism Forecasting via Multi-Scale Decomposed Transformers**

The model integrates Scaleformer-style iterative multi-scale refinement with FEDformer-style Fourier-enhanced decomposition to forecast tourism demand under climate variability.

## Overview

This repository provides a complete, self-contained implementation of the proposed architecture:

- Climate-aware multivariate embedding (16 meteorological variables + tourism target)
- Three-level temporal pyramid (daily / weekly / seasonal)
- Fourier Enhanced Blocks with top-k frequency mode selection and moving-average trend extraction
- Coarse-to-fine iterative refinement decoder
- Chronological train / validation / test splits
- Training-set-only Z-score standardization
- Deterministic multi-seed evaluation protocol

Core files:
- `model.py` — full Scaleformer-FEDformer pipeline (Algorithm 1)
- `preprocess.py` — data loading, daily aggregation, PCHIP interpolation, windowing
- `main.py` — training / evaluation loop

## Dataset

The pipeline fuses high-resolution climate observations with regional tourism arrivals for the Thuringia region of Germany (2009-01-01 to 2016-12-31).

### Climate features (16 variables)
Source: Max Planck Institute for Biogeochemistry, Beutenberg Weather Station (Jena).  
Original resolution: 10-minute. Aggregated to daily in `preprocess.py`.

- Thermodynamic: \(T\), \(T_{\mathrm{pot}}\), \(T_{\mathrm{dew}}\), \(RH\)
- Vapor mechanics: \(VP_{\max}\), \(VP_{\mathrm{act}}\), \(VP_{\mathrm{def}}\), \(sh\), \(H_2OC\)
- Kinematics & dynamics: \(p\), \(\rho\), \(w_v\), \(w_{\max}\), \(w_d\), \(R\), \(R_{\mathrm{dur}}\)

The file `dataset.csv` (hosted in this repository) is used by default.

### Tourism target
Source: Thüringer Landesamt für Statistik (Table ID ge000802) — monthly guest arrivals / overnight stays.

**Processing**: The monthly series is converted to daily resolution via monotonic cubic Hermite (PCHIP) spline interpolation. This produces a smooth latent macro-demand trajectory that serves as the forecasting target. The interpolation is performed on the full available monthly series before chronological splitting.

Files provided:
- `tourism_thuringia_2009_2016.csv`
- `tourism_monthly_dataset.csv`

### Chronological splits (exact date boundaries)

| Split       | Period                          | Days  | Approx. % |
|-------------|---------------------------------|-------|-----------|
| Training    | 2009-01-01 – 2014-12-07         | 2,167 | 74.2 %    |
| Validation  | 2014-12-08 – 2015-09-25         | 292   | 10.0 %    |
| Test        | 2015-09-26 – 2016-12-31         | 463   | 15.8 %    |

Total: 2,922 daily observations.

## Preprocessing Pipeline (`preprocess.py`)

1. Download / load climate data and aggregate 10-minute observations to daily frequency (exactly 16 variables).
2. Load monthly tourism arrivals and apply PCHIP interpolation onto the daily climate timeline.
3. Merge climate and interpolated tourism series by date.
4. Apply Z-score standardization using statistics computed **exclusively on the training period**.
5. Construct sliding-window supervised samples (default look-back \(L = 96\)).

Default forecast horizons: 24 / 48 / 96 days.

## Model Architecture (`model.py`)

The implementation follows Algorithm 1 of the paper:

1. **Climate-aware embedding**  
   Independent linear projections for the target channel and the 16-dimensional climate vector, summed with sinusoidal positional encoding.

2. **Multi-scale pyramid**  
   Non-overlapping average pooling produces daily (factor 1), weekly (factor 7) and seasonal (factor 30) representations.

3. **Fourier Enhanced Block (per scale)**  
   Moving-average trend extraction + real FFT → top-k dominant mode selection → inverse FFT. Trend and denoised seasonal components are recombined.

4. **Iterative refinement decoder**  
   Forecasts begin at the coarsest (seasonal) scale and are progressively upsampled and refined at the weekly and daily scales. The daily-scale output is the primary prediction.

## Configuration

Hyper-parameters are defined as module-level constants in `main.py` and mirrored in `config.json`:

| Parameter                    | Value                  |
|-----------------------------|------------------------|
| Input sequence length       | 96                     |
| Forecast horizons           | 24 / 48 / 96           |
| Number of scales            | 3                      |
| Transformer layers per scale| 2–3                    |
| Attention heads             | 8                      |
| Model dimension             | 512                    |
| Dropout                     | 0.1                    |
| Batch size                  | 32                     |
| Learning rate               | \(1 \times 10^{-4}\)   |
| Max epochs                  | 100                    |
| Early-stopping patience     | 10                     |
| Independent runs            | 30 (seeds 42 … 71)     |
| Multi-scale loss weight \(\lambda\) | 0.5              |
| Frequency loss weight \(\gamma\)    | 0.5              |

## Installation & Usage

```bash
git clone https://github.com/matinkhah/tourism.git
cd tourism
pip install -r requirements.txt
```

Run the full experimental protocol (three horizons × 30 seeds):

```bash
python main.py
```

The scripts automatically use the climate and tourism CSV files hosted in the repository. CUDA is used when available; otherwise execution falls back to CPU.

## Reproducibility

- Explicit seeding of Python, NumPy and PyTorch RNGs.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False`.
- 30 independent runs with seeds 42–71.
- No pre-trained checkpoints or result logs are distributed; all metrics must be regenerated by running the training loop.

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The model produces a univariate forecast of the (interpolated) tourism target.

## Citation

```bibtex
@article{matinkhah2026tourism,
  title   = {Mitigating Weather Anomalies in Tourism Forecasting via Multi-Scale Decomposed Transformers},
  author  = {Matinkhah, S. Mojtaba and Shahbazi, A.},
  journal = {<FINAL VENUE>},
  year    = {2026}
}
```

## License

MIT License.

## Contact

Dr. S. Mojtaba Matinkhah  
matinkhah@yazd.ac.ir
