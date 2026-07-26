# Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting

Official implementation of the **Multi-Scale Frequency-Enhanced Model** for multivariate tourism demand forecasting using climate-related temporal features.

The proposed framework combines multi-scale temporal representations (Scaleformer-style iterative refinement) with Fourier-based frequency decomposition (FEDformer-style) to model temporal dependencies and periodic patterns in tourism demand.

## Overview

The implementation contains the core components of the proposed forecasting framework:

* Multi-scale temporal representations (daily / weekly / seasonal pyramid)
* Fourier-based frequency decomposition with moving-average trend extraction
* Climate-aware feature embedding (separate target/climate projections + positional encoding)
* Coarse-to-fine iterative refinement decoder
* Long-horizon multivariate forecasting (horizons of 24 / 48 / 96 days)
* Chronological train/validation/test splitting
* Training-set-only standardization
* Deterministic execution across multiple random seeds

The main model is implemented in `model.py`, data preprocessing and temporal window construction are implemented in `preprocess.py`, and the training/evaluation loop is implemented in `main.py`.

## Repository Structure

```text
tourism-forecasting/
├── main.py
├── model.py
├── preprocess.py
├── README.md
└── data/
    ├── jena_climate.csv        # not included — see Dataset section
    └── tourism_thuringia.csv   # not included — see Dataset section
```

> This repository currently contains only the three Python scripts above. No `config.json`, checkpoints, or data files are included — see the notes in the relevant sections below before attempting to run the pipeline.

## Dataset

The forecasting pipeline combines daily climate information derived from the **Jena Climate Dataset** (Beutenberg station, Max Planck Institute for Biogeochemistry) with a tourism-demand target.

The climate data are originally recorded at 10-minute resolution. `preprocess.py` aggregates the observations to daily frequency across the 16 meteorological variables used by the model: temperature, potential temperature, dew point, relative humidity, saturation/actual/deficit vapor pressure, specific humidity, water vapor concentration, atmospheric pressure, air density, wind velocity, peak wind gust, wind direction, and cumulative rainfall depth/duration. Most continuous variables are averaged over each day, peak wind gust is aggregated using the daily maximum, and precipitation-related variables are accumulated.

The tourism target is represented by the variable:

```text
tourist_count
```

Because the source registry is compiled monthly, `preprocess.py` converts it to a daily series using **monotonic cubic Hermite (PCHIP) spline interpolation** rather than a plain merge — this is a deliberate low-pass step, not a placeholder. The climate and interpolated tourism series are then aligned by date before the forecasting dataset is constructed.

### Important

This repository does **not** include the raw dataset files (`jena_climate.csv`, `tourism_thuringia.csv`). You must supply them yourself, pointed at by the `raw_jena_path` / `tourism_csv_path` arguments of `run_experiment_pipeline()` in `main.py`. The pipeline no longer silently substitutes a synthetic tourism target when the file is missing — `preprocess.prepare_pipeline` raises a `FileNotFoundError` instead, so a missing file will fail loudly rather than produce misleading results.

## Preprocessing

`preprocess.py` performs the following operations:

1. Aggregate the raw 10-minute climate observations to daily frequency (16 variables).
2. Interpolate the monthly tourism registry to a daily series via monotonic cubic spline (PCHIP).
3. Merge the daily climate variables with the interpolated tourism target by date.
4. Sort observations chronologically.
5. Split the data chronologically by exact date boundary, not a fixed percentage cutoff:

   * Training: 2009-01-01 – 2014-12-07 (2,167 days, ≈74.2%)
   * Validation: 2014-12-08 – 2015-09-25 (292 days, ≈10.0%)
   * Test: 2015-09-26 – 2016-12-31 (463 days, ≈15.8%)
6. Fit Z-score standardization statistics using the training partition only, then apply to all three partitions.
7. Generate supervised sliding windows for forecasting.

The current implementation uses a look-back sequence length of 96 (`SEQ_LEN` in `main.py`) and forecasting horizons of 24, 48, and 96 days, looped over in `main.py`'s `__main__` block.

## Model Architecture

The implementation (`model.py`) contains four principal components, matching Algorithm 1 of the manuscript:

### Climate-Aware Embedding

The target channel and the 16-dimensional climate vector are projected with **independent** learned linear layers (`E_y`, `E_w`), then combined with a sinusoidal positional encoding (`E_p`): `E_t = E_y + E_w + E_p`.

### Fourier-Enhanced Block

Each scale's representation is split into a moving-average trend component and a seasonal component. The seasonal component is transformed into the frequency domain using the real-valued FFT, retains only the top-k dominant frequency modes, and is reconstructed via inverse FFT. Trend and denoised seasonal components are summed to form the block's output.

### Multi-Scale Forecasting Pipeline

The model constructs daily, weekly (factor 7), and seasonal (factor 30) representations using non-overlapping temporal average pooling. Each scale is encoded with a dedicated Transformer self-attention layer, then processed by its own Fourier-enhanced block.

### Iterative Refinement Decoder

Forecasts are produced coarse-to-fine: the seasonal-scale forecast is generated first, then upsampled and fed into the weekly-scale decoder, and finally upsampled again and fed into the daily-scale decoder, which produces the model's primary output.

## Configuration

There is currently no `config.json` in this repository. The experimental settings live directly in `main.py` as module-level constants:

* Model architecture: `ScaleformerFEDformerPipeline`
* Target: `tourist_count`
* Input sequence length: 96
* Forecast horizons: 24, 48, 96
* Number of scales: 3 (daily / weekly / seasonal)
* Transformer layers per scale: 3
* Attention heads: 8
* Model dimension: 512
* Dropout: 0.1
* Batch size: 32
* Learning rate: `1e-4`
* Training epochs: 100 (with early stopping)
* Early stopping patience: 10
* Independent runs: 30
* Base random seed: 42
* Multi-scale / frequency loss weights (`λ`, `γ`): 0.5, 0.5

If a `config.json` is added later, it must be reconciled against these constants — `main.py` does not currently read from a config file.

## Reproducibility

The implementation supports deterministic execution through explicit seeding of Python, NumPy, and PyTorch random-number generators (`preprocess.enforce_determinism`) and deterministic CuDNN settings (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`).

The experimental protocol specifies 30 independent runs using seeds:

```text
42, 43, ..., 71
```

## Training

Run the experiment using:

```bash
python main.py
```

This will loop over the three forecast horizons (24, 48, 96) and, for each, execute 30 deterministic training runs with validation-based early stopping.

Before running the experiment, verify that:

1. `raw_jena_path` and `tourism_csv_path` in `main.py`'s `run_experiment_pipeline()` call point to your actual dataset files.
2. The required tourism and climate data are available locally.
3. CUDA is available if GPU execution is intended (the scripts fall back to CPU automatically otherwise).
4. `torch` and `scipy` are installed (`scipy` is required by `preprocess.py` for the PCHIP interpolation).

## Evaluation

The implementation computes:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

The target variable is evaluated from the final feature position of the multivariate window (OT convention) — the model's raw output is univariate (the `tourist_count` forecast only).

## Checkpoints and Results

`main.py` writes a checkpoint every 10 runs (seeds 42, 52, 62) to a local `checkpoints/` directory, created automatically at runtime. No pretrained checkpoints or result files are included in this repository — none have been generated or verified yet.

## Citation

Please replace the following placeholder with the final bibliographic information before publication:

```bibtex
@article{matinkhah2026tourism,
  title={Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting},
  author={Matinkhah, S. Mojtaba and Shahbazi, A.},
  journal={<FINAL VENUE>},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Contact

**Dr. S. Mojtaba Matinkhah**

Email: [matinkhah@yazd.ac.ir](mailto:matinkhah@yazd.ac.ir)
