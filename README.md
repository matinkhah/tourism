```markdown
# Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting

Official implementation of the **Multi-Scale Frequency-Enhanced Model** for multivariate tourism demand forecasting under climate volatility.

This repository accompanies the paper:

> **Mitigating Weather Anomalies in Tourism Forecasting via Multi-Scale Decomposed Transformers**

The framework combines multi-scale temporal representations (Scaleformer-style iterative refinement) with Fourier-based frequency decomposition (FEDformer-style) to model short-term weather-induced fluctuations and long-term climate-driven seasonal patterns in tourism demand.

## Overview

The implementation contains the core components of the proposed forecasting framework:

- Multi-scale temporal representations (daily / weekly / seasonal pyramid)
- Fourier-based frequency decomposition with moving-average trend extraction
- Climate-aware feature embedding (separate target/climate projections + positional encoding)
- Coarse-to-fine iterative refinement decoder
- Long-horizon multivariate forecasting (horizons of 24 / 48 / 96 days)
- Chronological train/validation/test splitting
- Training-set-only standardization
- Deterministic execution across multiple random seeds

The main model is implemented in `model.py`, data preprocessing and temporal window construction are implemented in `preprocess.py`, and the training/evaluation loop is implemented in `main.py`.

## Key Contributions (from the paper)

1. A multi-scale tourism forecasting framework integrating Scaleformer’s iterative refinement with FEDformer’s frequency-domain decomposition.
2. Climate-aware multivariate modeling using 16 high-resolution meteorological variables, including thermodynamic indicators such as saturation vapor pressure (\(VP_{max}\)) and dew point.
3. Systematic evaluation against ARIMA, LSTM, GRU, Transformer, Informer, Autoformer, FEDformer, and Scaleformer.
4. Ablation studies quantifying the contribution of multi-scale learning, frequency decomposition, and advanced climate variables.

Empirical results indicate a **21.7–24.1% MSE reduction** relative to the strongest individual baseline (Scaleformer) across short-, medium-, and long-term horizons.

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

> **Important**: This repository currently contains only the three Python scripts above. No `config.json`, pretrained checkpoints, or data files are included. See the notes in the relevant sections below before attempting to run the pipeline.

## Dataset

The forecasting pipeline combines daily climate information derived from the **Jena Climate Dataset** (Beutenberg station, Max Planck Institute for Biogeochemistry) with a tourism-demand target from the Thuringia region of Germany.

### Climate Features
- **Source**: Max Planck Institute for Biogeochemistry – Weather Station Beutenberg  
  (https://www.bgc-jena.mpg.de/wetter/)
- **Coordinates**: \(50^\circ 54'32''N\), \(11^\circ 34'05''E\), elevation 155 m
- **Original sampling**: 10-minute resolution
- **Aggregation**: `preprocess.py` converts observations to daily frequency across **16 meteorological variables**:
  - Thermodynamic: Surface Air Temperature (\(T\)), Potential Temperature (\(T_{pot}\)), Dew Point (\(T_{dew}\)), Relative Humidity (\(RH\))
  - Vapor mechanics: Saturation Vapor Pressure (\(VP_{max}\)), Actual Vapor Pressure (\(VP_{act}\)), Vapor Pressure Deficit (\(VP_{def}\)), Specific Humidity (\(sh\)), Water Vapor Concentration (\(H_2OC\))
  - Kinematics & dynamics: Atmospheric Pressure (\(p\)), Air Density (\(\rho\)), Wind Velocity (\(w_v\)), Peak Wind Gust (\(w_{max}\)), Wind Direction (\(w_d\)), Cumulative Precipitation Depth (\(R\)), Active Rainfall Duration (\(R_{dur}\))

Most continuous variables are averaged daily; peak wind gust uses the daily maximum; precipitation-related variables are accumulated.

### Tourism Target
- **Source**: Thüringer Landesamt für Statistik – Beherbergungsstatistik (Table ID: ge000802)
- **Variable name in code**: `tourist_count`
- **Native frequency**: Monthly hospitality volume (guest arrivals / overnight stays)
- **Processing**: Converted to daily resolution via **monotonic cubic Hermite (PCHIP) spline interpolation**. This is a deliberate low-pass filter that produces a smooth latent macro-demand baseline free of daily operational noise.

### Temporal Coverage & Splits
Exact chronological date boundaries (not percentage cutoffs):

| Split       | Period                          | Days  | Percentage |
|-------------|----------------------------------|-------|------------|
| Training    | 2009-01-01 – 2014-12-07         | 2,167 | ≈74.2%     |
| Validation  | 2014-12-08 – 2015-09-25         | 292   | ≈10.0%     |
| Test        | 2015-09-26 – 2016-12-31         | 463   | ≈15.8%     |

**Total**: 2,922 daily observations (1 January 2009 – 31 December 2016).

### Important Notes on Data
This repository does **not** include the raw dataset files (`jena_climate.csv`, `tourism_thuringia.csv`). You must supply them yourself and point to them via the `raw_jena_path` / `tourism_csv_path` arguments of `run_experiment_pipeline()` in `main.py`.

The pipeline raises a `FileNotFoundError` if the files are missing (it no longer silently substitutes a synthetic target). A missing file will therefore fail loudly rather than produce misleading results.

## Preprocessing (`preprocess.py`)

1. Aggregate the raw 10-minute climate observations to daily frequency (16 variables).
2. Interpolate the monthly tourism registry to a daily series via monotonic cubic spline (PCHIP).
3. Merge the daily climate variables with the interpolated tourism target by date.
4. Sort observations chronologically.
5. Split the data by exact date boundaries (see table above).
6. Fit Z-score standardization statistics on the **training partition only**, then apply to all three partitions.
7. Generate supervised sliding windows for forecasting.

Default settings:
- Look-back sequence length (`SEQ_LEN`): **96**
- Forecast horizons: **24, 48, 96** days

## Model Architecture (`model.py`)

The implementation matches **Algorithm 1** of the manuscript and contains four principal components:

### 1. Climate-Aware Embedding
The target channel and the 16-dimensional climate vector are projected with **independent** learned linear layers (\(E_y\), \(E_w\)), then combined with sinusoidal positional encoding (\(E_p\)):

\[
E_t = E_y + E_w + E_p
\]

### 2. Fourier-Enhanced Block
Each scale’s representation is split into a moving-average trend component and a seasonal component. The seasonal component is transformed via real-valued FFT, retains only the top-\(k\) dominant frequency modes, and is reconstructed via inverse FFT. Trend and denoised seasonal components are summed.

### 3. Multi-Scale Forecasting Pipeline
The model constructs three resolution levels using non-overlapping temporal average pooling:
- Daily (scale factor 1)
- Weekly (scale factor 7)
- Seasonal (scale factor 30)

Each scale is encoded with a dedicated Transformer self-attention layer and processed by its own Fourier-enhanced block.

### 4. Iterative Refinement Decoder
Forecasts are produced coarse-to-fine:
1. Seasonal-scale forecast is generated first
2. Upsampled and fed into the weekly-scale decoder
3. Upsampled again and fed into the daily-scale decoder (primary output)

## Configuration

There is currently **no `config.json`**. Experimental settings live as module-level constants in `main.py`:

| Parameter                        | Value                          |
|----------------------------------|--------------------------------|
| Model                            | `ScaleformerFEDformerPipeline` |
| Target variable                  | `tourist_count`                |
| Input sequence length            | 96                             |
| Forecast horizons                | 24 / 48 / 96                   |
| Number of scales                 | 3 (daily / weekly / seasonal)  |
| Transformer layers per scale     | 3                              |
| Attention heads                  | 8                              |
| Model dimension                  | 512                            |
| Dropout                          | 0.1                            |
| Batch size                       | 32                             |
| Learning rate                    | \(1 \times 10^{-4}\)           |
| Training epochs                  | 100 (with early stopping)      |
| Early stopping patience          | 10                             |
| Independent runs                 | 30                             |
| Base random seed                 | 42                             |
| Multi-scale loss weight (\(\lambda\)) | 0.5                       |
| Frequency loss weight (\(\gamma\))    | 0.5                       |

## Reproducibility

Deterministic execution is enforced via:
- Explicit seeding of Python, NumPy, and PyTorch RNGs (`preprocess.enforce_determinism`)
- CuDNN deterministic settings (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`)

The experimental protocol uses 30 independent runs with seeds:
```text
42, 43, ..., 71
```

## Training

```bash
python main.py
```

This loops over the three forecast horizons (24, 48, 96) and, for each, executes 30 deterministic training runs with validation-based early stopping.

**Before running**, verify that:
1. `raw_jena_path` and `tourism_csv_path` in `main.py` point to your actual dataset files.
2. The required tourism and climate data are available locally.
3. CUDA is available if GPU execution is intended (scripts fall back to CPU automatically).
4. `torch` and `scipy` are installed (`scipy` is required for PCHIP interpolation).

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The target is evaluated from the final feature position of the multivariate window. The model’s raw output is univariate (the `tourist_count` forecast only).

## Checkpoints and Results

`main.py` writes a checkpoint every 10 runs (seeds 42, 52, 62) to a local `checkpoints/` directory (created automatically at runtime).

No pretrained checkpoints or result files are included in this repository.

## Citation

Please replace the placeholder below with the final bibliographic information before publication:

```bibtex
@article{matinkhah2026tourism,
  title   = {Mitigating Weather Anomalies in Tourism Forecasting via Multi-Scale Decomposed Transformers},
  author  = {Matinkhah, S. Mojtaba and Shahbazi, A.},
  journal = {<FINAL VENUE>},
  year    = {2026}
}
```

## License

This project is released under the MIT License.

## Contact

**Dr. S. Mojtaba Matinkhah**  
Email: [matinkhah@yazd.ac.ir](mailto:matinkhah@yazd.ac.ir)
```
