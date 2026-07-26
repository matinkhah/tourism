# Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting

Official implementation of the **Multi-Scale Frequency-Enhanced Model** for multivariate tourism demand forecasting using climate-related temporal features.

The proposed framework combines multi-scale temporal representations with Fourier-based frequency decomposition to model temporal dependencies and periodic patterns in tourism demand.

## Overview

The implementation contains the core components of the proposed forecasting framework:

* Multi-scale temporal representations
* Fourier-based frequency decomposition
* Climate-aware feature embedding
* Long-horizon multivariate forecasting
* Chronological train/validation/test splitting
* Training-set-only standardization
* Deterministic execution across multiple random seeds

The main model is implemented in `model.py`, while data preprocessing and temporal window construction are implemented in `preprocess.py`.

## Repository Structure

```text
tourism-forecasting/
├── main.py
├── model.py
├── preprocess.py
├── config.json
├── README.md
└── data/
    ├── jena_climate.csv
    └── tourism.csv
```

> The exact data files and directory structure should be updated to match the files actually released with this repository.

## Dataset

The forecasting pipeline combines daily climate information derived from the **Jena Climate Dataset** with a tourism-demand target.

The climate data are originally recorded at 10-minute resolution. The preprocessing pipeline aggregates the observations to daily frequency. Most continuous climate variables are averaged over each day, while maximum wind speed is aggregated using the daily maximum and precipitation-related variables are accumulated. 

The tourism target is represented by the variable:

```text
tourist_count
```

The climate and tourism data are aligned by date before the forecasting dataset is constructed. 

### Important

The repository should contain the actual dataset files, or provide precise instructions for obtaining and constructing them. The released code should not silently substitute synthetic tourism targets for missing data when reproducing the reported experiments.

## Preprocessing

The preprocessing pipeline performs the following operations:

1. Aggregate the original climate observations to daily frequency.
2. Merge the daily climate variables with tourism observations by date.
3. Sort observations chronologically.
4. Split the data chronologically into:

   * 70% training
   * 10% validation
   * 20% test
5. Fit standardization statistics using the training partition only.
6. Generate supervised sliding windows for forecasting. 

The current implementation uses a look-back sequence length of 96 and a forecasting horizon of 30 in `main.py`. These values must be kept consistent with the final manuscript and configuration file. 

## Model Architecture

The implementation contains three principal components.

### Climate-Aware Embedding

Input features are projected into the model's latent representation using a learned linear projection. 

### Fourier-Enhanced Block

The Fourier-enhanced block transforms the temporal representation into the frequency domain using the real-valued FFT, retains dominant frequency components, and reconstructs the selected periodic component using the inverse FFT. 

### Multi-Scale Forecasting Pipeline

The model constructs daily, weekly, and seasonal representations using temporal average pooling. Each scale is processed by the Fourier-enhanced block before being mapped back into the feature space and passed to the forecasting head. 

## Configuration

The configuration file specifies the principal experimental settings, including:

* Model architecture: `Scaleformer-FEDformerMS`
* Target: `tourist_count`
* Input sequence length: 96
* Number of scales: 3
* Batch size: 32
* Learning rate: `1e-4`
* Training epochs: 100
* Early stopping patience: 10
* Independent runs: 30
* Base random seed: 42 

Before publication, the configuration file should be validated as JSON and reconciled with the executable training script.

## Reproducibility

The implementation supports deterministic execution through explicit seeding of Python, NumPy, and PyTorch random-number generators and deterministic CuDNN settings. 

The experimental protocol specifies 30 independent runs using seeds:

```text
42, 43, ..., 71
```

The final repository should contain the exact scripts and configuration files required to reproduce the reported experiments.

## Training

Run the experiment using:

```bash
python main.py
```

Before running the experiment, verify that:

1. The dataset paths in `main.py` are correct.
2. The required tourism and climate data are available.
3. CUDA is available if GPU execution is intended.
4. The values in `config.json` agree with the values used by `main.py`.

## Evaluation

The implementation computes:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

The target variable is evaluated from the final feature position of the multivariate forecasting output.  

## Checkpoints and Results

Trained model checkpoints and experimental results should be added to the repository only after the corresponding files have been verified to reproduce the reported results.

The repository should not claim that trained checkpoints or complete reproduction artifacts are available unless those files are actually included.

## Citation

Please replace the following placeholder with the final bibliographic information before publication:

```bibtex
@article{matinkhah2026tourism,
  title={Multi-Scale Frequency-Enhanced Model for Tourism Demand Forecasting},
  author={Matinkhah, S. Mojtaba},
  journal={<FINAL VENUE>},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Contact

**Dr. S. Mojtaba Matinkhah**

Email: [matinkhah@yazd.ac.ir](mailto:matinkhah@yazd.ac.ir)
