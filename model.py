# model.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Three-level temporal pyramid (K = 2 -> daily / weekly / seasonal),
# Section III-C.
SCALE_NAMES = ("seasonal", "weekly", "daily")   # coarse -> fine, matches
                                                 # the refinement order of
                                                 # Eq. (13)-(14)
DOWNSAMPLE_FACTORS = {"daily": 1, "weekly": 7, "seasonal": 30}


class PositionalEmbedding(nn.Module):
    """Standard sinusoidal positional encoding -> E_p in Equation (21)."""

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, length):
        return self.pe[:, :length, :]


class ClimateAwareEmbedding(nn.Module):
    """
    Climate feature embedding of Section III-G / Equations (19)-(21).
    The target channel y_t and the 16-dim climate vector w_t are
    projected with independent learnable weights (E_w = W_t^T W_e + b_e,
    E_y = Y_t W_y + b_y) and combined with a positional encoding:
        E_t = E_y + E_w + E_p
    """

    def __init__(self, n_climate_vars, d_model):
        super().__init__()
        self.climate_proj = nn.Linear(n_climate_vars, d_model)   # W_e, b_e
        self.target_proj = nn.Linear(1, d_model)                  # W_y, b_y
        self.pos_embedding = PositionalEmbedding(d_model)

    def forward(self, x_raw):
        # x_raw: (B, T, c_in) where the LAST channel is the target y_t
        # (see preprocess.generate_sliding_windows OT convention).
        w_t = x_raw[..., :-1]
        y_t = x_raw[..., -1:]

        e_w = self.climate_proj(w_t)
        e_y = self.target_proj(y_t)
        e_p = self.pos_embedding(x_raw.shape[1])

        return e_y + e_w + e_p  # E_t, Equation (21)


class FourierEnhancedBlock(nn.Module):
    """
    FEDformer-style Frequency Enhanced Block (Section III-F).
    Splits an input scale representation R_s into:
      - a trend component T_s via moving-average decomposition
        (Eq. 17's T_t^s term), and
      - a denoised seasonal component S_hat via FFT -> top-k dominant
        mode selection sigma_k(.) -> IFFT (Eq. 15-16),
    then fuses them: Z_s = T_s + S_hat (Eq. 17). Because sigma_k prunes
    non-dominant modes, S_hat != S and therefore Z_s != R_s, which is
    the identity-collapse guard described directly under Equation (17).
    """

    def __init__(self, d_model, top_k=8, trend_kernel=25):
        super().__init__()
        self.top_k = top_k
        self.trend_kernel = trend_kernel

    def _moving_average_trend(self, x):
        # x: (B, T, D) -> centered moving average along the time axis
        k = min(self.trend_kernel, x.shape[1])
        if k % 2 == 0:
            k += 1
        pad = k // 2
        x_t = x.transpose(1, 2)  # (B, D, T)
        trend = F.avg_pool1d(
            F.pad(x_t, (pad, pad), mode="replicate"), kernel_size=k, stride=1
        )
        return trend.transpose(1, 2)

    def forward(self, r_s):
        # r_s: (B, T, D)
        t_s = self._moving_average_trend(r_s)
        seasonal_raw = r_s - t_s

        # FFT along the time dimension
        xf = torch.fft.rfft(seasonal_raw, dim=1)

        # sigma_k(.): retain only the top-k dominant frequency modes,
        # ranked by amplitude averaged over batch and channel (Eq. 15)
        k = min(self.top_k, xf.shape[1])
        amplitude = torch.abs(xf).mean(dim=(0, 2))
        _, top_idx = torch.topk(amplitude, k)

        xf_refined = torch.zeros_like(xf)
        xf_refined[:, top_idx, :] = xf[:, top_idx, :]

        s_hat = torch.fft.irfft(xf_refined, n=r_s.shape[1], dim=1)  # Eq. (16)
        z_s = t_s + s_hat  # Eq. (17)
        return z_s, s_hat, t_s


class ScaleEncoder(nn.Module):
    """Self-attention encoder H^(k) = Encoder(X^(k)), Equations (11)-(12)."""

    def __init__(self, d_model, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


class ScaleformerFEDformerPipeline(nn.Module):
    """
    Full climate-aware multi-scale frequency-domain forecaster
    (Algorithm 1). c_in must equal 17: 16 meteorological channels
    (Eq. 3) + 1 tourism target channel, appended last per the OT
    convention used by preprocess.generate_sliding_windows.
    """

    def __init__(
        self,
        c_in=15,
        seq_len=96,
        pred_len=24,
        d_model=512,
        n_heads=8,
        n_layers=2,
        dropout=0.1,
        top_k=8,
    ):
        super().__init__()
        assert c_in >= 2, "c_in must include the 16 climate channels + 1 target"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # 1. Climate-aware multivariate input embedding module (Eq. 19-21)
        self.embedding = ClimateAwareEmbedding(n_climate_vars=c_in - 1, d_model=d_model)

        # Per-scale self-attention encoders (Eq. 11-12)
        self.encoders = nn.ModuleDict(
            {s: ScaleEncoder(d_model, n_heads, n_layers, dropout) for s in SCALE_NAMES}
        )

        # 3. Frequency-enhanced decomposition module (Eq. 15-18)
        self.febs = nn.ModuleDict(
            {s: FourierEnhancedBlock(d_model, top_k=top_k) for s in SCALE_NAMES}
        )

        # 4. Hierarchical forecasting decoder: one linear decoder per
        # scale that consumes [H^(k) ; Up(Y_hat^(k+1))] and maps it back
        # to the target's forecast horizon (Eq. 13-14).
        self.decoders = nn.ModuleDict(
            {s: nn.Linear(2 * d_model, 1) for s in SCALE_NAMES[1:]}
        )
        self.coarsest_decoder = nn.Linear(d_model, 1)

        # Final projection of the fine-scale refined sequence to the
        # forecast horizon H (Algorithm 1, Step 5 / Eq. 30-equivalent).
        self.horizon_heads = nn.ModuleDict(
            {
                s: nn.Linear(self._scale_len(s), pred_len)
                for s in SCALE_NAMES
            }
        )

    def _scale_len(self, scale_name):
        # Must match the output length produced by F.avg_pool1d(...,
        # kernel=factor, stride=factor, ceil_mode=True) in _downsample:
        # for non-overlapping windows (kernel == stride) that length is
        # ceil(seq_len / factor), NOT floor(seq_len / factor).
        factor = DOWNSAMPLE_FACTORS[scale_name]
        return max(1, math.ceil(self.seq_len / factor))

    @staticmethod
    def _downsample(x, factor):
        # X^(k) = D_k(X^(k-1)), Equation (10): non-overlapping average
        # pooling that shrinks the temporal resolution.
        if factor == 1:
            return x
        x_t = x.transpose(1, 2)  # (B, D, T)
        pooled = F.avg_pool1d(x_t, kernel_size=factor, stride=factor, ceil_mode=True)
        return pooled.transpose(1, 2)

    @staticmethod
    def _upsample(y_hat, target_len):
        # Up(.) temporal upsampling used in the iterative refinement
        # decoder, Equation (14).
        y_t = y_hat.transpose(1, 2)  # (B, 1, T_coarse)
        up = F.interpolate(y_t, size=target_len, mode="linear", align_corners=False)
        return up.transpose(1, 2)

    def forward(self, x_raw):
        # Step 1: Input Fusion & Feature Embedding (Algorithm 1, L1-4)
        e0 = self.embedding(x_raw)  # (B, L, d_model)

        # Step 2: Multi-Scale Transformation / resolution pyramid
        # (Algorithm 1, L6-9; Eq. 9-10)
        pyramid = {
            s: self._downsample(e0, DOWNSAMPLE_FACTORS[s]) for s in SCALE_NAMES
        }

        # Step 3: FEDformer frequency decomposition per scale
        # (Algorithm 1, L11-21; Eq. 15-18)
        z, s_hat, t_s = {}, {}, {}
        for s in SCALE_NAMES:
            h_s = self.encoders[s](pyramid[s])          # Eq. (12)
            z[s], s_hat[s], t_s[s] = self.febs[s](h_s)   # Eq. (15)-(17)

        # Step 4: Scaleformer iterative refinement, coarse -> fine
        # (Algorithm 1, L23-27; Eq. 13-14)
        y_hat_scale = {}
        coarse_name = SCALE_NAMES[0]  # "seasonal"
        y_prev = self.coarsest_decoder(z[coarse_name])  # Eq. (13)
        y_hat_scale[coarse_name] = y_prev

        for s in SCALE_NAMES[1:]:
            up_prev = self._upsample(y_prev, z[s].shape[1])
            up_prev_proj = up_prev.expand(-1, -1, self.d_model)
            decoder_in = torch.cat([z[s], up_prev_proj], dim=-1)
            y_cur = self.decoders[s](decoder_in)         # Eq. (14)
            y_hat_scale[s] = y_cur
            y_prev = y_cur

        # Step 5: Final decoding to the forecast horizon, per scale,
        # then the fine (daily) scale is the model's primary output.
        horizon_outputs = {}
        for s in SCALE_NAMES:
            seq = y_hat_scale[s].squeeze(-1)              # (B, T_s)
            horizon_outputs[s] = self.horizon_heads[s](seq).unsqueeze(-1)  # (B, H, 1)

        fine_out = horizon_outputs[SCALE_NAMES[-1]]  # (B, pred_len, 1)

        # Returned alongside the coarser-scale forecasts and the
        # seasonal/trend decomposition so that the training loop can
        # compute the full multi-scale + frequency loss of Eq. (22)-(25).
        aux = {
            "scale_forecasts": horizon_outputs,  # for L_scale, Eq. (24)
            "seasonal": s_hat,                   # for L_freq, Eq. (25)
            "trend": t_s,
        }
        return fine_out, aux
