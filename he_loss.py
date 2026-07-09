# he_loss.py
#
# Differentiable Haseman-Elston (HE) regression heritability loss for the
# joint reconstruction + H^2 training of engine_128_T1_cnn_h2joint.py.
#
# The math here is a vectorized, closed-form, autograd-friendly rewrite of
# AGENT/model/heritability_regression.py::he_regression_h2 (per-latent-dim
# univariate HE slope: regress y_c[i]*y_c[j] on GRM_ij over the upper
# triangle, no intercept, h2 = slope / var(y_c)), summed across latent dims
# to get H_total. Verified against that reference to float64 machine
# precision via the matrix identity:
#
#   sum_{i<j} K_ij y_i y_j = 0.5 * (y^T K y - sum_i K_ii y_i^2)
#
# which avoids ever materializing the O(N^2) pairwise cross-product tensor.

import math
import torch
from contextlib import contextmanager


@contextmanager
def _force_fp32(*device_types):
    """Disable AMP autocast for this block. Lightning's training_step/validation_step run
    inside an active autocast region under precision=16; torch.no_grad() does NOT disable
    autocast, so matmuls here would otherwise get silently cast to fp16 -- and
    torch.linalg.solve's LU factorization has no fp16/Half CUDA kernel (cusolver), which
    crashes with NotImplementedError. This loss math also wants full precision regardless."""
    seen = {dt for dt in device_types if dt in ("cuda", "cpu")}
    if not seen:
        seen = {"cuda"} if torch.cuda.is_available() else {"cpu"}
    with torch.autocast(device_type="cuda", enabled=False) if "cuda" in seen else _nullctx():
        yield


@contextmanager
def _nullctx():
    yield


def fit_covariate_beta(X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Closed-form OLS beta = (X^T X)^-1 X^T Z. Call under torch.no_grad()."""
    with _force_fp32(X.device.type):
        X32, Z32 = X.float(), Z.float()
        XtX = X32.t() @ X32
        XtZ = X32.t() @ Z32
        return torch.linalg.solve(XtX, XtZ)


def residualize(Z: torch.Tensor, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    with _force_fp32(Z.device.type):
        return Z.float() - X.float() @ beta.float()


def he_total_and_per_dim(
    Y: torch.Tensor,
    K: torch.Tensor,
    K_diag: torch.Tensor,
    he_denom: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Y:        (N, k) latents for every subject in the cohort half (covariate-residualized).
    K:        (N, N) GRM, arbitrary diagonal (GCTA self-relatedness, not assumed 1).
    K_diag:   (N,) diagonal of K.
    he_denom: scalar = 0.5 * (sum(K**2) - sum(K_diag**2)) == sum_{i<j} K_ij^2, precomputed once.

    Returns (h2_total: scalar, h2_dim: (k,)) matching he_regression_h2 per dim, summed.
    """
    with _force_fp32(Y.device.type):
        Y, K, K_diag, he_denom = Y.float(), K.float(), K_diag.float(), he_denom.float() if torch.is_tensor(he_denom) else he_denom

        y_c = Y - Y.mean(dim=0, keepdim=True)
        std = y_c.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        y_c = y_c / std

        KY = K @ y_c                                               # (N, k)
        diag_YtKY = (y_c * KY).sum(dim=0)                          # (k,)
        self_term = (K_diag.unsqueeze(1) * y_c.pow(2)).sum(dim=0)  # (k,)
        cross_sum = 0.5 * (diag_YtKY - self_term)                 # sum_{i<j} K_ij y_i y_j, (k,)

        slope = cross_sum / he_denom
        var_y = y_c.var(dim=0, unbiased=False).clamp_min(eps)      # ~1 after standardization
        h2_dim = slope / var_y
        return h2_dim.sum(), h2_dim


def lambda_schedule(epoch: int, warmup_epochs: int, ramp_epochs: int, lambda_target: float) -> float:
    """Phase 1 (warm-up): lambda=0. Phase 2 (ramp): linear 0 -> target. Phase 3 (steady): target."""
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return lambda_target
    if epoch < warmup_epochs + ramp_epochs:
        frac = (epoch - warmup_epochs + 1) / float(ramp_epochs)
        return lambda_target * min(1.0, frac)
    return lambda_target


def lambda_schedule_cosine(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    lambda_target: float,
    cycle_epochs: int = 15,
) -> float:
    """Cosine-annealing lambda schedule to prevent proxy gaming.

    Phase 1 (warmup): lambda = 0.
    Phase 2 (fast ramp): linear 0 → lambda_target over ramp_epochs.
    Phase 3 (cosine cycles): SGDR-style decay within each cycle of length
      cycle_epochs: lambda decays from lambda_target → 0 following a cosine,
      then restarts at lambda_target for the next cycle. The periodic drop to 0
      forces reconstruction recovery and prevents the model from sustaining
      a gaming solution against the training GRM.
    """
    if epoch < warmup_epochs:
        return 0.0
    t = epoch - warmup_epochs
    if t < ramp_epochs:
        frac = (t + 1) / float(max(ramp_epochs, 1))
        return lambda_target * min(1.0, frac)
    t_post_ramp = t - ramp_epochs
    phase = (t_post_ramp % cycle_epochs) / float(cycle_epochs)
    return lambda_target * 0.5 * (1.0 + math.cos(math.pi * phase))
