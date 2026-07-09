# h2joint — Heritability-Joint Brain Encoder

Joint training of a 3D CNN autoencoder with a coordinate-free heritability
objective on T1 brain MRI (128³ voxels).

## Goal

Maximize `H_total = tr(P⁻¹G)` — the sum of heritabilities across all latent
dimensions — while preserving reconstruction quality.  The result is a 128-dim
latent `z` where genetically similar subjects have similar representations.

## Files

| File | Description |
|---|---|
| `engine_128_T1_cnn_h2joint_A.py` | **Option A** — maximize H_total over full 128-dim z |
| `engine_128_T1_cnn_h2joint_C.py` | **Option C** — split z 64+64; push z_h heritable, z_r non-heritable |
| `engine_128_T1_cnn_h2joint.py` | Original prototype (run 1–3) |
| `he_loss.py` | Covariate residualization, per-dimension HE utilities |
| `he_loss_svd.py` | `he_total_from_mixed_y` — full-population mixed-y estimator |
| `cohort_build_h2.py` | Build/cache GRM, covariate matrix, train/val split |
| `dataset.py` | PyTorch dataset for 3D MRI with EID lookup |
| `run_idp_pipeline_king_hereg.py` | GCTA HEreg pipeline (official val h² eval) |

## Training loss

**Option A:**
```
L = L_recon(z)  −  λ · H_total(z)
```

**Option C:**
```
L = L_recon(z)  −  λ · H_total(z_h)  +  λ · H_total(z_r)
```

## H_total estimator — `he_total_from_mixed_y`

1. Periodic SVD of the full-N queue → whitening basis V, S (detached)
2. Whiten current batch: `w_B = z_resid_B @ V / S`  (differentiable)
3. Build mixed W (N×m): queue rows = detached L_m; batch rows = live w_B
4. `KW = K @ W`  — genetic-neighbourhood-weighted phenotype
5. `cross_sum = 0.5 · (tr(WᵀKW) − Σᵢ Kᵢᵢ‖Wᵢ‖²)` — HE numerator over N(N−1)/2 pairs
6. `H_total = (N−c) · cross_sum / Σᵢ<ⱼ Kᵢⱼ²`

## Lambda schedule (run 8, current)

| Phase | Epochs | λ |
|---|---|---|
| Warmup | 0–1 | 0.0 |
| Linear ramp | 2–6 | 0.004 → 0.020 |
| Fixed | 7+ | 0.020 |

## Run history (summary)

| Run | Key change | A val h² | C val h² |
|---|---|---|---|
| run1 | λ=0, baseline | — | — |
| run2–3 | batch-cross estimator (biased) | gamed / crashed | — |
| run4–5 | mixed-y, proj_head bug | ~12 | ~17 |
| run6 | no proj_head, SVD_MODES=16 | 35→31 ↓ | 35→31 ↓ |
| run7 | SVD_MODES=64, λ=0.002 cosine | plateau 31.4 | plateau 30.6 |
| **run8** | λ=0.02 fixed | **34.5 (ep9)** | 32.3 (ep9) ↓ |
