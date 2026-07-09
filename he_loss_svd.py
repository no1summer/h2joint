# he_loss_svd.py
#
# Coordinate-free multivariate HE regression:  H_total = tr(P^{-1} G)
# as derived in Heritability Optimization.md.
#
# Replaces the per-dimension sum Σ_k h²_k used in he_loss.py.  That sum is
# equivalent to tr(G) / tr(P) (approximately), which is coordinate-dependent
# and gameable: the model can inflate G by rotating latents onto GRM
# eigenvectors without genuinely increasing heritability.
#
# tr(P^{-1} G) is coordinate-free (invariant to invertible linear remap of
# the latent space) and properly normalises genetic variance by phenotypic
# variance in each direction.  Combined with a reconstruction loss that keeps
# P full-rank, the model must genuinely increase G relative to P.
#
# Key identity (§5 of the doc):
#   Y = L Σ V^T  (compact SVD of the N×k residualised latent queue)
#   Y P^{-1} Y^T = (N-c) L L^T
#   → R_ij = y_i^T P^{-1} y_j = (N-c) L_{m,i} · L_{m,j}
#
# Single HE regression of R_ij on K_ij gives slope = H_total.  The intercept
# correction vanishes because Σ_{i<j} R_ij = 0 when L_m is column-orthonormal.
#
# Gradient path (§6):
#   z_new → Δ → (C_R, R_R via QR of E_R) → B → K_small → SVD → U_K_m
#         → L_m_plus → H_total
#
# The left-side objects (C_L, Q_L, R_L) involve only detached buffers and the
# fixed selection matrix S_B, so they carry no gradient.

import torch
from contextlib import contextmanager


# ── fp32 guard (same rationale as he_loss.py) ─────────────────────────────────

@contextmanager
def _fp32(device_type):
    if device_type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    else:
        yield


# ── H_total from retained SVD modes ───────────────────────────────────────────

def he_total_from_L(L_m, K, K_diag, he_denom, n_subjects, n_covar, eps=1e-8):
    """
    H_hat = (N-c) * Σ_{i<j} K_ij (L_{m,i}·L_{m,j}) / he_denom
          = (N-c) * 0.5 * (tr(L_m^T K L_m) - Σ_i K_ii ‖L_{m,i}‖²) / he_denom

    L_m:      (N, m) left singular modes of residualised queue — orthonormal cols
    K:        (N, N) GRM (training split)
    K_diag:   (N,)   diagonal of K
    he_denom: scalar Σ_{i<j} K_ij²  (precomputed once in cohort_build_h2)
    n_covar:  number of covariate columns (for the (N-c) factor)

    Returns differentiable scalar.  Main cost: O(N² m) for K @ L_m.
    """
    with _fp32(L_m.device.type):
        L_m    = L_m.float()
        K      = K.float()
        K_diag = K_diag.float()

        KL        = K @ L_m                                      # (N, m)
        tr_LtKL   = (L_m * KL).sum()                            # scalar
        self_term = (K_diag * (L_m ** 2).sum(dim=1)).sum()      # scalar
        cross_sum = 0.5 * (tr_LtKL - self_term)
        return (n_subjects - n_covar) * cross_sum / (he_denom.float() + eps)


# ── Differentiable rank-b SVD update ──────────────────────────────────────────

def rank_b_svd_update(L_m, S_m, V_m, Delta, batch_idx, N):
    """
    Rank-b update of the stored low-rank SVD  Y ≈ L_m diag(S_m) V_m^T
    given that rows batch_idx changed by Delta = Y_new - stopgrad(Y_old).

    Gradient flows:
      Delta → C_R (= V_m^T Δ^T) → E_R (= Δ^T - V_m C_R) → QR(E_R) → R_R
            → B (= cat[C_R, R_R]) → K_small (= diag(S_m) ⊕ 0 + A B^T)
            → SVD(K_small) → U_K[:,:m] → L_m_plus (= cat[L_m,Q_L] @ U_K_m)

    The left-side objects (C_L, Q_L, R_L) are detached (no Δ dependence).

    Args
    ----
    L_m:       (N, m)  detached left singular modes
    S_m:       (m,)    detached singular values
    V_m:       (k, m)  detached right singular modes
    Delta:     (b, k)  differentiable row update  Y_new - stopgrad(Y_old)
    batch_idx: (b,)    long tensor — rows being updated
    N:         int     total subjects in cohort

    Returns
    -------
    L_m_plus:  (N, m)  updated left modes  [gradient w.r.t. Delta]
    S_m_plus:  (m,)    updated singular values
    V_m_plus:  (k, m)  updated right modes
    """
    with _fp32(Delta.device.type):
        L_m   = L_m.float().detach()
        S_m   = S_m.float().detach()
        V_m   = V_m.float().detach()
        Delta = Delta.float()

        b, k = Delta.shape
        m    = L_m.shape[1]
        dev  = Delta.device

        # ── Left side: no gradient (S_B and L_m are fixed) ──────────────────
        with torch.no_grad():
            C_L  = L_m[batch_idx].T                     # (m, b)
            E_L  = -(L_m @ C_L)                         # (N, b)
            # scatter +1 at the diagonal positions (one 1 per column j at row idx_j)
            E_L[batch_idx, torch.arange(b, device=dev)] += 1.0
            Q_L, R_L = torch.linalg.qr(E_L, mode='reduced')   # (N,b), (b,b)

        # ── Right side: gradient flows through Delta ──────────────────────────
        C_R = V_m.T @ Delta.T                           # (m, b)
        E_R = Delta.T - V_m @ C_R                       # (k, b)
        Q_R, R_R = torch.linalg.qr(E_R, mode='reduced')       # (k,b), (b,b)

        # ── Small (m+b)×(m+b) matrix ─────────────────────────────────────────
        A = torch.cat([C_L, R_L.detach()], dim=0)      # (m+b, b)  no grad
        B = torch.cat([C_R, R_R],          dim=0)      # (m+b, b)  grad from Delta

        K_s = torch.zeros(m + b, m + b, device=dev, dtype=torch.float32)
        K_s[:m, :m] = torch.diag(S_m)
        K_s = K_s + A @ B.T                            # (m+b, m+b)

        # ── Top-m SVD of K_s ─────────────────────────────────────────────────
        U_K, S_K, Vh_K = torch.linalg.svd(K_s, full_matrices=False)
        U_K_m  = U_K[:, :m]          # (m+b, m)
        S_K_m  = S_K[:m]             # (m,)
        V_K_m  = Vh_K[:m, :].T       # (m+b, m)

        # ── Updated L, S, V ──────────────────────────────────────────────────
        LQ = torch.cat([L_m, Q_L.detach()], dim=1)    # (N, m+b)  no grad
        L_m_plus = LQ @ U_K_m                          # (N, m)  grad via U_K_m

        VQ = torch.cat([V_m, Q_R.detach()], dim=1)    # (k, m+b)
        V_m_plus = VQ @ V_K_m                          # (k, m)

        return L_m_plus, S_K_m, V_m_plus


# ── Full-population mixed-y HE estimator ─────────────────────────────────────
#
# The user's key insight: the phenotype matrix Y should include ALL N subjects,
# not just the minibatch.  For the b batch subjects, use their NEW (differentiable)
# z values; for the remaining N-b queue subjects, use their OLD (detached) values.
# Then run the standard full HE regression over all N*(N-1)/2 pairs.
#
# This fixes two problems with the pure batch-cross approach:
#   1. Numerator and denominator both cover the same N*(N-1)/2 pairs → valid slope
#   2. Gradient for batch subject i = (N-c)/denom * Σ_{j≠i} K_ij W_j uses ALL
#      N-1 other subjects, not just the b in the batch.
#
# Construction of W (the full whitened phenotype matrix, N×m):
#   W[j] = L_m[j]                   for j not in batch  (detached queue SVD row)
#   W[i] = Σ^{-1} V^T z_new_i       for i in batch      (differentiable)
#
# The out-of-place index_put creates a new tensor whose grad_fn is IndexPutBackward,
# so autograd correctly propagates ∂H/∂W[batch_idx] → ∂H/∂z_new_B.

def he_total_from_mixed_y(z_resid_B, batch_idx, V_m, S_m, L_m,
                           K, K_diag, he_denom, n_subjects, n_covar, eps=1e-8):
    """
    H_total = tr(P^{-1}G) using full N phenotype vector:
      - batch rows:  current differentiable z_resid_B (whitened via stable V, S)
      - other rows:  detached L_m rows from the periodic queue SVD

    Numerator and denominator both cover all N*(N-1)/2 pairs → valid OLS slope.
    Gradient = (N-c)/denom * Σ_{j≠i} K_ij W_j  for each batch subject i.

    z_resid_B: (b, k) residualised latents  — DIFFERENTIABLE
    batch_idx: (b,)   long tensor of training indices for this batch
    V_m:       (k, m) right singular modes  — detached
    S_m:       (m,)   singular values       — detached
    L_m:       (N, m) left singular modes   — detached
    K:         (N, N) GRM                   — detached
    K_diag:    (N,)   diagonal of K         — detached
    he_denom:  scalar Σ_{i<j} K_ij²        — detached
    """
    with _fp32(z_resid_B.device.type):
        V = V_m.float().detach().clone()
        S = S_m.float().detach().clone().clamp_min(eps)

        # Whitened batch representations (differentiable)
        w_B = (z_resid_B.float() @ V) / S.unsqueeze(0)    # (b, m)

        # Full whitened matrix: queue rows everywhere, batch rows replaced.
        # index_put (out-of-place) preserves gradient through w_B.
        W_base = L_m.float().detach()                       # (N, m)  no grad
        W = W_base.index_put((batch_idx,), w_B)             # (N, m)  grad via w_B

        # H_total = (N-c) * 0.5 * (tr(WᵀKW) - Σ_i K_ii‖W_i‖²) / he_denom
        KW        = K.float() @ W                           # (N, m)
        tr_WtKW   = (W * KW).sum()                          # scalar
        self_term = (K_diag.float() * (W ** 2).sum(dim=1)).sum()
        cross_sum = 0.5 * (tr_WtKW - self_term)
        return (n_subjects - n_covar) * cross_sum / (he_denom.float() + eps)


# ── Periodic full SVD refresh ──────────────────────────────────────────────────

def full_svd_refresh(Y_resid, m):
    """
    Full thin SVD of Y_resid (N, k) → top-m (L, S, V). All detached.
    Used after every covariate beta refit and every SVD_REFRESH_EVERY steps
    to correct accumulated floating-point drift in the incremental updates.
    """
    with torch.no_grad():
        Y32 = Y_resid.float()
        U, S, Vh = torch.linalg.svd(Y32, full_matrices=False)
        L = U[:, :m].contiguous()
        S = S[:m].contiguous()
        V = Vh[:m, :].T.contiguous()          # (k, m)
    return L, S, V
