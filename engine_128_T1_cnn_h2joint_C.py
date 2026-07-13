# engine_128_T1_cnn_h2joint_C.py  — Option C
#
# Disentangled latent: the 128-dim bottleneck z is split into
#   z_h  (first 64 dims)  — pushed to be HERITABLE    (maximise H_total)
#   z_r  (last  64 dims)  — pushed to be NON-HERITABLE (minimise H_total)
#
# The decoder receives the full concat(z_h, z_r) = z (128-dim), unchanged.
# Reconstruction is therefore unaffected by the split; the network is free to
# partition brain variation between heritable and residual axes.
#
# Loss:
#   L = L_recon  -  lambda_h(t) * H_total(z_h)  +  lambda_r(t) * H_total(z_r)
#
# Both lambda_h and lambda_r follow the same SGDR cosine schedule so that the
# heritable/residual split is applied symmetrically and periodically relaxed
# (preventing either side from gaming its proxy indefinitely).
#
# H_total is computed via the SVD-based estimator (he_loss_svd.py):
#   H_total = tr(P^{-1}G), coordinate-free, bounded by SVD_MODES.
# Two independent queues + SVD systems are maintained (one per half).

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataset import aedataset, transforms_monai
import he_loss
import he_loss_svd
from cohort_build_h2 import build_or_load as build_or_load_cohort, CACHE_DIR

_AGENT_IDP_DIR = "/data484_4/txia2/AGENT/experiments/IDP"
if _AGENT_IDP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_IDP_DIR)
from run_idp_pipeline_king_hereg import run_hereg_on_features, KinPreset  # noqa: E402

# ── paths & hyperparameters ───────────────────────────────────────────────────

PRETRAINED_CKPT = "/data484_4/txia2/DeepENDO/training/T1_128/epoch=39-train_loss=0.265290-val_loss=0.291595.ckpt"
DIR_NAME    = "/data484_4/txia2/h2joint/output/cnn_h2joint_C"
GCTA_BIN    = "/data4012/zxie3/gcta/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1"
CCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_ccovar_discovery"
QCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_qcovar_discovery"

LEARNING_RATE = 0.0005248074602497723
BATCH_SIZE    = 18
HIDDEN_DIM    = 128       # total encoder bottleneck; split 64+64
HALF_DIM      = 64        # z_h and z_r each have this many dims
SVD_MODES     = 64        # retained SVD modes per half; H_total bounded by this; captures full 64-dim half-space

WARMUP_EPOCHS       = 2
RAMP_EPOCHS         = 5
LAMBDA_TARGET       = 0.02   # run8: 10× higher, fixed (no cosine)
COSINE_CYCLE_EPOCHS = 20     # unused in run8
COVAR_REFIT_EVERY   = 200
GCTA_THREADS        = 8
GCTA_PARALLEL_JOBS  = 8


# ── dataset ───────────────────────────────────────────────────────────────────

class aedataset_with_eid(torch.utils.data.Dataset):
    def __init__(self, datafile, modality, transforms):
        self.base = aedataset(datafile=datafile, modality=modality, transforms=transforms)
        self.eids = pd.read_csv(datafile)["eid"].astype(str).tolist()

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, mask = self.base[idx]
        return img, mask, self.eids[idx]


# ── lightning module ──────────────────────────────────────────────────────────

class engine_AE_H2Joint_C(pl.LightningModule):
    def __init__(self, lr, train_eids, val_eids,
                 train_grm, train_grm_diag, train_he_denom, train_X,
                 val_grm,   val_grm_diag,   val_he_denom,   val_X):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "train_eids","val_eids","train_grm","train_grm_diag","train_he_denom","train_X",
            "val_grm","val_grm_diag","val_he_denom","val_X",
        ])
        self.hidden_dim = HIDDEN_DIM

        # ── encoder (identical to engine_128_T1_cnn.py) ──────────────────────
        self.first_cnn        = self.first_CNN_block(1, 16)
        self.first_max_poold  = self.max_poold((1,1,1))
        self.first_encoder    = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0,1,0))
        self.second_encoder   = self.encoder_block(32, 64)
        self.third_max_poold  = self.max_poold((1,0,1))
        self.third_encoder    = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0,0,0))
        self.fourth_encoder   = self.encoder_block(128, 256)
        self.encoding_mlp     = nn.Linear(256*12*14*12, HIDDEN_DIM)  # outputs 128; split below

        # ── decoder (identical to engine_128_T1_cnn.py) ──────────────────────
        self.decoding_mlp     = nn.Linear(HIDDEN_DIM, 256*12*14*12)  # takes full 128
        self.first_decoder    = self.decoder_block(256, 128)
        self.first_transconv  = self.conv_transpose(128, (0,0,0))
        self.second_decoder   = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(64,  (1,0,1))
        self.third_decoder    = self.decoder_block(64,  32)
        self.third_transconv  = self.conv_transpose(32,  (0,1,0))
        self.fourth_decoder   = self.decoder_block(32,  16)
        self.fourth_transconv = self.conv_transpose(16,  (1,1,1))
        self.last_cnn         = self.last_CNN_block(16,  1)
        self.recon_loss_fn    = nn.MSELoss(reduction="none")

        # ── cohort mappings ───────────────────────────────────────────────────
        self.train_eid_to_idx = {e: i for i, e in enumerate(train_eids)}
        self.val_eid_to_idx   = {e: i for i, e in enumerate(val_eids)}
        n_train, n_val = len(train_eids), len(val_eids)
        n_covar = train_X.shape[1]
        self.n_train = n_train
        self.n_covar = n_covar

        # ── non-persistent buffers: TWO SVD systems (heritable + residual) ───
        for half in ("h", "r"):
            self.register_buffer(f"queue_{half}",     torch.zeros(n_train, HALF_DIM), persistent=False)
            self.register_buffer(f"beta_{half}",      torch.zeros(n_covar, HALF_DIM), persistent=False)
            self.register_buffer(f"svd_L_{half}",     torch.zeros(n_train, SVD_MODES), persistent=False)
            self.register_buffer(f"svd_S_{half}",     torch.ones(SVD_MODES),           persistent=False)
            self.register_buffer(f"svd_V_{half}",     torch.zeros(HALF_DIM, SVD_MODES),persistent=False)
            self.register_buffer(f"svd_ready_{half}", torch.tensor(False),             persistent=False)

        self.register_buffer("queue_filled", torch.zeros(n_train, dtype=torch.bool), persistent=False)

        # GRM / covariate
        self.register_buffer("train_grm",      torch.as_tensor(train_grm,     dtype=torch.float32), persistent=False)
        self.register_buffer("train_grm_diag", torch.as_tensor(train_grm_diag,dtype=torch.float32), persistent=False)
        self.register_buffer("train_he_denom", torch.as_tensor(float(train_he_denom), dtype=torch.float32), persistent=False)
        self.register_buffer("train_X",        torch.as_tensor(train_X,       dtype=torch.float32), persistent=False)

        # validation — store full z (128-dim) so GCTA sees all features → baseline ~35
        self.register_buffer("val_queue_full", torch.zeros(n_val, HIDDEN_DIM), persistent=False)
        self.register_buffer("val_grm",        torch.as_tensor(val_grm,      dtype=torch.float32), persistent=False)
        self.register_buffer("val_grm_diag",   torch.as_tensor(val_grm_diag, dtype=torch.float32), persistent=False)
        self.register_buffer("val_he_denom",   torch.as_tensor(float(val_he_denom), dtype=torch.float32), persistent=False)
        self.register_buffer("val_X",          torch.as_tensor(val_X,        dtype=torch.float32), persistent=False)

    # ── architecture helpers ──────────────────────────────────────────────────

    def max_poold(self, p): return nn.MaxPool3d(kernel_size=2, padding=p)
    def encoder_block(self, ic, oc, pad=1):
        return nn.Sequential(
            nn.Conv3d(ic, oc, 3, padding=pad), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
            nn.Conv3d(oc, oc, 3, padding=pad), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
        )
    def conv_transpose(self, oc, ip):
        return nn.ConvTranspose3d(oc, oc, 2, stride=2, padding=ip)
    def decoder_block(self, ic, oc):
        return nn.Sequential(
            nn.Conv3d(ic, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
            nn.Conv3d(oc, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
        )
    def first_CNN_block(self, ic, oc, pad=1):
        return nn.Sequential(
            nn.Conv3d(ic, oc, 3, padding=pad), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
            nn.Conv3d(oc, oc, 3, padding=pad), nn.BatchNorm3d(oc), nn.LeakyReLU(inplace=False),
        )
    def last_CNN_block(self, ic, oc, pad=1):
        return nn.Sequential(
            nn.Conv3d(ic, ic, 3, padding=pad), nn.BatchNorm3d(ic), nn.LeakyReLU(inplace=False),
            nn.Conv3d(ic, ic, 3, padding=pad), nn.BatchNorm3d(ic), nn.LeakyReLU(inplace=False),
            nn.Conv3d(ic, oc, 1),
        )

    def forward(self, x):
        x = self.first_cnn(x)
        x = self.first_max_poold(x)
        x = self.first_encoder(x)
        x = self.second_max_poold(x)
        x = self.second_encoder(x)
        x = self.third_max_poold(x)
        x = self.third_encoder(x)
        x = self.fourth_max_poold(x)
        x = self.fourth_encoder(x)
        enc = torch.flatten(x, 1)
        z   = self.encoding_mlp(enc)          # (B, 128)
        dec = self.decoding_mlp(z).view(x.size())
        dec = self.first_transconv(self.first_decoder(dec))
        dec = self.second_transconv(self.second_decoder(dec))
        dec = self.third_transconv(self.third_decoder(dec))
        dec = self.fourth_transconv(self.fourth_decoder(dec))
        recon = self.last_cnn(dec)
        return recon, z

    # ── schedule ──────────────────────────────────────────────────────────────

    def current_lambda(self):
        ep = self.current_epoch
        if ep < WARMUP_EPOCHS:
            return 0.0
        ramp_ep = ep - WARMUP_EPOCHS
        if ramp_ep < RAMP_EPOCHS:
            return LAMBDA_TARGET * (ramp_ep + 1) / RAMP_EPOCHS
        return LAMBDA_TARGET  # fixed; no cosine cycling

    # ── SVD helpers ───────────────────────────────────────────────────────────

    def _refresh_svd(self, half):
        """Refit beta and full SVD refresh for one half ('h' or 'r')."""
        queue = getattr(self, f"queue_{half}")
        with torch.no_grad():
            beta = he_loss.fit_covariate_beta(self.train_X, queue)
            setattr(self, f"beta_{half}", beta)
            Y_resid = queue.float() - self.train_X.float() @ beta.float()
            L, S, V = he_loss_svd.full_svd_refresh(Y_resid, SVD_MODES)
            getattr(self, f"svd_L_{half}").copy_(L.to(self.device))
            getattr(self, f"svd_S_{half}").copy_(S.to(self.device))
            getattr(self, f"svd_V_{half}").copy_(V.to(self.device))
            getattr(self, f"svd_ready_{half}").fill_(True)

    def _he_for_half(self, z_half, idx_t, half):
        """
        Full-population mixed-y H_total for one latent half.
        Batch rows = new differentiable z; all other rows = detached queue L_m.
        Returns H_total scalar (differentiable).
        """
        beta  = getattr(self, f"beta_{half}")
        svd_V = getattr(self, f"svd_V_{half}")
        svd_S = getattr(self, f"svd_S_{half}")
        svd_L = getattr(self, f"svd_L_{half}")

        # Residualise current batch (differentiable)
        z_new = he_loss.residualize(z_half, self.train_X[idx_t], beta)

        h2 = he_loss_svd.he_total_from_mixed_y(
            z_new, idx_t,
            svd_V, svd_S, svd_L,
            self.train_grm, self.train_grm_diag, self.train_he_denom,
            self.n_train, self.n_covar,
        )

        # Update raw queue (for beta refit at next refresh)
        with torch.no_grad():
            getattr(self, f"queue_{half}")[idx_t] = z_half.detach().float()

        return h2

    # ── queue warm-up ─────────────────────────────────────────────────────────

    def warmup_queue(self, dataloader, device="cuda"):
        """One no-grad pass to pre-fill both half-queues and SVDs.
        Call BEFORE trainer.fit() — PL 2.x exposes the dataloader too late for on_fit_start.
        """
        if bool(self.queue_filled.all()):
            return
        print(f"\nPre-warming SVD queue on {device} ...", flush=True)
        self.to(device).eval()
        with torch.no_grad():
            for batch in dataloader:
                x, _mask, eids = batch
                x = x.to(device)
                _, z = self(x)
                z = z.float()
                idx_t = torch.as_tensor(
                    [self.train_eid_to_idx[e] for e in eids],
                    device=device, dtype=torch.long,
                )
                self.queue_h[idx_t] = z[:, :HALF_DIM].cpu()
                self.queue_r[idx_t] = z[:, HALF_DIM:].cpu()
                self.queue_filled[idx_t] = True
        self._refresh_svd("h")
        self._refresh_svd("r")
        self.train()
        K = self.train_grm.cpu(); K_diag = self.train_grm_diag.cpu(); he_d = self.train_he_denom.cpu()
        h2_h = he_loss_svd.he_total_from_L(self.svd_L_h.cpu(), K, K_diag, he_d, self.n_train, self.n_covar)
        h2_r = he_loss_svd.he_total_from_L(self.svd_L_r.cpu(), K, K_diag, he_d, self.n_train, self.n_covar)
        print(f"Queue warm-up done. Baseline H_h={h2_h.item():.2f}  H_r={h2_r.item():.2f}", flush=True)

    # ── training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx_pl):
        x, mask, eids = batch
        recon, z = self(x)
        z = z.float()

        recon_loss = self.recon_loss_fn(x, recon).squeeze(1) * mask
        recon_loss = recon_loss.sum() / mask.sum()

        z_h = z[:, :HALF_DIM]    # heritable half
        z_r = z[:, HALF_DIM:]    # residual half

        idx_t = torch.as_tensor([self.train_eid_to_idx[e] for e in eids],
                                 device=z.device, dtype=torch.long)
        self.queue_filled[idx_t] = True
        lambda_t   = self.current_lambda()
        queue_full = bool(self.queue_filled.all())
        svd_ready  = bool(self.svd_ready_h) and bool(self.svd_ready_r)

        h2_h = z_h.sum() * 0.0   # zero placeholder, stays in graph
        h2_r = z_r.sum() * 0.0
        h2_h_log = torch.zeros(1, device=z.device, dtype=torch.float32)
        h2_r_log = torch.zeros(1, device=z.device, dtype=torch.float32)

        if queue_full and svd_ready:
            if lambda_t > 0.0:
                h2_h = self._he_for_half(z_h, idx_t, "h")
                h2_r = self._he_for_half(z_r, idx_t, "r")
                h2_h_log = h2_h.detach()
                h2_r_log = h2_r.detach()
            else:
                # Warmup: monitor only, no gradient
                with torch.no_grad():
                    h2_h_log = self._he_for_half(z_h.detach(), idx_t, "h")
                    h2_r_log = self._he_for_half(z_r.detach(), idx_t, "r")
                    self.queue_h[idx_t] = z_h.detach().float()
                    self.queue_r[idx_t] = z_r.detach().float()
        else:
            with torch.no_grad():
                self.queue_h[idx_t] = z_h.detach().float()
                self.queue_r[idx_t] = z_r.detach().float()

        # Periodic refresh for both halves
        if queue_full and self.global_step > 0 and self.global_step % COVAR_REFIT_EVERY == 0:
            self._refresh_svd("h")
            self._refresh_svd("r")

        loss = recon_loss - lambda_t * h2_h + lambda_t * h2_r
        self.log("train_loss",      loss,      prog_bar=True,  on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False, on_epoch=True)
        self.log("train_h2_h",       h2_h_log,  prog_bar=True,  on_epoch=True)
        self.log("train_h2_r",       h2_r_log,  prog_bar=True,  on_epoch=True)
        self.log("lambda_h2",        lambda_t,  prog_bar=False, on_epoch=True)
        return loss

    # ── validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self.val_queue_full.zero_()
        self._val_recon_sum   = 0.0
        self._val_recon_count = 0

    def validation_step(self, batch, batch_idx_pl):
        x, mask, eids = batch
        recon, z = self(x)
        z = z.float()
        recon_loss = self.recon_loss_fn(x, recon).squeeze(1) * mask
        recon_loss = recon_loss.sum() / mask.sum()

        idx_t = torch.as_tensor([self.val_eid_to_idx[e] for e in eids],
                                 device=z.device, dtype=torch.long)
        with torch.no_grad():
            self.val_queue_full[idx_t] = z.detach().float()

        self._val_recon_sum   += recon_loss.item() * x.shape[0]
        self._val_recon_count += x.shape[0]
        self.log("val_recon_loss", recon_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return recon_loss

    def on_validation_epoch_end(self):
        lambda_t = self.current_lambda()

        # H² proxy on z_h only (monitors the heritable half during training)
        with torch.no_grad():
            val_z_h = self.val_queue_full[:, :HALF_DIM]
            beta_v  = he_loss.fit_covariate_beta(self.val_X, val_z_h)
            Y_v     = val_z_h.float() - self.val_X.float() @ beta_v.float()
            L_v, _, _ = he_loss_svd.full_svd_refresh(Y_v, SVD_MODES)
            L_v = L_v.to(self.device)
            h2_proxy = he_loss_svd.he_total_from_L(
                L_v, self.val_grm, self.val_grm_diag,
                self.val_he_denom, len(self.val_eid_to_idx), self.n_covar,
            )
        self.log("val_h2_total_proxy", h2_proxy, prog_bar=True, sync_dist=True)

        recon_epoch = self._val_recon_sum / max(self._val_recon_count, 1)
        self.log("val_loss", recon_epoch - lambda_t * h2_proxy.item(), prog_bar=True, sync_dist=True)

        # GCTA HEreg on full z (all 128 dims) → baseline ~35, comparable to original model
        epoch_dir    = Path(DIR_NAME) / "gcta_eval" / f"epoch_{self.current_epoch:04d}"
        features_dir = epoch_dir / "pca_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        val_eids = list(self.val_eid_to_idx.keys())
        z_np = self.val_queue_full.detach().cpu().numpy()
        for i in range(HIDDEN_DIM):
            pd.DataFrame({"FID": val_eids, "IID": val_eids, str(i): z_np[:, i]}).to_csv(
                features_dir / f"Feature_{i}.csv", sep=" ", index=False
            )
        preset = KinPreset(
            label="val_half",
            keep_file=Path(CACHE_DIR) / "val_eids.txt",
            grm_prefix=str(Path(CACHE_DIR) / "val_grm_gcta"),
            sample_ids_file=Path(CACHE_DIR) / "val_eids.txt",
        )
        try:
            _, hereg_sum_h2 = run_hereg_on_features(
                features_dir=features_dir, out_root=epoch_dir, preset=preset,
                gcta_bin=GCTA_BIN, gcta_threads=GCTA_THREADS,
                ccovar_path=CCOVAR_PATH, qcovar_path=QCOVAR_PATH,
                max_features=HIDDEN_DIM, parallel_jobs=GCTA_PARALLEL_JOBS,
            )
        except Exception as e:
            print(f"[h2joint_C] GCTA HEreg failed at epoch {self.current_epoch}: {e}")
            hereg_sum_h2 = float("nan")
        self.log("val_h2_total_gcta", hereg_sum_h2 if hereg_sum_h2 is not None else float("nan"),
                 prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        sched = {
            "scheduler": ReduceLROnPlateau(opt, "min", patience=4,
                                           min_lr=self.hparams["lr"]/1000, factor=0.5),
            "interval": "epoch", "frequency": 1, "monitor": "val_loss", "strict": True,
        }
        return {"optimizer": opt, "lr_scheduler": sched}


# ── cohort / data ─────────────────────────────────────────────────────────────

cohort_meta = build_or_load_cohort()
cache_dir   = Path(cohort_meta["cache_dir"])

with open(cache_dir / "train_eids.txt") as f: train_eids = [l.strip() for l in f if l.strip()]
with open(cache_dir / "val_eids.txt")   as f: val_eids   = [l.strip() for l in f if l.strip()]

train_grm      = np.load(cache_dir / "train_grm.npy")
train_grm_diag = np.load(cache_dir / "train_grm_diag.npy")
train_he_denom = np.load(cache_dir / "train_he_denom.npy")
train_X        = np.load(cache_dir / "train_covar_X.npy")
val_grm        = np.load(cache_dir / "val_grm.npy")
val_grm_diag   = np.load(cache_dir / "val_grm_diag.npy")
val_he_denom   = np.load(cache_dir / "val_he_denom.npy")
val_X          = np.load(cache_dir / "val_covar_X.npy")

train_dataset = aedataset_with_eid(str(cache_dir/"train_cohort.csv"), "T1_unbiased_linear", transforms_monai)
val_dataset   = aedataset_with_eid(str(cache_dir/"val_cohort.csv"),   "T1_unbiased_linear", transforms_monai)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               pin_memory=True, num_workers=12, shuffle=True)
val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                               pin_memory=True, num_workers=12, shuffle=False)

# ── model & trainer ───────────────────────────────────────────────────────────

AE_model = engine_AE_H2Joint_C(
    lr=LEARNING_RATE,
    train_eids=train_eids, val_eids=val_eids,
    train_grm=train_grm, train_grm_diag=train_grm_diag,
    train_he_denom=train_he_denom, train_X=train_X,
    val_grm=val_grm, val_grm_diag=val_grm_diag,
    val_he_denom=val_he_denom, val_X=val_X,
)

lr_monitor       = LearningRateMonitor(logging_interval="epoch")
model_checkpoint = ModelCheckpoint(dirpath=DIR_NAME, monitor="val_loss", save_last=True,
                                   filename="{epoch}-{train_loss:.6f}-{val_loss:.6f}", save_top_k=5)
tb_logger  = TensorBoardLogger(save_dir=DIR_NAME + "/tb_logs")
csv_logger = CSVLogger(save_dir=DIR_NAME + "/csv_logs")

if __name__ == "__main__":
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
    print(f"HALF_DIM={HALF_DIM}  SVD_MODES={SVD_MODES}")
    print(f"WARMUP={WARMUP_EPOCHS}  RAMP={RAMP_EPOCHS}  LAMBDA={LAMBDA_TARGET}  CYCLE={COSINE_CYCLE_EPOCHS}")

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        accelerator="gpu", devices=[0],
        callbacks=[lr_monitor, model_checkpoint, TQDMProgressBar()],
        log_every_n_steps=20, benchmark=True, max_epochs=300, precision=16,
    )

    _ckpt_dir = Path(DIR_NAME)
    _resume = None
    for _name in ("last.ckpt", "last-v1.ckpt"):
        _p = _ckpt_dir / _name
        if _p.is_file():
            _resume = str(_p)
            break

    if _resume is None:
        print(f"Loading pretrained weights from {PRETRAINED_CKPT}")
        ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
        missing, unexpected = AE_model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  missing: {missing}")
        print(f"  unexpected: {unexpected}")

    _warmup_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True,
        num_workers=12, shuffle=False,
    )
    AE_model.warmup_queue(_warmup_dl, device="cuda:0")

    trainer.fit(AE_model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader, ckpt_path=_resume)
