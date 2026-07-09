# engine_128_T1_cnn_h2joint.py
#
# engine_128_T1_cnn.py's 3D-CNN autoencoder, fine-tuned from its best checkpoint
# with a joint loss:
#
#   L = L_recon  -  lambda(epoch) * H_total
#
# H_total is a differentiable Haseman-Elston (HE) regression heritability proxy
# (he_loss.py), computed from a MoCo-style queue holding the latent (128-dim)
# for every subject in the training cohort -- one row per subject, the row
# updated whenever that subject is seen in a minibatch, every other row reused
# from its last known (detached) value. The GRM-based HE regression needs the
# whole cohort at once, unlike MoCo's queue-as-subset-of-negatives.
#
# Curriculum (see he_loss.lambda_schedule_cosine):
#   Phase 1 (warm-up, epoch < WARMUP_EPOCHS):              lambda = 0, pure reconstruction.
#   Phase 2 (ramp, WARMUP_EPOCHS <= epoch < +RAMP_EPOCHS): lambda linearly 0 -> LAMBDA_TARGET.
#   Phase 3 (cosine cycles):  SGDR-style cosine decay per cycle of COSINE_CYCLE_EPOCHS length;
#     lambda decays LAMBDA_TARGET -> 0 then restarts. Prevents sustained proxy gaming: the
#     periodic lambda=0 forces reconstruction recovery and resets the queue toward biologically
#     plausible latents before the next heritability pressure phase.
# Since we resume from an already well-trained reconstruction checkpoint (not
# random init), WARMUP_EPOCHS only needs to be long enough to fill the queue
# with this model's representations (one full pass over the training cohort).
#
# Cohort: half of the over5-kinship GRM discovery cohort for training, the
# other half for validation (cohort_build_h2.py; see that file for why the
# existing splits_large/* CSVs could not be reused -- different sub-cohort,
# zero eid overlap with the GRM).
#
# Covariates (age, sex, age^2, sex x age, ancestry PCs, ICV/head-size, scanner
# site, batch) are already in T1_ccovar_discovery / T1_qcovar_discovery and are
# used in two different ways:
#   - in-process queue residualization (closed-form OLS, refit periodically,
#     he_loss.fit_covariate_beta) for the fast differentiable training loss.
#   - passed straight to GCTA (--covar/--qcovar) for the per-epoch official
#     validation H^2 metric, which reuses run_idp_pipeline_king_hereg.py's own
#     run_hereg_on_features() so GCTA does its own internal covariate
#     projection -- no double-residualization.
#
# Single GPU (queue + GRM buffers live on one process; no DDP queue sync needed).

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from dataset import aedataset, transforms_monai
import he_loss
from cohort_build_h2 import build_or_load as build_or_load_cohort, CACHE_DIR

_AGENT_IDP_DIR = "/data484_4/txia2/AGENT/experiments/IDP"
if _AGENT_IDP_DIR not in sys.path:
    sys.path.insert(0, _AGENT_IDP_DIR)
from run_idp_pipeline_king_hereg import run_hereg_on_features, KinPreset  # noqa: E402

# ── paths & hyperparameters ──────────────────────────────────────────────────

PRETRAINED_CKPT = "/data484_4/txia2/DeepENDO/training/T1_128/epoch=39-train_loss=0.265290-val_loss=0.291595.ckpt"
DIR_NAME = "/data484_4/txia2/DeepENDO/training/T1_128/output/cnn_h2joint"
GCTA_BIN = "/data4012/zxie3/gcta/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1"
CCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_ccovar_discovery"
QCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_qcovar_discovery"

LEARNING_RATE = 0.0005248074602497723
BATCH_SIZE = 18
HIDDEN_DIM = 128

WARMUP_EPOCHS = 2          # phase 1: lambda = 0, just refill the queue with this model's latents
RAMP_EPOCHS = 3            # phase 2: fast linear ramp 0 -> LAMBDA_TARGET
LAMBDA_TARGET = 0.001      # peak lambda -- at baseline h2~35 contributes ~0.035 vs recon~0.30 (~10%)
COSINE_CYCLE_EPOCHS = 15   # phase 3: SGDR cosine cycle length; lambda decays target->0 then restarts
COVAR_REFIT_EVERY_STEPS = 200
GCTA_THREADS = 8
GCTA_PARALLEL_JOBS = 8


# ── dataset wrapper: same as aedataset, also returns eid for queue placement ──

class aedataset_with_eid(torch.utils.data.Dataset):
    def __init__(self, datafile, modality, transforms):
        self.base = aedataset(datafile=datafile, modality=modality, transforms=transforms)
        self.eids = pd.read_csv(datafile)["eid"].astype(str).tolist()
        assert len(self.eids) == len(self.base), f"CSV rows {len(self.eids)} != dataset length {len(self.base)}"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        return img, mask, self.eids[idx]


# ── lightning module ─────────────────────────────────────────────────────────

class engine_AE_H2Joint(pl.LightningModule):
    def __init__(self, lr, train_eids, val_eids, train_grm, train_grm_diag, train_he_denom,
                 train_X, val_grm, val_grm_diag, val_he_denom, val_X):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "train_eids", "val_eids", "train_grm", "train_grm_diag", "train_he_denom", "train_X",
            "val_grm", "val_grm_diag", "val_he_denom", "val_X",
        ])

        self.hidden_dim = HIDDEN_DIM

        # encoder
        self.first_cnn = self.first_CNN_block(1, 16)
        self.first_max_poold = self.max_poold((1, 1, 1))
        self.first_encoder = self.encoder_block(16, 32)
        self.second_max_poold = self.max_poold((0, 1, 0))
        self.second_encoder = self.encoder_block(32, 64)
        self.third_max_poold = self.max_poold((1, 0, 1))
        self.third_encoder = self.encoder_block(64, 128)
        self.fourth_max_poold = self.max_poold((0, 0, 0))
        self.fourth_encoder = self.encoder_block(128, 256)
        self.encoding_mlp = torch.nn.Linear(256 * 12 * 14 * 12, self.hidden_dim)

        # decoder
        self.decoding_mlp = torch.nn.Linear(self.hidden_dim, 256 * 12 * 14 * 12)
        self.first_decoder = self.decoder_block(256, 128)
        self.first_transconv = self.conv_transpose(128, input_padding=(0, 0, 0))
        self.second_decoder = self.decoder_block(128, 64)
        self.second_transconv = self.conv_transpose(64, input_padding=(1, 0, 1))
        self.third_decoder = self.decoder_block(64, 32)
        self.third_transconv = self.conv_transpose(32, input_padding=(0, 1, 0))
        self.fourth_decoder = self.decoder_block(32, 16)
        self.fourth_transconv = self.conv_transpose(16, input_padding=(1, 1, 1))
        self.last_cnn = self.last_CNN_block(16, 1)

        self.recon_loss_fn = torch.nn.MSELoss(reduction="none")

        # ── queue / GRM / covariate state (not part of the checkpointed model) ──
        self.train_eid_to_idx = {e: i for i, e in enumerate(train_eids)}
        self.val_eid_to_idx = {e: i for i, e in enumerate(val_eids)}
        n_train, n_val = len(train_eids), len(val_eids)

        self.register_buffer("queue_z", torch.zeros(n_train, self.hidden_dim), persistent=False)
        self.register_buffer("queue_filled", torch.zeros(n_train, dtype=torch.bool), persistent=False)
        self.register_buffer("train_grm", torch.as_tensor(train_grm, dtype=torch.float32), persistent=False)
        self.register_buffer("train_grm_diag", torch.as_tensor(train_grm_diag, dtype=torch.float32), persistent=False)
        self.register_buffer("train_he_denom", torch.as_tensor(float(train_he_denom), dtype=torch.float32), persistent=False)
        self.register_buffer("train_X", torch.as_tensor(train_X, dtype=torch.float32), persistent=False)
        self.register_buffer("train_beta", torch.zeros(train_X.shape[1], self.hidden_dim), persistent=False)

        self.register_buffer("val_queue", torch.zeros(n_val, self.hidden_dim), persistent=False)
        self.register_buffer("val_grm", torch.as_tensor(val_grm, dtype=torch.float32), persistent=False)
        self.register_buffer("val_grm_diag", torch.as_tensor(val_grm_diag, dtype=torch.float32), persistent=False)
        self.register_buffer("val_he_denom", torch.as_tensor(float(val_he_denom), dtype=torch.float32), persistent=False)
        self.register_buffer("val_X", torch.as_tensor(val_X, dtype=torch.float32), persistent=False)

    # ── architecture helpers (identical to engine_128_T1_cnn.py) ─────────────

    def max_poold(self, max_padding):
        return nn.MaxPool3d(kernel_size=2, padding=max_padding)

    def encoder_block(self, input_channels, output_channels, padding=1):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
        )

    def conv_transpose(self, output_channels, input_padding):
        return nn.ConvTranspose3d(output_channels, output_channels, kernel_size=2, stride=2, padding=input_padding)

    def decoder_block(self, input_channels, output_channels, input_padding=(0, 0, 0)):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
        )

    def first_CNN_block(self, input_channels, output_channels, padding=1):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(output_channels), nn.LeakyReLU(inplace=False),
        )

    def last_CNN_block(self, input_channels, output_channels, padding=1):
        return nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels), nn.LeakyReLU(inplace=False),
            nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=padding),
            nn.BatchNorm3d(input_channels), nn.LeakyReLU(inplace=False),
            nn.Conv3d(input_channels, output_channels, kernel_size=1),
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
        shape = x.size()

        enc_features = torch.flatten(x, start_dim=1, end_dim=-1)
        lin1 = self.encoding_mlp(enc_features)

        dec = self.decoding_mlp(lin1).view(shape)
        dec = self.first_transconv(self.first_decoder(dec))
        dec = self.second_transconv(self.second_decoder(dec))
        dec = self.third_transconv(self.third_decoder(dec))
        dec = self.fourth_transconv(self.fourth_decoder(dec))
        recon = self.last_cnn(dec)
        return recon, lin1

    # ── curriculum ────────────────────────────────────────────────────────────

    def current_lambda(self) -> float:
        return he_loss.lambda_schedule_cosine(
            self.current_epoch, WARMUP_EPOCHS, RAMP_EPOCHS, LAMBDA_TARGET, COSINE_CYCLE_EPOCHS
        )

    # ── training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, mask, eids = batch
        recon, z = self(x)
        z = z.float()

        recon_loss = self.recon_loss_fn(x, recon).squeeze(1) * mask
        recon_loss = recon_loss.sum() / mask.sum()

        idx_batch = torch.as_tensor([self.train_eid_to_idx[e] for e in eids], device=z.device, dtype=torch.long)
        self.queue_filled[idx_batch] = True
        lambda_t = self.current_lambda()

        h2_total = z.sum() * 0.0  # zero but keeps a graph node if lambda=0
        if lambda_t > 0.0 and bool(self.queue_filled.all()):
            queue_live = self.queue_z.index_copy(0, idx_batch, z)  # autograd flows into z's rows only
            resid_live = he_loss.residualize(queue_live, self.train_X, self.train_beta)
            h2_total, _ = he_loss.he_total_and_per_dim(
                resid_live, self.train_grm, self.train_grm_diag, self.train_he_denom
            )

        loss = recon_loss - lambda_t * h2_total

        # bookkeeping: stale (detached) latents for next step's queue lookup
        with torch.no_grad():
            self.queue_z[idx_batch] = z.detach()
            if (
                bool(self.queue_filled.all())
                and self.global_step > 0
                and self.global_step % COVAR_REFIT_EVERY_STEPS == 0
            ):
                self.train_beta = he_loss.fit_covariate_beta(self.train_X, self.queue_z)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=False, on_epoch=True)
        self.log("train_h2_total", h2_total, prog_bar=True, on_epoch=True)
        self.log("lambda_h2", lambda_t, prog_bar=False, on_epoch=True)
        return loss

    # ── validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self.val_queue.zero_()
        self._val_recon_loss_sum = 0.0
        self._val_recon_loss_count = 0

    def validation_step(self, batch, batch_idx):
        x, mask, eids = batch
        recon, z = self(x)
        z = z.float()

        recon_loss = self.recon_loss_fn(x, recon).squeeze(1) * mask
        recon_loss = recon_loss.sum() / mask.sum()

        idx_batch = torch.as_tensor([self.val_eid_to_idx[e] for e in eids], device=z.device, dtype=torch.long)
        with torch.no_grad():
            self.val_queue[idx_batch] = z.detach()

        self._val_recon_loss_sum += recon_loss.item() * x.shape[0]
        self._val_recon_loss_count += x.shape[0]
        self.log("val_recon_loss", recon_loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return recon_loss

    def on_validation_epoch_end(self):
        lambda_t = self.current_lambda()

        # fast in-process proxy (own closed-form beta fit on the val-half queue, no train leakage)
        with torch.no_grad():
            beta_val = he_loss.fit_covariate_beta(self.val_X, self.val_queue)
            resid_val = he_loss.residualize(self.val_queue, self.val_X, beta_val)
            h2_total_proxy, _ = he_loss.he_total_and_per_dim(
                resid_val, self.val_grm, self.val_grm_diag, self.val_he_denom
            )
        self.log("val_h2_total_proxy", h2_total_proxy, prog_bar=True, sync_dist=True)

        recon_loss_epoch = self._val_recon_loss_sum / max(self._val_recon_loss_count, 1)
        self.log("val_loss", recon_loss_epoch - lambda_t * h2_total_proxy.item(), prog_bar=True, sync_dist=True)

        # official GCTA HEreg eval, reusing run_idp_pipeline_king_hereg.run_hereg_on_features directly
        # (bypassing its PCA/QR preprocessing step -- our 128-dim latent IS already the phenotype, no
        # further dimensionality reduction wanted before HEreg).
        epoch_dir = Path(DIR_NAME) / "gcta_eval" / f"epoch_{self.current_epoch:04d}"
        features_dir = epoch_dir / "pca_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        val_eids = list(self.val_eid_to_idx.keys())
        z_np = self.val_queue.detach().cpu().numpy()
        for i in range(self.hidden_dim):
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
                features_dir=features_dir,
                out_root=epoch_dir,
                preset=preset,
                gcta_bin=GCTA_BIN,
                gcta_threads=GCTA_THREADS,
                ccovar_path=CCOVAR_PATH,
                qcovar_path=QCOVAR_PATH,
                max_features=self.hidden_dim,
                parallel_jobs=GCTA_PARALLEL_JOBS,
            )
        except Exception as e:  # GCTA subprocess issues should not crash training
            print(f"[h2joint] GCTA HEreg eval failed at epoch {self.current_epoch}: {e}")
            hereg_sum_h2 = float("nan")

        self.log("val_h2_total_gcta", hereg_sum_h2 if hereg_sum_h2 is not None else float("nan"), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer, "min", patience=4, min_lr=self.hparams["lr"] / 1000, factor=0.5),
            "interval": "epoch", "frequency": 1, "monitor": "val_loss", "strict": True, "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


# ── cohort / data ─────────────────────────────────────────────────────────────

cohort_meta = build_or_load_cohort()
cache_dir = Path(cohort_meta["cache_dir"])

with open(cache_dir / "train_eids.txt") as f:
    train_eids = [l.strip() for l in f if l.strip()]
with open(cache_dir / "val_eids.txt") as f:
    val_eids = [l.strip() for l in f if l.strip()]

train_grm = np.load(cache_dir / "train_grm.npy")
train_grm_diag = np.load(cache_dir / "train_grm_diag.npy")
train_he_denom = np.load(cache_dir / "train_he_denom.npy")
train_X = np.load(cache_dir / "train_covar_X.npy")
val_grm = np.load(cache_dir / "val_grm.npy")
val_grm_diag = np.load(cache_dir / "val_grm_diag.npy")
val_he_denom = np.load(cache_dir / "val_he_denom.npy")
val_X = np.load(cache_dir / "val_covar_X.npy")

train_dataset = aedataset_with_eid(
    datafile=str(cache_dir / "train_cohort.csv"), modality="T1_unbiased_linear", transforms=transforms_monai,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=12, shuffle=True,
)

val_dataset = aedataset_with_eid(
    datafile=str(cache_dir / "val_cohort.csv"), modality="T1_unbiased_linear", transforms=transforms_monai,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=12, shuffle=False,
)

# ── model & trainer ───────────────────────────────────────────────────────────

AE_model = engine_AE_H2Joint(
    lr=LEARNING_RATE,
    train_eids=train_eids, val_eids=val_eids,
    train_grm=train_grm, train_grm_diag=train_grm_diag, train_he_denom=train_he_denom, train_X=train_X,
    val_grm=val_grm, val_grm_diag=val_grm_diag, val_he_denom=val_he_denom, val_X=val_X,
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
model_checkpoint = ModelCheckpoint(
    dirpath=DIR_NAME, monitor="val_loss", save_last=True,
    filename="{epoch}-{train_loss:.6f}-{val_loss:.6f}", save_top_k=5,
)
tb_logger = TensorBoardLogger(save_dir=DIR_NAME + "/tb_logs")
csv_logger = CSVLogger(save_dir=DIR_NAME + "/csv_logs")
pb = TQDMProgressBar()

if __name__ == "__main__":
    print(f"Train subjects: {len(train_dataset)}  Val subjects: {len(val_dataset)}")
    print(f"WARMUP_EPOCHS={WARMUP_EPOCHS} RAMP_EPOCHS={RAMP_EPOCHS} LAMBDA_TARGET={LAMBDA_TARGET}")

    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        accelerator="gpu",
        devices=[0],  # physical GPU selected via CUDA_VISIBLE_DEVICES at launch time
        callbacks=[lr_monitor, model_checkpoint, pb],
        log_every_n_steps=20,
        benchmark=True,
        max_epochs=300,
        precision=16,
    )

    _ckpt_dir = Path(DIR_NAME)
    _resume = None
    for _name in ("last.ckpt", "last-v1.ckpt"):
        _p = _ckpt_dir / _name
        if _p.is_file():
            _resume = str(_p)
            break

    if _resume is None:
        print(f"Initializing weights from pretrained checkpoint: {PRETRAINED_CKPT}")
        ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
        missing, unexpected = AE_model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  missing keys: {missing}")
        print(f"  unexpected keys: {unexpected}")

    trainer.fit(
        AE_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=_resume,
    )
