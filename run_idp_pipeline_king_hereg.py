#!/usr/bin/env python3
"""
IDP heritability pipeline — GCTA HE-regression / REML variant (no covariate pre-residualization).

=== KEY DIFFERENCE FROM run_idp_pipeline_king_standard_optimized.py ===

  Standard: Python residualizes covariates THEN does PCA/QR → in-process HE or GCTA --reml
  This:     PCA/QR on RAW phenotype (no residualization) → GCTA --HEreg or --reml
            (GCTA handles covariate projection internally via --covar / --qcovar)

=== IN-HOUSE DEFAULT PATHS (no flags required for a standard run) ===

  Phenotype CSV  : /data484_4/txia2/gwas_practice/T1_IDP_pheno_discovery.csv
  Categorical cov: /data484_4/txia2/gwas_practice/T1_ccovar_discovery   (→ gcta --covar)
  Quantitative cov: /data484_4/txia2/gwas_practice/T1_qcovar_discovery  (→ gcta --qcovar)
  GRM            : AGENT/grm_subset_king/king_over4p5_gcta_discovery.grm.{id,bin}

=== EXAMPLE COMMAND LINES ===

  # Feature directory → write output into that directory:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /path/to/feature_dir/

  # Merged CSV → --output_root required:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /data484_4/txia2/gwas_practice/T1_IDP_pheno_discovery.csv \\
      --output_root /my/output/dir

  # QR decomposition instead of PCA:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /path/to/feature_dir/ --dim-reduction qr

  # REML instead of HEreg:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /path/to/feature_dir/ --h2-method reml

  # Both HEreg and REML:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /path/to/feature_dir/ --h2-method both

  # Tune parallelism:
  python run_idp_pipeline_king_hereg.py \\
      --phenotype_csv /path/to/feature_dir/ \\
      --hereg-parallel-jobs 8 --reml-parallel-jobs 8 --save-features-workers 8

=== INPUTS ===

GRM / KING preset (--cohort + --kin):  same as standard pipeline.

Phenotype (--phenotype_mode csv [default] | npy):  same loaders.
  Values are kept RAW — no covariate residualization, no whitening.
  --dim-reduction pca (default) | qr  performs PCA/QR purely for dimensionality
  reduction on the raw features before passing to GCTA.

Covariates:
  --ccovar  <path>  (default: gwas_practice/T1_ccovar_discovery)
  --qcovar  <path>  (default: gwas_practice/T1_qcovar_discovery)
  Passed directly as gcta64 --covar / --qcovar (no merging needed).

=== PROCESSING ===

  1. PCA/QR on raw phenotype  (same dim-reduction options as standard pipeline)
       pca: randomized PCA, truncated to --n_pca components
       qr:  economy QR — all min(n_samples, n_features) orthonormal Q columns
  2. Feature_*.csv written to --pca-features-dir (or <output_root>/pca_features/)
  3. Heritability estimation  --h2-method hereg (default) | reml | both
       hereg: gcta64 --HEreg --covar ccovar --qcovar qcovar
              → gcta_hereg_out/Feature_i.HEreg  +  hereg_h2_summary.csv  +  hereg_aggregate.json
       reml:  gcta64 --reml  --covar ccovar --qcovar qcovar
              → gcta_reml_out/Feature_i.hsq  +  reml_h2_summary.csv    +  reml_aggregate.json
       both:  run HEreg then REML, report both in arena_manifest.json

=== OUTPUTS ===

  pca_features/          Feature_*.csv  +  pca_explained_variance_ratio.json  (PCA only)
  gcta_hereg_out/        Feature_i.HEreg  +  Feature_i.HEreg.log  (hereg/both)
  hereg_h2_summary.csv   Per-feature HEreg h² table  (hereg/both)
  hereg_aggregate.json   hereg_sum_h2  (hereg/both)
  gcta_reml_out/         Feature_i.hsq  +  Feature_i.log  (reml/both)
  reml_h2_summary.csv    Per-feature REML h² table  (reml/both)
  reml_aggregate.json    reml_sum_h2 + PCA-weighted mean_h2  (reml/both)
  timing_resources.csv / .md
  arena_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA as _PCA
except ImportError:
    _PCA = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_agent_root = _script_dir.parent.parent

GWAS_PRACTICE = Path("/data484_4/txia2/gwas_practice")
DEFAULT_CCOVAR = GWAS_PRACTICE / "T1_ccovar_discovery"
DEFAULT_QCOVAR = GWAS_PRACTICE / "T1_qcovar_discovery"
DEFAULT_PHENOTYPE_MERGED = GWAS_PRACTICE / "T1_IDP_pheno_discovery.csv"
SOURCE_PHENOTYPE_FOR_BUILD = Path("/data484_4/txia2/mocov2/IDP_PhenoWAS/merged_IDP_result_filtered.csv")
DEFAULT_PHENOTYPE_NPY_DIR = Path("/data484_4/txia2/mesh/z_graph_fsaverage4_t1_gwas_mask0p75")

GRM_KING_DIR = _agent_root / "grm_subset_king"
DEFAULT_GCTA_BIN = "/data4012/zxie3/gcta/gcta-1.94.1-linux-kernel-3-x86_64/gcta-1.94.1"

PCA_N_COMPONENTS = 128
PCA_EXPLAINED_VARIANCE_RATIO_FILE = "pca_explained_variance_ratio.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_iid(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _record_step(timing_rows: list[dict], label: str, step_name: str, t0: float, r0) -> None:
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    timing_rows.append({
        "label": label,
        "step": step_name,
        "wall_sec": round(time.perf_counter() - t0, 3),
        "cpu_user_sec": round(r1.ru_utime - r0.ru_utime, 3),
        "cpu_sys_sec": round(r1.ru_stime - r0.ru_stime, 3),
        "max_rss_mb": round(r1.ru_maxrss / 1024.0, 2),
    })


def _write_timing_table(timing_rows: list[dict], out_root: Path, kin_label: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(timing_rows)
    df.to_csv(out_root / "timing_resources.csv", index=False)
    md_path = out_root / "timing_resources.md"
    with open(md_path, "w") as f:
        f.write("# IDP KING HEreg pipeline – time and resource usage\n\n")
        f.write(f"Kin preset: `{kin_label}`.\n\n")
        f.write("| label | step | wall_sec | cpu_user_sec | cpu_sys_sec | max_rss_mb |\n")
        f.write("|-------|------|----------|---------------|--------------|------------|\n")
        for _, row in df.iterrows():
            step_cell = f"**{row['step']}**" if row["step"] == "total" else str(row["step"])
            f.write(
                f"| {row['label']} | {step_cell} | {row['wall_sec']} | {row['cpu_user_sec']} | "
                f"{row['cpu_sys_sec']} | {row['max_rss_mb']} |\n"
            )
    print(f"[IDP-HEREG] Timing table: {out_root / 'timing_resources.csv'}")


# ---------------------------------------------------------------------------
# KinPreset / sample-ID helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KinPreset:
    label: str
    keep_file: Path
    grm_prefix: str
    sample_ids_file: Path


def _kin_preset(*, cohort: str, kin: str) -> KinPreset:
    if kin not in ("over4", "over4p5", "over5"):
        raise ValueError(f"Unsupported --kin {kin!r}")
    if cohort not in ("discovery", "full"):
        raise ValueError(f"Unsupported --cohort {cohort!r}")
    suf = "_discovery" if cohort == "discovery" else ""
    label = f"{kin}{suf}"
    keep = GRM_KING_DIR / f"king_cutoff_{kin}{suf}.txt"
    grm = str(GRM_KING_DIR / f"king_{kin}_gcta{suf}")
    sid = GRM_KING_DIR / f"sample_ids_king_{label}.txt"
    return KinPreset(label=label, keep_file=keep, grm_prefix=grm, sample_ids_file=sid)


def _ensure_sample_ids(preset: KinPreset, timing_rows: list[dict]) -> Path:
    grm_id = Path(preset.grm_prefix + ".grm.id")
    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)

    if preset.sample_ids_file.is_file():
        _record_step(timing_rows, preset.label, "ensure_sample_ids", t0, r0)
        return preset.sample_ids_file

    if grm_id.is_file():
        df = pd.read_csv(grm_id, sep=r"\s+", header=None, names=["FID", "IID"])
    else:
        keep_file = preset.keep_file
        if not keep_file.is_file():
            raise FileNotFoundError(f"Need {grm_id} or {keep_file} to create {preset.sample_ids_file}")
        df = pd.read_csv(keep_file, sep=r"\s+", header=None, names=["FID", "IID"])
        if len(df) and str(df.iloc[0]["FID"]).upper() == "FID":
            df = df.iloc[1:].reset_index(drop=True)

    GRM_KING_DIR.mkdir(parents=True, exist_ok=True)
    with open(preset.sample_ids_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{_norm_iid(row['IID'])}\n")

    _record_step(timing_rows, preset.label, "ensure_sample_ids", t0, r0)
    print(f"[IDP-HEREG] Wrote {preset.sample_ids_file} ({len(df)} samples)")
    return preset.sample_ids_file


# ---------------------------------------------------------------------------
# Dimensionality reduction on RAW features (no covariate residualization)
# ---------------------------------------------------------------------------

def pca_only(
    features: np.ndarray,
    eids: list,
    n_components: int = PCA_N_COMPONENTS,
    standardize: bool = False,
) -> Tuple[np.ndarray, list, list]:
    """
    PCA on raw (non-residualized) features.
    Returns (pc_scores, eids, explained_variance_ratio).
    """
    if _PCA is None:
        raise ImportError("sklearn required: pip install scikit-learn")

    Y = features.astype(np.float64)
    if standardize:
        Y = Y - Y.mean(axis=0)
        Y = Y / (Y.std(axis=0) + 1e-10)

    n_sub, n_feat = Y.shape
    n_comp = min(n_components, n_sub - 1, n_feat)
    use_randomized = n_comp < int(0.8 * min(n_sub, n_feat)) and n_sub > 500
    if use_randomized:
        pca = _PCA(n_components=n_comp, svd_solver="randomized", iterated_power=4, random_state=42)
    else:
        pca = _PCA(n_components=n_comp)
    pc_scores = pca.fit_transform(Y)

    if n_comp < n_components:
        padded = np.zeros((n_sub, n_components), dtype=np.float64)
        padded[:, :n_comp] = pc_scores
        ratio = np.zeros(n_components)
        ratio[:n_comp] = pca.explained_variance_ratio_
        return padded, eids, ratio.tolist()

    return pc_scores, eids, pca.explained_variance_ratio_.tolist()


def qr_only(
    features: np.ndarray,
    eids: list,
    standardize: bool = False,
) -> Tuple[np.ndarray, list]:
    """
    Economy QR on raw (non-residualized) features.
    Returns (Q, eids) — all min(n_samples, n_features) orthonormal columns.
    """
    Y = features.astype(np.float64)
    if standardize:
        Y = Y - Y.mean(axis=0)
        Y = Y / (Y.std(axis=0) + 1e-10)
    Q, _ = np.linalg.qr(Y, mode="reduced")
    return np.ascontiguousarray(Q), eids


# ---------------------------------------------------------------------------
# HEreg output parser + runner
# ---------------------------------------------------------------------------

def _read_h2_from_hereg(hereg_path: Path) -> Optional[float]:
    """Parse V(G)/Vp from a GCTA .HEreg file."""
    if not hereg_path.is_file():
        return None
    with open(hereg_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "V(G)/Vp":
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def _run_one_hereg_feature(
    i: int,
    *,
    features_dir: Path,
    hereg_out_dir: Path,
    preset: KinPreset,
    gcta_bin: str,
    threads: int,
    ccovar_path: str,
    qcovar_path: str,
    reuse_hereg: bool,
) -> dict:
    pheno_file = features_dir / f"Feature_{i}.csv"
    out_prefix = hereg_out_dir / f"Feature_{i}"
    hereg_path = hereg_out_dir / f"Feature_{i}.HEreg"

    if reuse_hereg and hereg_path.is_file():
        h2 = _read_h2_from_hereg(hereg_path)
        log_txt = "[IDP-HEREG] reuse_hereg: skipped GCTA; using existing .HEreg\n"
    else:
        cmd = [
            gcta_bin, "--HEreg",
            "--grm", preset.grm_prefix,
            "--pheno", str(pheno_file),
            "--covar", str(ccovar_path),
            "--qcovar", str(qcovar_path),
            "--thread-num", str(int(threads)),
            "--out", str(out_prefix),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_txt = proc.stdout
        h2 = _read_h2_from_hereg(hereg_path)

    with open(hereg_out_dir / f"Feature_{i}.HEreg.log", "w") as f:
        f.write(log_txt)
    print(f"[IDP-HEREG] HEreg Feature_{i}: h2={h2}")
    return {"feature_index": i, "h2": h2}


def run_hereg_on_features(
    *,
    features_dir: Path,
    out_root: Path,
    preset: KinPreset,
    gcta_bin: str,
    gcta_threads: int,
    ccovar_path: str,
    qcovar_path: str,
    max_features: Optional[int] = None,
    reuse_hereg: bool = False,
    parallel_jobs: int = 1,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """Run GCTA --HEreg on each Feature_i.csv. Returns (df, sum_h2)."""
    indices = _discover_feature_indices(features_dir)
    if not indices:
        raise FileNotFoundError(f"No Feature_*.csv under {features_dir}")

    hereg_out_dir = out_root / "gcta_hereg_out"
    hereg_out_dir.mkdir(parents=True, exist_ok=True)

    stop = len(indices) if max_features is None else min(int(max_features), len(indices))
    jobs = max(1, int(parallel_jobs))
    threads_per = max(1, int(gcta_threads) // jobs)
    if jobs > 1:
        print(f"[IDP-HEREG] HEreg parallel_jobs={jobs}, thread-num per job={threads_per}")

    worker = partial(
        _run_one_hereg_feature,
        features_dir=features_dir,
        hereg_out_dir=hereg_out_dir,
        preset=preset,
        gcta_bin=gcta_bin,
        threads=threads_per,
        ccovar_path=ccovar_path,
        qcovar_path=qcovar_path,
        reuse_hereg=reuse_hereg,
    )
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(worker, [indices[k] for k in range(stop)]))

    hereg_df = pd.DataFrame(rows).sort_values("feature_index").reset_index(drop=True)
    hereg_df.to_csv(out_root / "hereg_h2_summary.csv", index=False)

    valid = hereg_df["h2"].notna()
    hereg_sum_h2 = float(hereg_df["h2"].sum(skipna=True)) if valid.any() else None
    agg = {
        "hereg_sum_h2": hereg_sum_h2,
        "n_features_hereg": int(len(hereg_df)),
        "ccovar": str(ccovar_path),
        "qcovar": str(qcovar_path),
    }
    with open(out_root / "hereg_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[IDP-HEREG] HEreg: sum h2={hereg_sum_h2}")
    return hereg_df, hereg_sum_h2


# ---------------------------------------------------------------------------
# REML helpers (covar/qcovar passed to GCTA)
# ---------------------------------------------------------------------------

def _read_h2_from_hsq(hsq_path: Path) -> Optional[float]:
    if not hsq_path.is_file():
        return None
    with open(hsq_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "V(G)/Vp":
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def _discover_feature_indices(features_dir: Path) -> list[int]:
    idx: list[int] = []
    for p in sorted(features_dir.glob("Feature_*.csv")):
        m = re.match(r"Feature_(\d+)\.csv$", p.name, re.I)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def _load_evr(features_dir: Path, n_features: int) -> list[float]:
    evr_path = features_dir / PCA_EXPLAINED_VARIANCE_RATIO_FILE
    evr: list[float] = []
    if evr_path.is_file():
        raw = json.loads(evr_path.read_text())
        if isinstance(raw, list):
            evr = [float(x) for x in raw]
    while len(evr) < n_features:
        evr.append(0.0)
    return evr[:n_features]


def _run_one_reml_feature(
    i: int,
    *,
    features_dir: Path,
    reml_out_dir: Path,
    preset: KinPreset,
    gcta_bin: str,
    threads: int,
    ccovar_path: str,
    qcovar_path: str,
    explained: list[float],
    reuse_hsq: bool,
) -> dict:
    pheno_file = features_dir / f"Feature_{i}.csv"
    if not pheno_file.is_file():
        raise FileNotFoundError(pheno_file)
    out_prefix = reml_out_dir / f"Feature_{i}"
    hsq_path = reml_out_dir / f"Feature_{i}.hsq"

    if reuse_hsq and hsq_path.is_file():
        h2 = _read_h2_from_hsq(hsq_path)
        log_txt = "[IDP-HEREG] reuse_hsq: skipped GCTA; using existing .hsq\n"
    else:
        cmd = [
            gcta_bin, "--reml",
            "--grm", preset.grm_prefix,
            "--pheno", str(pheno_file),
            "--covar", str(ccovar_path),
            "--qcovar", str(qcovar_path),
            "--thread-num", str(int(threads)),
            "--out", str(out_prefix),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_txt = proc.stdout
        h2 = _read_h2_from_hsq(hsq_path)

    with open(reml_out_dir / f"Feature_{i}.log", "w") as f:
        f.write(log_txt)
    evr = float(explained[i]) if i < len(explained) else 0.0
    print(f"[IDP-HEREG] REML Feature_{i}: h2={h2}")
    return {"feature_index": i, "h2": h2, "explained_variance_ratio": evr}


def run_reml_on_features(
    *,
    features_dir: Path,
    out_root: Path,
    preset: KinPreset,
    gcta_bin: str,
    gcta_threads: int,
    ccovar_path: str,
    qcovar_path: str,
    max_features: Optional[int] = None,
    reuse_hsq: bool = False,
    parallel_jobs: int = 1,
) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    """Run GCTA --reml with --covar/--qcovar on each Feature_i.csv.
    Returns (df, reml_mean_h2_pca_weighted, reml_sum_h2)."""
    indices = _discover_feature_indices(features_dir)
    if not indices:
        raise FileNotFoundError(f"No Feature_*.csv under {features_dir}")
    explained = _load_evr(features_dir, max(indices) + 1)

    reml_out_dir = out_root / "gcta_reml_out"
    reml_out_dir.mkdir(parents=True, exist_ok=True)

    stop = len(indices) if max_features is None else min(int(max_features), len(indices))
    jobs = max(1, int(parallel_jobs))
    threads_per = max(1, int(gcta_threads) // jobs)
    if jobs > 1:
        print(f"[IDP-HEREG] REML parallel_jobs={jobs}, thread-num per job={threads_per}")

    worker = partial(
        _run_one_reml_feature,
        features_dir=features_dir,
        reml_out_dir=reml_out_dir,
        preset=preset,
        gcta_bin=gcta_bin,
        threads=threads_per,
        ccovar_path=ccovar_path,
        qcovar_path=qcovar_path,
        explained=explained,
        reuse_hsq=reuse_hsq,
    )
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(worker, [indices[k] for k in range(stop)]))

    reml_df = pd.DataFrame(rows).sort_values("feature_index").reset_index(drop=True)
    reml_df.to_csv(out_root / "reml_h2_summary.csv", index=False)

    valid = reml_df["h2"].notna()
    h2v = reml_df.loc[valid, "h2"].astype(float).values
    w = reml_df.loc[valid, "explained_variance_ratio"].astype(float).values
    w_sum = float(w.sum())
    reml_mean_h2 = float((h2v * (w / w_sum)).sum()) if len(h2v) and w_sum > 0 else None
    reml_sum_h2 = float(reml_df["h2"].sum(skipna=True)) if valid.any() else None

    agg = {
        "reml_sum_h2": reml_sum_h2,
        "mean_h2_reml_pca_weighted": reml_mean_h2,
        "n_features_reml": int(len(reml_df)),
        "ccovar": str(ccovar_path),
        "qcovar": str(qcovar_path),
    }
    with open(out_root / "reml_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[IDP-HEREG] REML: sum h2={reml_sum_h2}, PCA-weighted mean h2={reml_mean_h2}")
    return reml_df, reml_mean_h2, reml_sum_h2


# ---------------------------------------------------------------------------
# Phenotype loaders (identical to standard pipeline)
# ---------------------------------------------------------------------------

def _choose_numeric_feature_cols(df: pd.DataFrame, id_col: str) -> list[str]:
    candidates = [c for c in df.columns if c != id_col]
    kept: list[str] = []
    for c in candidates:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            df[c] = numeric
            kept.append(c)
    return kept


def _load_feature_dir_fast(
    feature_dir: Path, sample_ids: set[str]
) -> tuple[list[str], np.ndarray, list[str]]:
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    feature_paths: list[tuple[int, Path]] = []
    for fp in feature_dir.iterdir():
        if not fp.is_file():
            continue
        name = fp.name
        if not name.startswith("Feature_"):
            continue
        idx_str = name[len("Feature_"):]
        if idx_str.endswith(".csv"):
            idx_str = idx_str[:-4]
        if not idx_str.isdigit():
            continue
        feature_paths.append((int(idx_str), fp))
    feature_paths.sort(key=lambda x: x[0])
    if not feature_paths:
        raise ValueError(f"No Feature_* files found under {feature_dir}.")

    n_features = len(feature_paths)
    feature_cols = [f"feature_{idx}" for idx, _ in feature_paths]

    _, first_fp = feature_paths[0]
    first_df = pd.read_csv(first_fp, sep=r"\s+")
    if "IID" not in first_df.columns:
        raise ValueError(f"{first_fp} missing IID column.")
    first_df["_iid"] = first_df["IID"].apply(_norm_iid)
    valid_iids: list[str] = sorted(
        {iid for iid in first_df["_iid"].unique() if iid in sample_ids and iid != ""},
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    if not valid_iids:
        raise ValueError(
            f"No overlap between Feature_*.csv IIDs in {feature_dir} and KING sample set ({len(sample_ids)})."
        )
    iid_to_row = {iid: i for i, iid in enumerate(valid_iids)}
    n = len(valid_iids)
    Y = np.full((n, n_features), np.nan, dtype=np.float64)

    def _read_col(args: tuple[int, tuple[int, Path]]) -> None:
        col_idx, (_, fp) = args
        df = pd.read_csv(fp, sep=r"\s+")
        if "IID" not in df.columns:
            raise ValueError(f"{fp} missing IID column.")
        pheno_cols = [c for c in df.columns if c not in ("FID", "IID")]
        if len(pheno_cols) != 1:
            raise ValueError(f"{fp}: expected one phenotype column; got {pheno_cols}.")
        col = pheno_cols[0]
        iid_norm = df["IID"].apply(_norm_iid)
        keep = iid_norm.isin(iid_to_row)
        if keep.any():
            row_idx = iid_norm[keep].map(iid_to_row).to_numpy(dtype=int)
            vals = pd.to_numeric(df.loc[keep, col], errors="coerce").to_numpy(dtype=np.float64)
            Y[row_idx, col_idx] = vals

    n_workers = min(16, max(2, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_read_col, enumerate(feature_paths)))

    print(
        f"[IDP-HEREG] Feature dir: {n} samples × {n_features} features from {feature_dir.name}"
        f" (parallel, {n_workers} workers)."
    )
    return valid_iids, Y, feature_cols


def _load_mesh_phenotype_npy(
    mesh_dir: Path, sample_ids: set[str], *, name_suffix: str = "20263_2_0"
) -> tuple[pd.DataFrame, list[str]]:
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Phenotype .npy directory not found: {mesh_dir}")

    by_eid: dict[str, np.ndarray] = {}
    n_feat: int | None = None
    if name_suffix:
        for eid in sample_ids:
            p = mesh_dir / f"{eid}_{name_suffix}.npy"
            if not p.is_file():
                continue
            arr = np.asarray(np.load(p), dtype=np.float64).ravel()
            if n_feat is None:
                n_feat = int(arr.size)
            elif int(arr.size) != n_feat:
                raise ValueError(f"Inconsistent .npy length in {p.name}")
            by_eid[eid] = arr
    else:
        for p in mesh_dir.glob("*.npy"):
            eid = _norm_iid(p.stem.split("_")[0])
            if eid not in sample_ids or not eid:
                continue
            arr = np.asarray(np.load(p), dtype=np.float64).ravel()
            if n_feat is None:
                n_feat = int(arr.size)
            elif int(arr.size) != n_feat:
                raise ValueError(f"Inconsistent .npy length in {p.name}")
            if eid in by_eid:
                raise ValueError(f"More than one .npy for eid {eid} under {mesh_dir}")
            by_eid[eid] = arr

    if not by_eid or n_feat is None:
        raise ValueError(f"No .npy files in {mesh_dir} overlapped KING sample set ({len(sample_ids)} IIDs).")

    feat_names = [f"mesh_{i}" for i in range(n_feat)]
    eids_sorted = sorted(by_eid)
    Y = np.stack([by_eid[e] for e in eids_sorted], axis=0)
    df = pd.DataFrame(Y, columns=feat_names, dtype=np.float64)
    df.insert(0, "IID_norm", eids_sorted)
    print(f"[IDP-HEREG] Mesh .npy: {len(eids_sorted)} samples × {n_feat} features from {mesh_dir.name}.")
    return df, feat_names


# ---------------------------------------------------------------------------
# Arena manifest
# ---------------------------------------------------------------------------

def _canonical_payload_for_hash(manifest: dict) -> str:
    payload = {
        "run_id": manifest.get("run_id"),
        "timestamp_utc": manifest.get("timestamp_utc"),
        "kin_label": manifest.get("kin_label"),
        "grm_prefix": manifest.get("grm_prefix"),
        "n_samples": manifest.get("n_samples"),
        "n_phenotype_columns": manifest.get("n_phenotype_columns"),
        "sum_h2": manifest.get("sum_h2"),
        "reml_sum_h2": manifest.get("reml_sum_h2"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_arena_manifest(
    out_root: Path,
    *,
    preset: KinPreset,
    phenotype_csv: Path | None,
    phenotype_mode: str,
    phenotype_npy_dir: Path | None,
    ccovar: Path,
    qcovar: Path,
    n_samples: int,
    n_features: int,
    pca_features_dir: Path | None,
    dim_reduction: str,
    h2_method: str,
    hereg_sum_h2: float | None = None,
    reml_sum_h2: float | None = None,
    reml_mean_h2: float | None = None,
) -> None:
    artifacts = ["timing_resources.csv", "timing_resources.md"]
    if pca_features_dir is not None:
        artifacts.insert(0, "pca_features/")
    if hereg_sum_h2 is not None:
        artifacts.insert(-2, "gcta_hereg_out/")
    if reml_sum_h2 is not None:
        artifacts.insert(-2, "gcta_reml_out/")

    manifest = {
        "run_id": out_root.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kin_label": preset.label,
        "grm_prefix": preset.grm_prefix,
        "keep_file": str(preset.keep_file),
        "phenotype_mode": phenotype_mode,
        "phenotype_csv": str(phenotype_csv) if phenotype_csv else "",
        "phenotype_npy_dir": str(phenotype_npy_dir) if phenotype_npy_dir else "",
        "ccovar": str(ccovar),
        "qcovar": str(qcovar),
        "covariate_handling": "gcta_internal",
        "dim_reduction": dim_reduction,
        "h2_method": h2_method,
        "n_samples": n_samples,
        "n_phenotype_columns": n_features,
        "sum_h2": hereg_sum_h2,
        "reml_sum_h2": reml_sum_h2,
        "reml_mean_h2_pca_weighted": reml_mean_h2,
        "pca_features_dir": str(pca_features_dir.resolve()) if pca_features_dir else "",
        "artifacts_relative": artifacts,
    }
    canonical = _canonical_payload_for_hash(manifest)
    manifest["signed_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = out_root / "arena_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[IDP-HEREG] Arena manifest: {path}")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _ensure_merged_phenotype(pheno_path: Path, source: Path) -> None:
    if pheno_path.is_file():
        return
    if not source.is_file():
        raise FileNotFoundError(f"Phenotype missing ({pheno_path}) and source not found: {source}")
    pheno_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, pheno_path)
    print(f"[IDP-HEREG] Copied phenotype → {pheno_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    _cpu = os.cpu_count() or 8
    _default_sf_workers = min(8, max(2, max(1, _cpu // 2)))
    _default_parallel_jobs = min(8, max(2, _cpu))

    parser = argparse.ArgumentParser(
        description=(
            "IDP heritability pipeline: PCA/QR on raw features → GCTA --HEreg or --reml "
            "(no Python-side covariate residualization; covariates passed to GCTA directly)."
        )
    )
    parser.add_argument("--cohort", choices=["discovery", "full"], default="discovery")
    parser.add_argument("--kin", choices=["over4", "over4p5", "over5"], default="over4p5")
    parser.add_argument(
        "--custom-grm-prefix", type=str, default=None,
        help=(
            "Override the resolved KING grm_prefix with this path prefix (ad hoc cohort, "
            "e.g. the unfiltered whole-discovery GRM). Must be paired with "
            "--custom-sample-ids-file; --cohort/--kin are ignored when set."
        ),
    )
    parser.add_argument(
        "--custom-sample-ids-file", type=str, default=None,
        help="Sample-ID pool file paired with --custom-grm-prefix (one IID per line).",
    )
    parser.add_argument("--phenotype_mode", choices=["csv", "npy"], default="csv")
    parser.add_argument(
        "--phenotype_npy_dir", type=str, default=str(DEFAULT_PHENOTYPE_NPY_DIR),
        help="Directory of {eid}_*.npy files (used when --phenotype_mode npy).",
    )
    parser.add_argument("--mesh_npy_suffix", type=str, default="20263_2_0")
    parser.add_argument("--mesh_npy_glob_all", action="store_true")
    parser.add_argument(
        "--phenotype_csv", type=str, default=str(DEFAULT_PHENOTYPE_MERGED),
        help=(
            "Merged CSV (eid/EID + columns) or directory of Feature_*.csv / Feature_* files."
        ),
    )
    parser.add_argument(
        "--ccovar", type=str, default=str(DEFAULT_CCOVAR),
        help="Categorical covariate file (FID IID + columns) passed to gcta64 --covar.",
    )
    parser.add_argument(
        "--qcovar", type=str, default=str(DEFAULT_QCOVAR),
        help="Quantitative covariate file (FID IID + columns) passed to gcta64 --qcovar.",
    )
    parser.add_argument(
        "--phenotype_source", type=str, default=str(SOURCE_PHENOTYPE_FOR_BUILD),
        help="Source CSV copied to --phenotype_csv when it is missing.",
    )
    parser.add_argument("--n_pca", type=int, default=128, help="Max PCA components (ignored for QR).")
    parser.add_argument("--max_features", type=int, default=None, help="Cap Feature_*.csv used.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help=(
            "Output directory. If --phenotype_csv is a directory, defaults to that directory. "
            "Required for CSV-file or npy modes."
        ),
    )
    parser.add_argument(
        "--pca-features-dir", type=str, default=None,
        help="Write PCA Feature_*.csv here instead of <output_root>/pca_features/.",
    )
    parser.add_argument(
        "--standardize-features", action="store_true",
        help="Demean and z-score each feature column before PCA/QR (no covariate residualization).",
    )
    parser.add_argument(
        "--random-sample-fraction", type=float, default=None,
        help="Keep a random subset of the KING preset sample IDs (fraction in (0,1]).",
    )
    parser.add_argument("--random-sample-seed", type=int, default=42)
    parser.add_argument(
        "--save-features-workers", type=int, default=_default_sf_workers,
        help=f"Thread pool size for writing Feature_*.csv (default {_default_sf_workers}).",
    )
    parser.add_argument(
        "--dim-reduction", choices=["pca", "qr"], default="pca",
        help=(
            "'pca': randomized PCA to --n_pca components (default). "
            "'qr': economy QR — all min(n_samples, n_features) orthonormal Q columns."
        ),
    )
    parser.add_argument(
        "--h2-method", choices=["hereg", "reml", "both"], default="hereg",
        help=(
            "'hereg': GCTA --HEreg (default). "
            "'reml': GCTA --reml. "
            "'both': run HEreg then REML."
        ),
    )
    parser.add_argument(
        "--gcta-bin", type=str, default=DEFAULT_GCTA_BIN,
        help="Path to gcta64 binary.",
    )
    parser.add_argument("--gcta-threads", type=int, default=8, help="Total GCTA thread budget.")
    parser.add_argument(
        "--hereg-parallel-jobs", type=int, default=_default_parallel_jobs,
        help=f"Concurrent gcta64 --HEreg processes (default {_default_parallel_jobs}).",
    )
    parser.add_argument(
        "--reml-parallel-jobs", type=int, default=_default_parallel_jobs,
        help=f"Concurrent gcta64 --reml processes (default {_default_parallel_jobs}).",
    )
    parser.add_argument("--reuse-hereg", action="store_true",
                        help="Skip --HEreg for features with existing .HEreg files.")
    parser.add_argument("--reuse-hsq", action="store_true",
                        help="Skip --reml for features with existing .hsq files.")
    return parser


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_king_hereg(args: argparse.Namespace) -> None:
    if args.custom_grm_prefix or args.custom_sample_ids_file:
        if not (args.custom_grm_prefix and args.custom_sample_ids_file):
            raise SystemExit("--custom-grm-prefix and --custom-sample-ids-file must be given together.")
        preset = KinPreset(
            label=f"custom_{Path(args.custom_grm_prefix).name}",
            keep_file=Path(args.custom_sample_ids_file),
            grm_prefix=args.custom_grm_prefix,
            sample_ids_file=Path(args.custom_sample_ids_file),
        )
    else:
        preset = _kin_preset(cohort=args.cohort, kin=args.kin)
    ccovar_path = Path(args.ccovar)
    qcovar_path = Path(args.qcovar)
    phenotype_csv = Path(args.phenotype_csv)
    pheno_source = Path(args.phenotype_source)
    phenotype_npy_dir = Path(args.phenotype_npy_dir)
    use_npy = args.phenotype_mode == "npy"
    dim_reduction = getattr(args, "dim_reduction", "pca")
    h2_method = getattr(args, "h2_method", "hereg")

    if not ccovar_path.is_file():
        raise FileNotFoundError(f"Categorical covariate file not found: {ccovar_path}")
    if not qcovar_path.is_file():
        raise FileNotFoundError(f"Quantitative covariate file not found: {qcovar_path}")

    if not use_npy and not phenotype_csv.is_dir():
        _ensure_merged_phenotype(phenotype_csv, pheno_source)
        if not phenotype_csv.is_file():
            raise FileNotFoundError(f"Phenotype CSV not found: {phenotype_csv}")

    if args.output_root:
        out_root = Path(args.output_root)
    elif not use_npy and phenotype_csv.is_dir():
        out_root = phenotype_csv
    else:
        raise SystemExit(
            "[IDP-HEREG] --output_root is required when --phenotype_csv is a file "
            "or --phenotype_mode npy is used."
        )

    features_dir = (
        Path(args.pca_features_dir).expanduser().resolve()
        if args.pca_features_dir
        else (out_root / "pca_features")
    )

    grm_id = Path(preset.grm_prefix + ".grm.id")
    grm_bin = Path(preset.grm_prefix + ".grm.bin")
    grm_sp = Path(preset.grm_prefix + ".grm.sp")
    if not grm_id.is_file():
        raise FileNotFoundError(f"Missing KING GRM .grm.id: {grm_id}")
    if not (grm_bin.is_file() or grm_sp.is_file()):
        raise FileNotFoundError(f"Missing KING GRM: need {grm_bin} or {grm_sp}")
    if not preset.keep_file.is_file():
        print(f"[IDP-HEREG] WARNING: keep file not found: {preset.keep_file} (will rely on .grm.id).")

    timing_rows: list[dict] = []
    t_run_start = time.perf_counter()
    r_run_start = resource.getrusage(resource.RUSAGE_SELF)

    sample_ids_path = _ensure_sample_ids(preset, timing_rows)
    with open(sample_ids_path) as f:
        sample_ids = set(_norm_iid(line) for line in f if line.strip()) - {""}
    n_preset_ids = len(sample_ids)

    if args.random_sample_fraction is not None:
        frac = float(args.random_sample_fraction)
        if frac <= 0 or frac > 1:
            raise SystemExit("--random-sample-fraction must be in (0, 1].")
        if frac < 1:
            out_root.mkdir(parents=True, exist_ok=True)
            ids_sorted = sorted(sample_ids)
            rng = np.random.default_rng(int(args.random_sample_seed))
            k = max(1, int(round(frac * len(ids_sorted))))
            pick = rng.choice(np.array(ids_sorted, dtype=object), size=k, replace=False)
            sample_ids = {str(x) for x in pick.tolist()}
            subset_path = out_root / "sample_ids_subset.txt"
            with open(subset_path, "w") as f:
                for iid in sorted(sample_ids):
                    f.write(f"{iid}\n")
            print(
                f"[IDP-HEREG] Random subset {len(sample_ids)}/{n_preset_ids} "
                f"(fraction={frac}, seed={args.random_sample_seed}) → {subset_path}"
            )
    print(f"[IDP-HEREG] {preset.label}: {len(sample_ids)} samples for phenotype alignment.")

    # ---- Load phenotype -------------------------------------------------------
    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)

    if use_npy:
        suff = "" if args.mesh_npy_glob_all else (args.mesh_npy_suffix or "").strip()
        df_idp, feature_cols = _load_mesh_phenotype_npy(
            phenotype_npy_dir, sample_ids, name_suffix=suff
        )
        eids = df_idp["IID_norm"].astype(str).tolist()
        Y = df_idp[feature_cols].values.astype(np.float64)

    elif phenotype_csv.is_dir():
        iids_loaded, Y, feature_cols = _load_feature_dir_fast(phenotype_csv, sample_ids)
        eids = iids_loaded

    else:
        df_idp = pd.read_csv(phenotype_csv)
        id_col = "eid" if "eid" in df_idp.columns else "EID" if "EID" in df_idp.columns else None
        if id_col is None:
            raise ValueError(f"Phenotype CSV must contain 'eid' or 'EID'; got {list(df_idp.columns[:8])}...")
        feature_cols = _choose_numeric_feature_cols(df_idp, id_col=id_col)
        if not feature_cols:
            raise ValueError(f"No usable numeric phenotype columns in {phenotype_csv}.")
        df_idp["IID_norm"] = df_idp[id_col].apply(_norm_iid)
        df_idp = df_idp[df_idp["IID_norm"].isin(sample_ids)].copy()
        if df_idp.empty:
            raise ValueError("No overlap between phenotype IIDs and KING sample set.")
        eids = df_idp["IID_norm"].astype(str).tolist()
        Y = df_idp[feature_cols].values.astype(np.float64)

    # Impute NaN with column means
    nan_mask = np.isnan(Y)
    if nan_mask.any():
        col_means = np.nanmean(Y, axis=0)
        nan_where = np.where(nan_mask)
        Y[nan_where] = col_means[nan_where[1]]

    _record_step(timing_rows, preset.label, "load_phenotype", t0, r0)
    print(f"[IDP-HEREG] Loaded {Y.shape[0]} samples × {Y.shape[1]} features.")

    # ---- Dimensionality reduction (raw features, no residualization) ----------
    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    standardize = bool(args.standardize_features)
    explained_variance_ratio: list[float] = []

    if dim_reduction == "qr":
        pc_scores, eids_sub = qr_only(Y, eids, standardize=standardize)
        _record_step(timing_rows, preset.label, "dim_reduction_qr", t0, r0)
        print(f"[IDP-HEREG] QR: {pc_scores.shape[1]} orthonormal components.")
    else:
        n_comp = min(int(args.n_pca), PCA_N_COMPONENTS)
        pc_scores, eids_sub, explained_variance_ratio = pca_only(
            Y, eids, n_components=n_comp, standardize=standardize
        )
        _record_step(timing_rows, preset.label, "dim_reduction_pca", t0, r0)
        total_var = float(np.sum(explained_variance_ratio)) if explained_variance_ratio else 0.0
        print(f"[IDP-HEREG] PCA: {pc_scores.shape[1]} components, total explained variance={total_var:.4f}")

    n_save = pc_scores.shape[1]
    max_features_n = n_save if args.max_features is None else min(int(args.max_features), n_save)

    # Determine write_dir:
    #   PCA: always pca_features/ (permanent)
    #   QR + hereg/reml/both: temp dir (deleted after GCTA), or pca_features/ if explicitly set
    if dim_reduction == "pca":
        write_dir: Path = features_dir
    elif args.pca_features_dir:
        write_dir = features_dir  # user explicitly chose a dir
    else:
        write_dir = Path(tempfile.mkdtemp(prefix="qr_tmp_", dir=str(out_root)))

    _sfw = max(1, int(getattr(args, "save_features_workers", 1)))

    def _flush_features_to_disk() -> None:
        write_dir.mkdir(parents=True, exist_ok=True)

        def _write_one(i: int) -> None:
            out_df = pd.DataFrame({"FID": eids_sub, "IID": eids_sub, str(i): pc_scores[:, i]})
            out_df.to_csv(write_dir / f"Feature_{i}.csv", sep=" ", index=False)

        if _sfw <= 1:
            for i in range(max_features_n):
                _write_one(i)
        else:
            with ThreadPoolExecutor(max_workers=_sfw) as ex:
                list(ex.map(_write_one, range(max_features_n)))

        if explained_variance_ratio and dim_reduction == "pca":
            with open(write_dir / PCA_EXPLAINED_VARIANCE_RATIO_FILE, "w") as f:
                json.dump(explained_variance_ratio, f)

    t0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    _flush_features_to_disk()
    _record_step(timing_rows, preset.label, "save_features", t0, r0)
    _extra = f" (workers={_sfw})" if _sfw > 1 else ""
    print(f"[IDP-HEREG] Wrote {max_features_n} Feature_*.csv to {write_dir}{_extra}")

    hereg_sum_h2: float | None = None
    reml_sum_h2: float | None = None
    reml_mean_h2: float | None = None

    # ---- GCTA --HEreg ---------------------------------------------------------
    if h2_method in ("hereg", "both"):
        t0 = time.perf_counter()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        _, hereg_sum_h2 = run_hereg_on_features(
            features_dir=write_dir,
            out_root=out_root,
            preset=preset,
            gcta_bin=args.gcta_bin,
            gcta_threads=int(args.gcta_threads),
            ccovar_path=str(ccovar_path),
            qcovar_path=str(qcovar_path),
            max_features=max_features_n,
            reuse_hereg=bool(args.reuse_hereg),
            parallel_jobs=int(args.hereg_parallel_jobs),
        )
        _record_step(timing_rows, preset.label, "heritability_hereg", t0, r0)

    # ---- GCTA --reml ----------------------------------------------------------
    if h2_method in ("reml", "both"):
        t0 = time.perf_counter()
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        _, reml_mean_h2, reml_sum_h2 = run_reml_on_features(
            features_dir=write_dir,
            out_root=out_root,
            preset=preset,
            gcta_bin=args.gcta_bin,
            gcta_threads=int(args.gcta_threads),
            ccovar_path=str(ccovar_path),
            qcovar_path=str(qcovar_path),
            max_features=max_features_n,
            reuse_hsq=bool(args.reuse_hsq),
            parallel_jobs=int(args.reml_parallel_jobs),
        )
        _record_step(timing_rows, preset.label, "heritability_reml", t0, r0)

    # Clean up QR temp dir if it was not an explicit user path
    if dim_reduction == "qr" and not args.pca_features_dir:
        shutil.rmtree(str(write_dir), ignore_errors=True)
        print(f"[IDP-HEREG] Removed temp QR feature files from {write_dir}")

    # ---- Timing summary -------------------------------------------------------
    t_run_end = time.perf_counter()
    r_run_end = resource.getrusage(resource.RUSAGE_SELF)
    wall_total = round(t_run_end - t_run_start, 3)
    cpu_u_total = round(r_run_end.ru_utime - r_run_start.ru_utime, 3)
    cpu_s_total = round(r_run_end.ru_stime - r_run_start.ru_stime, 3)
    peak_rss = max((float(r["max_rss_mb"]) for r in timing_rows), default=round(r_run_end.ru_maxrss / 1024.0, 2))
    timing_rows.append({
        "label": preset.label, "step": "total",
        "wall_sec": wall_total, "cpu_user_sec": cpu_u_total,
        "cpu_sys_sec": cpu_s_total, "max_rss_mb": round(peak_rss, 2),
    })
    print(
        f"[IDP-HEREG] Total: wall {wall_total}s, "
        f"CPU user {cpu_u_total}s, CPU sys {cpu_s_total}s, peak RSS {peak_rss} MiB"
    )

    _write_timing_table(timing_rows, out_root, preset.label)
    permanent_pca_dir = features_dir if dim_reduction == "pca" else (
        features_dir if args.pca_features_dir else None
    )
    _write_arena_manifest(
        out_root,
        preset=preset,
        phenotype_csv=phenotype_csv if not use_npy else None,
        phenotype_mode="npy" if use_npy else "csv",
        phenotype_npy_dir=phenotype_npy_dir if use_npy else None,
        ccovar=ccovar_path,
        qcovar=qcovar_path,
        n_samples=len(eids_sub),
        n_features=len(feature_cols),
        pca_features_dir=permanent_pca_dir,
        dim_reduction=dim_reduction,
        h2_method=h2_method,
        hereg_sum_h2=float(hereg_sum_h2) if hereg_sum_h2 is not None else None,
        reml_sum_h2=float(reml_sum_h2) if reml_sum_h2 is not None else None,
        reml_mean_h2=float(reml_mean_h2) if reml_mean_h2 is not None else None,
    )


def main() -> None:
    run_king_hereg(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
