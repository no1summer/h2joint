# cohort_build_h2.py
#
# One-time (cached) cohort builder for engine_128_T1_cnn_h2joint.py.
#
# The existing DeepENDO training CSVs (splits_large/*, cv_splits_mixed_ethnicity/*)
# use a "mixed ethnicity" imaging cohort that has ZERO overlap with the
# over5-kinship GRM cohort used for heritability (different sub-cohorts of UKB,
# same eid numbering space). volume_manifest.csv (built for a prior IDP feature
# extraction run) covers 8019/8126 of the over5_discovery GRM eids with real,
# existing T1 paths -- that's the manifest used here instead.
#
# Produces, under CACHE_DIR (run once; subsequent runs reuse the cache):
#   train_cohort.csv / val_cohort.csv      eid, T1_unbiased_linear  (aedataset-compatible)
#   train_eids.txt / val_eids.txt          canonical queue row order (one IID per line)
#   train_grm.npy / val_grm.npy            dense float32 GRM submatrix, same row order as eids.txt
#   train_grm_diag.npy / val_grm_diag.npy  GRM diagonal (GCTA self-relatedness, not assumed 1)
#   train_he_denom.npy / val_he_denom.npy  scalar = sum_{i<j} K_ij^2, for he_loss.py
#   train_covar_X.npy / val_covar_X.npy    covariate design matrix (intercept+sex+site+qcovar, z-scored)
#   val_grm_gcta.grm.bin / .grm.id         GCTA dense-binary format, for reusing
#                                          run_idp_pipeline_king_hereg.run_hereg_on_features
#                                          as the per-epoch validation H^2 metric.
#   meta.json

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AGENT_MODEL_DIR = "/data484_4/txia2/AGENT/model"
if _AGENT_MODEL_DIR not in sys.path:
    sys.path.insert(0, _AGENT_MODEL_DIR)
from heritability_regression import load_grm_dense_bin  # noqa: E402

GRM_PREFIX = "/data484_4/txia2/AGENT/grm_subset_king/king_over5_gcta_discovery"
VOLUME_MANIFEST = "/data484_4/txia2/gwas_practice/individual_phenos/z_graph_fusion_residual_distill/volume_manifest.csv"
CCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_ccovar_discovery"
QCOVAR_PATH = "/data484_4/txia2/gwas_practice/T1_qcovar_discovery"
KING_KIN0 = "/data484_4/txia2/gwas_practice/KING/king_output.kin0"  # pairwise KING kinship
CACHE_DIR = Path("/data484_4/txia2/DeepENDO/training/T1_128/h2_joint_cohort")
SEED = 42
# Kinship threshold for clustering: 0.0221 ≈ 3rd-degree relatives.
# All subjects in the same kinship cluster are assigned to the same split
# so that train and val GRMs have no cross-split genetic covariance.
KINSHIP_SPLIT_THRESHOLD = 0.0221

# Age, sex, age^2, sex x age (+ sex x age^2), 10 ancestry PCs, ICV/head-size and
# scanner-related UKB fields -- all already present in T1_qcovar_discovery.
QCOVAR_NUMERIC_COLS = [
    "PC0", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9",
    "AGE", "25735", "AGE^2", "SEXxAGE", "SEXxAGE^2",
    "25000", "25756", "25757", "25758", "25759", "53", "53^2",
]


def _required_files(cache_dir: Path) -> list:
    names = [
        "train_cohort.csv", "val_cohort.csv",
        "train_eids.txt", "val_eids.txt",
        "train_grm.npy", "train_grm_diag.npy", "train_he_denom.npy",
        "val_grm.npy", "val_grm_diag.npy", "val_he_denom.npy",
        "val_grm_gcta.grm.bin", "val_grm_gcta.grm.id",
        "train_covar_X.npy", "val_covar_X.npy",
        "meta.json",
    ]
    return [cache_dir / n for n in names]


def _write_gcta_dense_bin(G: np.ndarray, ids: pd.DataFrame, out_prefix: Path) -> None:
    """GCTA --make-grm-bin format: lower triangle, row-major, float32. Round-trip
    verified against heritability_regression.load_grm_dense_bin and a live gcta64 run."""
    n = G.shape[0]
    with open(str(out_prefix) + ".grm.id", "w") as f:
        for _, row in ids.iterrows():
            f.write(f"{row['FID']} {row['IID']}\n")
    vals = np.empty(n * (n + 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n):
        vals[idx: idx + i + 1] = G[i, : i + 1]
        idx += i + 1
    vals.tofile(str(out_prefix) + ".grm.bin")


def _build_covar_design(eids_ordered: list, ccovar: pd.DataFrame, qcovar: pd.DataFrame) -> np.ndarray:
    eid_int = [int(e) for e in eids_ordered]
    ccov = ccovar.set_index("IID").loc[eid_int]
    qcov = qcovar.set_index("IID").loc[eid_int]

    sex = ccov["SEX"].to_numpy(dtype=np.float64).reshape(-1, 1)
    site_dummies = pd.get_dummies(ccov["54"], prefix="site", drop_first=True).to_numpy(dtype=np.float64)

    quant = qcov[QCOVAR_NUMERIC_COLS].to_numpy(dtype=np.float64)
    quant = (quant - quant.mean(axis=0, keepdims=True)) / (quant.std(axis=0, keepdims=True) + 1e-8)

    intercept = np.ones((len(eids_ordered), 1), dtype=np.float64)
    X = np.concatenate([intercept, sex, site_dummies, quant], axis=1)
    return X.astype(np.float32)


def _kinship_split(usable_positions, df_id, kin0_path, threshold, seed):
    """
    Split subjects into train/val so that all subjects in the same kinship
    cluster (KING kinship > threshold) land in the same split.

    Algorithm:
      1. Build a union-find over usable subjects.
      2. Read the pairwise KING kin0 file; union any usable pair with kinship > threshold.
      3. Extract connected components (clusters).
      4. Sort clusters largest-first; greedily assign each to whichever split is
         currently smaller, breaking ties by RNG → balanced ~50/50 split.
    """
    from collections import defaultdict

    usable_eids = set(str(df_id.iloc[p]["IID"]) for p in usable_positions)
    pos_by_eid  = {str(df_id.iloc[p]["IID"]): p for p in usable_positions}

    # Union-Find
    parent = {e: e for e in usable_eids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    # Read only relevant columns; filter to usable-subject pairs above threshold
    kin = pd.read_csv(kin0_path, sep="\t", usecols=["ID1", "ID2", "Kinship"],
                      dtype={"ID1": str, "ID2": str, "Kinship": float})
    related = kin[(kin["Kinship"] > threshold) &
                  kin["ID1"].isin(usable_eids) & kin["ID2"].isin(usable_eids)]
    for _, row in related.iterrows():
        union(row["ID1"], row["ID2"])
    n_pairs = len(related)

    # Collect clusters
    clusters = defaultdict(list)
    for e in usable_eids:
        clusters[find(e)].append(e)
    clusters = sorted(clusters.values(), key=len, reverse=True)  # largest first

    # Greedy balanced assignment
    rng = np.random.default_rng(seed)
    train_eids_set, val_eids_set = set(), set()
    for cluster in clusters:
        if len(train_eids_set) < len(val_eids_set):
            train_eids_set.update(cluster)
        elif len(val_eids_set) < len(train_eids_set):
            val_eids_set.update(cluster)
        else:
            (train_eids_set if rng.random() < 0.5 else val_eids_set).update(cluster)

    train_pos = np.sort([pos_by_eid[e] for e in train_eids_set])
    val_pos   = np.sort([pos_by_eid[e] for e in val_eids_set])
    print(f"[cohort_build_h2] Kinship split (threshold={threshold}): "
          f"{len(related)} related pairs clustered → "
          f"train={len(train_pos)}, val={len(val_pos)}")
    return train_pos, val_pos


def build_or_load(cache_dir: Path = CACHE_DIR, seed: int = SEED, force: bool = False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not force and all(p.is_file() for p in _required_files(cache_dir)):
        with open(cache_dir / "meta.json") as f:
            meta = json.load(f)
        print(f"[cohort_build_h2] Using cached cohort artifacts in {cache_dir} ({meta['n_train']}/{meta['n_val']} train/val)")
        return {"cache_dir": cache_dir, **meta}

    print("[cohort_build_h2] Building cohort from scratch ...")
    G_full, df_id = load_grm_dense_bin(GRM_PREFIX + ".grm.id", GRM_PREFIX + ".grm.bin")
    df_id["IID"] = df_id["IID"].astype(int)

    vm = pd.read_csv(VOLUME_MANIFEST)
    vm["eid"] = vm["sid"].str.split("_").str[0].astype(int)
    path_by_eid = dict(zip(vm["eid"], vm["t1_path"]))

    ccovar = pd.read_csv(CCOVAR_PATH, sep=r"\s+")
    qcovar = pd.read_csv(QCOVAR_PATH, sep=r"\s+")
    covar_ids = set(ccovar["IID"]) & set(qcovar["IID"])

    usable_mask = df_id["IID"].isin(path_by_eid.keys()).to_numpy() & df_id["IID"].isin(covar_ids).to_numpy()
    usable_positions = np.where(usable_mask)[0]
    print(
        f"[cohort_build_h2] {len(usable_positions)}/{len(df_id)} over5_discovery subjects have "
        f"both a T1 path (volume_manifest.csv) and covariate coverage (ccovar/qcovar)."
    )

    train_pos, val_pos = _kinship_split(
        usable_positions, df_id, KING_KIN0, KINSHIP_SPLIT_THRESHOLD, seed
    )

    meta = {
        "seed": seed,
        "split_method": "kinship_cluster",
        "kinship_split_threshold": KINSHIP_SPLIT_THRESHOLD,
        "king_kin0": KING_KIN0,
        "n_over5_discovery": int(len(df_id)),
        "n_usable_with_t1_path": int(len(usable_positions)),
        "grm_prefix": GRM_PREFIX,
        "volume_manifest": VOLUME_MANIFEST,
        "ccovar": CCOVAR_PATH,
        "qcovar": QCOVAR_PATH,
    }

    for split_name, positions in (("train", train_pos), ("val", val_pos)):
        ids_sub = df_id.iloc[positions].reset_index(drop=True)
        eids_ordered = [str(x) for x in ids_sub["IID"].tolist()]

        G_sub = G_full[np.ix_(positions, positions)].astype(np.float32)
        diag_sub = np.diagonal(G_sub).copy()
        he_denom = 0.5 * (
            float(np.sum(G_sub.astype(np.float64) ** 2)) - float(np.sum(diag_sub.astype(np.float64) ** 2))
        )

        np.save(cache_dir / f"{split_name}_grm.npy", G_sub)
        np.save(cache_dir / f"{split_name}_grm_diag.npy", diag_sub)
        np.save(cache_dir / f"{split_name}_he_denom.npy", np.array(he_denom, dtype=np.float64))

        with open(cache_dir / f"{split_name}_eids.txt", "w") as f:
            f.write("\n".join(eids_ordered) + "\n")

        cohort_df = pd.DataFrame({
            "eid": eids_ordered,
            "T1_unbiased_linear": [path_by_eid[int(e)] for e in eids_ordered],
        })
        cohort_df.to_csv(cache_dir / f"{split_name}_cohort.csv", index=False)

        X = _build_covar_design(eids_ordered, ccovar, qcovar)
        np.save(cache_dir / f"{split_name}_covar_X.npy", X)

        if split_name == "val":
            _write_gcta_dense_bin(G_sub, ids_sub, cache_dir / "val_grm_gcta")

        meta[f"n_{split_name}"] = int(len(eids_ordered))
        print(f"[cohort_build_h2] {split_name}: {len(eids_ordered)} subjects, GRM {G_sub.shape}, covar X {X.shape}")

    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[cohort_build_h2] Done. Artifacts cached in {cache_dir}")
    return {"cache_dir": cache_dir, **meta}


if __name__ == "__main__":
    build_or_load(force=False)
