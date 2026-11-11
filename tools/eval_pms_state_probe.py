#!/usr/bin/env python
# encoding: utf-8
"""
eval_pms_state_probe.py
=======================
Post-hoc linear state probe on PMS probe features (.npz).

Only reads the .npz produced by ``export_pms_probe_features.py`` (numpy +
sklearn only, no torch / no model code), so PMS state labels can never update
the ReID model or participate in model selection. Probe protocol:

  * identity-disjoint CV: ``GroupKFold(n_splits=5)``, ``groups=pid``
    -> no identity leakage between train/test folds (verified per fold)
  * ``StandardScaler`` (fit on the train fold only) + balanced
    ``LogisticRegression``  -> Linear State Probe
  * metrics: Balanced Accuracy and ROC-AUC, per-fold + overall (mean +/- std
    over folds, plus pooled out-of-fold metrics)

Checks (fail hard with a clear message unless stated as diagnostic):
  * NaN / Inf in features
  * state labels in {0,1} and both classes present
  * feature matrix 2-D, dim consistent, all arrays same length
  * PID leakage between train/test identities in every fold
  * feature L2-norm statistics (diagnostic; zero-norm rows are warned)

Usage
-----
  python tools/eval_pms_state_probe.py --npz output2/pms_probe/cdg_pwc_probe_features.npz
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

STATE_NAMES = {0: "walk", 1: "cycle"}


def parse_args():
    parser = argparse.ArgumentParser(description="Linear PMS state probe on exported .npz")
    parser.add_argument("--npz", type=str, required=True,
                        help="path to *_pwc_probe_features.npz")
    parser.add_argument("--out_json", type=str, default=None,
                        help="output json (default: <npz>.state_probe.json)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="GroupKFold n_splits (default 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed before GroupKFold")
    parser.add_argument("--max_iter", type=int, default=2000,
                        help="LogisticRegression max_iter")
    return parser.parse_args()


def load_and_check(npz_path):
    """Load .npz and run the hard checks; returns a dict of arrays + diagnostics."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError("npz not found: {}".format(npz_path))
    with np.load(npz_path, allow_pickle=True) as d:
        required = ["feat", "pid", "camid", "state", "path"]
        missing = [k for k in required if k not in d]
        if missing:
            raise RuntimeError(
                "npz missing keys {}; is it an export of "
                "export_pms_probe_features.py?".format(missing))
        feat = d["feat"].astype(np.float64)
        pid = d["pid"].astype(np.int64)
        camid = d["camid"].astype(np.int64)
        state = d["state"].astype(np.int64)
        path = list(d["path"])
        model_type = str(d["model_type"]) if "model_type" in d else "?"
        feat_dim_meta = int(d["feat_dim"][0]) if "feat_dim" in d else None

    n = feat.shape[0]
    if feat.ndim != 2:
        raise RuntimeError("feat must be 2-D, got shape {}".format(feat.shape))
    lens = {"feat": n, "pid": len(pid), "camid": len(camid),
            "state": len(state), "path": len(path)}
    bad = [k for k, v in lens.items() if v != n]
    if bad:
        raise RuntimeError("length mismatch: {}".format(
            ", ".join("{}={}".format(k, v) for k, v in lens.items())))
    if feat_dim_meta is not None and feat_dim_meta != feat.shape[1]:
        raise RuntimeError(
            "feat dim {} disagrees with metadata {}".format(feat.shape[1], feat_dim_meta))
    if not np.isfinite(feat).all():
        raise RuntimeError("feature matrix contains NaN or Inf")
    if not set(np.unique(state)).issubset({0, 1}):
        raise RuntimeError("state labels outside {{0,1}}: {}".format(
            np.unique(state).tolist()))
    if len(np.unique(state)) < 2:
        raise RuntimeError(
            "only one state class present ({}); a state probe needs both walk "
            "and cycle".format(np.unique(state).tolist()))

    norms = np.linalg.norm(feat, axis=1)
    norm_report = {
        "overall": {"mean": float(norms.mean()), "std": float(norms.std()),
                    "min": float(norms.min()), "max": float(norms.max())},
    }
    for s in (0, 1):
        m = norms[state == s]
        norm_report[STATE_NAMES[s]] = {
            "mean": float(m.mean()), "std": float(m.std()),
            "min": float(m.min()), "max": float(m.max())}
    n_zero = int((norms == 0).sum())
    if n_zero:
        print("[warn] {} zero-norm rows (StandardScaler cannot normalize these)".format(n_zero))

    checks = {
        "nan_inf": True,
        "state_labels_valid": True,
        "both_classes_present": True,
        "shape_consistent": True,
        "dim_matches_metadata": feat_dim_meta is None or feat_dim_meta == feat.shape[1],
        "zero_norm_rows": int(n_zero),
    }
    return dict(feat=feat, pid=pid, camid=camid, state=state, path=path,
                model_type=model_type, norm_report=norm_report, checks=checks)


def run_probe(data, n_splits, seed, max_iter):
    """Identity-disjoint GroupKFold + StandardScaler + balanced LogisticRegression."""
    feat, pid, state = data["feat"], data["pid"], data["state"]
    n = len(feat)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    X, y, groups = feat[order], state[order], pid[order]

    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    oof_pred = np.full(n, np.nan)
    oof_proba = np.full(n, np.nan)
    oof_mask = np.zeros(n, dtype=bool)

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=groups)):
        tr_pids, te_pids = set(groups[tr_idx]), set(groups[te_idx])
        leak = tr_pids & te_pids
        if leak:
            raise RuntimeError(
                "fold {}: PID leakage between train/test identities: {}".format(
                    fold, sorted(leak)))
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            print("[warn] fold {}: degenerate class coverage (train {}, test {}); "
                  "skipping fold".format(fold, np.unique(y_tr).tolist(),
                                         np.unique(y_te).tolist()))
            folds.append({"fold": fold, "train_n": int(len(tr_idx)),
                          "test_n": int(len(te_idx)), "train_ids": int(len(tr_pids)),
                          "test_ids": int(len(te_pids)),
                          "balanced_acc": None, "roc_auc": None, "skipped": True})
            continue

        scaler = StandardScaler().fit(X[tr_idx])
        X_tr = scaler.transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=max_iter)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        proba = clf.predict_proba(X_te)[:, 1]
        ba = float(balanced_accuracy_score(y_te, pred))
        auc = float(roc_auc_score(y_te, proba))
        folds.append({"fold": fold, "train_n": int(len(tr_idx)),
                      "test_n": int(len(te_idx)), "train_ids": int(len(tr_pids)),
                      "test_ids": int(len(te_pids)),
                      "balanced_acc": ba, "roc_auc": auc, "skipped": False})
        oof_pred[te_idx] = pred
        oof_proba[te_idx] = proba
        oof_mask[te_idx] = True

    valid = [f for f in folds if not f["skipped"]]
    if not valid:
        raise RuntimeError("no valid fold produced a probe result (all skipped)")
    ba_vals = [f["balanced_acc"] for f in valid]
    auc_vals = [f["roc_auc"] for f in valid]

    def ms(v):
        return [float(np.mean(v)), float(np.std(v))]

    summary = {
        "method": "GroupKFold(n_splits={}, groups=pid, seed={})".format(n_splits, seed),
        "scaler": "StandardScaler (train-fold only)",
        "classifier": "LogisticRegression(class_weight='balanced', max_iter={})".format(max_iter),
        "balanced_acc_mean_std": ms(ba_vals),
        "roc_auc_mean_std": ms(auc_vals),
        "n_valid_folds": len(valid),
        "pooled_oof_balanced_acc": float(balanced_accuracy_score(
            y[oof_mask], oof_pred[oof_mask])),
        "pooled_oof_roc_auc": float(roc_auc_score(y[oof_mask], oof_proba[oof_mask])),
    }
    return folds, summary


def main():
    args = parse_args()
    data = load_and_check(args.npz)
    folds, summary = run_probe(data, args.n_splits, args.seed, args.max_iter)

    # ---- console report ------------------------------------------------------
    vals, cnts = np.unique(data["state"], return_counts=True)
    print("\n=== PMS Linear State Probe (post-hoc, no model update) ===")
    print("npz        : {}".format(args.npz))
    print("model_type : {} | samples {} | ids {} | cams {}".format(
        data["model_type"], len(data["feat"]), len(np.unique(data["pid"])),
        len(np.unique(data["camid"]))))
    print("classes    : {}".format({STATE_NAMES[int(v)]: int(c)
                                    for v, c in zip(vals, cnts)}))
    print("feat dim   : {}".format(data["feat"].shape[1]))
    print("L2 norm    : mean {:.2f} std {:.2f}  (walk {:.2f}+-{:.2f}, cycle {:.2f}+-{:.2f})".format(
        data["norm_report"]["overall"]["mean"], data["norm_report"]["overall"]["std"],
        data["norm_report"]["walk"]["mean"], data["norm_report"]["walk"]["std"],
        data["norm_report"]["cycle"]["mean"], data["norm_report"]["cycle"]["std"]))
    print("\n  fold | train_n | test_n | train_ids | test_ids | BalancedAcc | ROC-AUC")
    for f in folds:
        ba = "--" if f["balanced_acc"] is None else "{:.4f}".format(f["balanced_acc"])
        auc = "--" if f["roc_auc"] is None else "{:.4f}".format(f["roc_auc"])
        print("  {:4d} | {:7d} | {:6d} | {:9d} | {:8d} | {:<11s} | {}".format(
            f["fold"], f["train_n"], f["test_n"], f["train_ids"], f["test_ids"], ba, auc))
    print("\nBalanced Acc (mean+-std over folds) : {:.4f} +- {:.4f}".format(
        *summary["balanced_acc_mean_std"]))
    print("ROC-AUC     (mean+-std over folds) : {:.4f} +- {:.4f}".format(
        *summary["roc_auc_mean_std"]))
    print("Pooled out-of-fold                 : BA {:.4f} | AUC {:.4f}".format(
        summary["pooled_oof_balanced_acc"], summary["pooled_oof_roc_auc"]))

    # ---- json ---------------------------------------------------------------
    out_json = args.out_json or (args.npz.rsplit(".npz", 1)[0] + ".state_probe.json")
    result = {
        "npz": os.path.abspath(args.npz),
        "model_type": data["model_type"],
        "n_samples": int(len(data["feat"])),
        "n_ids": int(len(np.unique(data["pid"]))),
        "n_cams": int(len(np.unique(data["camid"]))),
        "feat_dim": int(data["feat"].shape[1]),
        "class_counts": {STATE_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)},
        "feature_norm": data["norm_report"],
        "checks": data["checks"],
        "cv": {"method": summary["method"], "scaler": summary["scaler"],
               "classifier": summary["classifier"],
               # leakage is checked per fold and raises on violation; reaching
               # this point means every fold had disjoint train/test identities
               "pid_leakage_free": True},
        "folds": folds,
        "summary": summary,
        "note": "post-hoc only: PMS state labels were never used to update the ReID model or select it",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n[save] wrote {}".format(out_json))


if __name__ == "__main__":
    main()
