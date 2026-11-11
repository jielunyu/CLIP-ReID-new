#!/usr/bin/env python
# encoding: utf-8
"""
export_pms_probe_features.py
============================
PMS (Pedestrian Motion State) State-Probe feature exporter.

Exports the *actual inference retrieval descriptor* of a trained ReID model
over the PMS dataset (PWC, walk=0 / cycle=1) into a single .npz for a
**post-hoc** linear state probe. This script only reads the ReID model
(``model.eval()`` + ``torch.no_grad()``); it never updates weights, and the
PMS state labels never touch training or model selection.

Descriptor
----------
Both pipelines export the SAME descriptor used by ``do_inference``:

    torch.cat([img_feature(768), img_feature_proj(512)], dim=1)  -> [B, 1280]

with ``NECK_FEAT='before'``. ``--model_type`` selects the model builder /
checkpoint family used to load the model:

    baseline -> ``model/make_model.py::make_model``   (no text/prompt branch)
    cdg      -> ``model/make_model_clipreid.py::make_model``  (CLIP-ReID, prompt)
    auto     -> detected from checkpoint keys (prompt_learner present => cdg)

Reuses (does not copy existing logic):
  - config            : ``config/defaults.py`` ``cfg`` + ``--config_file``
  - model + loader    : ``make_model``/``make_model_clipreid`` + ``load_param``
  - PMS dataset/loader: ``datasets/make_dataloader_clipreid`` (PWC carries the
    state in the trackid slot, which surfaces as the *viewids* element of the
    val_loader batches: 0 = walk, 1 = cycle).

Output
------
    <out_dir>/<model_type>_pwc_probe_features.npz
        feat  [N, 1280] float32
        pid   [N] int64
        camid [N] int64
        state [N] int64   (0 = walk, 1 = cycle)
        path  [N] object  (full image path)
        + metadata scalars (model_type, descriptor, feat_dim, n_samples)

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python tools/export_pms_probe_features.py \
      --config_file configs/person/vit_clipreid.yml --model_type cdg
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from config import cfg

STATE_NAMES = {0: "walk", 1: "cycle"}


def parse_args():
    parser = argparse.ArgumentParser(description="Export PMS probe features (post-hoc)")
    parser.add_argument(
        "--config_file", type=str, default="configs/person/vit_clipreid.yml",
        help="config to load model-architecture fields from (SIE, SIZE, NECK_FEAT, ...)")
    parser.add_argument(
        "--model_type", type=str, default="auto", choices=["auto", "baseline", "cdg"],
        help="which pipeline the checkpoint was trained with (auto detects from keys)")
    parser.add_argument(
        "--dataset", type=str, default="pwc",
        help="PMS dataset name registered in make_dataloader_clipreid.__factory")
    parser.add_argument(
        "--weight", type=str, default=None,
        help="checkpoint path (default: cfg.TEST.WEIGHT)")
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="output dir (default: output2/pms_probe)")
    parser.add_argument("--camera_num", type=int, default=None,
                        help="override camera_num (only matters if SIE_CAMERA on)")
    parser.add_argument("--view_num", type=int, default=None,
                        help="override view_num (only matters if SIE_VIEW on)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id override (default: cfg.MODEL.DEVICE_ID, "
                             "or existing CUDA_VISIBLE_DEVICES if set)")
    return parser.parse_args()


def detect_builder(model_type, weight):
    """Decide which model builder to use; derive num_class from the checkpoint."""
    sd = torch.load(weight, map_location="cpu")
    cls_key = "classifier.weight" if "classifier.weight" in sd else "module.classifier.weight"
    if cls_key not in sd:
        raise RuntimeError(
            "checkpoint {} has no 'classifier.weight'; expected a full ReID "
            "model state_dict".format(weight))
    num_class = int(sd[cls_key].shape[0])
    has_prompt = any("prompt_learner" in k for k in sd)

    if model_type == "auto":
        model_type = "cdg" if has_prompt else "baseline"
        print("[model] auto-detected model_type='{}' "
              "(prompt_learner {} in checkpoint)".format(
                  model_type, "present" if has_prompt else "absent"))
    elif model_type == "cdg" and not has_prompt:
        raise RuntimeError(
            "--model_type cdg but checkpoint has no prompt_learner keys "
            "(it is a baseline model). Pass --model_type baseline.")
    elif model_type == "baseline" and has_prompt:
        raise RuntimeError(
            "--model_type baseline but checkpoint has prompt_learner keys "
            "(it is a cdg/CLIP-ReID model). Pass --model_type cdg.")
    return model_type, num_class


def main():
    args = parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    overrides = ["DATASETS.NAMES", args.dataset]
    if args.weight is not None:
        overrides += ["TEST.WEIGHT", args.weight]
    out_dir = args.out_dir or os.path.join("output2", "pms_probe")
    overrides += ["OUTPUT_DIR", out_dir]
    cfg.merge_from_list(overrides)
    cfg.freeze()

    # respect an externally-set CUDA_VISIBLE_DEVICES; otherwise use --gpu /
    # cfg.MODEL.DEVICE_ID (set the env directly; yacs cannot coerce int keys)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu = args.gpu if args.gpu is not None else cfg.MODEL.DEVICE_ID
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda"

    weight = cfg.TEST.WEIGHT
    if not os.path.exists(weight):
        raise FileNotFoundError(
            "checkpoint not found: {} (set --weight)".format(weight))

    # ---- PMS dataloader (reuse; carries state in the viewids slot) --------
    from datasets.make_dataloader_clipreid import make_dataloader
    train_loader, train_loader_normal, val_loader, num_query, num_classes, \
        cam_num, view_num = make_dataloader(cfg)
    print("[data] PMS dataset '{}' | val_loader {} samples (query+gallery)".format(
        cfg.DATASETS.NAMES, len(val_loader.dataset)))

    # ---- model builder (reuse) --------------------------------------------
    model_type, num_class = detect_builder(args.model_type, weight)
    if model_type == "cdg":
        from model.make_model_clipreid import make_model
    else:
        from model.make_model import make_model
    camera_num = args.camera_num if args.camera_num is not None else cam_num
    view_num_use = args.view_num if args.view_num is not None else view_num
    model = make_model(cfg, num_class=num_class, camera_num=camera_num,
                       view_num=view_num_use)
    model.load_param(weight)          # reuse existing checkpoint loader
    model.eval()
    model.to(device)

    # ---- descriptor extraction (mirrors do_inference) ---------------------
    feat_list, state_loader = [], []
    with torch.no_grad():
        for n_iter, (img, pid, camid, camids_batch, viewids, imgpath) in \
                enumerate(val_loader):
            img = img.to(device)
            camids_in = camids_batch.to(device) if cfg.MODEL.SIE_CAMERA else None
            viewids_in = viewids.to(device) if cfg.MODEL.SIE_VIEW else None
            feat = model(img, cam_label=camids_in, view_label=viewids_in)
            feat_list.append(feat.detach().cpu().numpy().astype(np.float32))
            state_loader.append(viewids.cpu().numpy())
    feat = np.concatenate(feat_list, axis=0)
    state_loader = np.concatenate(state_loader, axis=0)
    print("[model] descriptor shape {} | model_type '{}' | checkpoint {}".format(
        feat.shape, model_type, os.path.basename(weight)))

    # ---- metadata, index-aligned to the loader order ----------------------
    from datasets.make_dataloader_clipreid import __factory
    dset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    recs = dset.query + dset.gallery
    if len(recs) != len(feat):
        raise RuntimeError(
            "loader produced {} samples but {} records in the dataset; "
            "order misalignment".format(len(feat), len(recs)))
    state_recs = np.array([r[3] for r in recs], dtype=np.int64)
    if not np.array_equal(state_loader, state_recs):
        raise RuntimeError("state labels from loader disagree with dataset records")

    pid = np.array([r[1] for r in recs], dtype=np.int64)
    camid = np.array([r[2] for r in recs], dtype=np.int64)
    path = np.array([r[0] for r in recs], dtype=object)

    # ---- basic sanity checks (full checks run in the evaluator) -----------
    if not np.isfinite(feat).all():
        raise RuntimeError("feature matrix contains NaN/Inf")
    bad = np.setdiff1d(np.unique(state_loader), [0, 1])
    if bad.size:
        raise RuntimeError("state labels outside {{0,1}}: {}".format(bad.tolist()))
    if len(np.unique(state_loader)) < 2:
        raise RuntimeError("PMS features cover only one state class")

    # ---- save ---------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "{}_pwc_probe_features.npz".format(model_type))
    np.savez_compressed(
        out_path,
        feat=feat,
        pid=pid,
        camid=camid,
        state=state_loader,
        path=path,
        model_type=np.array(model_type, dtype=object),
        descriptor=np.array(
            "cat([img_feature(768), img_feature_proj(512)], dim=1)", dtype=object),
        feat_dim=np.array([feat.shape[1]], dtype=np.int64),
        n_samples=np.array([len(feat)], dtype=np.int64),
    )
    meta = {
        "model_type": model_type,
        "checkpoint": weight,
        "config_file": args.config_file,
        "dataset": cfg.DATASETS.NAMES,
        "descriptor": "cat([img_feature(768), img_feature_proj(512)], dim=1)",
        "feat_dim": int(feat.shape[1]),
        "n_samples": int(len(feat)),
        "n_ids": int(len(np.unique(pid))),
        "n_cams": int(len(np.unique(camid))),
        "class_counts": {STATE_NAMES[int(v)]: int(c)
                         for v, c in zip(*np.unique(state_loader, return_counts=True))},
        "neck_feat": cfg.TEST.NECK_FEAT,
        "sie_camera": bool(cfg.MODEL.SIE_CAMERA),
        "sie_view": bool(cfg.MODEL.SIE_VIEW),
        "note": "post-hoc only: state labels were not used to update the ReID model",
    }
    with open(out_path.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    vals, cnts = np.unique(state_loader, return_counts=True)
    print("[save] wrote {}  ({} samples | ids {} | states {})".format(
        out_path, len(feat), len(np.unique(pid)),
        {STATE_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}))
    print("[next] python tools/eval_pms_state_probe.py --npz {}".format(out_path))


if __name__ == "__main__":
    main()
