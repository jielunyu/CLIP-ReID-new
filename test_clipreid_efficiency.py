# -*- coding: utf-8 -*-
"""
test_clipreid_efficiency.py

CLIP-ReID retrieval evaluation + unified inference-efficiency benchmark.

Reported metrics:
1. Inference Params (M): parameters actually executed in the evaluation forward path.
2. GFLOPs: one forward pass for one 256x128 image.
3. Latency, bs=1: ms/image.
4. Throughput, bs=64: images/s.

Timing excludes data loading, preprocessing, CPU-to-GPU transfer,
distance computation, and ranking.
"""

import os
import csv
import json
import math
import time
import argparse
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger


def extract_first_tensor(output):
    """Extract a tensor from common ReID output structures."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        preferred = (
            "features", "feature", "feat", "embedding",
            "embeddings", "global_feat", "inference_feat",
        )
        for key in preferred:
            if key in output:
                try:
                    return extract_first_tensor(output[key])
                except TypeError:
                    pass
        for value in output.values():
            try:
                return extract_first_tensor(value)
            except TypeError:
                continue

    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return extract_first_tensor(value)
            except TypeError:
                continue

    raise TypeError(
        "No tensor was found in the model output. "
        "Please adapt extract_first_tensor() to your forward output."
    )


class CLIPReIDInferenceWrapper(nn.Module):
    """
    Wrap CLIP-ReID into a single-input inference module.

    Camera/view labels are supplied only when the corresponding SIE option is
    enabled. This mirrors the project's original do_inference() behavior and
    avoids entering an SIE branch when cv_embed was not constructed.
    """

    def __init__(self, model, use_camera_sie=False, use_view_sie=False):
        super().__init__()
        self.model = model
        self.use_camera_sie = bool(use_camera_sie)
        self.use_view_sie = bool(use_view_sie)

        # Fail early when the configuration requests SIE but the model does
        # not contain the required embedding parameter.
        if (self.use_camera_sie or self.use_view_sie) and not hasattr(
            self.model, "cv_embed"
        ):
            raise AttributeError(
                "SIE is enabled in cfg, but model.cv_embed does not exist. "
                "Check MODEL.SIE_CAMERA, MODEL.SIE_VIEW, and the checkpoint/config pair."
            )

    def forward(self, images):
        batch_size = images.size(0)

        cam_label = None
        view_label = None

        if self.use_camera_sie:
            cam_label = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=images.device,
            )

        if self.use_view_sie:
            view_label = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=images.device,
            )

        output = self.model(
            images,
            cam_label=cam_label,
            view_label=view_label,
        )
        return extract_first_tensor(output)


def count_loaded_params(model):
    """All parameters stored in the loaded checkpoint/model."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.inference_mode()
def count_executed_inference_params(model, sample):
    """
    Count unique parameters belonging to modules executed by one evaluation
    forward pass. This excludes classifier/text/training-only branches that
    remain stored in the model but are not called during inference.
    """
    called_modules = set()
    handles = []

    def hook(module, _inputs, _output):
        called_modules.add(module)

    for module in model.modules():
        handles.append(module.register_forward_hook(hook))

    try:
        _ = model(sample)
        torch.cuda.synchronize(sample.device)
    finally:
        for handle in handles:
            handle.remove()

    unique_parameters = {}
    for module in called_modules:
        for parameter in module.parameters(recurse=False):
            unique_parameters[id(parameter)] = parameter

    return sum(p.numel() for p in unique_parameters.values()) / 1e6


def compute_gflops(model, sample, backend="fvcore"):
    """
    Compute GFLOPs using one common backend for every compared model.

    fvcore is recommended:
        pip install fvcore

    thop is supported as a fallback:
        pip install thop

    For THOP, MACs are converted using FLOPs = 2 x MACs.
    Do not mix backends across CLIP-ReID, TransReID, and CDG-ReID.
    """
    if backend == "fvcore":
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, sample)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)

        total_flops = float(analysis.total())
        unsupported = dict(analysis.unsupported_ops())
        return total_flops / 1e9, unsupported

    if backend == "thop":
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return 2.0 * float(macs) / 1e9, {}

    raise ValueError(f"Unsupported FLOPs backend: {backend}")


@torch.inference_mode()
def warmup(model, images, iterations):
    for _ in range(iterations):
        _ = model(images)
    torch.cuda.synchronize(images.device)


@torch.inference_mode()
def measure_batch_time(model, images, iterations):
    """Return the average GPU forward time in seconds per batch."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize(images.device)
    start_event.record()

    for _ in range(iterations):
        _ = model(images)

    end_event.record()
    torch.cuda.synchronize(images.device)

    elapsed_ms = start_event.elapsed_time(end_event)
    return (elapsed_ms / 1000.0) / iterations


@torch.inference_mode()
def benchmark_speed(
    model,
    device,
    batch_size,
    image_height,
    image_width,
    warmup_iterations,
    timed_iterations,
    repeats,
    precision,
):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    images = torch.randn(
        batch_size,
        3,
        image_height,
        image_width,
        device=device,
        dtype=dtype,
    )

    # Validate the forward path before timing.
    output = model(images)
    if output.shape[0] != batch_size:
        raise RuntimeError(
            f"Output batch dimension {output.shape[0]} does not match "
            f"input batch dimension {batch_size}."
        )

    warmup(model, images, warmup_iterations)

    per_batch_seconds = []
    for _ in range(repeats):
        per_batch_seconds.append(
            measure_batch_time(model, images, timed_iterations)
        )

    latency_ms_per_image = [
        seconds * 1000.0 / batch_size for seconds in per_batch_seconds
    ]
    throughput_images_per_second = [
        batch_size / seconds for seconds in per_batch_seconds
    ]

    return {
        "latency_ms_per_image_mean": statistics.mean(latency_ms_per_image),
        "latency_ms_per_image_std": (
            statistics.stdev(latency_ms_per_image)
            if len(latency_ms_per_image) > 1 else 0.0
        ),
        "throughput_img_s_mean": statistics.mean(
            throughput_images_per_second
        ),
        "throughput_img_s_std": (
            statistics.stdev(throughput_images_per_second)
            if len(throughput_images_per_second) > 1 else 0.0
        ),
        "output_shape": list(output.shape),
    }


def save_efficiency_results(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "clipreid_efficiency.json"
    csv_path = output_dir / "clipreid_efficiency.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    return json_path, csv_path


def benchmark_clipreid(model, args, output_dir, logger):
    device = torch.device("cuda:0")
    model = model.to(device).eval()

    if args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()

    wrapper = CLIPReIDInferenceWrapper(
        model,
        use_camera_sie=cfg.MODEL.SIE_CAMERA,
        use_view_sie=cfg.MODEL.SIE_VIEW,
    ).to(device).eval()

    # FLOPs and parameter-path tracing use bs=1.
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    sample = torch.randn(
        1,
        3,
        args.img_height,
        args.img_width,
        device=device,
        dtype=dtype,
    )

    loaded_params_m = count_loaded_params(model)
    inference_params_m = count_executed_inference_params(wrapper, sample)

    try:
        gflops, unsupported_ops = compute_gflops(
            wrapper, sample, backend=args.flops_backend
        )
        unsupported_text = (
            json.dumps(unsupported_ops, ensure_ascii=False)
            if unsupported_ops else ""
        )
    except Exception as exc:
        logger.warning("FLOPs calculation failed: %s", exc)
        gflops = float("nan")
        unsupported_text = str(exc)

    bs1 = benchmark_speed(
        model=wrapper,
        device=device,
        batch_size=1,
        image_height=args.img_height,
        image_width=args.img_width,
        warmup_iterations=args.warmup,
        timed_iterations=args.latency_iters,
        repeats=args.repeats,
        precision=args.precision,
    )

    try:
        bs64 = benchmark_speed(
            model=wrapper,
            device=device,
            batch_size=args.throughput_bs,
            image_height=args.img_height,
            image_width=args.img_width,
            warmup_iterations=args.warmup,
            timed_iterations=args.throughput_iters,
            repeats=args.repeats,
            precision=args.precision,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning(
            "CUDA OOM at throughput batch size %d.",
            args.throughput_bs,
        )
        bs64 = {
            "throughput_img_s_mean": float("nan"),
            "throughput_img_s_std": float("nan"),
            "output_shape": ["OOM"],
        }

    result = {
        "method": "CLIP-ReID",
        "checkpoint": cfg.TEST.WEIGHT,
        "precision": args.precision,
        "input_height": args.img_height,
        "input_width": args.img_width,
        "loaded_params_M": loaded_params_m,
        "inference_params_M": inference_params_m,
        "gflops": gflops,
        "flops_backend": args.flops_backend,
        "latency_bs1_ms_mean": bs1["latency_ms_per_image_mean"],
        "latency_bs1_ms_std": bs1["latency_ms_per_image_std"],
        "throughput_bs64_img_s_mean": bs64["throughput_img_s_mean"],
        "throughput_bs64_img_s_std": bs64["throughput_img_s_std"],
        "warmup_iterations": args.warmup,
        "latency_iterations": args.latency_iters,
        "throughput_iterations": args.throughput_iters,
        "repeats": args.repeats,
        "timing_scope": (
            "GPU forward only; excludes data loading, preprocessing, "
            "host-to-device transfer, distance computation, and ranking"
        ),
        "unsupported_flop_ops": unsupported_text,
    }

    logger.info("=" * 68)
    logger.info("CLIP-ReID inference-efficiency results")
    logger.info("Loaded Params       : %.3f M", loaded_params_m)
    logger.info("Inference Params    : %.3f M", inference_params_m)
    logger.info(
        "GFLOPs              : %s",
        "--" if not math.isfinite(gflops) else f"{gflops:.3f}",
    )
    logger.info(
        "Latency, bs=1       : %.3f +/- %.3f ms/image",
        result["latency_bs1_ms_mean"],
        result["latency_bs1_ms_std"],
    )
    logger.info(
        "Throughput, bs=%d  : %.1f +/- %.1f images/s",
        args.throughput_bs,
        result["throughput_bs64_img_s_mean"],
        result["throughput_bs64_img_s_std"],
    )
    logger.info("=" * 68)

    json_path, csv_path = save_efficiency_results(output_dir, result)
    logger.info("Efficiency JSON saved to %s", json_path)
    logger.info("Efficiency CSV saved to %s", csv_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP-ReID evaluation and efficiency benchmark"
    )
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        help="path to config file",
        type=str,
    )

    # Efficiency options. Put these options before trailing cfg opts.
    parser.add_argument("--efficiency_only", action="store_true")
    parser.add_argument("--skip_efficiency", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--flops_backend",
        choices=["fvcore", "thop"],
        default="fvcore",
    )
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--latency_iters", type=int, default=1000)
    parser.add_argument("--throughput_iters", type=int, default=300)
    parser.add_argument("--throughput_bs", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)

    parser.add_argument(
        "opts",
        help="Modify config options using the command line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("clipreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file:
        logger.info("Loaded configuration file %s", args.config_file)
        with open(args.config_file, "r", encoding="utf-8") as config_file:
            logger.info("\n%s", config_file.read())

    logger.info("Running with config:\n%s", cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    (
        train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)

    # Preserve the original project settings. For other datasets, consider replacing
    # the hard-coded values with num_classes and camera_num after confirming that the
    # checkpoint was trained with those dimensions.
    model = make_model(
        cfg,
        num_class=6603,
        camera_num=8,
        view_num=view_num,
    )
    model.load_param(cfg.TEST.WEIGHT)

    if not args.skip_efficiency:
        benchmark_clipreid(model, args, output_dir, logger)

    if args.efficiency_only:
        raise SystemExit(0)

    if cfg.DATASETS.NAMES == "VehicleID":
        all_rank_1 = None
        all_rank_5 = None
        all_mAP = None

        for trial in range(10):
            (
                train_loader,
                train_loader_normal,
                val_loader,
                num_query,
                num_classes,
                camera_num,
                view_num,
            ) = make_dataloader(cfg)

            rank_1, rank_5, mAP = do_inference(
                cfg,
                model,
                val_loader,
                num_query,
            )

            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank_5
                all_mAP = mAP
            else:
                all_rank_1 += rank_1
                all_rank_5 += rank_5
                all_mAP += mAP

            logger.info(
                "rank_1: %s, rank_5: %s, mAP: %s, trial: %d",
                rank_1,
                rank_5,
                mAP,
                trial,
            )

        logger.info(
            "mean_rank_1: %.1f%%, mean_rank_5: %.1f%%, mean_mAP: %.1f%%",
            all_rank_1.sum() / 10.0 * 100.0,
            all_rank_5.sum() / 10.0 * 100.0,
            all_mAP.sum() / 10.0 * 100.0,
        )
    else:
        do_inference(
            cfg,
            model,
            val_loader,
            num_query,
        )