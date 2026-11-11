# # test.py
# import argparse
# import csv
# import itertools
# import os
# import re
# from functools import lru_cache
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image, ImageDraw, ImageFont

# from config import cfg
# from datasets.make_dataloader_clipreid import make_dataloader
# from model.make_model_clipreid import make_model
# from utils.logger import setup_logger
# from utils.metrics import R1_mAP_eval


# # ============================================================
# # Basic helpers
# # ============================================================

# def feat_norm_enabled(config):
#     value = config.TEST.FEAT_NORM
#     if isinstance(value, bool):
#         return value
#     return str(value).lower() in {"yes", "true", "1", "on"}


# def to_int_list(values):
#     if torch.is_tensor(values):
#         return [int(value) for value in values.detach().cpu().tolist()]
#     return [int(value) for value in values]


# def load_font(size):
#     font_candidates = [
#         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#         "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
#         "DejaVuSans.ttf",
#     ]
#     for font_path in font_candidates:
#         try:
#             return ImageFont.truetype(font_path, size=size)
#         except Exception:
#             continue
#     return ImageFont.load_default()


# def parse_pms_state(image_path):
#     """
#     PMS filename:
#         001-cam2-01-0.jpg

#     State convention:
#         0 = walking
#         1 = cycling

#     This function is called only after retrieval ranking has been fixed.
#     State is never used in model forward or distance calculation.
#     """
#     filename = os.path.basename(str(image_path))
#     pattern = re.compile(
#         r"^([0-9]+)-cam([0-9]+)-([0-9]+)-([01])\.(jpg|jpeg|png|bmp)$",
#         flags=re.IGNORECASE,
#     )
#     match = pattern.match(filename)
#     if match is None:
#         raise ValueError(
#             "Invalid PMS filename '{}'. Expected format such as "
#             "001-cam2-01-0.jpg.".format(filename)
#         )
#     return int(match.group(4))


# def state_name(state):
#     return "Walk" if int(state) == 0 else "Cycle"


# def get_visualization_dataset_root(config, override_root=None):
#     """
#     Prefer the exact PMS/PWC_val folder to avoid resolving duplicate
#     filenames from PWC, PWC1, PWC3, ... under the parent data root.
#     """
#     if override_root:
#         root = os.path.abspath(os.path.expanduser(str(override_root)))
#         if not os.path.isdir(root):
#             raise RuntimeError(
#                 "Visualization dataset root does not exist: {}".format(root)
#             )
#         return root

#     base_root = os.path.abspath(
#         os.path.expanduser(str(config.DATASETS.ROOT_DIR))
#     )
#     preferred_root = os.path.join(base_root, "PWC_val")

#     if os.path.isdir(preferred_root):
#         return preferred_root

#     if not os.path.isdir(base_root):
#         raise RuntimeError(
#             "Dataset root does not exist: {}".format(base_root)
#         )

#     return base_root


# @lru_cache(maxsize=8)
# def build_image_path_index(dataset_root):
#     dataset_root = os.path.abspath(os.path.expanduser(str(dataset_root)))
#     if not os.path.isdir(dataset_root):
#         raise RuntimeError(
#             "Dataset root does not exist: {}".format(dataset_root)
#         )

#     valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
#     filename_index = {}

#     for current_root, _, filenames in os.walk(dataset_root):
#         for filename in filenames:
#             extension = os.path.splitext(filename)[1].lower()
#             if extension not in valid_extensions:
#                 continue
#             full_path = os.path.join(current_root, filename)
#             filename_index.setdefault(filename.lower(), []).append(full_path)

#     return filename_index


# def resolve_reid_image_path(image_path, dataset_root, preferred_split=None):
#     """
#     Resolve an image basename to a full path.

#     When dataset_root points directly to PWC_val, duplicated filenames
#     from PWC1/PWC3/PWC6 will not be considered.
#     """
#     image_path = os.path.expanduser(str(image_path))

#     if os.path.isfile(image_path):
#         return os.path.abspath(image_path)

#     filename = os.path.basename(image_path).lower()
#     filename_index = build_image_path_index(dataset_root)
#     matched_paths = filename_index.get(filename, [])

#     if len(matched_paths) == 0:
#         raise FileNotFoundError(
#             "Cannot find image '{}' under '{}'.".format(
#                 image_path, dataset_root
#             )
#         )

#     if preferred_split is not None:
#         preferred_split = preferred_split.lower()
#         split_matches = []

#         for path in matched_paths:
#             path_parts = [part.lower() for part in Path(path).parts]
#             if preferred_split in path_parts:
#                 split_matches.append(path)

#         if len(split_matches) == 1:
#             return split_matches[0]

#         if len(split_matches) > 1:
#             raise RuntimeError(
#                 "Ambiguous {} image '{}'. Multiple files were found:\n{}"
#                 .format(
#                     preferred_split,
#                     filename,
#                     "\n".join(sorted(split_matches)),
#                 )
#             )

#     if len(matched_paths) > 1:
#         raise RuntimeError(
#             "Ambiguous image '{}'. Multiple files were found:\n{}"
#             .format(
#                 filename,
#                 "\n".join(sorted(matched_paths)),
#             )
#         )

#     return matched_paths[0]


# def read_and_resize_image(
#         image_path,
#         width=180,
#         height=360,
#         background_color=(255, 255, 255),
# ):
#     """
#     Scale the complete image into a fixed display canvas without cropping.

#     The original aspect ratio is preserved. Empty areas are padded with a
#     white background so that every source pixel remains visible.
#     """
#     if not os.path.isfile(image_path):
#         raise FileNotFoundError(
#             "Image does not exist: {}".format(image_path)
#         )

#     image = Image.open(image_path).convert("RGB")

#     try:
#         resample_method = Image.Resampling.BILINEAR
#     except AttributeError:
#         resample_method = Image.BILINEAR

#     original_width, original_height = image.size

#     if original_width <= 0 or original_height <= 0:
#         raise ValueError(
#             "Invalid image size {} for '{}'."
#             .format(image.size, image_path)
#         )

#     scale = min(
#         float(width) / float(original_width),
#         float(height) / float(original_height),
#     )

#     resized_width = max(
#         1,
#         int(round(original_width * scale)),
#     )
#     resized_height = max(
#         1,
#         int(round(original_height * scale)),
#     )

#     resized_image = image.resize(
#         (resized_width, resized_height),
#         resample=resample_method,
#     )

#     canvas = Image.new(
#         "RGB",
#         (width, height),
#         color=background_color,
#     )

#     paste_x = (width - resized_width) // 2
#     paste_y = (height - resized_height) // 2

#     canvas.paste(
#         resized_image,
#         (paste_x, paste_y),
#     )

#     return canvas


# # ============================================================
# # Standard inference and feature extraction
# # ============================================================

# def extract_reid_features(config, model, val_loader, num_query, logger):
#     """
#     Standard ReID evaluation plus feature/path caching.

#     Required validation batch:
#         img, pid, camid, camids, target_view, imgpath

#     To guarantee state-free inference, MODEL.SIE_VIEW must be False.
#     The target_view tensor is read from the loader but never sent to
#     the model.
#     """
#     if bool(config.MODEL.SIE_VIEW):
#         raise ValueError(
#             "MODEL.SIE_VIEW must be False for PMS state-free inference. "
#             "The PMS view/state field must not be passed into the model."
#         )

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     evaluator = R1_mAP_eval(
#         num_query=num_query,
#         max_rank=50,
#         feat_norm=config.TEST.FEAT_NORM,
#     )
#     evaluator.reset()

#     if (
#         torch.cuda.device_count() > 1
#         and not isinstance(model, nn.DataParallel)
#     ):
#         logger.info(
#             "Using {} GPUs for inference".format(
#                 torch.cuda.device_count()
#             )
#         )
#         model = nn.DataParallel(model)

#     model.to(device)
#     model.eval()

#     feature_bank = []
#     pid_bank = []
#     camid_bank = []
#     image_path_bank = []

#     with torch.no_grad():
#         for batch_index, batch_data in enumerate(val_loader):
#             if len(batch_data) != 6:
#                 raise ValueError(
#                     "Validation loader must return exactly six items:\n"
#                     "img, pid, camid, camids, target_view, imgpath\n"
#                     "Received {} items.".format(len(batch_data))
#                 )

#             (
#                 images,
#                 pids,
#                 camids_for_eval,
#                 camids_for_sie,
#                 _target_views,
#                 image_paths,
#             ) = batch_data

#             images = images.to(device, non_blocking=True)

#             if config.MODEL.SIE_CAMERA:
#                 camera_inputs = camids_for_sie.to(
#                     device, non_blocking=True
#                 )
#             else:
#                 camera_inputs = None

#             # State/view information is deliberately not provided.
#             features = model(
#                 images,
#                 cam_label=camera_inputs,
#                 view_label=None,
#             )

#             evaluator.update(
#                 (features, pids, camids_for_eval)
#             )

#             feature_bank.append(features.detach().cpu())
#             pid_bank.extend(to_int_list(pids))
#             camid_bank.extend(to_int_list(camids_for_eval))
#             image_path_bank.extend(
#                 [str(path) for path in image_paths]
#             )

#             if (batch_index + 1) % 50 == 0:
#                 logger.info(
#                     "Extracted features from {}/{} validation batches"
#                     .format(batch_index + 1, len(val_loader))
#                 )

#     cmc, mAP, _, _, _, _, _ = evaluator.compute()

#     all_features = torch.cat(feature_bank, dim=0).float()

#     if all_features.shape[0] != len(image_path_bank):
#         raise RuntimeError(
#             "Feature count {} does not match path count {}."
#             .format(
#                 all_features.shape[0],
#                 len(image_path_bank),
#             )
#         )

#     if all_features.shape[0] <= num_query:
#         raise RuntimeError(
#             "num_query={} leaves no gallery features."
#             .format(num_query)
#         )

#     return {
#         "cmc": cmc,
#         "mAP": mAP,
#         "features": all_features,
#         "pids": pid_bank,
#         "camids": camid_bank,
#         "paths": image_path_bank,
#     }


# # ============================================================
# # Query selection
# # ============================================================

# def select_query_index(
#         query_paths,
#         query_pids,
#         query_camids,
#         query_image=None,
#         query_pid=None,
#         query_camid=None,
#         query_index=None,
# ):
#     if query_index is not None:
#         selected_index = int(query_index)
#         if not 0 <= selected_index < len(query_paths):
#             raise IndexError(
#                 "query_index={} is outside [0, {})."
#                 .format(selected_index, len(query_paths))
#             )
#         return selected_index

#     if query_image is not None:
#         query_name = os.path.basename(str(query_image)).lower()
#         matched_indices = [
#             index
#             for index, path in enumerate(query_paths)
#             if os.path.basename(str(path)).lower() == query_name
#         ]

#         if len(matched_indices) == 0:
#             raise ValueError(
#                 "Cannot find query image '{}' in the query set."
#                 .format(query_image)
#             )

#         return matched_indices[0]

#     if query_pid is not None:
#         target_pid = int(query_pid)
#         candidates = []

#         for index, current_pid in enumerate(query_pids):
#             if int(current_pid) != target_pid:
#                 continue

#             if query_camid is not None:
#                 current_human_camid = int(query_camids[index]) + 1
#                 if current_human_camid != int(query_camid):
#                     continue

#             candidates.append(index)

#         if len(candidates) == 0:
#             raise ValueError(
#                 "No query found for pid={} and camid={}."
#                 .format(query_pid, query_camid)
#             )

#         return candidates[0]

#     raise ValueError(
#         "Specify --query_image, --query_pid, or --query_index."
#     )


# # ============================================================
# # Retrieval and Top-10 state-aware visualization selection
# # ============================================================

# def retrieve_ranked_gallery(
#         query_feature,
#         gallery_features,
#         query_pid,
#         query_camid,
#         gallery_pids,
#         gallery_camids,
#         gallery_paths,
#         true_topk=5,
#         search_topk=10,
# ):
#     """
#     Calculate one unchanged retrieval ranking.

#     The first true_topk entries are the true Top-5 used for Rank-5.
#     The first search_topk entries are retained only as a visualization
#     fallback pool. No state is used in distance calculation or sorting.
#     """
#     if query_feature.ndim != 2 or query_feature.shape[0] != 1:
#         raise ValueError(
#             "query_feature must have shape [1, D], got {}."
#             .format(tuple(query_feature.shape))
#         )

#     if gallery_features.ndim != 2:
#         raise ValueError(
#             "gallery_features must have shape [N, D], got {}."
#             .format(tuple(gallery_features.shape))
#         )

#     if gallery_features.shape[0] != len(gallery_paths):
#         raise ValueError(
#             "Gallery feature count does not match path count."
#         )

#     true_topk = int(true_topk)
#     search_topk = max(int(search_topk), true_topk)

#     # Unchanged squared Euclidean distance.
#     distances = torch.sum(
#         (gallery_features - query_feature) ** 2,
#         dim=1,
#     )

#     gallery_pid_tensor = torch.tensor(
#         gallery_pids, dtype=torch.long
#     )
#     gallery_camid_tensor = torch.tensor(
#         gallery_camids, dtype=torch.long
#     )

#     valid_mask = gallery_pid_tensor != -1
#     same_pid_same_camera = (
#         (gallery_pid_tensor == int(query_pid))
#         & (gallery_camid_tensor == int(query_camid))
#     )
#     valid_mask = valid_mask & (~same_pid_same_camera)

#     valid_indices = torch.nonzero(
#         valid_mask, as_tuple=False
#     ).flatten()

#     if valid_indices.numel() < true_topk:
#         raise RuntimeError(
#             "Fewer than {} valid gallery images remain."
#             .format(true_topk)
#         )

#     valid_distances = distances[valid_indices]
#     sorted_local_indices = torch.argsort(
#         valid_distances, descending=False
#     )
#     sorted_gallery_indices = valid_indices[
#         sorted_local_indices
#     ]

#     retained_count = min(
#         search_topk,
#         int(sorted_gallery_indices.numel()),
#     )

#     ranked_results = []

#     for original_rank, gallery_index in enumerate(
#             sorted_gallery_indices[:retained_count].tolist(),
#             start=1,
#         ):
#         gallery_pid = int(gallery_pids[gallery_index])
#         gallery_camid = int(gallery_camids[gallery_index])

#         ranked_results.append(
#             {
#                 "original_rank": original_rank,
#                 "gallery_index": int(gallery_index),
#                 "path": str(gallery_paths[gallery_index]),
#                 "pid": gallery_pid,
#                 "camid": gallery_camid,
#                 "distance": float(
#                     distances[gallery_index].item()
#                 ),
#                 "correct": bool(
#                     gallery_pid == int(query_pid)
#                 ),
#             }
#         )

#     true_top_results = [
#         dict(item)
#         for item in ranked_results[:true_topk]
#     ]

#     correct_flags = [
#         item["correct"] for item in true_top_results
#     ]

#     first_correct_rank = None
#     for rank, gallery_index in enumerate(
#             sorted_gallery_indices.tolist(),
#             start=1,
#         ):
#         if int(gallery_pids[gallery_index]) == int(query_pid):
#             first_correct_rank = rank
#             break

#     return {
#         "true_topk": true_topk,
#         "search_topk": retained_count,
#         "rank1_hit": int(correct_flags[0]),
#         "rank5_hit": int(any(correct_flags)),
#         "precision_at_5": sum(correct_flags) / float(true_topk),
#         "first_correct_rank": first_correct_rank,
#         "true_top_results": true_top_results,
#         "ranked_results": ranked_results,
#     }


# def annotate_states_after_ranking(query_path, ranked_results):
#     """
#     Parse query/gallery states only after ranking has been produced.
#     """
#     query_state = parse_pms_state(query_path)

#     annotated = []
#     for item in ranked_results:
#         copied = dict(item)
#         copied["state"] = parse_pms_state(copied["path"])
#         annotated.append(copied)

#     return query_state, annotated


# def failure_case_statistics(
#         results,
#         query_state,
#         min_unique_cameras,
# ):
#     """
#     Check whether a displayed set is a mixed-state failure case.

#     Required failure pattern:
#         1. Gallery results contain both walking and cycling images.
#         2. At least one gallery image is incorrectly retrieved.
#         3. For a walking query, at least one incorrectly retrieved image
#            must be a cycling image.
#         4. For a cycling query, at least one incorrectly retrieved image
#            must be a walking image.
#         5. The displayed gallery images cover at least the requested
#            number of cameras.
#     """
#     query_state = int(query_state)
#     opposite_state = 1 - query_state

#     walk_count = sum(
#         int(item["state"]) == 0
#         for item in results
#     )
#     cycle_count = sum(
#         int(item["state"]) == 1
#         for item in results
#     )
#     correct_count = sum(
#         bool(item["correct"])
#         for item in results
#     )
#     wrong_count = len(results) - correct_count
#     wrong_opposite_state_count = sum(
#         (not bool(item["correct"]))
#         and int(item["state"]) == opposite_state
#         for item in results
#     )
#     unique_camera_count = len(
#         {int(item["camid"]) for item in results}
#     )

#     mixed_states = (
#         walk_count > 0
#         and cycle_count > 0
#     )

#     ok = (
#         mixed_states
#         and wrong_opposite_state_count >= 1
#         and unique_camera_count >= int(min_unique_cameras)
#     )

#     return {
#         "ok": bool(ok),
#         "walk_count": int(walk_count),
#         "cycle_count": int(cycle_count),
#         "correct_count": int(correct_count),
#         "wrong_count": int(wrong_count),
#         "wrong_opposite_state_count": int(
#             wrong_opposite_state_count
#         ),
#         "unique_camera_count": int(
#             unique_camera_count
#         ),
#     }


# def choose_failure_visual_results(
#         query_state,
#         annotated_ranked_results,
#         true_topk=5,
#         search_topk=10,
#         min_unique_cameras=2,
# ):
#     """
#     Select a mixed-state retrieval failure case for visualization.

#     Selection policy:
#         1. Check the unchanged true Top-5.
#         2. A walking query must contain both states and at least one
#            incorrectly retrieved cycling image.
#         3. A cycling query must contain both states and at least one
#            incorrectly retrieved walking image.
#         4. If true Top-5 does not satisfy the condition, search all
#            five-image combinations inside the unchanged true Top-10.
#         5. Preserve the original relative retrieval order.
#         6. Prefer the earliest valid combination by the smallest worst
#            original rank, then by the smallest sum of original ranks.

#     This affects only the qualitative panel. It does not alter feature
#     distances, the original ranking, or standard ReID metrics.
#     """
#     true_topk = int(true_topk)
#     search_topk = max(
#         int(search_topk),
#         true_topk,
#     )

#     true_top5 = [
#         dict(item)
#         for item in annotated_ranked_results[:true_topk]
#     ]

#     top5_statistics = failure_case_statistics(
#         results=true_top5,
#         query_state=query_state,
#         min_unique_cameras=min_unique_cameras,
#     )

#     if top5_statistics["ok"]:
#         for shown_order, item in enumerate(
#                 true_top5,
#                 start=1,
#             ):
#             item["shown_order"] = shown_order

#         return {
#             "source_mode": "true_top5_failure_case",
#             "fallback_used": False,
#             "failure_case_found": True,
#             "query_state": int(query_state),
#             **top5_statistics,
#             "selected_results": true_top5,
#         }

#     pool = annotated_ranked_results[:search_topk]
#     valid_combinations = []

#     for combination in itertools.combinations(
#             pool,
#             true_topk,
#         ):
#         combination = list(combination)

#         statistics = failure_case_statistics(
#             results=combination,
#             query_state=query_state,
#             min_unique_cameras=min_unique_cameras,
#         )

#         if not statistics["ok"]:
#             continue

#         original_ranks = [
#             int(item["original_rank"])
#             for item in combination
#         ]

#         score = (
#             max(original_ranks),
#             sum(original_ranks),
#             tuple(original_ranks),
#         )

#         valid_combinations.append(
#             (
#                 score,
#                 combination,
#                 statistics,
#             )
#         )

#     if len(valid_combinations) == 0:
#         for shown_order, item in enumerate(
#                 true_top5,
#                 start=1,
#             ):
#             item["shown_order"] = shown_order

#         return {
#             "source_mode": (
#                 "true_top5_no_mixed_state_failure_in_top10"
#             ),
#             "fallback_used": False,
#             "failure_case_found": False,
#             "query_state": int(query_state),
#             **top5_statistics,
#             "selected_results": true_top5,
#         }

#     valid_combinations.sort(
#         key=lambda item: item[0]
#     )

#     (
#         _score,
#         selected_results,
#         selected_statistics,
#     ) = valid_combinations[0]

#     # itertools.combinations preserves pool order. Therefore, selected
#     # images retain their original relative retrieval ranking.
#     selected_results = [
#         dict(item)
#         for item in selected_results
#     ]

#     for shown_order, item in enumerate(
#             selected_results,
#             start=1,
#         ):
#         item["shown_order"] = shown_order

#     return {
#         "source_mode": (
#             "mixed_state_failure_selected_from_true_top10"
#         ),
#         "fallback_used": True,
#         "failure_case_found": True,
#         "query_state": int(query_state),
#         **selected_statistics,
#         "selected_results": selected_results,
#     }


# # ============================================================
# # Visualization and CSV
# # ============================================================

# def draw_state_visualization(
#         query_path,
#         query_pid,
#         query_camid,
#         retrieval_result,
#         visual_result,
#         save_path,
# ):
#     """
#     Draw only the query and selected gallery images with colored borders.

#     Border convention:
#         blue  = query image
#         green = gallery image with the same PID as the query
#         red   = gallery image with a different PID from the query

#     No title, state strip, rank text, PID text, camera text, distance text,
#     or any other annotation is drawn on top of or below the images.
#     """
#     image_width = 180
#     image_height = 360
#     margin = 22
#     border_width = 6

#     selected_results = visual_result["selected_results"]
#     total_images = 1 + len(selected_results)

#     canvas_width = (
#         margin
#         + total_images * image_width
#         + total_images * margin
#     )
#     canvas_height = image_height + 2 * margin

#     canvas = Image.new(
#         "RGB",
#         (canvas_width, canvas_height),
#         color=(255, 255, 255),
#     )
#     draw = ImageDraw.Draw(canvas)

#     image_y = margin

#     # Query: blue border.
#     query_image = read_and_resize_image(
#         query_path,
#         image_width,
#         image_height,
#     )
#     query_x = margin
#     canvas.paste(query_image, (query_x, image_y))

#     draw.rectangle(
#         [
#             query_x,
#             image_y,
#             query_x + image_width - 1,
#             image_y + image_height - 1,
#         ],
#         outline=(30, 90, 210),
#         width=border_width,
#     )

#     # Gallery: green for the same PID, red for a different PID.
#     for offset, item in enumerate(selected_results, start=1):
#         gallery_x = (
#             margin
#             + offset * (image_width + margin)
#         )

#         gallery_image = read_and_resize_image(
#             item["path"],
#             image_width,
#             image_height,
#         )
#         canvas.paste(gallery_image, (gallery_x, image_y))

#         border_color = (
#             (20, 170, 70)
#             if item["correct"]
#             else (215, 45, 45)
#         )

#         draw.rectangle(
#             [
#                 gallery_x,
#                 image_y,
#                 gallery_x + image_width - 1,
#                 image_y + image_height - 1,
#             ],
#             outline=border_color,
#             width=border_width,
#         )

#     save_parent = os.path.dirname(save_path)
#     if save_parent:
#         os.makedirs(save_parent, exist_ok=True)

#     canvas.save(save_path, quality=95)


# def save_state_visualization_csv(
#         query_path,
#         query_pid,
#         query_camid,
#         retrieval_result,
#         visual_result,
#         save_path,
# ):
#     os.makedirs(
#         os.path.dirname(save_path),
#         exist_ok=True,
#     )

#     with open(
#         save_path,
#         mode="w",
#         newline="",
#         encoding="utf-8",
#     ) as csv_file:
#         writer = csv.writer(csv_file)

#         writer.writerow(
#             [
#                 "source_mode",
#                 "query_path",
#                 "query_pid",
#                 "query_camid",
#                 "query_state",
#                 "failure_case_found",
#                 "displayed_walk_count",
#                 "displayed_cycle_count",
#                 "displayed_correct_count",
#                 "displayed_wrong_count",
#                 "wrong_opposite_state_count",
#                 "displayed_unique_camera_count",
#                 "shown_order",
#                 "original_rank",
#                 "gallery_path",
#                 "gallery_pid",
#                 "gallery_camid",
#                 "gallery_state",
#                 "distance",
#                 "correct",
#                 "true_rank1_hit",
#                 "true_rank5_hit",
#                 "true_precision_at_5",
#             ]
#         )

#         for item in visual_result["selected_results"]:
#             writer.writerow(
#                 [
#                     visual_result["source_mode"],
#                     query_path,
#                     int(query_pid),
#                     int(query_camid) + 1,
#                     state_name(visual_result["query_state"]),
#                     int(visual_result["failure_case_found"]),
#                     int(visual_result["walk_count"]),
#                     int(visual_result["cycle_count"]),
#                     int(visual_result["correct_count"]),
#                     int(visual_result["wrong_count"]),
#                     int(
#                         visual_result[
#                             "wrong_opposite_state_count"
#                         ]
#                     ),
#                     int(visual_result["unique_camera_count"]),
#                     int(item["shown_order"]),
#                     int(item["original_rank"]),
#                     item["path"],
#                     int(item["pid"]),
#                     int(item["camid"]) + 1,
#                     state_name(item["state"]),
#                     float(item["distance"]),
#                     int(item["correct"]),
#                     retrieval_result["rank1_hit"],
#                     retrieval_result["rank5_hit"],
#                     retrieval_result["precision_at_5"],
#                 ]
#             )


# def save_failure_cases_summary_csv(cases, save_path):
#     """
#     Save one summary row for each matched failure case.
#     """
#     save_parent = os.path.dirname(save_path)
#     if save_parent:
#         os.makedirs(save_parent, exist_ok=True)

#     with open(
#         save_path,
#         mode="w",
#         newline="",
#         encoding="utf-8",
#     ) as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(
#             [
#                 "case_index",
#                 "query_index",
#                 "query_path",
#                 "query_pid",
#                 "query_camid",
#                 "query_state",
#                 "source_mode",
#                 "displayed_walk_count",
#                 "displayed_cycle_count",
#                 "displayed_correct_count",
#                 "displayed_wrong_count",
#                 "wrong_opposite_state_count",
#                 "displayed_unique_camera_count",
#                 "true_rank1_hit",
#                 "true_rank5_hit",
#                 "true_precision_at_5",
#                 "visualization_path",
#                 "case_csv_path",
#             ]
#         )

#         for case_index, case in enumerate(cases, start=1):
#             visual_result = case["visual_result"]

#             writer.writerow(
#                 [
#                     case_index,
#                     int(case["query_index"]),
#                     case["query_path"],
#                     int(case["query_pid"]),
#                     int(case["query_camid"]) + 1,
#                     state_name(case["query_state"]),
#                     visual_result["source_mode"],
#                     int(visual_result["walk_count"]),
#                     int(visual_result["cycle_count"]),
#                     int(visual_result["correct_count"]),
#                     int(visual_result["wrong_count"]),
#                     int(
#                         visual_result[
#                             "wrong_opposite_state_count"
#                         ]
#                     ),
#                     int(visual_result["unique_camera_count"]),
#                     int(case["rank1_hit"]),
#                     int(case["rank5_hit"]),
#                     float(case["precision_at_5"]),
#                     case["visualization_path"],
#                     case["csv_path"],
#                 ]
#             )


# # ============================================================
# # Multi-query failure-case visualization
# # ============================================================

# def visualize_selected_query(
#         inference_result,
#         num_query,
#         query_image=None,
#         query_pid=None,
#         query_camid=None,
#         query_index=None,
#         true_topk=5,
#         search_topk=10,
#         min_unique_cameras=2,
#         max_failure_cases=0,
#         save_dir="./topk_visualization",
#         visualization_dataset_root=None,
# ):
#     """
#     Find and save mixed-state retrieval failure cases.

#     Behavior:
#         1. When a query selector is provided, only that query is checked.
#         2. When no selector is provided, all queries are scanned.
#         3. Every matching query is saved instead of stopping after the
#            first match.
#         4. max_failure_cases=0 means no limit and therefore saves all
#            matching cases.
#     """
#     all_features = inference_result["features"].float()
#     all_pids = inference_result["pids"]
#     all_camids = inference_result["camids"]
#     all_paths = inference_result["paths"]

#     if num_query <= 0 or num_query >= all_features.shape[0]:
#         raise ValueError(
#             "Invalid num_query={} for {} total features."
#             .format(num_query, all_features.shape[0])
#         )

#     max_failure_cases = int(max_failure_cases)
#     if max_failure_cases < 0:
#         raise ValueError(
#             "max_failure_cases must be greater than or equal to 0."
#         )

#     if feat_norm_enabled(cfg):
#         all_features = F.normalize(
#             all_features,
#             p=2,
#             dim=1,
#         )

#     query_features = all_features[:num_query]
#     gallery_features = all_features[num_query:]

#     query_pids = all_pids[:num_query]
#     gallery_pids = all_pids[num_query:]

#     query_camids = all_camids[:num_query]
#     gallery_camids = all_camids[num_query:]

#     query_paths = all_paths[:num_query]
#     gallery_paths = all_paths[num_query:]

#     selector_provided = (
#         query_image is not None
#         or query_pid is not None
#         or query_index is not None
#     )

#     if selector_provided:
#         candidate_indices = [
#             select_query_index(
#                 query_paths=query_paths,
#                 query_pids=query_pids,
#                 query_camids=query_camids,
#                 query_image=query_image,
#                 query_pid=query_pid,
#                 query_camid=query_camid,
#                 query_index=query_index,
#             )
#         ]
#     else:
#         # Scan all queries. Do not stop at the first matching example.
#         candidate_indices = range(num_query)

#     dataset_root = get_visualization_dataset_root(
#         cfg,
#         override_root=visualization_dataset_root,
#     )

#     os.makedirs(save_dir, exist_ok=True)

#     matched_cases = []

#     for candidate_index in candidate_indices:
#         candidate_query_pid = int(
#             query_pids[candidate_index]
#         )
#         candidate_query_camid = int(
#             query_camids[candidate_index]
#         )
#         candidate_query_feature = query_features[
#             candidate_index:candidate_index + 1
#         ]

#         # Ranking is completed before any state is parsed.
#         candidate_retrieval_result = retrieve_ranked_gallery(
#             query_feature=candidate_query_feature,
#             gallery_features=gallery_features,
#             query_pid=candidate_query_pid,
#             query_camid=candidate_query_camid,
#             gallery_pids=gallery_pids,
#             gallery_camids=gallery_camids,
#             gallery_paths=gallery_paths,
#             true_topk=true_topk,
#             search_topk=search_topk,
#         )

#         (
#             candidate_query_state,
#             candidate_annotated_results,
#         ) = annotate_states_after_ranking(
#             query_path=query_paths[candidate_index],
#             ranked_results=candidate_retrieval_result[
#                 "ranked_results"
#             ],
#         )

#         candidate_visual_result = (
#             choose_failure_visual_results(
#                 query_state=candidate_query_state,
#                 annotated_ranked_results=(
#                     candidate_annotated_results
#                 ),
#                 true_topk=true_topk,
#                 search_topk=search_topk,
#                 min_unique_cameras=min_unique_cameras,
#             )
#         )

#         if not candidate_visual_result["failure_case_found"]:
#             continue

#         selected_query_path = resolve_reid_image_path(
#             image_path=query_paths[candidate_index],
#             dataset_root=dataset_root,
#             preferred_split="query",
#         )

#         for item in candidate_visual_result["selected_results"]:
#             item["path"] = resolve_reid_image_path(
#                 image_path=item["path"],
#                 dataset_root=dataset_root,
#                 preferred_split="gallery",
#             )

#         query_stem = Path(selected_query_path).stem
#         mode_tag = candidate_visual_result["source_mode"]
#         case_number = len(matched_cases) + 1

#         file_stem = (
#             "case_{:04d}_qidx_{:04d}_{}_{}"
#             .format(
#                 case_number,
#                 int(candidate_index),
#                 query_stem,
#                 mode_tag,
#             )
#         )

#         image_save_path = os.path.join(
#             save_dir,
#             "{}.jpg".format(file_stem),
#         )
#         csv_save_path = os.path.join(
#             save_dir,
#             "{}.csv".format(file_stem),
#         )

#         draw_state_visualization(
#             query_path=selected_query_path,
#             query_pid=candidate_query_pid,
#             query_camid=candidate_query_camid,
#             retrieval_result=candidate_retrieval_result,
#             visual_result=candidate_visual_result,
#             save_path=image_save_path,
#         )

#         save_state_visualization_csv(
#             query_path=selected_query_path,
#             query_pid=candidate_query_pid,
#             query_camid=candidate_query_camid,
#             retrieval_result=candidate_retrieval_result,
#             visual_result=candidate_visual_result,
#             save_path=csv_save_path,
#         )

#         candidate_retrieval_result.update(
#             {
#                 "query_index": int(candidate_index),
#                 "query_path": selected_query_path,
#                 "query_pid": candidate_query_pid,
#                 "query_camid": candidate_query_camid,
#                 "query_state": candidate_query_state,
#                 "visual_result": candidate_visual_result,
#                 "visualization_path": image_save_path,
#                 "csv_path": csv_save_path,
#             }
#         )

#         matched_cases.append(candidate_retrieval_result)

#         if (
#             max_failure_cases > 0
#             and len(matched_cases) >= max_failure_cases
#         ):
#             break

#     if len(matched_cases) == 0:
#         if selector_provided:
#             selected_description = (
#                 query_image
#                 if query_image is not None
#                 else (
#                     "PID {}".format(query_pid)
#                     if query_pid is not None
#                     else "query index {}".format(query_index)
#                 )
#             )
#             raise RuntimeError(
#                 "The selected query '{}' has no mixed-state failure "
#                 "case in its true Top-{}.".format(
#                     selected_description,
#                     search_topk,
#                 )
#             )

#         raise RuntimeError(
#             "No query in the validation set has the requested "
#             "mixed-state failure pattern within true Top-{}."
#             .format(search_topk)
#         )

#     summary_csv_path = os.path.join(
#         save_dir,
#         "mixed_state_failure_cases_summary.csv",
#     )

#     save_failure_cases_summary_csv(
#         cases=matched_cases,
#         save_path=summary_csv_path,
#     )

#     walk_query_count = sum(
#         int(case["query_state"]) == 0
#         for case in matched_cases
#     )
#     cycle_query_count = sum(
#         int(case["query_state"]) == 1
#         for case in matched_cases
#     )
#     true_top5_case_count = sum(
#         case["visual_result"]["source_mode"]
#         == "true_top5_failure_case"
#         for case in matched_cases
#     )
#     top10_fallback_case_count = sum(
#         case["visual_result"]["fallback_used"]
#         for case in matched_cases
#     )

#     return {
#         "matched_count": len(matched_cases),
#         "walk_query_count": int(walk_query_count),
#         "cycle_query_count": int(cycle_query_count),
#         "true_top5_case_count": int(true_top5_case_count),
#         "top10_fallback_case_count": int(
#             top10_fallback_case_count
#         ),
#         "summary_csv_path": summary_csv_path,
#         "cases": matched_cases,
#     }


# # ============================================================
# # Main
# # ============================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="ReID Testing and Mixed-State Failure Visualization"
#     )

#     parser.add_argument(
#         "--config_file",
#         default="configs/person/vit_clipreid.yml",
#         type=str,
#         help="Path to config file",
#     )

#     parser.add_argument(
#         "--checkpoint_num_classes",
#         type=int,
#         default=6603,
#         help="Training identity count used by the checkpoint.",
#     )
#     parser.add_argument(
#         "--checkpoint_camera_num",
#         type=int,
#         default=8,
#         help="Training camera count used by the checkpoint.",
#     )
#     parser.add_argument(
#         "--checkpoint_view_num",
#         type=int,
#         default=None,
#         help="Training view count used by the checkpoint.",
#     )

#     parser.add_argument(
#         "--visualize_topk",
#         action="store_true",
#         help=(
#             "Find and visualize mixed-state failure cases. Check true Top-5 "
#             "first; otherwise search five-image combinations inside true "
#             "Top-10. Without a query selector, all queries are scanned."
#         ),
#     )

#     query_group = parser.add_mutually_exclusive_group()
#     query_group.add_argument(
#         "--query_image",
#         type=str,
#         default=None,
#         help=(
#             "Exact query filename. When no query selector is given, all "
#             "queries are scanned and multiple matching cases are saved."
#         ),
#     )
#     query_group.add_argument(
#         "--query_pid",
#         type=int,
#         default=None,
#         help="Select the first query of this PID.",
#     )
#     query_group.add_argument(
#         "--query_index",
#         type=int,
#         default=None,
#         help="Select a query by zero-based query index.",
#     )

#     parser.add_argument(
#         "--query_camid",
#         type=int,
#         default=None,
#         help="Optional one-based camera number used with --query_pid.",
#     )
#     parser.add_argument(
#         "--topk",
#         type=int,
#         default=5,
#         help="True Top-K size. Keep this at 5 for failure visualization.",
#     )
#     parser.add_argument(
#         "--search_topk",
#         type=int,
#         default=10,
#         help="Fallback search range. Default: true Top-10.",
#     )
#     parser.add_argument(
#         "--min_vis_cameras",
#         type=int,
#         default=2,
#         help="Minimum distinct gallery cameras in the displayed five images.",
#     )
#     parser.add_argument(
#         "--max_failure_cases",
#         type=int,
#         default=0,
#         help=(
#             "Maximum number of matched failure cases to save. "
#             "Use 0 to scan all queries and save every matching case."
#         ),
#     )
#     parser.add_argument(
#         "--vis_dir",
#         type=str,
#         default=None,
#         help="Output directory. Default: OUTPUT_DIR/pms_mixed_state_failure.",
#     )
#     parser.add_argument(
#         "--vis_dataset_root",
#         type=str,
#         default=None,
#         help=(
#             "Exact visualization dataset root. "
#             "Default prefers DATASETS.ROOT_DIR/PWC_val."
#         ),
#     )

#     # Keep opts last because argparse.REMAINDER consumes all following tokens.
#     parser.add_argument(
#         "opts",
#         default=None,
#         nargs=argparse.REMAINDER,
#         help="YACS config overrides as KEY VALUE pairs.",
#     )

#     args = parser.parse_args()

#     if args.topk != 5:
#         raise ValueError(
#             "--topk must be 5 for the requested 3/2 state composition."
#         )
#     if args.search_topk < args.topk:
#         raise ValueError(
#             "--search_topk must be greater than or equal to --topk."
#         )
#     if args.max_failure_cases < 0:
#         raise ValueError(
#             "--max_failure_cases must be greater than or equal to 0."
#         )

#     if args.config_file:
#         cfg.merge_from_file(args.config_file)

#     clean_opts = [
#         item
#         for item in args.opts
#         if isinstance(item, str) and item.strip()
#     ]

#     if len(clean_opts) % 2 != 0:
#         raise ValueError(
#             "YACS override options must be KEY VALUE pairs: {}"
#             .format(clean_opts)
#         )

#     cfg.merge_from_list(clean_opts)
#     cfg.freeze()

#     output_dir = cfg.OUTPUT_DIR
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     logger = setup_logger(
#         "transreid",
#         output_dir,
#         if_train=False,
#     )

#     logger.info(args)
#     logger.info("Running with config:\n{}".format(cfg))

#     os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

#     (
#         train_loader,
#         train_loader_normal,
#         val_loader,
#         num_query,
#         target_num_classes,
#         target_camera_num,
#         target_view_num,
#     ) = make_dataloader(cfg)

#     logger.info(
#         "Target validation set: num_query={}, "
#         "target_num_classes={}, target_camera_num={}, "
#         "target_view_num={}"
#         .format(
#             num_query,
#             target_num_classes,
#             target_camera_num,
#             target_view_num,
#         )
#     )

#     checkpoint_view_num = (
#         args.checkpoint_view_num
#         if args.checkpoint_view_num is not None
#         else target_view_num
#     )

#     model = make_model(
#         cfg,
#         num_class=args.checkpoint_num_classes,
#         camera_num=args.checkpoint_camera_num,
#         view_num=checkpoint_view_num,
#     )
#     model.load_param(cfg.TEST.WEIGHT)

#     inference_result = extract_reid_features(
#         config=cfg,
#         model=model,
#         val_loader=val_loader,
#         num_query=num_query,
#         logger=logger,
#     )

#     cmc = inference_result["cmc"]
#     mAP = inference_result["mAP"]

#     logger.info("Validation Results")
#     logger.info("mAP: {:.1%}".format(mAP))

#     for rank in [1, 5, 10]:
#         if rank <= len(cmc):
#             logger.info(
#                 "CMC curve, Rank-{:<3}: {:.1%}"
#                 .format(rank, cmc[rank - 1])
#             )

#     if args.visualize_topk:
#         visualization_dir = (
#             args.vis_dir
#             if args.vis_dir is not None
#             else os.path.join(
#                 cfg.OUTPUT_DIR,
#                 "pms_mixed_state_failure",
#             )
#         )

#         batch_result = visualize_selected_query(
#             inference_result=inference_result,
#             num_query=num_query,
#             query_image=args.query_image,
#             query_pid=args.query_pid,
#             query_camid=args.query_camid,
#             query_index=args.query_index,
#             true_topk=args.topk,
#             search_topk=args.search_topk,
#             min_unique_cameras=args.min_vis_cameras,
#             max_failure_cases=args.max_failure_cases,
#             save_dir=visualization_dir,
#             visualization_dataset_root=args.vis_dataset_root,
#         )

#         logger.info(
#             "Matched failure cases: {}".format(
#                 batch_result["matched_count"]
#             )
#         )
#         logger.info(
#             "Walk-query cases: {}, Cycle-query cases: {}"
#             .format(
#                 batch_result["walk_query_count"],
#                 batch_result["cycle_query_count"],
#             )
#         )
#         logger.info(
#             "True Top-5 cases: {}, Top-10 fallback cases: {}"
#             .format(
#                 batch_result["true_top5_case_count"],
#                 batch_result["top10_fallback_case_count"],
#             )
#         )
#         logger.info(
#             "Summary CSV saved to: {}".format(
#                 batch_result["summary_csv_path"]
#             )
#         )
#         logger.info(
#             "All case images and per-case CSV files are saved under: {}"
#             .format(visualization_dir)
#         )


# test.py
import argparse
import csv
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from functools import lru_cache

from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval



# ============================================================
# Basic helpers
# ============================================================

def feat_norm_enabled(cfg):
    """
    Compatible with:
        TEST.FEAT_NORM = True
        TEST.FEAT_NORM = "yes"
        TEST.FEAT_NORM = "on"
    """
    value = cfg.TEST.FEAT_NORM

    if isinstance(value, bool):
        return value

    return str(value).lower() in {
        "yes",
        "true",
        "1",
        "on",
    }


def to_int_list(values):
    """
    Convert a tensor/list of IDs into a Python int list.
    """
    if torch.is_tensor(values):
        return [
            int(value)
            for value in values.detach().cpu().tolist()
        ]

    return [int(value) for value in values]


def load_font(size):
    """
    Load a visualization font with safe fallbacks.
    """
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/"
        "LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(
                font_path,
                size=size,
            )
        except Exception:
            continue

    return ImageFont.load_default()

def get_pwc_val_root(config):
    """
    Return the exact validation dataset root used for PMS/PWC testing.
    """
    base_root = os.path.abspath(
        os.path.expanduser(str(config.DATASETS.ROOT_DIR))
    )
    pwc_val_root = os.path.join(base_root, "PWC_val")
    query_root = os.path.join(pwc_val_root, "query")
    gallery_root = os.path.join(pwc_val_root, "gallery")

    if not os.path.isdir(query_root):
        raise RuntimeError(
            "PWC_val query directory does not exist: {}".format(query_root)
        )

    if not os.path.isdir(gallery_root):
        raise RuntimeError(
            "PWC_val gallery directory does not exist: {}".format(gallery_root)
        )

    return pwc_val_root


def validate_loader_full_paths(image_paths, num_query, pwc_val_root):
    """
    Detect full paths returned by the validation loader that are outside
    PWC_val/query or PWC_val/gallery.
    """
    violations = []

    for index, image_path in enumerate(image_paths):
        image_path = os.path.expanduser(str(image_path))

        if not os.path.isabs(image_path) or not os.path.isfile(image_path):
            continue

        expected_split = "query" if index < int(num_query) else "gallery"
        expected_root = os.path.abspath(
            os.path.join(pwc_val_root, expected_split)
        )
        actual_path = os.path.abspath(image_path)

        try:
            inside_expected_root = (
                os.path.commonpath([actual_path, expected_root])
                == expected_root
            )
        except ValueError:
            inside_expected_root = False

        if not inside_expected_root:
            violations.append(
                (index, expected_split, actual_path)
            )

    if violations:
        preview = "\n".join(
            "index={} expected={} actual={}".format(
                index,
                split,
                path,
            )
            for index, split, path in violations[:20]
        )

        raise RuntimeError(
            "The validation loader returned full paths outside PWC_val. "
            "Check the registered 'pwc' dataset class and ensure its "
            "dataset_dir/query_dir/gallery_dir point to PWC_val.\n{}"
            .format(preview)
        )


@lru_cache(maxsize=8)
def build_image_path_index(dataset_root):
    """
    Recursively index images under DATASETS.ROOT_DIR.

    Returns:
        {
            filename: [full_path_1, full_path_2, ...]
        }
    """
    dataset_root = os.path.abspath(
        os.path.expanduser(str(dataset_root))
    )

    if not os.path.isdir(dataset_root):
        raise RuntimeError(
            "Dataset root does not exist: {}".format(dataset_root)
        )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    filename_index = {}

    for current_root, _, filenames in os.walk(dataset_root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension not in valid_extensions:
                continue

            full_path = os.path.join(
                current_root,
                filename,
            )

            filename_index.setdefault(
                filename.lower(),
                [],
            ).append(full_path)

    return filename_index


def resolve_reid_image_path(
        image_path,
        dataset_root,
        preferred_split=None,
):
    """
    Resolve an image strictly inside the requested PWC_val split.
    """
    dataset_root = os.path.abspath(
        os.path.expanduser(str(dataset_root))
    )
    filename = os.path.basename(
        os.path.expanduser(str(image_path))
    )

    if preferred_split is not None:
        preferred_split = str(preferred_split).lower()

        if preferred_split not in {"query", "gallery"}:
            raise ValueError(
                "preferred_split must be 'query' or 'gallery', got '{}'."
                .format(preferred_split)
            )

        split_root = os.path.join(
            dataset_root,
            preferred_split,
        )
        candidate_path = os.path.join(
            split_root,
            filename,
        )

        if os.path.isfile(candidate_path):
            return os.path.abspath(candidate_path)

        filename_index = build_image_path_index(split_root)
        matched_paths = filename_index.get(
            filename.lower(),
            [],
        )

        if len(matched_paths) == 1:
            return os.path.abspath(matched_paths[0])

        if len(matched_paths) > 1:
            raise RuntimeError(
                "Ambiguous '{}' image '{}' inside '{}':\n{}"
                .format(
                    preferred_split,
                    filename,
                    split_root,
                    "\n".join(sorted(matched_paths)),
                )
            )

        raise FileNotFoundError(
            "Cannot find '{}' inside '{}'."
            .format(
                filename,
                split_root,
            )
        )

    expanded_path = os.path.abspath(
        os.path.expanduser(str(image_path))
    )

    if os.path.isfile(expanded_path):
        try:
            inside_dataset_root = (
                os.path.commonpath([expanded_path, dataset_root])
                == dataset_root
            )
        except ValueError:
            inside_dataset_root = False

        if inside_dataset_root:
            return expanded_path

    filename_index = build_image_path_index(dataset_root)
    matched_paths = filename_index.get(
        filename.lower(),
        [],
    )

    if len(matched_paths) == 1:
        return os.path.abspath(matched_paths[0])

    if len(matched_paths) > 1:
        raise RuntimeError(
            "Ambiguous image '{}' inside '{}':\n{}"
            .format(
                filename,
                dataset_root,
                "\n".join(sorted(matched_paths)),
            )
        )

    raise FileNotFoundError(
        "Cannot find image '{}' under '{}'."
        .format(
            filename,
            dataset_root,
        )
    )


def read_and_resize_image(
        image_path,
        width=144,
        height=288,
):
    """
    Read an image and resize it to a uniform ReID display size.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "Image does not exist: {}".format(image_path)
        )

    image = Image.open(image_path).convert("RGB")

    # Compatible with old and new Pillow versions.
    try:
        resample_method = Image.Resampling.BILINEAR
    except AttributeError:
        resample_method = Image.BILINEAR

    image = ImageOps.fit(
        image,
        (width, height),
        method=resample_method,
        centering=(0.5, 0.5),
    )

    return image


# ============================================================
# Standard inference and feature extraction
# ============================================================

def extract_reid_features(
        cfg,
        model,
        val_loader,
        num_query,
        logger,
):
    """
    Perform standard ReID validation and save all query/gallery
    features, person IDs, camera IDs, and image paths.

    Required validation batch format:
        img, pid, camid, camids, target_view, imgpath

    State labels are not accepted or used.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluator = R1_mAP_eval(
        num_query=num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
    )
    evaluator.reset()

    if (
        torch.cuda.device_count() > 1
        and not isinstance(model, nn.DataParallel)
    ):
        logger.info(
            "Using {} GPUs for inference".format(
                torch.cuda.device_count()
            )
        )
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    feature_bank = []
    pid_bank = []
    camid_bank = []
    image_path_bank = []

    with torch.no_grad():
        for batch_index, batch_data in enumerate(val_loader):
            # Validation must not explicitly contain state_label.
            if len(batch_data) != 6:
                raise ValueError(
                    "Validation loader must return exactly six items:\n"
                    "  img, pid, camid, camids, target_view, imgpath\n"
                    "Received {} items instead. Remove state_label "
                    "from the validation output."
                    .format(len(batch_data))
                )

            (
                images,
                pids,
                camids_for_eval,
                camids_for_sie,
                target_views,
                image_paths,
            ) = batch_data

            images = images.to(
                device,
                non_blocking=True,
            )

            # Camera information is used only when the trained model
            # explicitly enables SIE_CAMERA.
            if cfg.MODEL.SIE_CAMERA:
                camera_inputs = camids_for_sie.to(
                    device,
                    non_blocking=True,
                )
            else:
                camera_inputs = None

            # View information is used only when the trained model
            # explicitly enables SIE_VIEW.
            #
            # For state-free PMS inference, target_views should be a
            # genuine view label or a constant zero, not the filename's
            # final walking/cycling field.
            if cfg.MODEL.SIE_VIEW:
                view_inputs = target_views.to(
                    device,
                    non_blocking=True,
                )
            else:
                view_inputs = None

            features = model(
                images,
                cam_label=camera_inputs,
                view_label=view_inputs,
            )

            # Standard metric evaluation.
            evaluator.update(
                (
                    features,
                    pids,
                    camids_for_eval,
                )
            )

            # Cache the same features for single-query visualization.
            feature_bank.append(
                features.detach().cpu()
            )
            pid_bank.extend(
                to_int_list(pids)
            )
            camid_bank.extend(
                to_int_list(camids_for_eval)
            )
            image_path_bank.extend(
                [str(path) for path in image_paths]
            )

            if (batch_index + 1) % 50 == 0:
                logger.info(
                    "Extracted features from {}/{} validation batches"
                    .format(
                        batch_index + 1,
                        len(val_loader),
                    )
                )

    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    all_features = torch.cat(
        feature_bank,
        dim=0,
    ).float()

    if all_features.shape[0] != len(image_path_bank):
        raise RuntimeError(
            "Feature count {} does not match image path count {}."
            .format(
                all_features.shape[0],
                len(image_path_bank),
            )
        )

    if all_features.shape[0] <= num_query:
        raise RuntimeError(
            "num_query={} leaves no gallery features; "
            "total feature count is {}."
            .format(
                num_query,
                all_features.shape[0],
            )
        )

    return {
        "cmc": cmc,
        "mAP": mAP,
        "features": all_features,
        "pids": pid_bank,
        "camids": camid_bank,
        "paths": image_path_bank,
    }


# ============================================================
# Query selection
# ============================================================

def select_query_index(
        query_paths,
        query_pids,
        query_camids,
        query_image=None,
        query_pid=None,
        query_camid=None,
        query_index=None,
):
    """
    Select one query.

    Priority:
        1. query_index
        2. query_image
        3. query_pid + optional query_camid

    query_camid is human-readable and one-based:
        cam2 -> query_camid=2

    Dataset-internal camera IDs are normally zero-based.
    """

    if query_index is not None:
        selected_index = int(query_index)

        if not 0 <= selected_index < len(query_paths):
            raise IndexError(
                "query_index={} is outside [0, {})."
                .format(
                    selected_index,
                    len(query_paths),
                )
            )

        return selected_index

    if query_image is not None:
        query_name = os.path.basename(
            str(query_image)
        ).lower()

        matched_indices = [
            index
            for index, path in enumerate(query_paths)
            if os.path.basename(str(path)).lower() == query_name
        ]

        if len(matched_indices) == 0:
            available_examples = [
                os.path.basename(path)
                for path in query_paths[:10]
            ]

            raise ValueError(
                "Cannot find query image '{}' in the query set.\n"
                "First query filenames include: {}"
                .format(
                    query_image,
                    available_examples,
                )
            )

        if len(matched_indices) > 1:
            print(
                "Warning: multiple query samples have filename '{}'; "
                "the first one is selected."
                .format(query_image)
            )

        return matched_indices[0]

    if query_pid is not None:
        target_pid = int(query_pid)
        candidates = []

        for index, current_pid in enumerate(query_pids):
            if int(current_pid) != target_pid:
                continue

            if query_camid is not None:
                # Internal camera ID is zero-based.
                current_human_camid = (
                    int(query_camids[index]) + 1
                )

                if current_human_camid != int(query_camid):
                    continue

            candidates.append(index)

        if len(candidates) == 0:
            raise ValueError(
                "No query found for pid={} and camid={}."
                .format(
                    query_pid,
                    query_camid,
                )
            )

        return candidates[0]

    raise ValueError(
        "Top-K visualization requires one selector: "
        "--query_image, --query_pid, or --query_index."
    )


# ============================================================
# Top-K retrieval
# ============================================================

def retrieve_topk(
        query_feature,
        gallery_features,
        query_pid,
        query_camid,
        gallery_pids,
        gallery_camids,
        gallery_paths,
        topk=5,
):
    """
    Retrieve Top-K gallery images under the standard ReID protocol.

    Excluded gallery samples:
        1. pid == -1
        2. same PID and same camera as the query

    No walking/cycling state information is used.
    """
    if query_feature.ndim != 2:
        raise ValueError(
            "query_feature must have shape [1, D], got {}."
            .format(tuple(query_feature.shape))
        )

    if query_feature.shape[0] != 1:
        raise ValueError(
            "Only one query is supported at a time."
        )

    if gallery_features.ndim != 2:
        raise ValueError(
            "gallery_features must have shape [N, D], got {}."
            .format(tuple(gallery_features.shape))
        )

    if gallery_features.shape[0] != len(gallery_paths):
        raise ValueError(
            "Gallery feature count {} does not match path count {}."
            .format(
                gallery_features.shape[0],
                len(gallery_paths),
            )
        )

    # Squared Euclidean distance:
    #
    # d(q, g) = ||f_q - f_g||_2^2
    #
    # When features are L2-normalized, this produces the same
    # ranking as cosine similarity.
    distances = torch.sum(
        (
            gallery_features
            - query_feature
        ) ** 2,
        dim=1,
    )

    gallery_pid_tensor = torch.tensor(
        gallery_pids,
        dtype=torch.long,
    )

    gallery_camid_tensor = torch.tensor(
        gallery_camids,
        dtype=torch.long,
    )

    # Standard ReID valid-gallery mask.
    valid_mask = gallery_pid_tensor != -1

    same_pid_same_camera = (
        (gallery_pid_tensor == int(query_pid))
        & (gallery_camid_tensor == int(query_camid))
    )

    valid_mask = (
        valid_mask
        & (~same_pid_same_camera)
    )

    valid_indices = torch.nonzero(
        valid_mask,
        as_tuple=False,
    ).flatten()

    if valid_indices.numel() == 0:
        raise RuntimeError(
            "No valid gallery sample remains after filtering."
        )

    valid_distances = distances[valid_indices]

    sorted_local_indices = torch.argsort(
        valid_distances,
        descending=False,
    )

    sorted_gallery_indices = valid_indices[
        sorted_local_indices
    ]

    actual_topk = min(
        int(topk),
        int(sorted_gallery_indices.numel()),
    )

    top_gallery_indices = sorted_gallery_indices[
        :actual_topk
    ].tolist()

    top_results = []

    for rank, gallery_index in enumerate(
            top_gallery_indices,
            start=1):

        gallery_pid = int(
            gallery_pids[gallery_index]
        )

        gallery_camid = int(
            gallery_camids[gallery_index]
        )

        correct = (
            gallery_pid == int(query_pid)
        )

        top_results.append(
            {
                "rank": rank,
                "gallery_index": int(gallery_index),
                "path": str(
                    gallery_paths[gallery_index]
                ),
                "pid": gallery_pid,
                "camid": gallery_camid,
                "distance": float(
                    distances[gallery_index].item()
                ),
                "correct": bool(correct),
            }
        )

    correct_flags = [
        result["correct"]
        for result in top_results
    ]

    rank1_hit = int(
        len(correct_flags) > 0
        and correct_flags[0]
    )

    rankk_hit = int(
        any(correct_flags)
    )

    precision_at_k = (
        sum(correct_flags) / float(actual_topk)
        if actual_topk > 0
        else 0.0
    )

    first_correct_rank = None

    # Search the complete valid ranking for the first correct match.
    for rank, gallery_index in enumerate(
            sorted_gallery_indices.tolist(),
            start=1):

        if int(gallery_pids[gallery_index]) == int(query_pid):
            first_correct_rank = rank
            break

    return {
        "topk": actual_topk,
        "rank1_hit": rank1_hit,
        "rankk_hit": rankk_hit,
        "precision_at_k": precision_at_k,
        "first_correct_rank": first_correct_rank,
        "top_results": top_results,
    }


# ============================================================
# Visualization and CSV saving
# ============================================================

def draw_topk_visualization(
        query_path,
        query_pid,
        query_camid,
        retrieval_result,
        save_path,
):
    """
    Border colors:
        blue  = query image
        green = correct gallery match
        red   = incorrect gallery match

    State labels are neither displayed nor used.
    """
    image_width = 144
    image_height = 288
    margin = 18
    title_height = 74
    label_height = 90
    border_width = 5

    top_results = retrieval_result["top_results"]

    total_image_count = 1 + len(top_results)

    canvas_width = (
        margin
        + total_image_count * image_width
        + total_image_count * margin
    )

    canvas_height = (
        title_height
        + image_height
        + label_height
        + margin
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(canvas)

    title_font = load_font(18)
    label_font = load_font(14)

    title_text = (
        "Rank-1: {} | Rank-{}: {} | "
        "Precision@{}: {:.1%} | First correct rank: {}"
        .format(
            retrieval_result["rank1_hit"],
            retrieval_result["topk"],
            retrieval_result["rankk_hit"],
            retrieval_result["topk"],
            retrieval_result["precision_at_k"],
            (
                retrieval_result["first_correct_rank"]
                if retrieval_result["first_correct_rank"] is not None
                else "None"
            ),
        )
    )

    draw.text(
        (margin, 22),
        title_text,
        fill=(0, 0, 0),
        font=title_font,
    )

    image_y = title_height

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------
    query_image = read_and_resize_image(
        query_path,
        image_width,
        image_height,
    )

    query_x = margin

    canvas.paste(
        query_image,
        (query_x, image_y),
    )

    draw.rectangle(
        [
            query_x,
            image_y,
            query_x + image_width,
            image_y + image_height,
        ],
        outline=(30, 90, 210),
        width=border_width,
    )

    draw.multiline_text(
        (
            query_x,
            image_y + image_height + 8,
        ),
        "Query\nPID: {}  Cam: {}\n{}"
        .format(
            query_pid,
            int(query_camid) + 1,
            os.path.basename(query_path),
        ),
        fill=(0, 0, 0),
        font=label_font,
        spacing=3,
    )

    # --------------------------------------------------------
    # Gallery Top-K
    # --------------------------------------------------------
    for offset, item in enumerate(
            top_results,
            start=1):

        gallery_x = (
            margin
            + offset * (
                image_width + margin
            )
        )

        gallery_image = read_and_resize_image(
            item["path"],
            image_width,
            image_height,
        )

        canvas.paste(
            gallery_image,
            (
                gallery_x,
                image_y,
            ),
        )

        if item["correct"]:
            border_color = (20, 170, 70)
            status = "Correct"
        else:
            border_color = (215, 45, 45)
            status = "Wrong"

        draw.rectangle(
            [
                gallery_x,
                image_y,
                gallery_x + image_width,
                image_y + image_height,
            ],
            outline=border_color,
            width=border_width,
        )

        gallery_label = (
            "Rank-{}: {}\n"
            "PID: {}  Cam: {}\n"
            "Dist: {:.4f}"
            .format(
                item["rank"],
                status,
                item["pid"],
                int(item["camid"]) + 1,
                item["distance"],
            )
        )

        draw.multiline_text(
            (
                gallery_x,
                image_y + image_height + 8,
            ),
            gallery_label,
            fill=(0, 0, 0),
            font=label_font,
            spacing=3,
        )

    save_parent = os.path.dirname(save_path)

    if save_parent:
        os.makedirs(
            save_parent,
            exist_ok=True,
        )

    canvas.save(
        save_path,
        quality=95,
    )


def save_topk_csv(
        query_path,
        query_pid,
        query_camid,
        retrieval_result,
        save_path,
):
    """
    Save the Top-K ranking without any state field.
    """
    save_parent = os.path.dirname(save_path)

    if save_parent:
        os.makedirs(
            save_parent,
            exist_ok=True,
        )

    with open(
        save_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "query_path",
                "query_pid",
                "query_camid",
                "rank",
                "gallery_path",
                "gallery_pid",
                "gallery_camid",
                "distance",
                "correct",
            ]
        )

        for item in retrieval_result["top_results"]:
            writer.writerow(
                [
                    query_path,
                    int(query_pid),
                    int(query_camid) + 1,
                    int(item["rank"]),
                    item["path"],
                    int(item["pid"]),
                    int(item["camid"]) + 1,
                    float(item["distance"]),
                    int(item["correct"]),
                ]
            )


def visualize_selected_query(
        inference_result,
        num_query,
        query_image=None,
        query_pid=None,
        query_camid=None,
        query_index=None,
        topk=5,
        save_dir="./topk_visualization",
):
    """
    Split query/gallery features, retrieve Top-K, and save results.

    Validation order must be:
        first num_query samples = query
        remaining samples       = gallery
    """
    all_features = inference_result["features"]
    all_pids = inference_result["pids"]
    all_camids = inference_result["camids"]
    all_paths = inference_result["paths"]

    if num_query <= 0:
        raise ValueError(
            "num_query must be larger than zero."
        )

    if num_query >= all_features.shape[0]:
        raise ValueError(
            "num_query={} leaves no gallery samples."
            .format(num_query)
        )

    # Normalize once for visualization if enabled in the test config.
    if feat_norm_enabled(cfg):
        all_features = F.normalize(
            all_features,
            p=2,
            dim=1,
        )

    query_features = all_features[:num_query]
    gallery_features = all_features[num_query:]

    query_pids = all_pids[:num_query]
    gallery_pids = all_pids[num_query:]

    query_camids = all_camids[:num_query]
    gallery_camids = all_camids[num_query:]

    query_paths = all_paths[:num_query]
    gallery_paths = all_paths[num_query:]

    selected_index = select_query_index(
        query_paths=query_paths,
        query_pids=query_pids,
        query_camids=query_camids,
        query_image=query_image,
        query_pid=query_pid,
        query_camid=query_camid,
        query_index=query_index,
    )

    selected_query_feature = query_features[
        selected_index:selected_index + 1
    ]

    selected_query_pid = int(
        query_pids[selected_index]
    )

    selected_query_camid = int(
        query_camids[selected_index]
    )

    pwc_val_root = get_pwc_val_root(cfg)

    validate_loader_full_paths(
        image_paths=all_paths,
        num_query=num_query,
        pwc_val_root=pwc_val_root,
    )

    selected_query_path = resolve_reid_image_path(
        image_path=query_paths[selected_index],
        dataset_root=pwc_val_root,
        preferred_split="query",
    )

    retrieval_result = retrieve_topk(
        query_feature=selected_query_feature,
        gallery_features=gallery_features,
        query_pid=selected_query_pid,
        query_camid=selected_query_camid,
        gallery_pids=gallery_pids,
        gallery_camids=gallery_camids,
        gallery_paths=gallery_paths,
        topk=topk,
    )
    # Resolve gallery basenames into full paths only for Top-K results.
    for item in retrieval_result["top_results"]:
        item["path"] = resolve_reid_image_path(
            image_path=item["path"],
            dataset_root=pwc_val_root,
            preferred_split="gallery",
        )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    query_stem = Path(
        selected_query_path
    ).stem

    image_save_path = os.path.join(
        save_dir,
        "{}_top{}.jpg".format(
            query_stem,
            retrieval_result["topk"],
        ),
    )

    csv_save_path = os.path.join(
        save_dir,
        "{}_top{}.csv".format(
            query_stem,
            retrieval_result["topk"],
        ),
    )
    
    print("Resolved query path:", selected_query_path)
    print("Query image exists:", os.path.isfile(selected_query_path))

    for item in retrieval_result["top_results"]:
        print(
            "Rank-{} gallery path: {} | exists={}"
            .format(
                item["rank"],
                item["path"],
                os.path.isfile(item["path"]),
            )
        )
    draw_topk_visualization(
        query_path=selected_query_path,
        query_pid=selected_query_pid,
        query_camid=selected_query_camid,
        retrieval_result=retrieval_result,
        save_path=image_save_path,
    )

    save_topk_csv(
        query_path=selected_query_path,
        query_pid=selected_query_pid,
        query_camid=selected_query_camid,
        retrieval_result=retrieval_result,
        save_path=csv_save_path,
    )

    retrieval_result.update(
        {
            "query_index": selected_index,
            "query_path": selected_query_path,
            "query_pid": selected_query_pid,
            "query_camid": selected_query_camid,
            "visualization_path": image_save_path,
            "csv_path": csv_save_path,
        }
    )

    return retrieval_result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReID Testing and Top-K Visualization"
    )

    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid.yml",
        help="Path to config file",
        type=str,
    )

    # --------------------------------------------------------
    # Checkpoint architecture parameters
    # --------------------------------------------------------
    # These parameters must match the model structure used to
    # train the checkpoint, not necessarily the target PMS set.
    parser.add_argument(
        "--checkpoint_num_classes",
        type=int,
        default=6603,
        help=(
            "Number of training identities used by the checkpoint. "
            "It must match the classifier shape in TEST.WEIGHT."
        ),
    )

    parser.add_argument(
        "--checkpoint_camera_num",
        type=int,
        default=8,
        help=(
            "Number of cameras used when training the checkpoint. "
            "Relevant when MODEL.SIE_CAMERA=True."
        ),
    )

    parser.add_argument(
        "--checkpoint_view_num",
        type=int,
        default=None,
        help=(
            "Number of views used when training the checkpoint. "
            "If omitted, the target loader's view_num is used."
        ),
    )

    # --------------------------------------------------------
    # Top-K visualization arguments
    # --------------------------------------------------------
    parser.add_argument(
        "--visualize_topk",
        action="store_true",
        help="Generate a query and gallery Top-K visualization.",
    )

    query_group = parser.add_mutually_exclusive_group()

    query_group.add_argument(
        "--query_image",
        type=str,
        default=None,
        help=(
            "Exact query filename, e.g. "
            "001-cam2-01-0.jpg. The final state field is "
            "only part of the filename and is not used by inference."
        ),
    )

    query_group.add_argument(
        "--query_pid",
        type=int,
        default=None,
        help="Select the first query of the specified PID.",
    )

    query_group.add_argument(
        "--query_index",
        type=int,
        default=None,
        help="Select a query by its zero-based query-set index.",
    )

    parser.add_argument(
        "--query_camid",
        type=int,
        default=None,
        help=(
            "Optional one-based camera number used with --query_pid; "
            "for example, cam2 corresponds to --query_camid 2."
        ),
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of gallery results to visualize.",
    )

    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help=(
            "Visualization output directory. "
            "Default: OUTPUT_DIR/pms_topk"
        ),
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------
    if args.config_file:
        cfg.merge_from_file(
            args.config_file
        )

    cfg.merge_from_list(
        args.opts
    )
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    logger = setup_logger(
        "transreid",
        output_dir,
        if_train=False,
    )

    logger.info(args)

    if args.config_file:
        logger.info(
            "Loaded configuration file {}".format(
                args.config_file
            )
        )

        with open(
            args.config_file,
            mode="r",
            encoding="utf-8",
        ) as config_file:
            config_text = "\n" + config_file.read()
            logger.info(config_text)

    logger.info(
        "Running with config:\n{}".format(cfg)
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        cfg.MODEL.DEVICE_ID
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    (
        train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        target_num_classes,
        target_camera_num,
        target_view_num,
    ) = make_dataloader(cfg)

    logger.info(
        "Target validation set: num_query={}, "
        "target_num_classes={}, target_camera_num={}, "
        "target_view_num={}"
        .format(
            num_query,
            target_num_classes,
            target_camera_num,
            target_view_num,
        )
    )

    # The checkpoint structure must use its training-time sizes.
    checkpoint_view_num = (
        args.checkpoint_view_num
        if args.checkpoint_view_num is not None
        else target_view_num
    )

    model = make_model(
        cfg,
        num_class=args.checkpoint_num_classes,
        camera_num=args.checkpoint_camera_num,
        view_num=checkpoint_view_num,
    )

    model.load_param(
        cfg.TEST.WEIGHT
    )

    # --------------------------------------------------------
    # Standard ReID evaluation and feature extraction
    # --------------------------------------------------------
    inference_result = extract_reid_features(
        cfg=cfg,
        model=model,
        val_loader=val_loader,
        num_query=num_query,
        logger=logger,
    )

    cmc = inference_result["cmc"]
    mAP = inference_result["mAP"]

    logger.info("Validation Results")
    logger.info(
        "mAP: {:.1%}".format(mAP)
    )

    for rank in [1, 5, 10]:
        if rank <= len(cmc):
            logger.info(
                "CMC curve, Rank-{:<3}: {:.1%}"
                .format(
                    rank,
                    cmc[rank - 1],
                )
            )

    # --------------------------------------------------------
    # Optional single-query Top-K visualization
    # --------------------------------------------------------
    if args.visualize_topk:
        if (
            args.query_image is None
            and args.query_pid is None
            and args.query_index is None
        ):
            raise ValueError(
                "--visualize_topk requires --query_image, "
                "--query_pid, or --query_index."
            )

        visualization_dir = (
            args.vis_dir
            if args.vis_dir is not None
            else os.path.join(
                cfg.OUTPUT_DIR,
                "pms_topk",
            )
        )

        topk_result = visualize_selected_query(
            inference_result=inference_result,
            num_query=num_query,
            query_image=args.query_image,
            query_pid=args.query_pid,
            query_camid=args.query_camid,
            query_index=args.query_index,
            topk=args.topk,
            save_dir=visualization_dir,
        )

        logger.info(
            "Selected query: {}".format(
                topk_result["query_path"]
            )
        )

        logger.info(
            "Query PID: {}, Camera: {}"
            .format(
                topk_result["query_pid"],
                topk_result["query_camid"] + 1,
            )
        )

        logger.info(
            "Single-query Rank-1: {}"
            .format(
                topk_result["rank1_hit"]
            )
        )

        logger.info(
            "Single-query Rank-{}: {}"
            .format(
                topk_result["topk"],
                topk_result["rankk_hit"],
            )
        )

        logger.info(
            "Precision@{}: {:.2%}"
            .format(
                topk_result["topk"],
                topk_result["precision_at_k"],
            )
        )

        logger.info(
            "First correct rank: {}"
            .format(
                topk_result["first_correct_rank"]
            )
        )

        logger.info(
            "Visualization saved to: {}"
            .format(
                topk_result["visualization_path"]
            )
        )

        logger.info(
            "Top-K CSV saved to: {}"
            .format(
                topk_result["csv_path"]
            )
        )