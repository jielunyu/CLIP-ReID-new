import logging
import os
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from loss.supcontrast import SupConLoss
from loss.popd_loss import POPDLoss, fft_style_intervention


def build_unfixed_natural_batches(sta_list, batch_size):
    """
    无放回动态非固定比例采样器 (Without-Replacement Dynamic Natural Sampler):
    1. 严格保证每个 Epoch 内，多数类和少数类的每一张图都【只被采样 1 次】(0% 过采样, 0% 欠采样)。
    2. 每个 Batch 内的比例不固定 (围绕天然比例动态随机波动，如 59:5, 55:9, 61:3...)。
    3. 精确对齐每个 Batch 的容量上限，杜绝末尾余数 Batch 指针提前耗尽或出现单样本 Batch。
    """
    N = sta_list.size(0)
    if N <= batch_size:
        return [torch.randperm(N, device=sta_list.device)]

    # 计算 Batch 数量与每个 Batch 的精确容量 b_sizes
    num_batches = N // batch_size
    rem = N % batch_size
    if rem >= 4:
        # 余数 >= 4 时单独作为一个尾 Batch
        b_sizes = [batch_size] * num_batches + [rem]
        num_batches += 1
    else:
        # 余数 < 4 (如只有1~3张图) 时，直接并入最后一个 Batch，防止单样本对比损失退化
        b_sizes = [batch_size] * num_batches
        b_sizes[-1] += rem

    b_sizes_t = torch.tensor(b_sizes, dtype=torch.long, device=sta_list.device)
    unique_states, counts = torch.unique(sta_list, return_counts=True)

    # 若只有单状态，或者少数类总样本数不足以每个 Batch 分 1 个，退化为标准无放回打乱
    if unique_states.numel() < 2 or counts.min().item() < num_batches:
        perm = torch.randperm(N, device=sta_list.device)
        batches = []
        ptr = 0
        for sz in b_sizes:
            batches.append(perm[ptr: ptr + sz])
            ptr += sz
        return batches

    maj_state = unique_states[counts.argmax()]
    min_state = unique_states[counts.argmin()]

    # 1. 对多数类和少数类分别进行【无放回】随机打乱
    maj_idx = (sta_list == maj_state).nonzero(as_tuple=False).view(-1)
    min_idx = (sta_list == min_state).nonzero(as_tuple=False).view(-1)
    maj_perm = maj_idx[torch.randperm(maj_idx.size(0), device=sta_list.device)]
    min_perm = min_idx[torch.randperm(min_idx.size(0), device=sta_list.device)]

    N_min = min_perm.size(0)
    N_maj = maj_perm.size(0)

    # 2. 动态分配每个 Batch 的少数类样本数 counts_min，保证 1 <= counts_min[i] <= max_min_per_batch[i]
    max_min_per_batch = torch.clamp(b_sizes_t // 2, min=1)
    floor_min = min(2, N_min // num_batches)
    counts_min = torch.minimum(
        torch.full((num_batches,), floor_min, dtype=torch.long, device=sta_list.device),
        max_min_per_batch
    )
    rem_min = N_min - int(counts_min.sum().item())

    if rem_min > 0:
        # 用随机指数分布生成非固定比例权重
        weights = torch.empty(num_batches, device=sta_list.device).exponential_(1.0)
        extra = torch.multinomial(weights, rem_min, replacement=True)
        counts_min.scatter_add_(0, extra, torch.ones_like(extra, dtype=torch.long))
        counts_min = torch.minimum(counts_min, max_min_per_batch)

    # 3. 若因上限截断产生微小差额，均匀补齐到仍有余量的桶中，确保 sum(counts_min) == N_min
    diff = N_min - int(counts_min.sum().item())
    while diff > 0:
        avail_mask = (counts_min < max_min_per_batch).nonzero(as_tuple=False).view(-1)
        if avail_mask.numel() == 0:
            break
        take = min(diff, avail_mask.numel())
        chosen = avail_mask[torch.randperm(avail_mask.numel(), device=sta_list.device)[:take]]
        counts_min[chosen] += 1
        diff -= take

    # 多数类在每个 Batch 的数量严格等于 (桶总容量 - 少数类数量)，天然保证 sum(counts_maj) == N_maj
    counts_maj = b_sizes_t - counts_min

    # 4. 无放回精确切片：min_ptr 必停在 N_min，maj_ptr 必停在 N_maj
    batches = []
    min_ptr = 0
    maj_ptr = 0
    for i in range(num_batches):
        c_min = int(counts_min[i].item())
        c_maj = int(counts_maj[i].item())

        b_min = min_perm[min_ptr: min_ptr + c_min]
        b_maj = maj_perm[maj_ptr: maj_ptr + c_maj]
        min_ptr += c_min
        maj_ptr += c_maj

        b_combined = torch.cat([b_maj, b_min], dim=0)
        b_combined = b_combined[torch.randperm(b_combined.size(0), device=sta_list.device)]
        batches.append(b_combined)

    return batches


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info("start training (Stage1 with POPD Tri-Factor Prompt Disentanglement)")

    # -------------------------------------------------
    # device 设置
    # -------------------------------------------------
    device = "cuda"

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    # 获取底层模型实例，避免 DataParallel 在 x=None 或固定维度边际张量上报错
    base_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

    if not hasattr(base_model, "state_priors"):
        raise RuntimeError(
            "Stage1 POPD error: 'base_model' does not have 'state_priors'. "
            "Please make sure 'model/make_model_clipreid.py' has been updated with Module 1."
        )

    # 损失统计器（总损失 + 四个子损失项）
    loss_meter = AverageMeter()
    comp_meter = AverageMeter()
    intv_meter = AverageMeter()
    anch_meter = AverageMeter()
    orth_meter = AverageMeter()

    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    popd_criterion = POPDLoss(
        device=device,
        num_states=base_model.prompt_learner.num_states,
        feat_dim=base_model.state_priors.size(-1),
        tau_sem=0.85,
        lambda_interv=0.5,
        lambda_anchor=0.5,
        lambda_orth=1.0
    )

    # -------------------------------------------------
    # Stage1: 先提取所有原图特征与 FDA 风格干预特征并缓存
    # -------------------------------------------------
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    image_features = []
    image_features_fda = []
    labels = []
    dom_labels = []
    swapped_dom_labels = []
    raw_view_labels = []

    model.eval()
    amp_bank = {}  # 跨 Batch 低频幅度谱记忆池
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            cam = target_cam.to(device)
            view = target_view.to(device)

            # 传入 view 和 amp_bank，确保即使单 Batch 全是同一数据集也能跨域交换低频谱
            img_fda, cam_fda = fft_style_intervention(img, cam, sta_labels=view, amp_bank=amp_bank, beta=0.08)

            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                image_feature_fda = model(img_fda, target, get_image=True)

            for label_i, img_feat, img_feat_fda, d_i, d_fda_i, v_i in zip(
                target, image_feature, image_feature_fda, cam, cam_fda, view
            ):
                labels.append(label_i.detach().cpu())
                image_features.append(img_feat.detach().cpu())
                image_features_fda.append(img_feat_fda.detach().cpu())
                dom_labels.append(d_i.detach().cpu())
                swapped_dom_labels.append(d_fda_i.detach().cpu())
                raw_view_labels.append(v_i.detach().cpu())

    if len(labels) == 0 or len(image_features) == 0:
        raise RuntimeError(
            "Stage1 error: no image features were extracted. "
            "Please check train_loader_stage1 and dataset parsing."
        )

    labels_list = torch.stack(labels, dim=0).cuda()                         # [N]
    image_features_list = torch.stack(image_features, dim=0).cuda()         # [N, C]
    image_features_fda_list = torch.stack(image_features_fda, dim=0).cuda() # [N, C]
    dom_list = torch.stack(dom_labels, dim=0).cuda()                        # [N]
    swapped_dom_list = torch.stack(swapped_dom_labels, dim=0).cuda()        # [N]
    raw_view_list = torch.stack(raw_view_labels, dim=0).cuda()              # [N]

    # -------------------------------------------------
    # 全局判定状态标签来源：显式标签 vs. CLIP 零样本语义先验推断
    # -------------------------------------------------
    unique_views = torch.unique(raw_view_list)
    if unique_views.numel() >= 2 and raw_view_list.max() < base_model.prompt_learner.num_states:
        sta_list = raw_view_list.long()
        logger.info(
            "Stage1 POPD: Using explicit state labels from dataloader (unique states={}).".format(
                unique_views.tolist()
            )
        )
    else:
        with torch.no_grad():
            norm_feats = F.normalize(image_features_list.float(), dim=-1)
            norm_priors = F.normalize(base_model.state_priors.float(), dim=-1)
            sim_prior = torch.matmul(norm_feats, norm_priors.t())
            sta_list = sim_prior.argmax(dim=-1).long()
        sta_counts = torch.bincount(sta_list).tolist()
        logger.info(
            "Stage1 POPD: No explicit multi-state labels in target_view. "
            "Activated CLIP Zero-Shot State Discovery (pseudo state distribution={}).".format(sta_counts)
        )

    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]

    if num_image <= 0:
        raise RuntimeError(
            "Stage1 error: num_image <= 0. "
            "Please check whether the training dataset is empty."
        )

    if batch <= 0:
        raise RuntimeError(
            "Stage1 error: cfg.SOLVER.STAGE1.IMS_PER_BATCH must be > 0."
        )

    del labels, image_features, image_features_fda, dom_labels, swapped_dom_labels, raw_view_labels

    # -------------------------------------------------
    # Stage1: 用缓存的 image features 训练三元解耦提示词 (POPD)
    # -------------------------------------------------
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        comp_meter.reset()
        intv_meter.reset()
        anch_meter.reset()
        orth_meter.reset()

        scheduler.step(epoch)
        model.train()

        # 调用无放回动态非固定比例采样器生成当前 Epoch 的所有 Batch
        batch_list = build_unfixed_natural_batches(sta_list, batch)
        num_batches = len(batch_list)

        if epoch == 1:
            logger.info(
                "Stage1 feature cache & dynamic sampler ready: num_image={}, batch={}, num_batches={}".format(
                    num_image, batch, num_batches
                )
            )

        for i, b_list in enumerate(batch_list):
            optimizer.zero_grad()

            # 保护检查
            if b_list.numel() == 0:
                logger.warning(
                    "Skip empty batch in Stage1: epoch={}, iter={}, num_image={}".format(
                        epoch, i, num_image
                    )
                )
                continue

            target = labels_list[b_list]
            dom_t = dom_list[b_list]
            dom_fda_t = swapped_dom_list[b_list]
            sta_t = sta_list[b_list]
            image_feat = image_features_list[b_list]
            image_feat_fda = image_features_fda_list[b_list]

            if target.size(0) == 0 or image_feat.size(0) == 0:
                logger.warning(
                    "Skip invalid batch in Stage1: target={}, image_features={}".format(
                        target.shape, image_feat.shape
                    )
                )
                continue

            with amp.autocast(enabled=True):
                # 1. 前向获取真实组合提示 z_comp 以及三元边际独立提示 (z_D_all, z_S_all, z_X_batch)
                z_comp, z_D_all, z_S_all, z_X_batch, state_priors = base_model(
                    label=target,
                    get_text=True,
                    cam_label=dom_t,
                    view_label=sta_t,
                    return_popd=True
                )

                if z_comp.size(0) == 0:
                    logger.warning(
                        "Skip empty z_comp in Stage1: epoch={}, iter={}".format(
                            epoch, i
                        )
                    )
                    continue

                # 2. 前向获取 FDA 风格干预图像对应的重组提示 P_comp(d_swap, i, s_orig)
                z_comp_fda, _, _, _, _ = base_model(
                    label=target,
                    get_text=True,
                    cam_label=dom_fda_t,
                    view_label=sta_t,
                    return_popd=True
                )

                # 3. 计算 POPD 模块四项联合损失
                loss, loss_dict = popd_criterion(
                    image_feat, image_feat_fda,
                    z_comp, z_comp_fda,
                    z_D_all, z_S_all, z_X_batch, state_priors,
                    target, dom_t, dom_fda_t, sta_t
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_sz = target.size(0)
            loss_meter.update(loss.item(), batch_sz)
            comp_meter.update(loss_dict["L_comp"], batch_sz)
            intv_meter.update(loss_dict["L_intv"], batch_sz)
            anch_meter.update(loss_dict["L_anch"], batch_sz)
            orth_meter.update(loss_dict["L_orth"], batch_sz)

            torch.cuda.synchronize()

            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f} "
                    "(Comp: {:.3f}, Intv: {:.3f}, Anch: {:.3f}, Orth: {:.3f}), Base Lr: {:.2e}".format(
                        epoch,
                        i + 1,
                        num_batches,
                        loss_meter.avg,
                        comp_meter.avg,
                        intv_meter.avg,
                        anch_meter.avg,
                        orth_meter.avg,
                        scheduler._get_lr(epoch)[0]
                    )
                )

        # -------------------------------------------------
        # 保存 checkpoint
        # -------------------------------------------------
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            cfg.OUTPUT_DIR,
                            cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)
                        )
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR,
                        cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)
                    )
                )

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))