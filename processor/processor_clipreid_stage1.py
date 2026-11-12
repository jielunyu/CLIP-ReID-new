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
from loss.popd_loss import POPDLoss, fft_style_intervention

# ============================================================
# Original dynamic no-replacement batch construction
#
# IMPORTANT:
# The algorithm itself is unchanged.
#
# In POPD v1.1, its input is domain labels rather than
# state labels.
#
# Every image is used exactly once per epoch.
# ============================================================

def build_unfixed_natural_batches(sta_list, batch_size):
    """
    Original Without-Replacement Dynamic Natural Sampler.

    The input label is now domain_id.

    No oversampling.
    No undersampling.
    No fixed balanced sampling.
    """
    N = sta_list.size(0)
    if N <= batch_size:
        return [torch.randperm(N, device=sta_list.device)]
    # Calculate batch numbers and capacities.
    num_batches = N // batch_size
    rem = N % batch_size
    if rem >= 4:
        b_sizes = [batch_size] * num_batches + [rem]
        num_batches += 1
    else:
        b_sizes = [batch_size] * num_batches
        b_sizes[-1] += rem
    b_sizes_t = torch.tensor(b_sizes, dtype=torch.long, device=sta_list.device)
    unique_states, counts = torch.unique(sta_list, return_counts=True)
    # Single-domain fallback.
    if unique_states.numel() < 2 or counts.min().item() < num_batches:
        perm = torch.randperm(N, device=sta_list.device)
        batches = []
        ptr = 0
        for sz in b_sizes:
            batches.append(perm[ptr : ptr + sz])
            ptr += sz
        return batches
    # Identify majority and minority groups.
    maj_state = unique_states[counts.argmax()]
    min_state = unique_states[counts.argmin()]
    # Random permutation without replacement.
    maj_idx = (sta_list == maj_state).nonzero(as_tuple=False).view(-1)
    min_idx = (sta_list == min_state).nonzero(as_tuple=False).view(-1)
    maj_perm = maj_idx[torch.randperm(maj_idx.size(0), device=sta_list.device)]
    min_perm = min_idx[torch.randperm(min_idx.size(0), device=sta_list.device)]
    N_min = min_perm.size(0)
    N_maj = maj_perm.size(0)
    # --------------------------------------------------------
    # Dynamic allocation of minority-domain samples.
    # --------------------------------------------------------
    max_min_per_batch = torch.clamp(b_sizes_t // 2, min=1)
    floor_min = min(2, N_min // num_batches)
    counts_min = torch.minimum(
        torch.full((num_batches,), floor_min, dtype=torch.long, device=sta_list.device), max_min_per_batch
    )
    rem_min = N_min - int(counts_min.sum().item())
    if rem_min > 0:
        weights = torch.empty(num_batches, device=sta_list.device).exponential_(1.0)
        extra = torch.multinomial(weights, rem_min, replacement=True)
        counts_min.scatter_add_(0, extra, torch.ones_like(extra, dtype=torch.long))
        counts_min = torch.minimum(counts_min, max_min_per_batch)
    # --------------------------------------------------------
    # Correct allocation differences.
    # --------------------------------------------------------
    diff = N_min - int(counts_min.sum().item())
    while diff > 0:
        avail_mask = (counts_min < max_min_per_batch).nonzero(as_tuple=False).view(-1)
        if avail_mask.numel() == 0:
            break
        take = min(diff, avail_mask.numel())
        chosen = avail_mask[torch.randperm(avail_mask.numel(), device=sta_list.device)[:take]]
        counts_min[chosen] += 1
        diff -= take
    counts_maj = b_sizes_t - counts_min
    # --------------------------------------------------------
    # Construct batches without replacement.
    # --------------------------------------------------------
    batches = []
    min_ptr = 0
    maj_ptr = 0
    for i in range(num_batches):
        c_min = int(counts_min[i].item())
        c_maj = int(counts_maj[i].item())
        b_min = min_perm[min_ptr : min_ptr + c_min]
        b_maj = maj_perm[maj_ptr : maj_ptr + c_maj]
        min_ptr += c_min
        maj_ptr += c_maj
        b_combined = torch.cat([b_maj, b_min], dim=0)
        b_combined = b_combined[torch.randperm(b_combined.size(0), device=sta_list.device)]
        batches.append(b_combined)
    return batches
# ============================================================
# Domain-State diagnostic
#
# This is only a diagnostic.
# It does not introduce any additional loss.
# ============================================================

def domain_state_statistics(dom_list, sta_list, num_domains, num_states):
    contingency = torch.zeros(num_domains, num_states, dtype=torch.float32, device=dom_list.device)
    for d in range(num_domains):
        for s in range(num_states):
            contingency[d, s] = ((dom_list == d) & (sta_list == s)).sum()
    total = contingency.sum()
    if total.item() == 0:
        return (contingency.cpu().tolist(), 0.0)
    p_joint = contingency / total
    p_domain = p_joint.sum(dim=1)
    p_state = p_joint.sum(dim=0)
    mutual_info = torch.tensor(0.0, device=dom_list.device)
    for d in range(num_domains):
        for s in range(num_states):
            p = p_joint[d, s]
            if p.item() > 0:
                mutual_info += p * torch.log(p / (p_domain[d] * p_state[s]))
    entropy_domain = -(p_domain[p_domain > 0] * torch.log(p_domain[p_domain > 0])).sum()
    entropy_state = -(p_state[p_state > 0] * torch.log(p_state[p_state > 0])).sum()
    denominator = torch.sqrt(entropy_domain * entropy_state)
    if denominator.item() > 0:
        nmi = (mutual_info / denominator).item()
    else:
        nmi = 0.0
    return (contingency.cpu().tolist(), nmi)
# ============================================================
# POPD Stage1 Training
# ============================================================

def do_train_stage1(cfg, model, train_loader_stage1, optimizer, scheduler, local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    logger = logging.getLogger("transreid.train")
    logger.info("start training " "(Stage1 with POPD Tri-Factor Prompt Disentanglement)")
    # --------------------------------------------------------
    # 1. Device
    # --------------------------------------------------------
    device = "cuda"
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
    base_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
    if not hasattr(base_model, "state_priors"):
        raise RuntimeError("POPD error: state_priors not found.")
    # --------------------------------------------------------
    # 2. Freeze CLIP Text Encoder
    #
    # Important:
    # Frozen weights do not prevent gradients from flowing
    # back to learnable prompt embeddings.
    # --------------------------------------------------------
    for p in base_model.text_encoder.parameters():
        p.requires_grad_(False)
    base_model.text_encoder.eval()
    logger.info("Stage1: CLIP Text Encoder frozen; " "prompt parameters remain trainable.")
    num_domains = base_model.prompt_learner.num_domains
    num_states = base_model.prompt_learner.num_states
    if num_domains != 2:
        raise RuntimeError("Current POPD experiment expects " "two dataset domains.")
    if num_states != 2:
        raise RuntimeError("Current POPD experiment expects " "two semantic states.")
    # --------------------------------------------------------
    # 3. Loss and optimizer
    # --------------------------------------------------------
    loss_meter = AverageMeter()
    comp_meter = AverageMeter()
    intv_meter = AverageMeter()
    anch_meter = AverageMeter()
    orth_meter = AverageMeter()
    scaler = amp.GradScaler()
    popd_criterion = POPDLoss(
        device=device,
        num_states=num_states,
        feat_dim=base_model.state_priors.size(-1),
        tau_sem=0.85,
        lambda_interv=0.5,
        lambda_anchor=0.5,
        lambda_orth=1.0,
    )
    # --------------------------------------------------------
    # 4. Stage1 feature cache
    #
    # Each original image is extracted once.
    # FDA image features are cached correspondingly.
    # --------------------------------------------------------
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    image_features_fda = []
    labels = []
    dom_labels = []
    swapped_dom_labels = []
    model.eval()
    amp_bank = {}
    with torch.no_grad():
        for n_iter, batch_data in enumerate(train_loader_stage1):
            (img, vid, target_cam, target_track, target_domain) = batch_data
            img = img.to(device)
            target = vid.to(device)
            # Camera and track IDs are not used as
            # POPD domain/state supervision.
            domain = target_domain.to(device).long()
            # ------------------------------------------------
            # FDA:
            # Source phase + other-domain amplitude.
            # ------------------------------------------------
            img_fda, domain_fda = fft_style_intervention(images=img, dom_labels=domain, amp_bank=amp_bank, beta=0.08)
            # ------------------------------------------------
            # Extract frozen visual features.
            # ------------------------------------------------
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                image_feature_fda = model(img_fda, target, get_image=True)
            for label_i, img_feat, img_feat_fda, d_i, d_fda_i in zip(
                target, image_feature, image_feature_fda, domain, domain_fda
            ):
                labels.append(label_i.detach().cpu())
                image_features.append(img_feat.detach().cpu())
                image_features_fda.append(img_feat_fda.detach().cpu())
                dom_labels.append(d_i.detach().cpu())
                swapped_dom_labels.append(d_fda_i.detach().cpu())
    if len(labels) == 0 or len(image_features) == 0:
        raise RuntimeError("Stage1 error: no image features extracted.")
    # --------------------------------------------------------
    # 5. Convert feature cache to GPU tensors
    # --------------------------------------------------------
    labels_list = torch.stack(labels, dim=0).cuda()
    image_features_list = torch.stack(image_features, dim=0).cuda()
    image_features_fda_list = torch.stack(image_features_fda, dim=0).cuda()
    dom_list = torch.stack(dom_labels, dim=0).cuda().long()
    swapped_dom_list = torch.stack(swapped_dom_labels, dim=0).cuda().long()
    num_image = labels_list.shape[0]
    # --------------------------------------------------------
    # 6. Validate explicit dataset-domain labels
    # --------------------------------------------------------
    if dom_list.min().item() < 0 or dom_list.max().item() >= num_domains:
        raise RuntimeError("Invalid dataset-domain labels.")
    domain_counts = torch.bincount(dom_list, minlength=num_domains)
    if (domain_counts == 0).any():
        raise RuntimeError("At least one dataset domain has no images.")
    # --------------------------------------------------------
    # 7. Generate semantic-state pseudo labels
    #
    # IMPORTANT:
    # Do NOT use raw trackid here.
    #
    # The current Market/BikeReID trackid values are
    # dataset markers, not independent behavior labels.
    #
    # State pseudo labels are generated using frozen CLIP
    # visual-text semantic similarities.
    # --------------------------------------------------------
    with torch.no_grad():
        norm_feats = F.normalize(image_features_list.float(), dim=-1)
        norm_priors = F.normalize(base_model.state_priors.float(), dim=-1)
        sim_prior = torch.matmul(norm_feats, norm_priors.t())
        sta_list = sim_prior.argmax(dim=-1).long()
        mean_max_similarity = sim_prior.max(dim=-1).values.mean().item()
    state_counts = torch.bincount(sta_list, minlength=num_states)
    if (state_counts == 0).any():
        raise RuntimeError(
            "CLIP semantic-state pseudo labels collapsed "
            "to one class. Check state_priors or provide "
            "independent state annotations before training."
        )
    # --------------------------------------------------------
    # 8. Domain-State correlation diagnostic
    #
    # This is only a diagnostic, not a loss.
    # --------------------------------------------------------
    contingency, nmi = domain_state_statistics(dom_list, sta_list, num_domains, num_states)
    logger.info("Stage1 dataset-domain counts: {}".format(domain_counts.tolist()))
    logger.info("Stage1 CLIP semantic-state pseudo-label counts: {}".format(state_counts.tolist()))
    logger.info("Stage1 mean maximum CLIP semantic similarity: {:.4f}".format(mean_max_similarity))
    logger.info("Stage1 Domain-State contingency matrix: {}".format(contingency))
    logger.info("Stage1 Domain-State NMI: {:.4f}".format(nmi))
    if nmi > 0.95:
        logger.warning(
            "Domain-State pseudo labels remain highly correlated. "
            "FDA intervention and semantic-state assignments "
            "should be carefully inspected."
        )
    # --------------------------------------------------------
    # 9. Validate dataset size and batch size
    # --------------------------------------------------------
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    if num_image <= 0:
        raise RuntimeError("Stage1 error: num_image <= 0.")
    if batch <= 0:
        raise RuntimeError("Stage1 error: batch size must be > 0.")
    # Release temporary CPU lists.
    del labels
    del image_features
    del image_features_fda
    del dom_labels
    del swapped_dom_labels
    # --------------------------------------------------------
    # 10. Stage1 prompt optimization
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        comp_meter.reset()
        intv_meter.reset()
        anch_meter.reset()
        orth_meter.reset()
        scheduler.step(epoch)
        model.train()
        # Keep the frozen text encoder in evaluation mode.
        base_model.text_encoder.eval()
        # IMPORTANT:
        # Use domain IDs for dynamic batch construction.
        #
        # The original no-replacement algorithm is unchanged.
        # No image is oversampled or discarded.
        # ----------------------------------------------------
        batch_list = build_unfixed_natural_batches(dom_list, batch)
        num_batches = len(batch_list)
        # ----------------------------------------------------
        # Verify full-dataset coverage.
        # ----------------------------------------------------
        if epoch == 1:
            flat_indices = torch.cat(batch_list, dim=0)
            if flat_indices.numel() != num_image or torch.unique(flat_indices).numel() != num_image:
                raise RuntimeError("Stage1 sampler failed full-dataset " "no-replacement coverage.")
            logger.info(
                "Stage1 feature cache and dynamic "
                "domain sampler ready: "
                "num_image={}, batch={}, num_batches={}, "
                "domain_counts={}".format(num_image, batch, num_batches, domain_counts.tolist())
            )
        # ----------------------------------------------------
        # Batch training
        # ----------------------------------------------------
        for i, b_list in enumerate(batch_list):
            optimizer.zero_grad()
            if b_list.numel() == 0:
                logger.warning("Skip empty batch: epoch={}, iter={}".format(epoch, i))
                continue
            target = labels_list[b_list]
            dom_t = dom_list[b_list]
            dom_fda_t = swapped_dom_list[b_list]
            sta_t = sta_list[b_list]
            image_feat = image_features_list[b_list]
            image_feat_fda = image_features_fda_list[b_list]
            if target.size(0) == 0 or image_feat.size(0) == 0:
                continue
            # ------------------------------------------------
            # 1. Original compositional and marginal prompts
            # ------------------------------------------------
            with amp.autocast(enabled=True):
                (z_comp, z_D_all, z_S_all, z_X_batch, state_priors) = base_model(
                    label=target, get_text=True, cam_label=dom_t, view_label=sta_t, return_popd=True
                )
                # ------------------------------------------------
                # 2. Counterfactual compositional prompts
                #
                # Domain is swapped.
                # Identity and state remain unchanged.
                # ------------------------------------------------
                (z_comp_fda, _, _, _, _) = base_model(
                    label=target, get_text=True, cam_label=dom_fda_t, view_label=sta_t, return_popd=True
                )
                # ------------------------------------------------
                # 3. Four original POPD losses
                # ------------------------------------------------
                loss, loss_dict = popd_criterion(
                    image_feat,
                    image_feat_fda,
                    z_comp,
                    z_comp_fda,
                    z_D_all,
                    z_S_all,
                    z_X_batch,
                    state_priors,
                    target,
                    dom_t,
                    dom_fda_t,
                    sta_t,
                )
            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ------------------------------------------------
            # Loss statistics
            # ------------------------------------------------
            batch_sz = target.size(0)
            loss_meter.update(loss.item(), batch_sz)
            comp_meter.update(loss_dict["L_comp"], batch_sz)
            intv_meter.update(loss_dict["L_intv"], batch_sz)
            anch_meter.update(loss_dict["L_anch"], batch_sz)
            orth_meter.update(loss_dict["L_orth"], batch_sz)
            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] "
                    "Loss: {:.3f} "
                    "(Comp: {:.3f}, "
                    "Intv: {:.3f}, "
                    "Anch: {:.3f}, "
                    "Orth: {:.3f}), "
                    "Base Lr: {:.2e}".format(
                        epoch,
                        i + 1,
                        num_batches,
                        loss_meter.avg,
                        comp_meter.avg,
                        intv_meter.avg,
                        anch_meter.avg,
                        orth_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
        # --------------------------------------------------------
        # 11. Save Stage1 checkpoint
        # --------------------------------------------------------
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch)),
                    )
            else:
                torch.save(
                    model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_stage1_{}.pth".format(epoch))
                )
    # --------------------------------------------------------
    # 12. Finish
    # --------------------------------------------------------
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
