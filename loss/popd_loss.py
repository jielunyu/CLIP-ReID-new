import torch
import torch.nn as nn
import torch.nn.functional as F
from .supcontrast import SupConLoss

def fft_style_intervention(images, dom_labels, sta_labels=None, amp_bank=None, beta=0.08):
    """
    支持 1:10 极端不平衡数据集的跨状态/跨域频域风格干预 (FDA with Cross-Domain Memory Bank):
    - 优先从不同状态 (Walk vs. Cycle) 或不同域 (Market vs. CS-S) 的样本中获取低频幅度谱。
    - 维护一个跨 Batch 的 amp_bank 记忆池，哪怕当前 Batch 64 张图全是同一数据集，
      也能从 amp_bank 中取出另一数据集的低频幅度谱进行 100% 跨数据集风格干预！
    """
    B, C, H, W = images.shape
    h_crop = max(1, int(H * beta))
    w_crop = max(1, int(W * beta))
    c_h, c_w = H // 2, W // 2

    img_fp32 = images.float()
    fft_src = torch.fft.fftshift(torch.fft.fft2(img_fp32, dim=(-2, -1)), dim=(-2, -1))
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

    # 提取当前 Batch 的中心低频块 [B, C, 2*h_crop, 2*w_crop]
    low_freq_src = amp_src[:, :, c_h - h_crop:c_h + h_crop, c_w - w_crop:c_w + w_crop].detach()

    # 确定分组键：如果提供了双状态标签 (Walk vs. Cycle) 且有区分度，优先按状态跨域交换，否则按 dom_labels
    group_labels = sta_labels if (sta_labels is not None and sta_labels.max() > sta_labels.min()) else dom_labels

    # 1. 更新跨 Batch 低频幅度谱记忆池 amp_bank: {group_id: (low_freq_tensor, dom_label_tensor)}
    if amp_bank is not None:
        for g_val in torch.unique(group_labels):
            g_id = int(g_val.item())
            mask = (group_labels == g_val)
            g_amps = low_freq_src[mask].cpu()
            g_doms = dom_labels[mask].cpu()
            if g_id not in amp_bank:
                amp_bank[g_id] = (g_amps[:64], g_doms[:64])
            else:
                old_amps, old_doms = amp_bank[g_id]
                new_amps = torch.cat([g_amps, old_amps], dim=0)[:64] # 最多保留最近64个，几乎不占内存
                new_doms = torch.cat([g_doms, old_doms], dim=0)[:64]
                amp_bank[g_id] = (new_amps, new_doms)

    # 2. 为当前 Batch 每个样本匹配异域/异状态的低频幅度谱
    amp_mixed = amp_src.clone()
    swapped_dom_labels = dom_labels.clone()

    for i in range(B):
        cur_g = int(group_labels[i].item())
        cur_d = int(dom_labels[i].item())

        # 优先在当前 Batch 内找不同 group (或不同 domain) 的样本
        diff_mask = (group_labels != cur_g).nonzero(as_tuple=False).view(-1)
        if len(diff_mask) == 0:
            diff_mask = (dom_labels != cur_d).nonzero(as_tuple=False).view(-1)

        if len(diff_mask) > 0:
            rand_idx = diff_mask[torch.randint(0, len(diff_mask), (1,)).item()]
            amp_mixed[i, :, c_h - h_crop:c_h + h_crop, c_w - w_crop:c_w + w_crop] = low_freq_src[rand_idx]
            swapped_dom_labels[i] = dom_labels[rand_idx]
        elif amp_bank is not None and len(amp_bank) > 1:
            # 当前 Batch 全是同一数据集！直接从 amp_bank 记忆池中抽取另一数据集的低频谱
            other_keys = [k for k in amp_bank.keys() if k != cur_g]
            if len(other_keys) > 0:
                trg_k = other_keys[torch.randint(0, len(other_keys), (1,)).item()]
                bank_amps, bank_doms = amp_bank[trg_k]
                r_idx = torch.randint(0, bank_amps.size(0), (1,)).item()
                amp_mixed[i, :, c_h - h_crop:c_h + h_crop, c_w - w_crop:c_w + w_crop] = bank_amps[r_idx].to(images.device)
                swapped_dom_labels[i] = bank_doms[r_idx].to(images.device)

    # 3. 逆傅里叶变换重建风格干预图像
    fft_mixed = torch.polar(amp_mixed, pha_src)
    fft_mixed = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    images_fda = torch.real(torch.fft.ifft2(fft_mixed, dim=(-2, -1)))

    return images_fda.type_as(images), swapped_dom_labels


class POPDLoss(nn.Module):
    def __init__(self, device="cuda", num_states=2, feat_dim=512, ema_m=0.9,
                 tau_sem=0.85, lambda_interv=0.5, lambda_anchor=0.5, lambda_orth=1.0):
        super().__init__()
        self.supcon = SupConLoss(device)
        self.tau_sem = tau_sem
        self.lambda_interv = lambda_interv
        self.lambda_anchor = lambda_anchor
        self.lambda_orth = lambda_orth
        self.ema_m = ema_m

        # 跨 Batch 动量身份中心: 即使当前 Batch 比例是 62:2 甚至 64:0，也能提供稳定的状态组均值!
        self.register_buffer("ema_center_X", torch.zeros(num_states, feat_dim, device=device))
        self.register_buffer("ema_inited", torch.zeros(num_states, dtype=torch.bool, device=device))

    def _balanced_ce_loss(self, logits, targets):
        unique_classes = torch.unique(targets)
        loss_sum = 0.0
        for c in unique_classes:
            mask = (targets == c)
            loss_sum = loss_sum + F.cross_entropy(logits[mask], targets[mask])
        return loss_sum / len(unique_classes)

    def forward(self, img_feat, img_feat_fda, z_comp, z_comp_fda, 
                z_D_all, z_S_all, z_X_batch, state_priors, 
                labels, dom_labels, swapped_dom_labels, sta_labels):
        img_f = F.normalize(img_feat.float(), dim=-1)
        img_f_fda = F.normalize(img_feat_fda.float(), dim=-1)
        z_D = F.normalize(z_D_all.float(), dim=-1)         # [N_dom, 512]
        z_S = F.normalize(z_S_all.float(), dim=-1)         # [N_sta, 512]
        z_X = F.normalize(z_X_batch.float(), dim=-1)       # [B, 512]
        priors = F.normalize(state_priors.float(), dim=-1) # [2, 512]

        # 1. 全因子组合对比损失 L_comp
        loss_comp = self.supcon(img_feat, z_comp, labels, labels) + \
                    self.supcon(z_comp, img_feat, labels, labels)

        # 2. 跨数据集风格干预损失 L_interv
        num_dom = z_D.shape[0]
        logits_dom_fda = torch.matmul(img_f_fda, z_D.t()) / 0.07
        loss_dom_cls_fda = self._balanced_ce_loss(logits_dom_fda, swapped_dom_labels % num_dom)
        loss_interv = self.supcon(img_feat_fda, z_comp_fda, labels, labels) + \
                      self.supcon(z_comp_fda, img_feat_fda, labels, labels) + loss_dom_cls_fda

        # 3. 语义流形锚定损失 L_anchor
        cos_sim_prior = (z_S * priors).sum(dim=-1)
        loss_cone = F.relu(self.tau_sem - cos_sim_prior).mean()

        num_sta = z_S.shape[0]
        logits_sta = torch.matmul(img_f, z_S.t()) / 0.07
        loss_sta_cls = self._balanced_ce_loss(logits_sta, sta_labels % num_sta)
        
        logits_dom = torch.matmul(img_f, z_D.t()) / 0.07
        loss_dom_cls = self._balanced_ce_loss(logits_dom, dom_labels % num_dom)
        loss_anchor = loss_cone + 0.5 * (loss_sta_cls + loss_dom_cls)

        # 4. 基于跨 Batch 动量中心的差分正交约束 L_orth (完美适配不固定采样比例)
        delta_z_S = F.normalize(z_S[0:1] - z_S[1:2], dim=-1)
        z_D_mean = z_D.mean(dim=0, keepdim=True)
        delta_z_D = F.normalize(z_D - z_D_mean + 1e-6, dim=-1)

        delta_z_X_list = []
        for s_val in torch.unique(sta_labels):
            s_idx = int((s_val % num_sta).item())
            s_mask = (sta_labels == s_val)
            z_X_s = z_X[s_mask]
            batch_mean_s = z_X_s.detach().mean(dim=0)

            # 更新该状态的 EMA 动量中心
            with torch.no_grad():
                if not self.ema_inited[s_idx]:
                    self.ema_center_X[s_idx].copy_(batch_mean_s)
                    self.ema_inited[s_idx] = True
                else:
                    self.ema_center_X[s_idx].mul_(self.ema_m).add_(batch_mean_s * (1.0 - self.ema_m))

            # 使用稳定的 EMA 动量中心做去中心化，不受当前 Batch 内样本数过少的影响
            center_s = self.ema_center_X[s_idx].unsqueeze(0).detach()
            delta_z_X_s = F.normalize(z_X_s - center_s + 1e-6, dim=-1)
            delta_z_X_list.append(delta_z_X_s)

        delta_z_X = torch.cat(delta_z_X_list, dim=0)

        orth_S_D = torch.abs(torch.matmul(delta_z_S, delta_z_D.t())).mean()
        orth_X_S = torch.abs(torch.matmul(delta_z_X, delta_z_S.t())).mean()
        orth_X_D = torch.abs(torch.matmul(delta_z_X, delta_z_D.t())).mean()

        loss_orth = orth_S_D + orth_X_S + orth_X_D

        total_loss = loss_comp + \
                     self.lambda_interv * loss_interv + \
                     self.lambda_anchor * loss_anchor + \
                     self.lambda_orth * loss_orth

        loss_dict = {
            "L_comp": loss_comp.item(),
            "L_intv": loss_interv.item(),
            "L_anch": loss_anchor.item(),
            "L_orth": loss_orth.item()
        }
        return total_loss, loss_dict