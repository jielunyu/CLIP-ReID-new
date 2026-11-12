import torch
import torch.nn as nn
import torch.nn.functional as F
from .supcontrast import SupConLoss

# ============================================================
# Frequency Domain Adaptation
# Cross-Dataset Domain Intervention
# ============================================================

def fft_style_intervention(images, dom_labels, sta_labels=None, amp_bank=None, beta=0.08):
    """
    Cross-domain low-frequency amplitude intervention.

    Source:
        Original image phase.

    Donor:
        Low-frequency amplitude from another dataset domain.

    Domain labels:
        0 = Market
        1 = CS-S / BikeReID

    Important:
        Donor selection depends ONLY on domain IDs.

        Camera IDs and track IDs are not used.

    amp_bank:
        {
            domain_id: low_frequency_amplitude_tensor
        }

    The memory bank stores at most 64 amplitude crops
    per domain on CPU.
    """
    B, C, H, W = images.shape
    dom_labels = dom_labels.long()
    h_crop = max(1, int(H * beta))
    w_crop = max(1, int(W * beta))
    c_h = H // 2
    c_w = W // 2
    # --------------------------------------------------------
    # 1. Fourier transform
    # --------------------------------------------------------
    img_fp32 = images.float()
    fft_src = torch.fft.fftshift(torch.fft.fft2(img_fp32, dim=(-2, -1)), dim=(-2, -1))
    amp_src = torch.abs(fft_src)
    pha_src = torch.angle(fft_src)
    low_freq_src = amp_src[:, :, c_h - h_crop : c_h + h_crop, c_w - w_crop : c_w + w_crop].detach()
    # --------------------------------------------------------
    # 2. Update domain-specific memory bank
    # --------------------------------------------------------
    if amp_bank is not None:
        for d_val in torch.unique(dom_labels):
            d_id = int(d_val.item())
            d_amps = low_freq_src[dom_labels == d_val].cpu()
            if d_id not in amp_bank:
                amp_bank[d_id] = d_amps[:64]
            else:
                amp_bank[d_id] = torch.cat([d_amps, amp_bank[d_id]], dim=0)[:64]
    # --------------------------------------------------------
    # 3. Cross-domain donor selection
    # --------------------------------------------------------
    amp_mixed = amp_src.clone()
    swapped_dom_labels = dom_labels.clone()
    for i in range(B):
        cur_d = int(dom_labels[i].item())
        # First: select a donor from another domain
        # in the current batch.
        donor_indices = torch.nonzero(dom_labels != cur_d, as_tuple=False).view(-1)
        if donor_indices.numel() > 0:
            rand_pos = torch.randint(0, donor_indices.numel(), (1,), device=images.device)
            donor_idx = int(donor_indices[rand_pos].item())
            donor_amp = low_freq_src[donor_idx]
            donor_domain = int(dom_labels[donor_idx].item())
        else:
            # Second: use the cross-batch memory bank.
            other_domains = []
            if amp_bank is not None:
                other_domains = [d for d, bank in amp_bank.items() if d != cur_d and bank.numel() > 0]
            # No other domain available:
            # Keep the original image unchanged.
            if len(other_domains) == 0:
                continue
            domain_pos = torch.randint(0, len(other_domains), (1,)).item()
            donor_domain = other_domains[domain_pos]
            bank_amps = amp_bank[donor_domain]
            bank_idx = torch.randint(0, bank_amps.size(0), (1,)).item()
            donor_amp = bank_amps[bank_idx].to(images.device)
        # Replace only the low-frequency amplitude.
        amp_mixed[i, :, c_h - h_crop : c_h + h_crop, c_w - w_crop : c_w + w_crop] = donor_amp
        swapped_dom_labels[i] = donor_domain
    # --------------------------------------------------------
    # 4. Inverse Fourier transform
    # --------------------------------------------------------
    fft_mixed = torch.polar(amp_mixed, pha_src)
    fft_mixed = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    images_fda = torch.real(torch.fft.ifft2(fft_mixed, dim=(-2, -1)))
    return (images_fda.type_as(images), swapped_dom_labels)
# ============================================================
# POPD Loss
# ============================================================

class POPDLoss(nn.Module):

    def __init__(
        self,
        device="cuda",
        num_states=2,
        feat_dim=512,
        ema_m=0.9,
        tau_sem=0.85,
        lambda_interv=0.5,
        lambda_anchor=0.5,
        lambda_orth=1.0,
    ):
        super().__init__()
        self.supcon = SupConLoss(device)
        self.tau_sem = tau_sem
        self.lambda_interv = lambda_interv
        self.lambda_anchor = lambda_anchor
        self.lambda_orth = lambda_orth
        self.ema_m = ema_m
        # EMA identity centers conditioned on semantic states.
        self.register_buffer("ema_center_X", torch.zeros(num_states, feat_dim, device=device))
        self.register_buffer("ema_inited", torch.zeros(num_states, dtype=torch.bool, device=device))
    # ========================================================
    # Balanced classification
    # ========================================================

    def _balanced_ce_loss(self, logits, targets):
        targets = targets.long()
        unique_classes = torch.unique(targets)
        if unique_classes.numel() == 0:
            return logits.sum() * 0.0
        loss_sum = 0.0
        for c in unique_classes:
            mask = targets == c
            loss_sum = loss_sum + F.cross_entropy(logits[mask], targets[mask])
        return loss_sum / unique_classes.numel()
    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        img_feat,
        img_feat_fda,
        z_comp,
        z_comp_fda,
        z_D_all,
        z_S_all,
        z_X_batch,
        state_priors,
        labels,
        dom_labels,
        swapped_dom_labels,
        sta_labels,
    ):
        # ----------------------------------------------------
        # 0. Normalize representations
        # ----------------------------------------------------
        img_f = F.normalize(img_feat.float(), dim=-1)
        img_f_fda = F.normalize(img_feat_fda.float(), dim=-1)
        z_D = F.normalize(z_D_all.float(), dim=-1)
        z_S = F.normalize(z_S_all.float(), dim=-1)
        z_X = F.normalize(z_X_batch.float(), dim=-1)
        priors = F.normalize(state_priors.float(), dim=-1)
        labels = labels.long()
        dom_labels = dom_labels.long()
        swapped_dom_labels = swapped_dom_labels.long()
        sta_labels = sta_labels.long()
        num_dom = z_D.shape[0]
        num_sta = z_S.shape[0]
        # ----------------------------------------------------
        # 1. Compositional identity alignment
        #
        # L_comp =
        #     Compositional Prompt alignment
        #     +
        #     Identity Marginal Prompt alignment
        #
        # Both belong to the original L_comp.
        # ----------------------------------------------------
        loss_comp_pair = self.supcon(img_feat, z_comp, labels, labels) + self.supcon(z_comp, img_feat, labels, labels)
        loss_identity_pair = self.supcon(img_feat, z_X_batch, labels, labels) + self.supcon(
            z_X_batch, img_feat, labels, labels
        )
        loss_comp = 0.5 * (loss_comp_pair + loss_identity_pair)
        # ----------------------------------------------------
        # 2. Cross-domain intervention
        #
        # FDA image:
        #     D -> D'
        #     X -> X
        #     S -> S
        #
        # The source identity and state are retained.
        # ----------------------------------------------------
        if dom_labels.min().item() < 0 or dom_labels.max().item() >= num_dom:
            raise ValueError("Invalid source domain labels.")
        if swapped_dom_labels.min().item() < 0 or swapped_dom_labels.max().item() >= num_dom:
            raise ValueError("Invalid intervention domain labels.")
        logits_dom_fda = torch.matmul(img_f_fda, z_D.t()) / 0.07
        loss_dom_cls_fda = self._balanced_ce_loss(logits_dom_fda, swapped_dom_labels)
        loss_interv = (
            self.supcon(img_feat_fda, z_comp_fda, labels, labels)
            + self.supcon(z_comp_fda, img_feat_fda, labels, labels)
            + loss_dom_cls_fda
        )
        # ----------------------------------------------------
        # 3. Semantic manifold anchoring
        #
        # State:
        #     CLIP semantic pseudo-labels
        #
        # Domain:
        #     Explicit dataset-domain labels
        #
        # No additional loss is introduced.
        # ----------------------------------------------------
        cos_sim_prior = (z_S * priors).sum(dim=-1)
        loss_cone = F.relu(self.tau_sem - cos_sim_prior).mean()
        if sta_labels.min().item() < 0 or sta_labels.max().item() >= num_sta:
            raise ValueError("Invalid semantic-state labels.")
        logits_sta = torch.matmul(img_f, z_S.t()) / 0.07
        loss_sta_cls = self._balanced_ce_loss(logits_sta, sta_labels)
        logits_dom = torch.matmul(img_f, z_D.t()) / 0.07
        loss_dom_cls = self._balanced_ce_loss(logits_dom, dom_labels)
        loss_anchor = loss_cone + 0.5 * (loss_sta_cls + loss_dom_cls)
        # ----------------------------------------------------
        # 4. Difference-based orthogonality
        #
        # Domain difference:
        #     Delta_D = Z_D(1) - Z_D(0)
        #
        # State difference:
        #     Delta_S = Z_S(1) - Z_S(0)
        #
        # Identity difference:
        #     Delta_X = Z_X - EMA_center(X | S)
        # ----------------------------------------------------
        if num_sta != 2:
            raise ValueError("Current POPD implementation " "expects two semantic states.")
        if num_dom != 2:
            raise ValueError("Current POPD experiment " "expects two dataset domains.")
        delta_z_D = F.normalize(z_D[1:2] - z_D[0:1], dim=-1)
        delta_z_S = F.normalize(z_S[1:2] - z_S[0:1], dim=-1)
        delta_z_X_list = []
        for s_val in torch.unique(sta_labels):
            s_idx = int(s_val.item())
            s_mask = sta_labels == s_val
            z_X_s = z_X[s_mask]
            batch_mean_s = z_X_s.detach().mean(dim=0)
            # Update EMA center.
            with torch.no_grad():
                if not self.ema_inited[s_idx]:
                    self.ema_center_X[s_idx].copy_(batch_mean_s)
                    self.ema_inited[s_idx] = True
                else:
                    self.ema_center_X[s_idx].mul_(self.ema_m).add_(batch_mean_s * (1.0 - self.ema_m))
            center_s = self.ema_center_X[s_idx].unsqueeze(0).detach()
            delta_z_X_s = F.normalize(z_X_s - center_s, dim=-1)
            delta_z_X_list.append(delta_z_X_s)
        delta_z_X = torch.cat(delta_z_X_list, dim=0)
        # Pairwise difference-direction orthogonality.
        orth_S_D = torch.abs(torch.matmul(delta_z_S, delta_z_D.t())).mean()
        orth_X_S = torch.abs(torch.matmul(delta_z_X, delta_z_S.t())).mean()
        orth_X_D = torch.abs(torch.matmul(delta_z_X, delta_z_D.t())).mean()
        loss_orth = orth_S_D + orth_X_S + orth_X_D
        # ----------------------------------------------------
        # 5. Total POPD loss
        # ----------------------------------------------------
        total_loss = (
            loss_comp
            + self.lambda_interv * loss_interv
            + self.lambda_anchor * loss_anchor
            + self.lambda_orth * loss_orth
        )
        loss_dict = {
            "L_comp": loss_comp.item(),
            "L_intv": loss_interv.item(),
            "L_anch": loss_anchor.item(),
            "L_orth": loss_orth.item(),
        }
        return total_loss, loss_dict
