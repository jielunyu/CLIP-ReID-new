import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
# ============================================================
# CLIP Text Encoder
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
# ============================================================
# CLIP-ReID Transformer
# ============================================================

class build_transformer(nn.Module):

    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        # Explicit dataset-domain number.
        self.num_domains = int(cfg.DATASETS.NUM_DOMAINS)
        if self.num_domains < 1:
            raise ValueError("DATASETS.NUM_DOMAINS must be >= 1.")
        # ----------------------------------------------------
        # Classifiers
        # ----------------------------------------------------
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        # ----------------------------------------------------
        # BNNeck
        # ----------------------------------------------------
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        # ----------------------------------------------------
        # Load CLIP
        # ----------------------------------------------------
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual
        # ----------------------------------------------------
        # Original SIE
        # Camera and view embeddings are independent of POPD.
        # ----------------------------------------------------
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        # ----------------------------------------------------
        # POPD Prompt Learner
        #
        # [D] = dataset domain
        # [X] = identity
        # [S] = semantic state
        # ----------------------------------------------------
        self.prompt_learner = PromptLearner(num_classes, self.num_domains, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        # ----------------------------------------------------
        # Frozen CLIP semantic state priors
        # ----------------------------------------------------
        walk_concepts = [
            "a photo of a pedestrian walking upright " "on foot with full legs visible.",
            "a photo of a person standing or strolling " "on the street.",
            "a photo of a walking person " "without any bicycle.",
        ]
        cycle_concepts = [
            "a photo of a person riding a bicycle " "with bent posture.",
            "a photo of a cyclist sitting on a bike " "with lower body occluded by frame.",
            "a photo of a person cycling on the road.",
        ]
        with torch.no_grad():
            walk_tok = clip.tokenize(walk_concepts).cuda()
            cycle_tok = clip.tokenize(cycle_concepts).cuda()
            walk_emb = clip_model.encode_text(walk_tok).float()
            walk_emb = F.normalize(walk_emb, dim=-1).mean(dim=0, keepdim=True)
            cycle_emb = clip_model.encode_text(cycle_tok).float()
            cycle_emb = F.normalize(cycle_emb, dim=-1).mean(dim=0, keepdim=True)
            state_priors = F.normalize(torch.cat([walk_emb, cycle_emb], dim=0), dim=-1)
        self.register_buffer("state_priors", state_priors)
    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None, return_popd=False
    ):
        # ----------------------------------------------------
        # Text branch
        # ----------------------------------------------------
        if get_text:
            if return_popd:
                dom_label = cam_label if cam_label is not None else torch.zeros_like(label)
                sta_label = view_label if view_label is not None else torch.zeros_like(label)
                p_comp, tok_comp = self.prompt_learner.forward_comp(label, dom_label, sta_label)
                z_comp = self.text_encoder(p_comp, tok_comp)
                ((p_dom, tok_dom), (p_sta, tok_sta), (p_cls, tok_cls)) = self.prompt_learner.forward_marginals(label)
                z_D_all = self.text_encoder(p_dom, tok_dom)
                z_S_all = self.text_encoder(p_sta, tok_sta)
                z_X_batch = self.text_encoder(p_cls, tok_cls)
                return (z_comp, z_D_all, z_S_all, z_X_batch, self.state_priors)
            else:
                prompts = self.text_encoder(self.prompt_learner(label), self.prompt_learner.tokenized_prompts)
                return prompts
        # ----------------------------------------------------
        # Image feature extraction for Stage1
        # ----------------------------------------------------
        if get_image:
            (image_features_last, image_features, image_features_proj) = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]
        # ----------------------------------------------------
        # Original CLIP-ReID image forward
        # ----------------------------------------------------
        if self.model_name == 'RN50':
            (image_features_last, image_features, image_features_proj) = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1
            )
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            (image_features_last, image_features, image_features_proj) = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return ([cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj)
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)
    # ========================================================
    # Checkpoint loading
    # ========================================================

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        for key in param_dict:
            key_new = key.replace('module.', '')
            if key_new in self.state_dict():
                self.state_dict()[key_new].copy_(param_dict[key])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location="cpu")
        for key in param_dict:
            if key in self.state_dict():
                self.state_dict()[key].copy_(param_dict[key])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
# ============================================================
# Model factory
# ============================================================

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model
# ============================================================
# CLIP loading
# ============================================================

from .clip import clip

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model
# ============================================================
# POPD Prompt Learner
# ============================================================

class PromptLearner(nn.Module):

    def __init__(self, num_class, num_domains, dtype, token_embedding, clip_model=None, num_states=2):
        super().__init__()
        n_dom_ctx = 2
        n_cls_ctx = 4
        n_sta_ctx = 2
        ctx_dim = token_embedding.weight.shape[1]
        self.num_class = num_class
        self.num_domains = int(num_domains)
        self.num_states = int(num_states)
        self.dtype = dtype
        if self.num_domains < 1:
            raise ValueError("num_domains must be >= 1.")
        if self.num_states != 2:
            raise ValueError("Current POPD implementation expects " "two semantic states.")
        # ----------------------------------------------------
        # 1. Learnable tri-factor prompts
        # ----------------------------------------------------
        dom_vectors = torch.empty(self.num_domains, n_dom_ctx, ctx_dim, dtype=dtype)
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        sta_vectors = torch.empty(num_states, n_sta_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(dom_vectors, std=0.02)
        nn.init.normal_(cls_vectors, std=0.02)
        nn.init.normal_(sta_vectors, std=0.02)
        self.dom_ctx = nn.Parameter(dom_vectors)
        self.cls_ctx = nn.Parameter(cls_vectors)
        self.sta_ctx = nn.Parameter(sta_vectors)
        # ----------------------------------------------------
        # 2. Compositional prompt
        #
        # A [D] [D] style photo of a [X] [X] [X] [X]
        # person in [S] [S] state.
        # ----------------------------------------------------
        comp_prompt_str = "A X X style photo of a " "X X X X person in X X state."
        tokenized_comp = clip.tokenize(comp_prompt_str).cuda()
        with torch.no_grad():
            comp_embedding = token_embedding(tokenized_comp).type(dtype)
        self.register_buffer("tokenized_comp", tokenized_comp)
        self.register_buffer("comp_prefix", comp_embedding[:, :2, :])
        self.register_buffer("comp_mid1", comp_embedding[:, 4:8, :])
        self.register_buffer("comp_mid2", comp_embedding[:, 12:14, :])
        self.register_buffer("comp_suffix", comp_embedding[:, 16:, :])
        # ----------------------------------------------------
        # 3. Domain marginal prompt
        # ----------------------------------------------------
        dom_str = "A X X style photo."
        tok_dom = clip.tokenize(dom_str).cuda()
        with torch.no_grad():
            emb_dom = token_embedding(tok_dom).type(dtype)
        self.register_buffer("tokenized_dom", tok_dom)
        self.register_buffer("dom_prefix", emb_dom[:, :2, :])
        self.register_buffer("dom_suffix", emb_dom[:, 4:, :])
        # ----------------------------------------------------
        # 4. State marginal prompt
        # ----------------------------------------------------
        sta_str = "A photo of a person in X X state."
        tok_sta = clip.tokenize(sta_str).cuda()
        with torch.no_grad():
            emb_sta = token_embedding(tok_sta).type(dtype)
        self.register_buffer("tokenized_sta", tok_sta)
        self.register_buffer("sta_prefix", emb_sta[:, :7, :])
        self.register_buffer("sta_suffix", emb_sta[:, 9:, :])
        # ----------------------------------------------------
        # 5. Identity marginal prompt
        # Same template as original CLIP-ReID.
        # ----------------------------------------------------
        cls_str = "A photo of a X X X X person."
        tok_cls = clip.tokenize(cls_str).cuda()
        with torch.no_grad():
            emb_cls = token_embedding(tok_cls).type(dtype)
        self.register_buffer("tokenized_prompts", tok_cls)
        self.register_buffer("cls_prefix", emb_cls[:, :5, :])
        self.register_buffer("cls_suffix", emb_cls[:, 9:, :])
    # ========================================================
    # Compositional prompt
    # ========================================================

    def forward_comp(self, label, dom_label, sta_label):
        b = label.shape[0]
        dom_label = dom_label.long()
        sta_label = sta_label.long()
        # Explicit range checking.
        # Do not silently map invalid IDs using modulo.
        if dom_label.min().item() < 0 or dom_label.max().item() >= self.num_domains:
            raise ValueError("Domain label out of range: " "valid range is [0, {}].".format(self.num_domains - 1))
        if sta_label.min().item() < 0 or sta_label.max().item() >= self.num_states:
            raise ValueError("State label out of range.")
        d_ctx = self.dom_ctx[dom_label]
        x_ctx = self.cls_ctx[label]
        s_ctx = self.sta_ctx[sta_label]
        prompts = torch.cat(
            [
                self.comp_prefix.expand(b, -1, -1),
                d_ctx,
                self.comp_mid1.expand(b, -1, -1),
                x_ctx,
                self.comp_mid2.expand(b, -1, -1),
                s_ctx,
                self.comp_suffix.expand(b, -1, -1),
            ],
            dim=1,
        )
        return (prompts, self.tokenized_comp.repeat(b, 1))
    # ========================================================
    # Marginal prompts
    # ========================================================

    def forward_marginals(self, label):
        b = label.shape[0]
        # Domain marginal
        p_dom = torch.cat(
            [
                self.dom_prefix.expand(self.num_domains, -1, -1),
                self.dom_ctx,
                self.dom_suffix.expand(self.num_domains, -1, -1),
            ],
            dim=1,
        )
        tok_dom = self.tokenized_dom.repeat(self.num_domains, 1)
        # State marginal
        p_sta = torch.cat(
            [
                self.sta_prefix.expand(self.num_states, -1, -1),
                self.sta_ctx,
                self.sta_suffix.expand(self.num_states, -1, -1),
            ],
            dim=1,
        )
        tok_sta = self.tokenized_sta.repeat(self.num_states, 1)
        # Identity marginal
        p_cls = torch.cat(
            [self.cls_prefix.expand(b, -1, -1), self.cls_ctx[label], self.cls_suffix.expand(b, -1, -1)], dim=1
        )
        tok_cls = self.tokenized_prompts.repeat(b, 1)
        return ((p_dom, tok_dom), (p_sta, tok_sta), (p_cls, tok_cls))
    # ========================================================
    # Original Stage2 identity prompt
    # ========================================================

    def forward(self, label):
        b = label.shape[0]
        prompts = torch.cat(
            [self.cls_prefix.expand(b, -1, -1), self.cls_ctx[label], self.cls_suffix.expand(b, -1, -1)], dim=1
        )
        return prompts
