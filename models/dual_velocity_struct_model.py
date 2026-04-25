import itertools
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .attention_phi import AttentionPhi, AttentionCompoundLoss
from .dino_attention import DinoAttentionExtractor
from .dual_velocity_model import LatentVelocityNet
from util.image_pool import ImagePool
import util.util as util


class StructureFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class PhiFilmConditioner(nn.Module):
    """Lightweight FiLM parameter generator conditioned on original MRI."""

    def __init__(self, in_channels=1, film_channels=1, hidden_channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_gamma_beta = nn.Linear(hidden_channels, film_channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, cond_image):
        feat = self.encoder(cond_image).flatten(1)
        gamma_beta = self.to_gamma_beta(feat)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


class ViTEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_w = self.attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class LatentViTFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=256, depth=4, num_heads=8, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([ViTEncoderBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.last_attn_map = None
        self.last_attn_weights = None

    def forward(self, x):
        b, _, h, w = x.shape
        tokens_2d = self.patch_embed(x)
        hp, wp = tokens_2d.shape[2], tokens_2d.shape[3]
        tokens = tokens_2d.flatten(2).transpose(1, 2)

        last_attn = None
        for block in self.blocks:
            tokens, attn_w = block(tokens)
            last_attn = attn_w

        if last_attn is not None:
            self.last_attn_weights = last_attn.detach()
            # last_attn: [B, heads, N, N], convert to token saliency over keys.
            token_score = last_attn.mean(dim=1).mean(dim=1)
            attn_2d = token_score.view(b, 1, hp, wp)
            if attn_2d.shape[-2:] != (h, w):
                attn_2d = F.interpolate(attn_2d, size=(h, w), mode="bilinear", align_corners=False)
            attn_2d = attn_2d - attn_2d.amin(dim=(2, 3), keepdim=True)
            attn_2d = attn_2d / attn_2d.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            self.last_attn_map = attn_2d
        else:
            self.last_attn_map = None
            self.last_attn_weights = None

        tokens_2d = tokens.transpose(1, 2).reshape(b, -1, hp, wp)
        feat = self.proj(tokens_2d)
        if feat.shape[-2:] != (h, w):
            feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return feat

    def get_last_attention_map(self):
        return self.last_attn_map

    def get_last_attention_weights(self):
        return self.last_attn_weights


class StructureVelocityGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _time_embedding(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(
            torch.linspace(0, -math.log(10000.0), half, device=t.device, dtype=t.dtype)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.shape[1]))
        return emb

    def forward(self, x, t):
        h = self.in_proj(x)
        t_emb = self.time_mlp(self._time_embedding(t)).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        h = self.body(h)
        return self.out_proj(h)


class DualVelocityStructModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--lambda_GAN", type=float, default=1.0, help="weight for GAN loss")
        parser.add_argument("--lambda_rec", type=float, default=5.0, help="weight for reconstruction loss")
        parser.add_argument("--lambda_idt", type=float, default=5.0, help="weight for identity loss")
        parser.add_argument("--lambda_kl", type=float, default=0.01, help="weight for latent KL proxy")
        parser.add_argument("--lambda_path", type=float, default=0.1, help="weight for path penalty on (v_g+v_s)")
        parser.add_argument("--lambda_pair", type=float, default=1.0, help="weight for paired latent MSE")
        parser.add_argument("--lambda_vs", type=float, default=0.01, help="weight for ||v_s||^2")
        parser.add_argument("--lambda_ortho", type=float, default=0.01, help="weight for orthogonality loss")
        parser.add_argument("--noise_std", type=float, default=1.0, help="std of Gaussian noise added to latents")
        parser.add_argument("--style_dim", type=int, default=8, help="style code dimensionality for AdaIN decoder")
        parser.add_argument("--ode_steps", type=int, default=8, help="number of unfolding steps")
        parser.add_argument("--warmup_epochs", type=int, default=10, help="warmup epochs without structure guidance")
        parser.add_argument("--force_unpaired", type=util.str2bool, nargs="?", const=True, default=False,
                help="if true, disable loss_pair term while keeping other losses unchanged")
        parser.add_argument("--struct_update_mode", type=str, default="sync", choices=["sync", "alternate"],
                    help="update mode for netGen/netVStruct during unpaired structure training")
        parser.add_argument("--struct_update_interval", type=int, default=1,
                    help="steps per phase when struct_update_mode=alternate")
        parser.add_argument("--struct_channels", type=int, default=64, help="structure feature channels for net_A")
        parser.add_argument("--a_backbone", type=str, default="vit", choices=["cnn", "vit", "dit"],
                            help="backbone for structure feature extractor net_A")
        parser.add_argument("--a_vit_dim", type=int, default=256, help="embed dim for ViT-based net_A")
        parser.add_argument("--a_vit_depth", type=int, default=4, help="encoder depth for ViT-based net_A")
        parser.add_argument("--a_vit_heads", type=int, default=8, help="attention heads for ViT-based net_A")
        parser.add_argument("--a_vit_patch", type=int, default=2, help="patch size for ViT-based net_A")
        parser.add_argument("--gen_hidden_channels", type=int, default=128,
                            help="hidden channels for latent velocity predictor netGen")
        parser.add_argument("--vgen_scale", type=float, default=1.0, help="scale factor applied to latent v_g prediction")
        parser.add_argument("--struct_velocity_mode", type=str, default="learned", choices=["learned", "perturb"],
                            help="how to construct the structural velocity field")
        parser.add_argument("--perturb_eps", type=float, default=0.02,
                            help="finite perturbation radius for structural energy estimation")
        parser.add_argument("--perturb_samples", type=int, default=1,
                            help="number of random perturbation directions per ODE step")
        parser.add_argument("--struct_grad_scale", type=float, default=0.1,
                            help="scale factor applied to perturbation-induced structural velocity")
        parser.add_argument("--attn_norm_eps", type=float, default=1e-6,
                            help="epsilon used when normalizing multi-channel structural features")
        parser.add_argument("--perturb_ortho_scale", type=float, default=0.1,
                            help="extra scale applied to lambda_ortho in perturb mode")
        parser.add_argument("--log_attention_map", type=util.str2bool, nargs="?", const=True, default=True,
                            help="log structure attention map in visual outputs")
        parser.add_argument("--use_structure_attention", type=util.str2bool, nargs="?", const=True, default=True,
                            help="enable structure attention branch in non-warmup phases")
        parser.add_argument("--structure_attention_source", type=str, default="rollout", choices=["internal", "rollout"],
                            help="source used to build structure reference in paired distillation")
        parser.add_argument("--lambda_v0_match", type=float, default=1.0,
                            help="weight for matching learned structural velocity against distilled V0")
        parser.add_argument("--lambda_phi_pair", type=float, default=1.0,
                            help="weight for paired structure feature consistency loss")
        parser.add_argument("--lambda_phi_attn", type=float, default=1.0,
                            help="weight for phi(attn_mri) -> attn_ct supervision during phi pretraining")
        parser.add_argument("--enable_vg_attn_align", type=util.str2bool, nargs="?", const=True, default=False,
                    help="enable optional Scheme-A loss: directly align vg trajectory attention to phi target")
        parser.add_argument("--lambda_vg_attn_align", type=float, default=0.0,
                    help="weight for optional Scheme-A vg attention alignment loss")
        parser.add_argument("--phi_gen_mode", type=str, default="direct", choices=["direct", "fm"],
                    help="Phi generation mode: direct mapping or FM (flow matching) integration")
        parser.add_argument("--phi_fm_steps", type=int, default=4,
                    help="Euler integration steps used when phi_gen_mode=fm")
        parser.add_argument("--phi_fm_train_noise", type=float, default=0.0,
                    help="Gaussian noise std added to x_t during FM training")
        parser.add_argument("--phi_fm_velocity_loss_mode", type=str, default="mse", choices=["mse", "cos", "mse_cos"],
                    help="velocity supervision for FM Phi: MSE, cosine, or a weighted combination")
        parser.add_argument("--lambda_phi_fm_velocity_cos", type=float, default=1.0,
                    help="weight for cosine term when phi_fm_velocity_loss_mode=mse_cos")
        parser.add_argument("--lambda_phi_fm_end", type=float, default=0.0,
                help="weight for optional FM endpoint consistency loss (L_end)")
        parser.add_argument("--phi_input_domain", type=str, default="attention", choices=["attention", "feature"],
                help="input domain for Phi: attention map only or DINO feature+attention")
        parser.add_argument("--phi_feature_dim", type=int, default=768,
                help="channel dimension for DINO patch feature map when phi_input_domain=feature")
        parser.add_argument("--debug_phi_shapes", type=util.str2bool, nargs="?", const=True, default=False,
                help="print Phi/DINO tensor shapes to debug feature concatenation")
        parser.add_argument("--phi_loss_mode", type=str, default="kl_cos_l1", choices=["kl_cos_l1", "kl", "clip", "kl_clip"],
                help="Phi supervision mode; default uses KL + cosine + L1")
        parser.add_argument("--phi_aux_mse_weight", type=float, default=0.0,
                    help="optional auxiliary MSE weight on normalized Phi attention map")
        parser.add_argument("--phi_kl_temperature", type=float, default=1.0,
                    help="temperature used to smooth Phi/CT attention distributions before KL")
        parser.add_argument("--phi_clip_temperature", type=float, default=0.07,
                    help="temperature for the weighted CLIP-style contrastive loss")
        parser.add_argument("--phi_clip_distance_sigma", type=float, default=4.0,
                    help="distance decay scale for same-patient positive weights in CLIP loss")
        parser.add_argument("--phi_clip_queue_size", type=int, default=512,
                    help="number of CT attention embeddings cached for CLIP loss")
        parser.add_argument("--phi_condition_mode", type=str, default="none", choices=["none", "film"],
                help="conditioning mode for Phi branch (default: none, keep original behavior)")
        parser.add_argument("--phi_film_hidden_channels", type=int, default=16,
                help="hidden channels for lightweight FiLM conditioner when phi_condition_mode=film")
        parser.add_argument("--lambda_phi_kl", type=float, default=1.0,
                help="weight for KL term in Phi supervision")
        parser.add_argument("--lambda_phi_cos", type=float, default=1.0,
                help="weight for cosine term in Phi supervision")
        parser.add_argument("--lambda_phi_l1", type=float, default=1.0,
                help="weight for L1 term in Phi supervision")
        parser.add_argument("--lambda_phi_dice", type=float, default=0.0,
                help="weight for Soft Dice term in Phi supervision")
        parser.add_argument("--v0_stopgrad_phi", type=util.str2bool, nargs="?", const=True, default=True,
                            help="stop gradients to net_A when computing distilled V0 labels")
        parser.add_argument("--phi_hidden_channels", type=int, default=32,
                            help="hidden channels for the MRI-to-CT attention mapper phi (d_model for transformer)")
        parser.add_argument("--phi_num_layers", type=int, default=4,
                            help="number of transformer layers for AttentionPhi")
        parser.add_argument("--phi_pretrain_epochs", type=int, default=5,
                            help="legacy alias for max phi pretrain epochs before warmup")
        parser.add_argument("--phi_pretrain_max_epochs", type=int, default=None,
                    help="maximum number of epochs used to pretrain phi; if omitted, uses phi_pretrain_epochs")
        parser.add_argument("--phi_pretrain_loss_threshold", type=float, default=-1.0,
                            help="stop phi pretraining early when EMA phi loss <= threshold; negative disables")
        parser.add_argument("--phi_pretrain_ema_momentum", type=float, default=0.9,
                            help="EMA momentum used for phi loss early-stop tracking")
        parser.add_argument("--phi_grad_clip_norm", type=float, default=1.0,
                    help="max grad norm for Phi optimizer; <=0 disables clipping")
        parser.add_argument("--phi_best_metric", type=str, default="main",
                choices=["main", "l1", "avg", "ema", "kl", "cos", "mse", "clip"],
                    help="metric used to select best_phi checkpoint during phi pretraining")
        parser.add_argument("--auto_load_best_phi", type=util.str2bool, nargs="?", const=True, default=True,
                    help="when resuming, replace netPhi with checkpoint from epoch having minimum phi_pretrain_ema_loss")
        parser.add_argument("--use_real_struct_guidance", type=util.str2bool, nargs="?", const=True, default=False,
                    help="use real B DINO attention as structural guidance for paired data (upper bound test)")
        parser.add_argument("--dino_model_name", type=str, default="dino_vitb8",
                            help="torch.hub DINO model name")
        parser.add_argument("--dino_image_size", type=int, default=224,
                            help="input size used for frozen DINO attention extraction")
        parser.add_argument("--dino_cache_dir", type=str, default="",
                    help="optional directory storing per-image DINO attention cache (.pt)")
        parser.add_argument("--dino_cache_rel_root", type=str, default="",
                    help="root used to compute relative image keys for dino cache (default: dataroot)")
        parser.add_argument("--dino_cache_strict", type=util.str2bool, nargs="?", const=True, default=False,
                    help="if true, fail on cache miss instead of online DINO fallback")
        parser.add_argument("--dino_cache_save_missing", type=util.str2bool, nargs="?", const=True, default=False,
                    help="if true, save online-computed DINO attention to cache on misses")
        parser.add_argument("--dino_cache_verbose", type=util.str2bool, nargs="?", const=True, default=False,
                    help="print DINO cache hit/miss information")
        parser.add_argument("--tag", type=str, default="dual_velocity_struct", help="experiment tag")

        parser.set_defaults(no_html=True, pool_size=0, controlled_pairing=True, paired_ratio=0.1)
        opt, _ = parser.parse_known_args()
        if opt.phase != "test":
            model_id = "%s" % opt.tag
            model_id += "/" + os.path.basename(opt.dataroot.strip("/")) + "_%s" % opt.direction
            model_id += "/pair%s_path%s_vs%s_ortho%s" % (
                opt.lambda_pair,
                opt.lambda_path,
                opt.lambda_vs,
                opt.lambda_ortho,
            )
            parser.set_defaults(name=model_id)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        requested_phi_condition_mode = str(getattr(opt, "phi_condition_mode", "none")).strip().lower()
        if requested_phi_condition_mode != "none":
            print(
                f"[phi] forcing no-conditioning mode (requested phi_condition_mode={requested_phi_condition_mode})."
            )
        self.phi_condition_mode = "none"
        self.use_phi_film = False
        self.phi_gen_mode = str(getattr(opt, "phi_gen_mode", "direct")).strip().lower()
        self.phi_input_domain = str(getattr(opt, "phi_input_domain", "attention")).strip().lower()
        self.phi_feature_dim = int(getattr(opt, "phi_feature_dim", 768))
        self.phi_grad_clip_norm = float(getattr(opt, "phi_grad_clip_norm", 1.0))
        self.struct_update_mode = str(getattr(opt, "struct_update_mode", "sync")).strip().lower()
        self.struct_update_interval = max(1, int(getattr(opt, "struct_update_interval", 1)))
        self._struct_update_step = 0

        self.loss_names = [
            "G_GAN",
            "D_real",
            "D_fake",
            "G_rec",
            "G_idt",
            "G_kl",
            "G_path",
            "G_vs",
            "G_ortho",
            "G_pair",
            "G_v0_match",
            "G_phi_main",
            "G_phi_l1",
            "G_phi_dice",
            "G_phi_kl",
            "G_phi_cos",
            "G_phi_fm_mse",
            "G_phi_fm_vel",
            "G_phi_fm_end",
            "G_phi_fm_velocity_norm",
            "G_phi_fm_xt_noise",
            "G_phi_pair",
            "G_vg_attn",
            "G_phase_gen",
            "G_phase_vstruct",
            "G_phase_sync",
            "G_total",
        ]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if opt.log_attention_map:
            self.visual_names.append("attn_map")

        self.use_learned_struct = opt.struct_velocity_mode == "learned"
        self.ortho_weight = float(opt.lambda_ortho)
        if not self.use_learned_struct:
            self.ortho_weight *= float(opt.perturb_ortho_scale)

        if self.isTrain:
            self.model_names = ["G", "Gen", "Phi", "D"]
            if self.use_phi_film:
                self.model_names.insert(3, "PhiCond")
            if self.use_learned_struct:
                self.model_names.insert(3, "VStruct")
        else:
            self.model_names = ["G", "Gen", "Phi"]
            if self.use_phi_film:
                self.model_names.append("PhiCond")
            if self.use_learned_struct:
                self.model_names.append("VStruct")

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )

        latent_channels = self._infer_latent_channels()

        self.netGen = networks.init_net(
            LatentVelocityNet(latent_channels, hidden_channels=max(64, int(opt.gen_hidden_channels))),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        phi_in_channels = 1 if self.phi_input_domain == "attention" else (1 + self.phi_feature_dim)
        phi_num_layers = int(getattr(opt, "phi_num_layers", 4))
        self.netPhi = networks.init_net(
            AttentionPhi(
                in_channels=phi_in_channels,
                d_model=max(8, int(opt.phi_hidden_channels)),
                num_layers=phi_num_layers
            ),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netA = self.netPhi
        self.netPhiCond = None
        if self.use_phi_film:
            self.netPhiCond = networks.init_net(
                PhiFilmConditioner(
                    in_channels=max(1, int(opt.input_nc)),
                    film_channels=1,
                    hidden_channels=max(4, int(getattr(opt, "phi_film_hidden_channels", 16))),
                ),
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
            )
        self.dino_extractor = DinoAttentionExtractor(
            model_name=opt.dino_model_name,
            image_size=opt.dino_image_size,
        ).to(self.device)
        self.dino_extractor.eval()
        for param in self.dino_extractor.parameters():
            param.requires_grad = False
        if self.use_learned_struct:
            self.netVStruct = networks.init_net(
                StructureVelocityGenerator(latent_channels + 1, latent_channels, hidden_channels=max(64, opt.ngf * 2)),
                opt.init_type,
                opt.init_gain,
                self.gpu_ids,
                initialize_weights=False,
            )
        else:
            self.netVStruct = None

        self.model_gen = self.netGen
        self.net_A = self.netPhi
        if self.netVStruct is not None:
            self.net_V_struct = self.netVStruct

        self.dino_cache_dir = None
        if str(getattr(self.opt, "dino_cache_dir", "")).strip():
            self.dino_cache_dir = os.path.abspath(self.opt.dino_cache_dir)
        cache_rel_root_opt = str(getattr(self.opt, "dino_cache_rel_root", "")).strip()
        self.dino_cache_rel_root = os.path.abspath(cache_rel_root_opt) if cache_rel_root_opt else os.path.abspath(self.opt.dataroot)
        self.dino_cache_strict = bool(getattr(self.opt, "dino_cache_strict", False))
        self.dino_cache_save_missing = bool(getattr(self.opt, "dino_cache_save_missing", False))
        self.dino_cache_verbose = bool(getattr(self.opt, "dino_cache_verbose", False))
        if self.dino_cache_dir is not None and self.dino_cache_save_missing:
            os.makedirs(self.dino_cache_dir, exist_ok=True)
        if self.dino_cache_dir is not None and self.isTrain and (not bool(getattr(self.opt, "no_flip", True))):
            print("[dino-cache] warning: no_flip=False may cause cache mismatch with online augmented inputs.")

        self.batch_A_paths = []
        self.batch_B_paths = []

        self.d_A = torch.zeros([1], device=self.device)
        self.d_B = torch.ones([1], device=self.device)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_gen = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netGen.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
            )
            phi_params = list(self.netPhi.parameters())
            if self.netPhiCond is not None:
                phi_params += list(self.netPhiCond.parameters())
            self.optimizer_Phi = torch.optim.Adam(phi_params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_A = self.optimizer_Phi
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.extend([self.optimizer_gen, self.optimizer_Phi, self.optimizer_D])
            self.optimizer_V_struct = None
            if self.netVStruct is not None:
                self.optimizer_V_struct = torch.optim.Adam(self.netVStruct.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.insert(2, self.optimizer_V_struct)

        self._init_loss_tensors()
        self.phi_pretrain_end_stage_epoch = None
        self.phi_pretrain_ema_loss = None
        self.phi_epoch_avg_loss = None
        self.phi_epoch_main_loss = None
        self.phi_epoch_l1_loss = None
        self.phi_epoch_kl_loss = None
        self.phi_epoch_cos_loss = None
        self._phi_epoch_loss_sum = 0.0
        self._phi_epoch_loss_sq_sum = 0.0
        self._phi_epoch_loss_count = 0
        self._phi_epoch_main_loss_sum = 0.0
        self._phi_epoch_main_loss_sq_sum = 0.0
        self._phi_epoch_main_loss_count = 0
        self._phi_clip_queue = []
        self._phi_clip_queue_size = max(0, int(getattr(self.opt, "phi_clip_queue_size", 0)))
        self.is_phi_pretrain_stage = False

    def _phase_state_path(self, epoch):
        return os.path.join(self.save_dir, f"{epoch}_phase_state.pth")

    def save_networks(self, epoch):
        super().save_networks(epoch)
        phase_state = {
            "phi_pretrain_end_stage_epoch": self.phi_pretrain_end_stage_epoch,
            "phi_pretrain_ema_loss": self.phi_pretrain_ema_loss,
            "phi_epoch_avg_loss": self.phi_epoch_avg_loss,
            "phi_epoch_main_loss": self.phi_epoch_main_loss,
            "phi_epoch_l1_loss": self.phi_epoch_l1_loss,
            "phi_epoch_kl_loss": self.phi_epoch_kl_loss,
            "phi_epoch_cos_loss": self.phi_epoch_cos_loss,
            "phi_epoch_fm_vel_loss": getattr(self, "loss_G_phi_fm_vel", None),
        }
        self._atomic_torch_save(phase_state, self._phase_state_path(epoch))

    def load_networks(self, epoch):
        super().load_networks(epoch)
        phase_state_path = self._phase_state_path(epoch)
        if os.path.exists(phase_state_path):
            phase_state = torch.load(phase_state_path, map_location=self.device)
            self.phi_pretrain_end_stage_epoch = phase_state.get("phi_pretrain_end_stage_epoch", None)
            self.phi_pretrain_ema_loss = phase_state.get("phi_pretrain_ema_loss", None)
            self.phi_epoch_avg_loss = phase_state.get("phi_epoch_avg_loss", None)
            self.phi_epoch_main_loss = phase_state.get("phi_epoch_main_loss", None)
            self.phi_epoch_l1_loss = phase_state.get("phi_epoch_l1_loss", phase_state.get("phi_epoch_mse_loss", None))
            self.phi_epoch_kl_loss = phase_state.get("phi_epoch_kl_loss", None)
            self.phi_epoch_cos_loss = phase_state.get("phi_epoch_cos_loss", phase_state.get("phi_epoch_clip_loss", None))
            self.loss_G_phi_fm_vel = phase_state.get("phi_epoch_fm_vel_loss", self.loss_G_phi_fm_vel)

        if self.isTrain and bool(getattr(self.opt, "continue_train", False)) and bool(getattr(self.opt, "auto_load_best_phi", False)):
            self._maybe_load_best_phi_by_phase_loss()

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._phi_epoch_loss_sum = 0.0
        self._phi_epoch_loss_sq_sum = 0.0
        self._phi_epoch_loss_count = 0
        self._phi_epoch_main_loss_sum = 0.0
        self._phi_epoch_main_loss_sq_sum = 0.0
        self._phi_epoch_main_loss_count = 0
        self.phi_epoch_avg_loss = None
        self.phi_epoch_main_loss = None
        self.phi_epoch_l1_loss = None
        self.phi_epoch_kl_loss = None
        self.phi_epoch_cos_loss = None

    def get_phi_epoch_avg_loss(self):
        if self._phi_epoch_loss_count <= 0:
            return None
        return self._phi_epoch_loss_sum / float(self._phi_epoch_loss_count)

    def get_phi_epoch_main_loss(self):
        if self._phi_epoch_main_loss_count <= 0:
            return None
        return self._phi_epoch_main_loss_sum / float(self._phi_epoch_main_loss_count)

    def get_phi_epoch_l1_loss(self):
        if self._phi_epoch_loss_count <= 0:
            return None
        return self._phi_epoch_loss_sq_sum / float(self._phi_epoch_loss_count)

    def get_phi_epoch_kl_loss(self):
        return self.phi_epoch_kl_loss

    def get_phi_epoch_cos_loss(self):
        return self.phi_epoch_cos_loss

    # Backward-compatible aliases
    def get_phi_epoch_mse_loss(self):
        return self.get_phi_epoch_l1_loss()

    def get_phi_epoch_clip_loss(self):
        return self.get_phi_epoch_cos_loss()

    @staticmethod
    def _to_python_str_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @staticmethod
    def _to_long_tensor(value, device):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device).long().view(-1)
        if isinstance(value, (list, tuple)):
            return torch.tensor(list(value), device=device, dtype=torch.long).view(-1)
        return torch.tensor([int(value)], device=device, dtype=torch.long)

    @staticmethod
    def _normalize_attention_distribution(logits, temperature=1.0, eps=1e-8):
        scaled = logits / max(float(temperature), eps)
        scaled = scaled - scaled.amin(dim=-1, keepdim=True)
        scaled = scaled + eps
        return scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _sum_normalize_attention(prob, eps=1e-8):
        prob = prob.clamp_min(eps)
        return prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _minmax_normalize_attention(prob, eps=1e-8):
        prob = prob - prob.amin(dim=-1, keepdim=True)
        return prob / prob.amax(dim=-1, keepdim=True).clamp_min(eps)

    def _extract_attention_features(self, images):
        attn_map, cls_attn = self.dino_extractor(images, return_cls_attn=True)
        return attn_map, cls_attn

    def _phi_attention_logits(self, attn_map):
        return attn_map.flatten(1)

    def _phi_attention_prob(self, attn_map, temperature=1.0):
        return self._normalize_attention_distribution(self._phi_attention_logits(attn_map), temperature=temperature)

    def _update_phi_clip_queue(self, attn_probs, patient_ids, slice_indices):
        if self._phi_clip_queue_size <= 0:
            return
        patient_list = self._to_python_str_list(patient_ids)
        slice_tensor = self._to_long_tensor(slice_indices, device=attn_probs.device)
        if slice_tensor is None:
            return
        probs_cpu = attn_probs.detach().cpu()
        slice_cpu = slice_tensor.detach().cpu()
        for idx in range(probs_cpu.shape[0]):
            self._phi_clip_queue.append((probs_cpu[idx].clone(), patient_list[idx], int(slice_cpu[idx].item())))
        if len(self._phi_clip_queue) > self._phi_clip_queue_size:
            self._phi_clip_queue = self._phi_clip_queue[-self._phi_clip_queue_size:]

    def _build_phi_clip_candidate_bank(self, current_ct_probs, patient_ids, slice_indices):
        patient_list = self._to_python_str_list(patient_ids)
        slice_tensor = self._to_long_tensor(slice_indices, device=current_ct_probs.device)
        if slice_tensor is None:
            raise RuntimeError("slice_indices are required for CLIP-style Phi loss.")

        candidate_probs = [current_ct_probs]
        candidate_patient_ids = list(patient_list)
        candidate_slice_indices = [int(item) for item in slice_tensor.detach().cpu().tolist()]

        for probs_cpu, patient_id, slice_idx in self._phi_clip_queue:
            candidate_probs.append(probs_cpu.to(current_ct_probs.device).unsqueeze(0))
            candidate_patient_ids.append(str(patient_id))
            candidate_slice_indices.append(int(slice_idx))

        candidate_probs = torch.cat(candidate_probs, dim=0)
        candidate_patient_ids = list(candidate_patient_ids)
        candidate_slice_indices = torch.tensor(candidate_slice_indices, device=current_ct_probs.device, dtype=torch.long)
        return candidate_probs, candidate_patient_ids, candidate_slice_indices

    def _compute_phi_kl_loss(self, pred_map, target_cls, temperature=None):
        temp = float(self.opt.phi_kl_temperature if temperature is None else temperature)
        pred_logits = self._phi_attention_logits(self._normalize_struct_features(pred_map))
        target_logits = self._minmax_normalize_attention(target_cls)
        pred_prob = self._normalize_attention_distribution(pred_logits, temperature=temp)
        target_prob = self._sum_normalize_attention(target_logits)
        return F.kl_div(torch.log(pred_prob.clamp_min(1e-8)), target_prob, reduction="batchmean")

    def _compute_phi_clip_loss(self, pred_map, target_cls, patient_ids, slice_indices):
        temp = float(self.opt.phi_clip_temperature)
        sigma = max(float(self.opt.phi_clip_distance_sigma), 1e-6)
        anchor_prob = self._phi_attention_prob(self._normalize_struct_features(pred_map), temperature=1.0)
        candidate_probs, candidate_patient_ids, candidate_slice_indices = self._build_phi_clip_candidate_bank(
            self._sum_normalize_attention(self._minmax_normalize_attention(target_cls)),
            patient_ids,
            slice_indices,
        )

        anchor_patient_ids = self._to_python_str_list(patient_ids)
        anchor_slice_indices = self._to_long_tensor(slice_indices, device=pred_map.device)
        if anchor_slice_indices is None:
            raise RuntimeError("slice_indices are required for CLIP-style Phi loss.")

        anchor_emb = F.normalize(anchor_prob, dim=-1)
        candidate_emb = F.normalize(candidate_probs, dim=-1)
        logits = anchor_emb @ candidate_emb.t()
        logits = logits / max(temp, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)

        candidate_patient_tensor = candidate_patient_ids
        candidate_slice_tensor = candidate_slice_indices
        losses = []
        for row_idx, (patient_id, slice_idx) in enumerate(zip(anchor_patient_ids, anchor_slice_indices.detach().cpu().tolist())):
            positive_mask = torch.tensor(
                [cand_patient == patient_id for cand_patient in candidate_patient_tensor],
                device=pred_map.device,
                dtype=torch.float32,
            )
            if positive_mask.sum() <= 0:
                continue
            distance = (candidate_slice_tensor - int(slice_idx)).abs().float()
            positive_weights = torch.exp(-distance / sigma) * positive_mask
            positive_weights = positive_weights / positive_weights.sum().clamp_min(1e-8)
            losses.append(-(positive_weights * log_probs[row_idx]).sum())

        if not losses:
            return torch.tensor(0.0, device=pred_map.device)
        return torch.stack(losses).mean()

    def _compute_phi_supervision_losses(self, pred_map, target_map, target_cls, patient_ids, slice_indices):
        pred_norm = self._normalize_struct_features(pred_map)
        target_norm = self._normalize_struct_features(target_map)
        loss_l1 = F.l1_loss(pred_norm, target_norm)

        # Soft Dice Loss
        batch_size = pred_norm.size(0)
        p = pred_norm.view(batch_size, -1)
        t = target_norm.view(batch_size, -1)
        intersection = torch.sum(p * t, dim=1)
        cardinality = torch.sum(p**2 + t**2, dim=1)
        dice_score = (2. * intersection + 1e-5) / (cardinality + 1e-5)
        loss_dice = 1.0 - dice_score.mean()

        loss_cos = (1.0 - F.cosine_similarity(pred_norm.flatten(1), target_norm.flatten(1), dim=1, eps=1e-8)).mean()
        loss_kl = self._compute_phi_kl_loss(pred_map, target_cls)
        _ = self._compute_phi_clip_loss(pred_map, target_cls, patient_ids, slice_indices)

        lambda_kl = float(getattr(self.opt, "lambda_phi_kl", 1.0))
        lambda_cos = float(getattr(self.opt, "lambda_phi_cos", 1.0))
        lambda_l1 = float(getattr(self.opt, "lambda_phi_l1", 1.0))
        lambda_dice = float(getattr(self.opt, "lambda_phi_dice", 0.0))

        mode = str(getattr(self.opt, "phi_loss_mode", "kl_cos_l1"))
        if mode in {"kl_cos_l1", "kl_clip"}:
            main_loss = lambda_kl * loss_kl + lambda_cos * loss_cos + lambda_l1 * loss_l1 + lambda_dice * loss_dice
        elif mode == "kl":
            main_loss = lambda_kl * loss_kl + lambda_dice * loss_dice
        elif mode == "clip":
            main_loss = lambda_cos * loss_cos + lambda_dice * loss_dice
        else:
            raise ValueError(f"Unsupported phi_loss_mode: {mode}")

        total_loss = main_loss
        return total_loss, main_loss, loss_l1, loss_kl, loss_cos, loss_dice

    def _phi_build_input(self, attn_xt, cond_feature_map=None):
        if self.phi_input_domain != "feature":
            return attn_xt
        if cond_feature_map is None:
            raise RuntimeError("phi_input_domain=feature requires DINO patch features as conditioning input.")
        if cond_feature_map.shape[-2:] != attn_xt.shape[-2:]:
            cond_feature_map = F.interpolate(cond_feature_map, size=attn_xt.shape[-2:], mode="bilinear", align_corners=False)
        if cond_feature_map.shape[1] != self.phi_feature_dim:
            raise RuntimeError(
                f"DINO feature dim mismatch: expected {self.phi_feature_dim}, got {cond_feature_map.shape[1]}"
            )
        phi_input = torch.cat([cond_feature_map, attn_xt], dim=1)
        if bool(getattr(self.opt, "debug_phi_shapes", False)):
            print(
                "[phi-shape] build_input "
                f"attn={tuple(attn_xt.shape)} feat={tuple(cond_feature_map.shape)} "
                f"phi_input={tuple(phi_input.shape)} expected_in={1 + self.phi_feature_dim}"
            )
        return phi_input

    def _phi_predict_velocity(self, attn_xt, cond_images=None, cond_feature_map=None, net_phi=None, net_phi_cond=None):
        net_phi = self.netPhi if net_phi is None else net_phi
        attn_xt = self._apply_phi_condition(attn_xt, cond_images, net_phi_cond=net_phi_cond)
        phi_input = self._phi_build_input(attn_xt, cond_feature_map=cond_feature_map)
        if bool(getattr(self.opt, "debug_phi_shapes", False)):
            print(f"[phi-shape] predict_velocity input={tuple(phi_input.shape)}")
        return net_phi(phi_input)

    def _phi_generate_attention_fm(self, attn_start, cond_images=None, cond_feature_map=None, net_phi=None, net_phi_cond=None, steps=None):
        steps = max(1, int(self.opt.phi_fm_steps if steps is None else steps))
        dt = 1.0 / float(steps)
        xt = attn_start
        for _ in range(steps):
            v = self._phi_predict_velocity(
                xt,
                cond_images=cond_images,
                cond_feature_map=cond_feature_map,
                net_phi=net_phi,
                net_phi_cond=net_phi_cond,
            )
            xt = xt + dt * v
        return xt

    def _compute_phi_fm_loss(self, attn_mri_map, attn_ct_map, cond_images=None, cond_feature_map=None):
        t = torch.rand(attn_mri_map.size(0), 1, 1, 1, device=attn_mri_map.device, dtype=attn_mri_map.dtype)
        xt = (1.0 - t) * attn_mri_map + t * attn_ct_map
        noise_std = float(getattr(self.opt, "phi_fm_train_noise", 0.0))
        noise = None
        if noise_std > 0:
            noise = noise_std * torch.randn_like(xt)
            xt = xt + noise
        v_target = attn_ct_map - attn_mri_map
        v_pred = self._phi_predict_velocity(xt, cond_images=cond_images, cond_feature_map=cond_feature_map)
        if v_pred.shape != v_target.shape:
            raise RuntimeError(
                "Phi FM velocity shape mismatch: "
                f"v_pred={tuple(v_pred.shape)} vs v_target={tuple(v_target.shape)}"
            )
        fm_mse = F.mse_loss(v_pred, v_target)
        fm_cos = (1.0 - F.cosine_similarity(v_pred.flatten(1), v_target.flatten(1), dim=1, eps=1e-8)).mean()
        vel_mode = str(getattr(self.opt, "phi_fm_velocity_loss_mode", "mse"))
        if vel_mode == "mse":
            fm_vel = fm_mse
        elif vel_mode == "cos":
            fm_vel = fm_cos
        elif vel_mode == "mse_cos":
            lambda_cos = float(getattr(self.opt, "lambda_phi_fm_velocity_cos", 1.0))
            fm_vel = fm_mse + lambda_cos * fm_cos
        else:
            raise ValueError(f"Unsupported phi_fm_velocity_loss_mode: {vel_mode}")
        lambda_end = float(getattr(self.opt, "lambda_phi_fm_end", 0.0))
        if lambda_end > 0.0:
            attn_ct_pred_end = self._phi_generate_attention_fm(
                attn_mri_map,
                cond_images=cond_images,
                cond_feature_map=cond_feature_map,
            )
            if attn_ct_pred_end.shape != attn_ct_map.shape:
                raise RuntimeError(
                    "Phi FM endpoint shape mismatch: "
                    f"attn_ct_pred_end={tuple(attn_ct_pred_end.shape)} vs attn_ct_map={tuple(attn_ct_map.shape)}"
                )
            fm_end = F.mse_loss(attn_ct_pred_end, attn_ct_map)
        else:
            fm_end = torch.tensor(0.0, device=attn_mri_map.device)
        velocity_norm = v_pred.abs().mean()
        xt_noise = noise.abs().mean() if noise is not None else torch.tensor(0.0, device=attn_mri_map.device)
        return fm_vel, fm_mse, fm_cos, fm_end, velocity_norm, xt_noise

    def _maybe_load_best_phi_by_phase_loss(self):
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir

        if not os.path.isdir(load_dir):
            return

        best_epoch = None
        best_loss = None
        best_loss_name = None
        phase_pat = re.compile(r"^(\d+)_phase_state\.pth$")

        for filename in os.listdir(load_dir):
            match = phase_pat.match(filename)
            if match is None:
                continue

            phase_path = os.path.join(load_dir, filename)
            try:
                state = torch.load(phase_path, map_location="cpu")
            except Exception:
                continue

            loss = state.get("phi_epoch_main_loss", None)
            loss_name = "main_loss"
            if loss is None:
                loss = state.get("phi_epoch_l1_loss", state.get("phi_epoch_mse_loss", None))
                loss_name = "l1_loss"
            if loss is None:
                loss = state.get("phi_epoch_avg_loss", None)
                loss_name = "avg_loss"
            if loss is None:
                loss = state.get("phi_pretrain_ema_loss", None)
                loss_name = "ema_loss"
            if loss is None:
                continue
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                continue

            epoch = int(match.group(1))
            phi_path = os.path.join(load_dir, f"{epoch}_net_Phi.pth")
            if not os.path.exists(phi_path):
                continue

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_loss_name = loss_name

        if best_epoch is None:
            return

        phi_path = os.path.join(load_dir, f"{best_epoch}_net_Phi.pth")
        net_phi = self.netPhi.module if isinstance(self.netPhi, torch.nn.DataParallel) else self.netPhi
        print(f"[resume] loading best-phi checkpoint from epoch={best_epoch}, {best_loss_name}={best_loss:.6f}: {phi_path}")
        state_dict = torch.load(phi_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net_phi.load_state_dict(state_dict)

    def _init_loss_tensors(self):
        zero = torch.tensor(0.0, device=self.device)
        self.loss_G_GAN = zero
        self.loss_D_real = zero
        self.loss_D_fake = zero
        self.loss_G_rec = zero
        self.loss_G_idt = zero
        self.loss_G_kl = zero
        self.loss_G_path = zero
        self.loss_G_vs = zero
        self.loss_G_ortho = zero
        self.loss_G_pair = zero
        self.loss_G_v0_match = zero
        self.loss_G_phi_main = zero
        self.loss_G_phi_l1 = zero
        self.loss_G_phi_dice = zero
        self.loss_G_phi_kl = zero
        self.loss_G_phi_cos = zero
        self.loss_G_phi_fm_mse = zero
        self.loss_G_phi_fm_vel = zero
        self.loss_G_phi_fm_end = zero
        self.loss_G_phi_fm_velocity_norm = zero
        self.loss_G_phi_fm_xt_noise = zero
        self.loss_G_phi_pair = zero
        self.loss_G_vg_attn = zero
        self.loss_G_phase_gen = zero
        self.loss_G_phase_vstruct = zero
        self.loss_G_phase_sync = zero
        self.loss_G_total = zero

    def _set_phase_flags(self, update_gen, update_v_struct):
        one = torch.tensor(1.0, device=self.device)
        zero = torch.tensor(0.0, device=self.device)
        self.loss_G_phase_gen = one if update_gen else zero
        self.loss_G_phase_vstruct = one if update_v_struct else zero
        self.loss_G_phase_sync = one if (update_gen and update_v_struct) else zero

    def _infer_latent_channels(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.opt.input_nc, self.opt.crop_size, self.opt.crop_size, device=self.device)
            latent = self.netG(dummy, [], "encode")
        return latent.shape[1]

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        self.batch_A_paths = self._to_python_str_list(input.get("A_paths" if AtoB else "B_paths", None))
        self.batch_B_paths = self._to_python_str_list(input.get("B_paths" if AtoB else "A_paths", None))
        self.patient_ids = self._to_python_str_list(input.get("patient_id", None))
        self.slice_indices = self._to_long_tensor(input.get("slice_idx", None), device=self.device)
        is_paired = input.get("is_paired", None)
        if is_paired is None:
            self.is_paired = torch.zeros(self.real_A.size(0), device=self.device, dtype=torch.bool)
        elif isinstance(is_paired, bool):
            self.is_paired = torch.full((self.real_A.size(0),), is_paired, device=self.device, dtype=torch.bool)
        else:
            self.is_paired = is_paired.to(self.device).bool().view(-1)

    def _select_batch_metadata(self, mask):
        if self.slice_indices is None:
            return [], None
        if torch.is_tensor(mask):
            mask_cpu = mask.detach().cpu().bool().view(-1)
            indices = mask_cpu.nonzero(as_tuple=False).view(-1).tolist()
        else:
            indices = [idx for idx, flag in enumerate(mask) if bool(flag)]
        patient_ids = [self.patient_ids[idx] for idx in indices] if self.patient_ids else []
        slice_indices = self.slice_indices[indices] if len(indices) > 0 else self.slice_indices.new_zeros((0,))
        return patient_ids, slice_indices

    def _select_batch_paths(self, mask, domain="A"):
        paths = self.batch_A_paths if domain == "A" else self.batch_B_paths
        if not paths:
            return []
        if torch.is_tensor(mask):
            mask_cpu = mask.detach().cpu().bool().view(-1)
            indices = mask_cpu.nonzero(as_tuple=False).view(-1).tolist()
        else:
            indices = [idx for idx, flag in enumerate(mask) if bool(flag)]
        return [paths[idx] for idx in indices]

    def _decode(self, latents, domain_value):
        domain = torch.full((latents.size(0), 1), float(domain_value), device=latents.device, dtype=latents.dtype)
        return self.netG((latents, domain), [], "decode")

    @staticmethod
    def _unwrap_module(net):
        return net.module if isinstance(net, torch.nn.DataParallel) else net

    def _apply_phi_condition(self, attn_map, cond_image, net_phi_cond=None):
        # Conditioning on original MRI is explicitly disabled.
        return attn_map

    def _build_attention_map(self, images_A):
        with torch.no_grad():
            attn, _, feat = self._extract_attention_features(images_A)
            attn = self._apply_phi_condition(attn, images_A)
            attn = self.netPhi(self._phi_build_input(attn, cond_feature_map=feat))
            attn = F.interpolate(attn, size=self.real_A.shape[2:], mode="bilinear", align_corners=False)
            return self._normalize_struct_features(attn)

    @staticmethod
    def _set_trainable(net, flag):
        for param in net.parameters():
            param.requires_grad = flag
        if flag:
            net.train()
        else:
            net.eval()

    def _encode_latents(self, images_A, images_B, use_noise=True):
        real = torch.cat([images_A, images_B], dim=0)
        latents = self.netG(real, [], "encode")
        mu = latents
        if self.isTrain and use_noise and self.opt.noise_std > 0:
            latents = latents + torch.randn_like(latents) * self.opt.noise_std
        latent_A, latent_B = latents.chunk(2, dim=0)
        return latent_A, latent_B, mu

    def _predict_v_g(self, latents, t_tensor, net_gen=None):
        net_gen = self.netGen if net_gen is None else net_gen
        v_g = net_gen(latents, t_tensor)
        return v_g * float(self.opt.vgen_scale)

    def _normalize_struct_features(self, feat):
        eps = float(self.opt.attn_norm_eps)
        feat_min = feat.amin(dim=(1, 2, 3), keepdim=True)
        feat_max = feat.amax(dim=(1, 2, 3), keepdim=True)
        return (feat - feat_min) / (feat_max - feat_min).clamp_min(eps)

    def _dino_cache_file_path(self, image_path):
        if self.dino_cache_dir is None:
            return None
        abs_path = os.path.abspath(str(image_path))
        rel_path = os.path.relpath(abs_path, self.dino_cache_rel_root)
        if rel_path.startswith(".."):
            rel_path = os.path.basename(abs_path)
        return os.path.join(self.dino_cache_dir, rel_path + ".pt")

    def _load_cached_attention_item(self, image_path):
        cache_path = self._dino_cache_file_path(image_path)
        if cache_path is None or (not os.path.exists(cache_path)):
            return None
        data = torch.load(cache_path, map_location="cpu")
        attn_map = data.get("attn_map", None)
        cls_attn = data.get("cls_attn", None)
        if attn_map is None or cls_attn is None:
            return None
        return attn_map, cls_attn

    def _save_cached_attention_item(self, image_path, attn_map, cls_attn):
        cache_path = self._dino_cache_file_path(image_path)
        if cache_path is None:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._atomic_torch_save(
            {
                "attn_map": attn_map.detach().cpu(),
                "cls_attn": cls_attn.detach().cpu(),
                "dino_model_name": self.opt.dino_model_name,
                "dino_image_size": int(self.opt.dino_image_size),
            },
            cache_path,
        )

    def _extract_attention_map(self, images, image_paths=None):
        if image_paths:
            attn_map, _, _ = self._extract_attention_features(images, image_paths=image_paths)
            return attn_map
        return self.dino_extractor(images)

    def _extract_attention_features(self, images, image_paths=None):
        need_patch_feat = (self.phi_input_domain == "feature")

        if need_patch_feat:
            out = self.dino_extractor(images, return_cls_attn=True, return_patch_feat=True)
            if bool(getattr(self.opt, "debug_phi_shapes", False)):
                attn_map, cls_attn, patch_feat = out
                print(
                    "[phi-shape] dino feature path "
                    f"attn={tuple(attn_map.shape)} cls={tuple(cls_attn.shape)} feat={tuple(patch_feat.shape)}"
                )
            return out

        if (not image_paths) or self.dino_cache_dir is None:
            attn_map, cls_attn = self.dino_extractor(images, return_cls_attn=True)
            if bool(getattr(self.opt, "debug_phi_shapes", False)):
                print(f"[phi-shape] dino attention path attn={tuple(attn_map.shape)} cls={tuple(cls_attn.shape)} feat=None")
            return attn_map, cls_attn, None

        cache_hits = []
        missing_indices = []
        for idx, image_path in enumerate(image_paths):
            cached = self._load_cached_attention_item(image_path)
            if cached is None:
                cache_hits.append(None)
                missing_indices.append(idx)
            else:
                cache_hits.append(cached)

        if len(missing_indices) == 0:
            attn_map = torch.stack([item[0] for item in cache_hits], dim=0).to(self.device, non_blocking=True).float()
            cls_attn = torch.stack([item[1] for item in cache_hits], dim=0).to(self.device, non_blocking=True).float()
            if self.dino_cache_verbose:
                print(f"[dino-cache] hit {len(image_paths)}/{len(image_paths)}")
            if bool(getattr(self.opt, "debug_phi_shapes", False)):
                print(f"[phi-shape] dino cache hit attn={tuple(attn_map.shape)} cls={tuple(cls_attn.shape)} feat=None")
            return attn_map, cls_attn, None

        if self.dino_cache_strict:
            miss_files = [str(image_paths[idx]) for idx in missing_indices]
            raise RuntimeError(f"DINO cache miss for {len(miss_files)} files (strict mode): {miss_files[:5]}")

        attn_map_online, cls_attn_online = self.dino_extractor(images, return_cls_attn=True)
        if self.dino_cache_verbose:
            print(f"[dino-cache] hit {len(image_paths)-len(missing_indices)}/{len(image_paths)}, miss {len(missing_indices)}")
        if bool(getattr(self.opt, "debug_phi_shapes", False)):
            print(
                "[phi-shape] dino cache miss fallback "
                f"attn={tuple(attn_map_online.shape)} cls={tuple(cls_attn_online.shape)} feat=None"
            )

        if self.dino_cache_save_missing:
            for idx in missing_indices:
                self._save_cached_attention_item(image_paths[idx], attn_map_online[idx], cls_attn_online[idx])

        return attn_map_online, cls_attn_online, None

    def _predict_target_attention(self, images_A, net_phi=None, net_phi_cond=None, detach=False, image_paths=None, images_B=None):
        if bool(getattr(self.opt, "use_real_struct_guidance", False)) and images_B is not None:
            attn_ct, _ = self.dino_extractor(images_B, return_cls_attn=True)
            if detach:
                attn_ct = attn_ct.detach()
            return attn_ct

        net_phi = self.netPhi if net_phi is None else net_phi
        attn_mri, _, feat_mri = self._extract_attention_features(images_A, image_paths=image_paths)
        if self.phi_gen_mode == "fm":
            attn_ct_pred = self._phi_generate_attention_fm(
                attn_mri,
                cond_images=images_A,
                cond_feature_map=feat_mri,
                net_phi=net_phi,
                net_phi_cond=net_phi_cond,
            )
        else:
            attn_mri = self._apply_phi_condition(attn_mri, images_A, net_phi_cond=net_phi_cond)
            phi_input = self._phi_build_input(attn_mri, cond_feature_map=feat_mri)
            attn_ct_pred = net_phi(phi_input)
        if detach:
            attn_ct_pred = attn_ct_pred.detach()
        return attn_ct_pred

    def _resize_attention_for_latent(self, attn_map, latents):
        return F.interpolate(attn_map, size=latents.shape[-2:], mode="bilinear", align_corners=False)

    def _build_struct_condition(self, images_A, latents, net_phi=None, detach=True, image_paths=None, images_B=None):
        attn_target = self._predict_target_attention(images_A, net_phi=net_phi, detach=detach, image_paths=image_paths, images_B=images_B)
        if self.phi_gen_mode != "fm" or (bool(getattr(self.opt, "use_real_struct_guidance", False)) and images_B is not None):
            attn_target = self._normalize_struct_features(attn_target)
        return attn_target

    def _predict_struct_velocity(self, latents, struct_condition, t_tensor, net_v_struct=None):
        if not self.use_learned_struct:
            return self._estimate_v_s_perturb(latents, struct_condition)

        net_v_struct = self.netVStruct if net_v_struct is None else net_v_struct
        latents_up = F.interpolate(latents, size=struct_condition.shape[-2:], mode="bilinear", align_corners=False)
        v_s_up = net_v_struct(torch.cat([latents_up, struct_condition], dim=1), t_tensor)
        return F.interpolate(v_s_up, size=latents.shape[-2:], mode="bilinear", align_corners=False)

    def _estimate_v_s_perturb(self, latents, ref_struct):
        eps = float(self.opt.perturb_eps)
        num_samples = max(1, int(self.opt.perturb_samples))
        g_hat = torch.zeros_like(latents)

        for _ in range(num_samples):
            direction = torch.empty_like(latents).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            e_plus = self._latent_attention_energy(latents + eps * direction, ref_struct)
            e_minus = self._latent_attention_energy(latents - eps * direction, ref_struct)
            coeff = ((e_plus - e_minus) / (2.0 * eps)).view(-1, 1, 1, 1)
            g_hat = g_hat + coeff * direction

        g_hat = g_hat / float(num_samples)
        g_hat = g_hat / g_hat.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return -float(self.opt.struct_grad_scale) * g_hat

    def _latent_attention_energy(self, latents_t, ref_attn):
        images_t = self._decode(latents_t, domain_value=1.0)
        attn_t = self._extract_attention_map(images_t)
        attn_t = self._normalize_struct_features(attn_t)
        ref_attn = self._normalize_struct_features(ref_attn.detach())
        return (attn_t - ref_attn).square().mean(dim=(1, 2, 3))

    def _compute_v0_label(self, latents_t, ref_attn):
        latents_t = latents_t.detach().requires_grad_(True)

        original_flags = None
        if bool(self.opt.v0_stopgrad_phi):
            original_flags = [param.requires_grad for param in self.netPhi.parameters()]
            for param in self.netPhi.parameters():
                param.requires_grad = False

        energy = self._latent_attention_energy(latents_t, ref_attn).sum()
        grad = torch.autograd.grad(energy, latents_t, create_graph=False, retain_graph=False)[0]
        v0 = -float(self.opt.struct_grad_scale) * grad
        v0 = v0 / v0.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)

        if original_flags is not None:
            for param, flag in zip(self.netPhi.parameters(), original_flags):
                param.requires_grad = flag
        return v0.detach()

    def _record_phi_epoch_metrics(self, main_loss, l1_loss, kl_loss, cos_loss, dice_loss=None):
        main_val = float(main_loss.item())
        l1_val = float(l1_loss.item())
        kl_val = float(kl_loss.item())
        cos_val = float(cos_loss.item())

        self._phi_epoch_main_loss_sum += main_val
        self._phi_epoch_main_loss_sq_sum += main_val * main_val
        self._phi_epoch_main_loss_count += 1
        self.phi_epoch_main_loss = self.get_phi_epoch_main_loss()

        self._phi_epoch_loss_sum += main_val
        self._phi_epoch_loss_sq_sum += main_val * main_val
        self._phi_epoch_loss_count += 1
        self.phi_epoch_avg_loss = self.get_phi_epoch_avg_loss()

        self.phi_epoch_l1_loss = l1_val if self.phi_epoch_l1_loss is None else (0.5 * self.phi_epoch_l1_loss + 0.5 * l1_val)
        self.phi_epoch_kl_loss = kl_val if self.phi_epoch_kl_loss is None else (0.5 * self.phi_epoch_kl_loss + 0.5 * kl_val)
        self.phi_epoch_cos_loss = cos_val if self.phi_epoch_cos_loss is None else (0.5 * self.phi_epoch_cos_loss + 0.5 * cos_val)
        if dice_loss is not None:
            dice_val = float(dice_loss.item())
            self.phi_epoch_dice_loss = dice_val if getattr(self, "phi_epoch_dice_loss", None) is None else (0.5 * self.phi_epoch_dice_loss + 0.5 * dice_val)

    def _phi_pretrain_step(self, images_A, images_B, patient_ids, slice_indices, image_paths_A=None, image_paths_B=None):
        if not self.isTrain:
            return None

        # Phi-only mode: all trainable modules except Phi are frozen.
        self._set_trainable(self.netG, False)
        self._set_trainable(self.netGen, False)
        self._set_trainable(self.netPhi, True)
        if self.netPhiCond is not None:
            self._set_trainable(self.netPhiCond, True)
        if self.netVStruct is not None:
            self._set_trainable(self.netVStruct, False)
        self._set_trainable(self.netD, False)

        # Ensure no stale gradients remain on non-Phi optimizers.
        self.optimizer_gen.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)
        if self.optimizer_V_struct is not None:
            self.optimizer_V_struct.zero_grad(set_to_none=True)
        self.optimizer_Phi.zero_grad(set_to_none=True)

        with torch.no_grad():
            attn_mri_map, _, feat_mri_map = self._extract_attention_features(images_A, image_paths=image_paths_A)
            attn_ct_map, attn_ct_cls, _ = self._extract_attention_features(images_B, image_paths=image_paths_B)
        if self.phi_gen_mode == "fm":
            fm_vel, fm_mse, fm_cos, fm_end, velocity_norm, xt_noise = self._compute_phi_fm_loss(
                attn_mri_map,
                attn_ct_map,
                cond_images=images_A,
                cond_feature_map=feat_mri_map,
            )
            total_loss = fm_vel + float(getattr(self.opt, "lambda_phi_fm_end", 0.0)) * fm_end
            main_loss = total_loss
            loss_l1 = torch.zeros_like(total_loss)
            loss_kl = torch.zeros_like(total_loss)
            loss_cos = torch.zeros_like(total_loss)
            self.loss_G_phi_fm_mse = fm_mse.detach()
            self.loss_G_phi_fm_vel = fm_vel.detach()
            self.loss_G_phi_fm_end = fm_end.detach()
            self.loss_G_phi_fm_velocity_norm = velocity_norm.detach()
            self.loss_G_phi_fm_xt_noise = xt_noise.detach()
        else:
            attn_mri_map = self._apply_phi_condition(attn_mri_map, images_A)
            pred_ct = self.netPhi(self._phi_build_input(attn_mri_map, cond_feature_map=feat_mri_map))
            total_loss, main_loss, loss_l1, loss_kl, loss_cos, loss_dice = self._compute_phi_supervision_losses(
                pred_ct,
                attn_ct_map,
                attn_ct_cls,
                patient_ids,
                slice_indices,
            )
            self.loss_G_phi_fm_mse = torch.zeros_like(total_loss)
            self.loss_G_phi_fm_end = torch.zeros_like(total_loss)
            self.loss_G_phi_fm_velocity_norm = torch.zeros_like(total_loss)
            self.loss_G_phi_fm_xt_noise = torch.zeros_like(total_loss)
        (self.opt.lambda_phi_attn * total_loss).backward()
        if self.phi_grad_clip_norm > 0.0:
            phi_params = []
            for group in self.optimizer_Phi.param_groups:
                phi_params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(phi_params, max_norm=self.phi_grad_clip_norm)
        self.optimizer_Phi.step()
        self._record_phi_epoch_metrics(main_loss.detach(), loss_l1.detach(), loss_kl.detach(), loss_cos.detach(), dice_loss=loss_dice.detach())
        self._update_phi_clip_queue(self._sum_normalize_attention(attn_ct_cls), patient_ids, slice_indices)

        self.loss_G_phi_main = main_loss.detach()
        self.loss_G_phi_l1 = loss_l1.detach()
        self.loss_G_phi_dice = loss_dice.detach()
        self.loss_G_phi_kl = loss_kl.detach()
        self.loss_G_phi_cos = loss_cos.detach()
        return total_loss.detach()

    def inference(self, latents_A, source_images=None, use_structure=True, detach_vg=False, net_gen=None, net_a=None, net_v_struct=None):
        latents = latents_A
        num_steps = max(1, int(self.opt.ode_steps))
        dt = 1.0 / num_steps

        net_gen = self.netGen if net_gen is None else net_gen
        net_phi = self.netPhi if net_a is None else net_a
        net_v_struct = self.netVStruct if net_v_struct is None else net_v_struct

        S_0 = None
        if use_structure and self.opt.use_structure_attention and source_images is not None:
            S_0 = self._build_struct_condition(source_images, latents_A, net_phi=net_phi, detach=True)
        v_gen_history = []
        v_struct_history = []

        for step in range(num_steps):
            t_val = float(step) / num_steps
            t_tensor = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
            v_g = self._predict_v_g(latents, t_tensor, net_gen=net_gen)
            if detach_vg:
                v_g = v_g.detach()

            if S_0 is not None:
                v_s = self._predict_struct_velocity(latents, S_0, t_tensor, net_v_struct=net_v_struct)
            else:
                v_s = torch.zeros_like(v_g)

            latents = latents + (v_g + v_s) * dt
            v_gen_history.append(v_g)
            v_struct_history.append(v_s)

        v_gen_stacked = torch.stack(v_gen_history, dim=0)
        v_struct_stacked = torch.stack(v_struct_history, dim=0)
        return latents, v_gen_stacked, v_struct_stacked

    def _integrate_latent_async_for_vis(self, latent, source_image, steps, net_gen=None, net_a=None, net_v_struct=None):
        """
        Asynchronous integration for visualization:
        use different time inputs for v_g and v_s, then decode after integration.
        """
        latents = latent
        dt = 1.0 / max(1, steps)
        net_gen = self.netGen if net_gen is None else net_gen
        net_phi = self.netPhi if net_a is None else net_a
        net_v_struct = self.netVStruct if net_v_struct is None else net_v_struct
        S_0 = None
        if self.opt.use_structure_attention:
            S_0 = self._build_struct_condition(source_image, latent, net_phi=net_phi, detach=True)

        states = [latents]
        for step in range(steps):
            t_g = float(step) / steps
            t_g_tensor = torch.full((latents.shape[0],), t_g, device=latents.device, dtype=latents.dtype)
            v_g = self._predict_v_g(latents, t_g_tensor, net_gen=net_gen)
            if S_0 is not None:
                t_s = (float(step) + 0.5) / steps
                if t_s >= 1.0:
                    t_s -= 1.0
                t_s_tensor = torch.full((latents.shape[0],), t_s, device=latents.device, dtype=latents.dtype)
                v_s = self._predict_struct_velocity(latents, S_0, t_s_tensor, net_v_struct=net_v_struct)
            else:
                v_s = torch.zeros_like(v_g)

            latents = latents + (v_g + v_s) * dt
            states.append(latents)
        return states

    def _path_penalty(self, v_gen_hist, v_struct_hist):
        return (v_gen_hist + v_struct_hist).square().mean()

    def _vs_l2_penalty(self, v_struct_hist):
        return v_struct_hist.square().mean()

    def _orthogonality_loss(self, v_gen_hist, v_struct_hist):
        v_gen_flat = v_gen_hist.flatten(2)
        v_struct_flat = v_struct_hist.flatten(2)
        cosine = F.cosine_similarity(v_gen_flat, v_struct_flat, dim=2, eps=1e-8)
        return cosine.abs().mean()

    def _compute_discriminator_loss(self, fake_B, real_B):
        fake_pool = self.fake_B_pool.query(fake_B.detach())
        pred_fake = self.netD(fake_pool)
        pred_real = self.netD(real_B)
        loss_fake = self.criterionGAN(pred_fake, False).mean()
        loss_real = self.criterionGAN(pred_real, True).mean()
        return 0.5 * (loss_fake + loss_real), loss_real, loss_fake

    def _unpaired_update_flags(self, use_structure):
        update_gen = True
        update_v_struct = False
        if use_structure and self.opt.use_structure_attention and self.optimizer_V_struct is not None:
            if self.struct_update_mode == "alternate":
                phase = (self._struct_update_step // self.struct_update_interval) % 2
                update_gen = (phase == 0)
                update_v_struct = (phase == 1)
                self._struct_update_step += 1
            else:
                update_v_struct = True
        self._set_phase_flags(update_gen, update_v_struct)
        return update_gen, update_v_struct

    def _integrate_latent_vg_only(self, latents_A, net_gen=None):
        latents = latents_A
        num_steps = max(1, int(self.opt.ode_steps))
        dt = 1.0 / num_steps
        net_gen = self.netGen if net_gen is None else net_gen
        for step in range(num_steps):
            t_val = float(step) / num_steps
            t_tensor = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
            v_g = self._predict_v_g(latents, t_tensor, net_gen=net_gen)
            latents = latents + v_g * dt
        return latents

    def _compute_vg_attention_alignment_loss(self, images_A, latent_A):
        if not bool(getattr(self.opt, "enable_vg_attn_align", False)):
            return torch.tensor(0.0, device=self.device)
        if float(getattr(self.opt, "lambda_vg_attn_align", 0.0)) <= 0.0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            target_attn = self._build_struct_condition(images_A, latent_A, detach=True)

        latents_vg = self._integrate_latent_vg_only(latent_A, net_gen=self.netGen)
        image_vg = self._decode(latents_vg, domain_value=1.0)
        attn_vg = self._extract_attention_map(image_vg)
        return F.mse_loss(
            self._normalize_struct_features(attn_vg),
            self._normalize_struct_features(target_attn.detach()),
        )

    def _unpaired_step(self, images_A, images_B, use_structure):
        update_gen, update_v_struct = self._unpaired_update_flags(use_structure)
        latent_A, latent_B, mu = self._encode_latents(images_A, images_B, use_noise=True)
        
        # 1. Initial inference for current path
        latents_fake, v_g_hist, v_s_hist = self.inference(
            latent_A,
            source_images=images_A,
            use_structure=use_structure,
            detach_vg=False,
        )
        fake_B = self._decode(latents_fake, domain_value=1.0)

        # 2. Update Discriminator (always needed for GAN realism)
        self._set_trainable(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D, loss_D_real, loss_D_fake = self._compute_discriminator_loss(fake_B, images_B)
        loss_D.backward()
        self.optimizer_D.step()
        self._set_trainable(self.netD, False)

        # 3. Decoupled / Alternating G Step
        self.optimizer_gen.zero_grad()
        if self.optimizer_V_struct is not None:
            self.optimizer_V_struct.zero_grad()

        # Initialize tracking tensors
        results = {
            "fake_B": fake_B,
            "loss_D_real": loss_D_real.detach(),
            "loss_D_fake": loss_D_fake.detach(),
            "loss_G_GAN": torch.tensor(0.0, device=self.device),
            "loss_G_rec": torch.tensor(0.0, device=self.device),
            "loss_G_idt": torch.tensor(0.0, device=self.device),
            "loss_G_kl": torch.tensor(0.0, device=self.device),
            "loss_G_path": torch.tensor(0.0, device=self.device),
            "loss_G_vs": torch.tensor(0.0, device=self.device),
            "loss_G_ortho": torch.tensor(0.0, device=self.device),
            "loss_G_vg_attn": torch.tensor(0.0, device=self.device),
            "loss_G_v0_match": torch.tensor(0.0, device=self.device),
            "loss_G_total": torch.tensor(0.0, device=self.device),
            "loss_G_pair": torch.tensor(0.0, device=self.device),
            "loss_G_phi_pair": torch.tensor(0.0, device=self.device),
        }

        total_loss = 0
        loss_applied = False

        if update_gen:
            # ... rest of gen logic ...
            total_loss = total_loss + loss_gen_total
            loss_applied = True
            # ... rest of gen update ...

        if update_v_struct and self.optimizer_V_struct is not None:
            # ... rest of struct logic ...
            total_loss = total_loss + loss_struct
            loss_applied = True
            # ... rest of struct update ...

        if loss_applied:
            total_loss.backward()
            if update_gen:
                self.optimizer_gen.step()
            if update_v_struct and self.optimizer_V_struct is not None:
                self.optimizer_V_struct.step()
            results["loss_G_total"] = total_loss.detach()

        return results

        return results

    def _paired_step(self, images_A, images_B):
        self._set_trainable(self.netG, False)
        self._set_trainable(self.netGen, False)
        self._set_trainable(self.netD, False)
        self._set_trainable(self.netPhi, False)
        if self.netVStruct is not None:
            self._set_trainable(self.netVStruct, True)

        pair_patient_ids, pair_slice_indices = self._select_batch_metadata(self.is_paired)
        pair_paths_A = self._select_batch_paths(self.is_paired, domain="A")
        pair_paths_B = self._select_batch_paths(self.is_paired, domain="B")
        with torch.no_grad():
            latent_A, latent_B, _ = self._encode_latents(images_A, images_B, use_noise=False)
            ref_attn = self._predict_target_attention(images_A, detach=True, image_paths=pair_paths_A)
            attn_ct_map, attn_ct_cls, _ = self._extract_attention_features(images_B, image_paths=pair_paths_B)
        loss_phi_pair, main_loss, loss_l1, loss_kl, loss_cos, loss_dice = self._compute_phi_supervision_losses(
            ref_attn,
            attn_ct_map,
            attn_ct_cls,
            pair_patient_ids,
            pair_slice_indices,
        )

        if self.optimizer_V_struct is not None:
            self.optimizer_V_struct.zero_grad()

        latents_fake, v_g_hist, v_s_hist = self.inference(
            latent_A.detach(),
            source_images=images_A,
            use_structure=True,
            detach_vg=True,
        )
        loss_pair = F.mse_loss(latents_fake, latent_B.detach())
        loss_vs = self._vs_l2_penalty(v_s_hist)
        loss_ortho = self._orthogonality_loss(v_g_hist, v_s_hist)
        struct_condition = self._build_struct_condition(images_A, latent_A.detach(), detach=True, image_paths=pair_paths_A)

        batch_size = latent_A.size(0)
        t_rand = torch.rand(batch_size, device=latent_A.device, dtype=latent_A.dtype)
        xt = torch.lerp(latent_A.detach(), latent_B.detach(), t_rand.view(-1, 1, 1, 1))
        v0_label = self._compute_v0_label(xt, ref_attn)
        v_s_pred = self._predict_struct_velocity(xt, struct_condition, t_rand, net_v_struct=self.netVStruct)
        loss_v0_match = F.mse_loss(v_s_pred, v0_label)

        pair_weight = 0.0 if bool(getattr(self.opt, "force_unpaired", False)) else float(self.opt.lambda_pair)
        loss_total = (
            loss_phi_pair
            + pair_weight * loss_pair
            + self.opt.lambda_vs * loss_vs
            + self.ortho_weight * loss_ortho
            + self.opt.lambda_v0_match * loss_v0_match
        )

        loss_total.backward()
        if self.optimizer_V_struct is not None:
            self.optimizer_V_struct.step()

        self.loss_G_phi_main = main_loss.detach()
        self.loss_G_phi_l1 = loss_l1.detach()
        self.loss_G_phi_dice = loss_dice.detach()
        self.loss_G_phi_kl = loss_kl.detach()
        self.loss_G_phi_cos = loss_cos.detach()
        self._record_phi_epoch_metrics(main_loss.detach(), loss_l1.detach(), loss_kl.detach(), loss_cos.detach(), dice_loss=loss_dice.detach())
        self._update_phi_clip_queue(self._sum_normalize_attention(attn_ct_cls), pair_patient_ids, pair_slice_indices)

        logged_pair = torch.zeros_like(loss_pair) if bool(getattr(self.opt, "force_unpaired", False)) else loss_pair.detach()
        return {
            "loss_G_pair": logged_pair,
            "loss_G_vs": loss_vs.detach(),
            "loss_G_ortho": loss_ortho.detach(),
            "loss_G_v0_match": loss_v0_match.detach(),
            "loss_G_phi_pair": loss_phi_pair.detach(),
        }

    def forward(self):
        latent_A, latent_B, _ = self._encode_latents(self.real_A, self.real_B, use_noise=False)
        latents_fake, _, _ = self.inference(latent_A, source_images=self.real_A, use_structure=True, detach_vg=False)
        self.fake_B = self._decode(latents_fake, domain_value=1.0)
        self.rec_A = self._decode(latent_A, domain_value=0.0)
        self.idt_B = self._decode(latent_B, domain_value=1.0)
        if self.opt.log_attention_map:
            self.attn_map = self._build_attention_map(self.real_A)

    def optimize_parameters(self):
        self._init_loss_tensors()
        self.is_phi_pretrain_stage = False
        epoch = int(self.get_epoch())
        stage_epoch = max(0, epoch - int(getattr(self.opt, "epoch_count", 1)))

        phi_pretrain_epochs = int(getattr(self.opt, "phi_pretrain_epochs", 0))
        phi_pretrain_max_epochs = getattr(self.opt, "phi_pretrain_max_epochs", None)
        if phi_pretrain_max_epochs is None:
            max_phi_epochs = phi_pretrain_epochs
        else:
            max_phi_epochs = int(phi_pretrain_max_epochs)
        max_phi_epochs = max(0, max_phi_epochs)
        phi_loss_threshold = float(getattr(self.opt, "phi_pretrain_loss_threshold", -1.0))

        if self.phi_pretrain_end_stage_epoch is None and stage_epoch >= max_phi_epochs:
            self.phi_pretrain_end_stage_epoch = stage_epoch
            print(f"[stage] phi pretraining finished by max epoch at stage_epoch={stage_epoch}")

        if (
            self.phi_pretrain_end_stage_epoch is None
            and phi_loss_threshold >= 0.0
            and self.phi_pretrain_ema_loss is not None
            and self.phi_pretrain_ema_loss <= phi_loss_threshold
        ):
            self.phi_pretrain_end_stage_epoch = stage_epoch
            print(
                "[stage] phi pretraining finished by loss threshold "
                f"(ema={self.phi_pretrain_ema_loss:.6f} <= {phi_loss_threshold:.6f}) at stage_epoch={stage_epoch}"
            )

        is_phi_pretrain = self.phi_pretrain_end_stage_epoch is None
        self.is_phi_pretrain_stage = bool(is_phi_pretrain)
        if self.phi_pretrain_end_stage_epoch is None:
            phi_done_offset = 0
        else:
            phi_done_offset = max(0, stage_epoch - int(self.phi_pretrain_end_stage_epoch))
        is_warmup = (not is_phi_pretrain) and (phi_done_offset < int(self.opt.warmup_epochs))

        if is_phi_pretrain:
            phi_loss = None
            if self.is_paired.any():
                pair_patient_ids, pair_slice_indices = self._select_batch_metadata(self.is_paired)
                pair_paths_A = self._select_batch_paths(self.is_paired, domain="A")
                pair_paths_B = self._select_batch_paths(self.is_paired, domain="B")
                phi_loss = self._phi_pretrain_step(
                    self.real_A[self.is_paired],
                    self.real_B[self.is_paired],
                    pair_patient_ids,
                    pair_slice_indices,
                    image_paths_A=pair_paths_A,
                    image_paths_B=pair_paths_B,
                )
                if phi_loss is not None:
                    momentum = float(getattr(self.opt, "phi_pretrain_ema_momentum", 0.9))
                    momentum = min(max(momentum, 0.0), 0.9999)
                    phi_loss_val = float(phi_loss.item())
                    if self.phi_pretrain_ema_loss is None:
                        self.phi_pretrain_ema_loss = phi_loss_val
                    else:
                        self.phi_pretrain_ema_loss = momentum * self.phi_pretrain_ema_loss + (1.0 - momentum) * phi_loss_val
                    self._phi_epoch_loss_sum += phi_loss_val
                    self._phi_epoch_loss_sq_sum += phi_loss_val * phi_loss_val
                    self._phi_epoch_loss_count += 1
                    self.phi_epoch_avg_loss = self.get_phi_epoch_avg_loss()
                    self.phi_epoch_l1_loss = self.get_phi_epoch_l1_loss()
                    self.phi_epoch_main_loss = self.get_phi_epoch_main_loss()
            self.loss_G_phi_pair = phi_loss if phi_loss is not None else torch.tensor(0.0, device=self.device)
            self.loss_G_total = self.loss_G_phi_pair
            self.fake_B = self.real_B.detach()
            self.rec_A = self.real_A.detach()
            self.idt_B = self.real_B.detach()
            if self.opt.log_attention_map:
                self.attn_map = self._build_attention_map(self.real_A)
            return

        if is_warmup:
            self._set_trainable(self.netG, True)
            self._set_trainable(self.netGen, True)
            self._set_trainable(self.netPhi, False)
            if self.netPhiCond is not None:
                self._set_trainable(self.netPhiCond, False)
            if self.netVStruct is not None:
                self._set_trainable(self.netVStruct, False)
            result = self._unpaired_step(self.real_A, self.real_B, use_structure=False)
            self.fake_B = result["fake_B"].detach()
            self.loss_D_real = result["loss_D_real"]
            self.loss_D_fake = result["loss_D_fake"]
            self.loss_G_GAN = result["loss_G_GAN"]
            self.loss_G_rec = result["loss_G_rec"]
            self.loss_G_idt = result["loss_G_idt"]
            self.loss_G_kl = result["loss_G_kl"]
            self.loss_G_path = result["loss_G_path"]
            self.loss_G_vs = result["loss_G_vs"]
            self.loss_G_ortho = result["loss_G_ortho"]
            self.loss_G_vg_attn = result["loss_G_vg_attn"]
            self.loss_G_pair = result["loss_G_pair"]
            self.loss_G_v0_match = result["loss_G_v0_match"]
            self.loss_G_phi_pair = result["loss_G_phi_pair"]
            self.loss_G_total = result["loss_G_total"]
            if self.opt.log_attention_map:
                self.attn_map = self._build_attention_map(self.real_A)
            return

        paired_mask = self.is_paired
        unpaired_mask = ~paired_mask
        fake_for_visual = None

        if unpaired_mask.any():
            self._set_trainable(self.netG, True)
            self._set_trainable(self.netGen, True)
            self._set_trainable(self.netPhi, False)
            if self.netPhiCond is not None:
                self._set_trainable(self.netPhiCond, False)
            if self.netVStruct is not None:
                self._set_trainable(self.netVStruct, True)
            unpaired_result = self._unpaired_step(self.real_A[unpaired_mask], self.real_B[unpaired_mask], use_structure=True)
            fake_for_visual = unpaired_result["fake_B"].detach()
            self.loss_D_real = unpaired_result["loss_D_real"]
            self.loss_D_fake = unpaired_result["loss_D_fake"]
            self.loss_G_GAN = unpaired_result["loss_G_GAN"]
            self.loss_G_rec = unpaired_result["loss_G_rec"]
            self.loss_G_idt = unpaired_result["loss_G_idt"]
            self.loss_G_kl = unpaired_result["loss_G_kl"]
            self.loss_G_path = unpaired_result["loss_G_path"]
            self.loss_G_vs = unpaired_result["loss_G_vs"]
            self.loss_G_ortho = unpaired_result["loss_G_ortho"]
            self.loss_G_vg_attn = unpaired_result["loss_G_vg_attn"]
            self.loss_G_v0_match = unpaired_result["loss_G_v0_match"]
            self.loss_G_phi_pair = unpaired_result["loss_G_phi_pair"]
            self.loss_G_total = unpaired_result["loss_G_total"]

        if paired_mask.any() and self.opt.use_structure_attention:
            paired_result = self._paired_step(self.real_A[paired_mask], self.real_B[paired_mask])
            self.loss_G_pair = paired_result["loss_G_pair"]
            self.loss_G_vs = self.loss_G_vs + paired_result["loss_G_vs"]
            self.loss_G_ortho = self.loss_G_ortho + paired_result["loss_G_ortho"]
            self.loss_G_v0_match = self.loss_G_v0_match + paired_result["loss_G_v0_match"]
            self.loss_G_phi_pair = self.loss_G_phi_pair + paired_result["loss_G_phi_pair"]
            pair_weight = 0.0 if bool(getattr(self.opt, "force_unpaired", False)) else float(self.opt.lambda_pair)
            self.loss_G_total = self.loss_G_total + (
                pair_weight * paired_result["loss_G_pair"]
                + self.opt.lambda_vs * paired_result["loss_G_vs"]
                + self.ortho_weight * paired_result["loss_G_ortho"]
                + self.opt.lambda_v0_match * paired_result["loss_G_v0_match"]
                + self.opt.lambda_phi_pair * paired_result["loss_G_phi_pair"]
            )

        if fake_for_visual is None:
            self.forward()
        else:
            self.fake_B = fake_for_visual
            self.rec_A = self.real_A.detach()
            self.idt_B = self.real_B.detach()
            if self.opt.log_attention_map:
                self.attn_map = self._build_attention_map(self.real_A)

    @torch.no_grad()
    def translate(self, x, *_unused):
        was_training = self.netG.training
        self.netG.eval()
        self.netGen.eval()
        self.netPhi.eval()
        if self.netVStruct is not None:
            self.netVStruct.eval()

        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netPhi = self._unwrap_module(self.netPhi)
        netVStruct = self._unwrap_module(self.netVStruct) if self.netVStruct is not None else None

        latent = netG(x, [], "encode")
        latents_fake, _, _ = self.inference(
            latent,
            source_images=x,
            use_structure=True,
            net_gen=netGen,
            net_a=netPhi,
            net_v_struct=netVStruct,
        )
        domain = torch.full((latents_fake.size(0), 1), 1.0, device=latents_fake.device, dtype=latents_fake.dtype)
        out = netG((latents_fake, domain), [], "decode")

        if was_training:
            self.netG.train()
            self.netGen.train()
            self.netPhi.train()
            if self.netVStruct is not None:
                self.netVStruct.train()
        return out

    @torch.no_grad()
    def interpolation(self, x_a, x_b):
        if self.opt.direction == "AtoB":
            x = x_a
        else:
            x = x_b

        self.netG.eval()
        self.netGen.eval()
        self.netPhi.eval()
        if self.netVStruct is not None:
            self.netVStruct.eval()
        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netPhi = self._unwrap_module(self.netPhi)
        netVStruct = self._unwrap_module(self.netVStruct) if self.netVStruct is not None else None

        interps = []
        for i in range(min(x.size(0), 2)):
            latent = netG(x[i].unsqueeze(0), [], "encode")
            steps = max(1, int(self.opt.ode_steps))
            states = self._integrate_latent_async_for_vis(
                latent,
                x[i].unsqueeze(0),
                steps,
                net_gen=netGen,
                net_a=netPhi,
                net_v_struct=netVStruct,
            )
            picks = torch.linspace(0, len(states) - 1, steps=6).long().tolist()
            snaps = []
            for idx in picks:
                domain = torch.full((states[idx].size(0), 1), 1.0, device=states[idx].device, dtype=states[idx].dtype)
                snaps.append(netG((states[idx], domain), [], "decode"))
            interps.append(torch.cat(snaps, dim=0))

        self.netG.train()
        self.netGen.train()
        self.netPhi.train()
        if self.netVStruct is not None:
            self.netVStruct.train()
        return interps

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        self.netGen.eval()
        self.netPhi.eval()
        if self.netVStruct is not None:
            self.netVStruct.eval()
        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netPhi = self._unwrap_module(self.netPhi)
        netVStruct = self._unwrap_module(self.netVStruct) if self.netVStruct is not None else None

        if self.opt.direction == "BtoA":
            x_a, x_b = x_b, x_a

        x_a_recon = []
        x_b_recon = []
        x_ab = []
        for i in range(x_a.size(0)):
            h_a = netG(x_a[i].unsqueeze(0), [], "encode")
            h_b = netG(x_b[i].unsqueeze(0), [], "encode")
            d0 = torch.full((h_a.size(0), 1), 0.0, device=h_a.device, dtype=h_a.dtype)
            d1 = torch.full((h_b.size(0), 1), 1.0, device=h_b.device, dtype=h_b.dtype)
            x_a_recon.append(netG((h_a, d0), [], "decode"))
            x_b_recon.append(netG((h_b, d1), [], "decode"))
            h_ab, _, _ = self.inference(
                h_a,
                source_images=x_a[i].unsqueeze(0),
                use_structure=True,
                net_gen=netGen,
                net_a=netPhi,
                net_v_struct=netVStruct,
            )
            d1_ab = torch.full((h_ab.size(0), 1), 1.0, device=h_ab.device, dtype=h_ab.dtype)
            x_ab.append(netG((h_ab, d1_ab), [], "decode"))

        x_a_recon = torch.cat(x_a_recon)
        x_b_recon = torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)

        self.netG.train()
        self.netGen.train()
        self.netPhi.train()
        if self.netVStruct is not None:
            self.netVStruct.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon
