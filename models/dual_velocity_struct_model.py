import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
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
        parser.add_argument("--struct_channels", type=int, default=64, help="structure feature channels for net_A")
        parser.add_argument("--vgen_scale", type=float, default=1.0, help="scale factor applied to latent v_g prediction")
        parser.add_argument("--log_attention_map", type=util.str2bool, nargs="?", const=True, default=True,
                            help="log structure attention map in visual outputs")
        parser.add_argument("--use_structure_attention", type=util.str2bool, nargs="?", const=True, default=True,
                            help="enable structure attention branch in non-warmup phases")
        parser.add_argument("--tag", type=str, default="dual_velocity_struct", help="experiment tag")

        parser.set_defaults(no_html=True, pool_size=0)
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
            "G_total",
        ]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if opt.log_attention_map:
            self.visual_names.append("attn_map")

        if self.isTrain:
            self.model_names = ["G", "Gen", "A", "VStruct", "D"]
        else:
            self.model_names = ["G", "Gen", "A", "VStruct"]

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
        struct_channels = max(1, int(opt.struct_channels))

        self.netGen = networks.init_net(
            LatentVelocityNet(latent_channels, hidden_channels=max(64, opt.ngf * 2)),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netA = networks.init_net(
            StructureFeatureExtractor(latent_channels, struct_channels),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netVStruct = networks.init_net(
            StructureVelocityGenerator(latent_channels + struct_channels, latent_channels, hidden_channels=max(64, opt.ngf * 2)),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
            initialize_weights=False,
        )

        self.model_gen = self.netGen
        self.net_A = self.netA
        self.net_V_struct = self.netVStruct
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
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_V_struct = torch.optim.Adam(self.netVStruct.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.extend([self.optimizer_gen, self.optimizer_A, self.optimizer_V_struct, self.optimizer_D])

        self._init_loss_tensors()

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
        self.loss_G_total = zero

    def _infer_latent_channels(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.opt.input_nc, self.opt.crop_size, self.opt.crop_size, device=self.device)
            latent = self.netG(dummy, mode="encode")
        return latent.shape[1]

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        is_paired = input.get("is_paired", None)
        if is_paired is None:
            self.is_paired = torch.zeros(self.real_A.size(0), device=self.device, dtype=torch.bool)
        elif isinstance(is_paired, bool):
            self.is_paired = torch.full((self.real_A.size(0),), is_paired, device=self.device, dtype=torch.bool)
        else:
            self.is_paired = is_paired.to(self.device).bool().view(-1)

    def _decode(self, latents, domain_value):
        domain = torch.full((latents.size(0), 1), float(domain_value), device=latents.device, dtype=latents.dtype)
        return self.netG((latents, domain), mode="decode")

    @staticmethod
    def _unwrap_module(net):
        return net.module if isinstance(net, torch.nn.DataParallel) else net

    def _build_attention_map(self, latent_A):
        with torch.no_grad():
            feat = self.netA(latent_A)
            attn = feat.abs().mean(dim=1, keepdim=True)
            attn = F.interpolate(attn, size=self.real_A.shape[2:], mode="bilinear", align_corners=False)
            attn = attn / (attn.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
            return attn * 2.0 - 1.0

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
        latents = self.netG(real, mode="encode")
        mu = latents
        if self.isTrain and use_noise and self.opt.noise_std > 0:
            latents = latents + torch.randn_like(latents) * self.opt.noise_std
        latent_A, latent_B = latents.chunk(2, dim=0)
        return latent_A, latent_B, mu

    def _predict_v_g(self, latents, t_tensor, net_gen=None):
        net_gen = self.netGen if net_gen is None else net_gen
        v_g = net_gen(latents, t_tensor)
        return v_g * float(self.opt.vgen_scale)

    def inference(self, latents_A, use_structure=True, detach_vg=False, net_gen=None, net_a=None, net_v_struct=None):
        latents = latents_A
        num_steps = max(1, int(self.opt.ode_steps))
        dt = 1.0 / num_steps

        net_gen = self.netGen if net_gen is None else net_gen
        net_a = self.netA if net_a is None else net_a
        net_v_struct = self.netVStruct if net_v_struct is None else net_v_struct

        S_0 = net_a(latents_A) if (use_structure and self.opt.use_structure_attention) else None
        v_gen_history = []
        v_struct_history = []

        for step in range(num_steps):
            t_val = float(step) / num_steps
            t_tensor = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
            v_g = self._predict_v_g(latents, t_tensor, net_gen=net_gen)
            if detach_vg:
                v_g = v_g.detach()

            if S_0 is not None:
                v_s = net_v_struct(torch.cat([latents, S_0], dim=1), t_tensor)
            else:
                v_s = torch.zeros_like(v_g)

            latents = latents + (v_g + v_s) * dt
            v_gen_history.append(v_g)
            v_struct_history.append(v_s)

        v_gen_stacked = torch.stack(v_gen_history, dim=0)
        v_struct_stacked = torch.stack(v_struct_history, dim=0)
        return latents, v_gen_stacked, v_struct_stacked

    def _integrate_latent_async_for_vis(self, latent, steps, net_gen=None, net_a=None, net_v_struct=None):
        """
        Asynchronous integration for visualization:
        use different time inputs for v_g and v_s, then decode after integration.
        """
        latents = latent
        dt = 1.0 / max(1, steps)
        net_gen = self.netGen if net_gen is None else net_gen
        net_a = self.netA if net_a is None else net_a
        net_v_struct = self.netVStruct if net_v_struct is None else net_v_struct
        S_0 = net_a(latent) if self.opt.use_structure_attention else None

        states = [latents]
        for step in range(steps):
            t_g = float(step) / steps
            # Half-step phase shift for structural velocity, making two fields asynchronous.
            t_s = (float(step) + 0.5) / steps
            if t_s >= 1.0:
                t_s -= 1.0

            t_g_tensor = torch.full((latents.shape[0],), t_g, device=latents.device, dtype=latents.dtype)
            t_s_tensor = torch.full((latents.shape[0],), t_s, device=latents.device, dtype=latents.dtype)

            v_g = self._predict_v_g(latents, t_g_tensor, net_gen=net_gen)
            if S_0 is not None:
                v_s = net_v_struct(torch.cat([latents, S_0], dim=1), t_s_tensor)
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

    def _unpaired_step(self, images_A, images_B, use_structure):
        latent_A, latent_B, mu = self._encode_latents(images_A, images_B, use_noise=True)
        latents_fake, v_g_hist, v_s_hist = self.inference(latent_A, use_structure=use_structure, detach_vg=False)

        fake_B = self._decode(latents_fake, domain_value=1.0)
        rec_A = self._decode(latent_A, domain_value=0.0)
        idt_B = self._decode(latent_B, domain_value=1.0)

        self._set_trainable(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D, loss_D_real, loss_D_fake = self._compute_discriminator_loss(fake_B, images_B)
        loss_D.backward()
        self.optimizer_D.step()

        self._set_trainable(self.netD, False)
        self.optimizer_gen.zero_grad()
        self.optimizer_V_struct.zero_grad()

        loss_gan = self.criterionGAN(self.netD(fake_B), True).mean()
        loss_rec = self.criterionIdt(rec_A, images_A).mean()
        loss_idt = self.criterionIdt(idt_B, images_B).mean()
        loss_kl = mu.square().mean() if self.opt.noise_std > 0 else torch.tensor(0.0, device=self.device)
        loss_path = self._path_penalty(v_g_hist, v_s_hist)
        loss_vs = self._vs_l2_penalty(v_s_hist)
        loss_ortho = self._orthogonality_loss(v_g_hist, v_s_hist)

        loss_g_total = (
            self.opt.lambda_GAN * loss_gan
            + self.opt.lambda_rec * loss_rec
            + self.opt.lambda_idt * loss_idt
            + self.opt.lambda_kl * loss_kl
            + self.opt.lambda_path * loss_path
            + self.opt.lambda_vs * loss_vs
            + self.opt.lambda_ortho * loss_ortho
        )

        loss_g_total.backward()
        self.optimizer_gen.step()
        if use_structure and self.opt.use_structure_attention:
            self.optimizer_V_struct.step()

        return {
            "fake_B": fake_B,
            "loss_D_real": loss_D_real.detach(),
            "loss_D_fake": loss_D_fake.detach(),
            "loss_G_GAN": loss_gan.detach(),
            "loss_G_rec": loss_rec.detach(),
            "loss_G_idt": loss_idt.detach(),
            "loss_G_kl": loss_kl.detach(),
            "loss_G_path": loss_path.detach(),
            "loss_G_vs": loss_vs.detach(),
            "loss_G_ortho": loss_ortho.detach(),
            "loss_G_total": loss_g_total.detach(),
            "loss_G_pair": torch.tensor(0.0, device=self.device),
        }

    def _paired_step(self, images_A, images_B):
        self._set_trainable(self.netG, False)
        self._set_trainable(self.netGen, False)
        self._set_trainable(self.netD, False)
        self._set_trainable(self.netA, True)
        self._set_trainable(self.netVStruct, True)

        with torch.no_grad():
            latent_A, latent_B, _ = self._encode_latents(images_A, images_B, use_noise=False)

        self.optimizer_A.zero_grad()
        self.optimizer_V_struct.zero_grad()

        latents_fake, v_g_hist, v_s_hist = self.inference(latent_A.detach(), use_structure=True, detach_vg=True)
        loss_pair = F.mse_loss(latents_fake, latent_B.detach())
        loss_vs = self._vs_l2_penalty(v_s_hist)
        loss_ortho = self._orthogonality_loss(v_g_hist, v_s_hist)
        loss_total = self.opt.lambda_pair * loss_pair + self.opt.lambda_vs * loss_vs + self.opt.lambda_ortho * loss_ortho

        loss_total.backward()
        self.optimizer_A.step()
        self.optimizer_V_struct.step()

        return {
            "loss_G_pair": loss_pair.detach(),
            "loss_G_vs": loss_vs.detach(),
            "loss_G_ortho": loss_ortho.detach(),
        }

    def forward(self):
        latent_A, latent_B, _ = self._encode_latents(self.real_A, self.real_B, use_noise=False)
        latents_fake, _, _ = self.inference(latent_A, use_structure=True, detach_vg=False)
        self.fake_B = self._decode(latents_fake, domain_value=1.0)
        self.rec_A = self._decode(latent_A, domain_value=0.0)
        self.idt_B = self._decode(latent_B, domain_value=1.0)
        if self.opt.log_attention_map:
            self.attn_map = self._build_attention_map(latent_A)

    def optimize_parameters(self):
        self._init_loss_tensors()
        epoch = int(self.get_epoch())
        is_warmup = epoch < int(self.opt.warmup_epochs)

        if is_warmup:
            self._set_trainable(self.netG, True)
            self._set_trainable(self.netGen, True)
            self._set_trainable(self.netA, False)
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
            self.loss_G_pair = result["loss_G_pair"]
            self.loss_G_total = result["loss_G_total"]
            if self.opt.log_attention_map:
                with torch.no_grad():
                    latent_A_vis, _, _ = self._encode_latents(self.real_A, self.real_B, use_noise=False)
                    self.attn_map = self._build_attention_map(latent_A_vis)
            return

        paired_mask = self.is_paired
        unpaired_mask = ~paired_mask
        fake_for_visual = None

        if unpaired_mask.any():
            self._set_trainable(self.netG, True)
            self._set_trainable(self.netGen, True)
            self._set_trainable(self.netA, False)
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
            self.loss_G_total = unpaired_result["loss_G_total"]

        if paired_mask.any() and self.opt.use_structure_attention:
            paired_result = self._paired_step(self.real_A[paired_mask], self.real_B[paired_mask])
            self.loss_G_pair = paired_result["loss_G_pair"]
            self.loss_G_vs = self.loss_G_vs + paired_result["loss_G_vs"]
            self.loss_G_ortho = self.loss_G_ortho + paired_result["loss_G_ortho"]
            self.loss_G_total = self.loss_G_total + (
                self.opt.lambda_pair * paired_result["loss_G_pair"]
                + self.opt.lambda_vs * paired_result["loss_G_vs"]
                + self.opt.lambda_ortho * paired_result["loss_G_ortho"]
            )

        if fake_for_visual is None:
            self.forward()
        else:
            self.fake_B = fake_for_visual
            self.rec_A = self.real_A.detach()
            self.idt_B = self.real_B.detach()
            if self.opt.log_attention_map:
                with torch.no_grad():
                    latent_A_vis, _, _ = self._encode_latents(self.real_A, self.real_B, use_noise=False)
                    self.attn_map = self._build_attention_map(latent_A_vis)

    @torch.no_grad()
    def translate(self, x, *_unused):
        was_training = self.netG.training
        self.netG.eval()
        self.netGen.eval()
        self.netA.eval()
        self.netVStruct.eval()

        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netA = self._unwrap_module(self.netA)
        netVStruct = self._unwrap_module(self.netVStruct)

        latent = netG(x, mode="encode")
        latents_fake, _, _ = self.inference(latent, use_structure=True, net_gen=netGen, net_a=netA, net_v_struct=netVStruct)
        domain = torch.full((latents_fake.size(0), 1), 1.0, device=latents_fake.device, dtype=latents_fake.dtype)
        out = netG((latents_fake, domain), mode="decode")

        if was_training:
            self.netG.train()
            self.netGen.train()
            self.netA.train()
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
        self.netA.eval()
        self.netVStruct.eval()
        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netA = self._unwrap_module(self.netA)
        netVStruct = self._unwrap_module(self.netVStruct)

        interps = []
        for i in range(min(x.size(0), 2)):
            latent = netG(x[i].unsqueeze(0), mode="encode")
            steps = max(1, int(self.opt.ode_steps))
            states = self._integrate_latent_async_for_vis(latent, steps, net_gen=netGen, net_a=netA, net_v_struct=netVStruct)
            picks = torch.linspace(0, len(states) - 1, steps=6).long().tolist()
            snaps = []
            for idx in picks:
                domain = torch.full((states[idx].size(0), 1), 1.0, device=states[idx].device, dtype=states[idx].dtype)
                snaps.append(netG((states[idx], domain), mode="decode"))
            interps.append(torch.cat(snaps, dim=0))

        self.netG.train()
        self.netGen.train()
        self.netA.train()
        self.netVStruct.train()
        return interps

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        self.netGen.eval()
        self.netA.eval()
        self.netVStruct.eval()
        netG = self._unwrap_module(self.netG)
        netGen = self._unwrap_module(self.netGen)
        netA = self._unwrap_module(self.netA)
        netVStruct = self._unwrap_module(self.netVStruct)

        if self.opt.direction == "BtoA":
            x_a, x_b = x_b, x_a

        x_a_recon = []
        x_b_recon = []
        x_ab = []
        for i in range(x_a.size(0)):
            h_a = netG(x_a[i].unsqueeze(0), mode="encode")
            h_b = netG(x_b[i].unsqueeze(0), mode="encode")
            d0 = torch.full((h_a.size(0), 1), 0.0, device=h_a.device, dtype=h_a.dtype)
            d1 = torch.full((h_b.size(0), 1), 1.0, device=h_b.device, dtype=h_b.dtype)
            x_a_recon.append(netG((h_a, d0), mode="decode"))
            x_b_recon.append(netG((h_b, d1), mode="decode"))
            h_ab, _, _ = self.inference(h_a, use_structure=True, net_gen=netGen, net_a=netA, net_v_struct=netVStruct)
            d1_ab = torch.full((h_ab.size(0), 1), 1.0, device=h_ab.device, dtype=h_ab.dtype)
            x_ab.append(netG((h_ab, d1_ab), mode="decode"))

        x_a_recon = torch.cat(x_a_recon)
        x_b_recon = torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)

        self.netG.train()
        self.netGen.train()
        self.netA.train()
        self.netVStruct.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon
