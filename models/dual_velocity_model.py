import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
import util.util as util


class VelocityIntegrationLayer(nn.Module):
    """Integrate stationary velocity fields with scaling-and-squaring."""

    def __init__(self, nsteps=7):
        super().__init__()
        self.nsteps = nsteps

    def _identity_grid(self, b, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.repeat(b, 1, 1, 1)

    @staticmethod
    def _to_normalized_velocity(v_struct):
        _, _, h, w = v_struct.shape
        scale_x = 2.0 / max(w - 1, 1)
        scale_y = 2.0 / max(h - 1, 1)
        v = v_struct.clone()
        v[:, 0] = v[:, 0] * scale_x
        v[:, 1] = v[:, 1] * scale_y
        return v

    def forward(self, v_struct):
        b, c, h, w = v_struct.shape
        if c != 2:
            raise ValueError(f"VelocityIntegrationLayer expects 2 channels, got {c}")

        ident = self._identity_grid(b, h, w, v_struct.device, v_struct.dtype)
        v0 = self._to_normalized_velocity(v_struct) / (2 ** self.nsteps)
        flow = ident + v0

        for _ in range(self.nsteps):
            grid = flow.permute(0, 2, 3, 1)
            flow = F.grid_sample(
                flow,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        return flow


class LatentStructUNet(nn.Module):
    """Lightweight UNet-style regressor for structural velocity field."""

    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(base_channels * 2, 2, 3, padding=1)

    def forward(self, x):
        skip = self.enc1(x)
        x = self.down(skip)
        x = self.mid(x)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.out(x)


class LatentVelocityNet(nn.Module):
    """Time-conditioned latent velocity predictor v_gen(c_t, t)."""

    def __init__(self, in_channels, hidden_channels=128, time_dim=64, num_heads=1):
        super().__init__()
        self.time_dim = time_dim
        self.num_heads = num_heads
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
        if num_heads == 1:
            self.out_proj = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        else:
            self.out_proj1 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
            self.out_proj2 = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

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

    def forward(self, latents, t):
        x = self.in_proj(latents)
        temb = self.time_mlp(self._time_embedding(t)).unsqueeze(-1).unsqueeze(-1)
        x = x + temb
        x = self.body(x)
        if self.num_heads == 1:
            return self.out_proj(x)
        else:
            return self.out_proj1(x), self.out_proj2(x)


class DualVelocityModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--lambda_GAN", type=float, default=1.0, help="weight for GAN loss")
        parser.add_argument("--lambda_rec", type=float, default=5.0, help="weight for reconstruction loss")
        parser.add_argument("--lambda_idt", type=float, default=5.0, help="weight for identity loss")
        parser.add_argument("--lambda_kl", type=float, default=0.01, help="weight for latent KL proxy")
        parser.add_argument("--lambda_struct", type=float, default=0.1, help="weight for structural smoothness")
        parser.add_argument("--lambda_kin", type=float, default=0.1, help="weight for kinetic regularization")
        parser.add_argument("--noise_std", type=float, default=1.0, help="std of Gaussian noise added to latents")
        parser.add_argument("--style_dim", type=int, default=8, help="style code dimensionality for AdaIN decoder")
        parser.add_argument("--ode_steps", type=int, default=8, help="number of unfolding steps for v_gen")
        parser.add_argument("--ss_steps", type=int, default=7, help="scaling-and-squaring steps for v_struct")
        parser.add_argument("--gen_hidden_channels", type=int, default=128,
                            help="hidden channels for latent velocity predictor netGen")
        parser.add_argument("--use_struct_flow", type=util.str2bool, nargs="?", const=True, default=True,
                            help="enable structural deformation flow before generative flow")
        parser.add_argument("--ode_solver", type=str, default="euler", choices=["euler", "heun"],
                            help="integration method for generative flow")
        parser.add_argument("--tag", type=str, default="dual_velocity", help="experiment tag")

        parser.set_defaults(no_html=True, pool_size=0)
        opt, _ = parser.parse_known_args()
        if opt.phase != "test":
            model_id = "%s" % opt.tag
            model_id += "/" + os.path.basename(opt.dataroot.strip("/")) + "_%s" % opt.direction
            model_id += "/rec%s_idt%s_noise%s_kl%s_struct%s_kin%s" % (
                opt.lambda_rec,
                opt.lambda_idt,
                opt.noise_std,
                opt.lambda_kl,
                opt.lambda_struct,
                opt.lambda_kin,
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
            "G_struct_smooth",
            "G_kin",
        ]
        self.visual_names = ["real_A", "fake_B", "real_B"]

        if self.isTrain:
            self.model_names = ["G", "Struct", "Gen", "D"]
        else:
            self.model_names = ["G", "Struct", "Gen"]

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
        self.netStruct = networks.init_net(
            LatentStructUNet(latent_channels, base_channels=max(32, opt.ngf // 2)),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )
        self.netGen = networks.init_net(
            LatentVelocityNet(latent_channels, hidden_channels=max(64, int(opt.gen_hidden_channels))),
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

        self.model_struct = self.netStruct
        self.model_gen = self.netGen
        self.integrator = VelocityIntegrationLayer(nsteps=opt.ss_steps).to(self.device)

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

            params_G = itertools.chain(
                self.netG.parameters(),
                self.netStruct.parameters(),
                self.netGen.parameters(),
            )
            self.optimizer_G = torch.optim.Adam(params_G, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def _infer_latent_channels(self):
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.opt.input_nc,
                self.opt.crop_size,
                self.opt.crop_size,
                device=self.device,
            )
            latent = self.netG(dummy, [], "encode")
        return latent.shape[1]

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _decode(self, latents, domain_value):
        domain = torch.full((latents.size(0), 1), float(domain_value), device=latents.device, dtype=latents.dtype)
        return self.netG((latents, domain), [], "decode")

    def forward(self):
        real = torch.cat([self.real_A, self.real_B], dim=0)
        latents = self.netG(real, [], "encode")

        if self.isTrain and self.opt.noise_std > 0:
            noise = torch.randn_like(latents) * self.opt.noise_std
            self.mu = latents
            latents = latents + noise

        self.latent_A, self.latent_B = latents.chunk(2, dim=0)

        self.rec_A = self._decode(self.latent_A, domain_value=0.0)
        self.idt_B = self._decode(self.latent_B, domain_value=1.0)

        latents_fake, v_struct, v_gen_stacked = self.inference(
            self.latent_A,
            apply_deformation=self.opt.use_struct_flow,
        )

        self.latents_fake = latents_fake
        self.v_struct = v_struct
        self.v_gen_stacked = v_gen_stacked
        self.fake_B = self._decode(self.latents_fake, domain_value=1.0)

    def _ode_step(self, latents, t_tensor, dt):
        v_gen = self.netGen(latents, t_tensor)
        if self.opt.ode_solver == "heun":
            latents_euler = latents + dt * v_gen
            t_next = (t_tensor + dt).clamp_max(1.0)
            v_next = self.netGen(latents_euler, t_next)
            latents_next = latents + 0.5 * dt * (v_gen + v_next)
            return latents_next, v_gen
        return latents + dt * v_gen, v_gen

    def inference(self, latents_A, apply_deformation=True, return_path=False):
        if apply_deformation:
            v_struct = self.netStruct(latents_A)
            phi_1 = self.integrator(v_struct)
            latents_warp = F.grid_sample(
                latents_A,
                phi_1.permute(0, 2, 3, 1),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        else:
            v_struct = None
            latents_warp = latents_A

        latents = latents_warp
        num_steps = max(1, int(self.opt.ode_steps))
        dt = 1.0 / num_steps

        v_gen_history = []
        path_states = [latents] if return_path else None

        for step in range(num_steps):
            t_val = float(step) / num_steps
            t_tensor = torch.full((latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype)
            latents, v_gen = self._ode_step(latents, t_tensor, dt)
            v_gen_history.append(v_gen)
            if return_path:
                path_states.append(latents)

        v_gen_stacked = torch.stack(v_gen_history, dim=0)

        if return_path:
            return latents, v_struct, v_gen_stacked, path_states
        return latents, v_struct, v_gen_stacked

    def compute_D_loss(self):
        fake = self.fake_B_pool.query(self.fake_B.detach())
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        self.loss_G_rec = self.criterionIdt(self.rec_A, self.real_A).mean()
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()

        if self.opt.noise_std > 0:
            self.loss_G_kl = torch.pow(self.mu, 2).mean()
        else:
            self.loss_G_kl = torch.tensor(0.0, device=self.device)

        if self.v_struct is None:
            self.loss_G_struct_smooth = torch.tensor(0.0, device=self.device)
        else:
            dy = torch.abs(self.v_struct[:, :, 1:, :] - self.v_struct[:, :, :-1, :])
            dx = torch.abs(self.v_struct[:, :, :, 1:] - self.v_struct[:, :, :, :-1])
            self.loss_G_struct_smooth = torch.mean(dx) + torch.mean(dy)

        dt = 1.0 / max(1, int(self.opt.ode_steps))
        per_step_energy = self.v_gen_stacked.pow(2).mean(dim=(1, 2, 3, 4))
        self.loss_G_kin = per_step_energy.sum() * dt

        self.loss_G = (
            self.opt.lambda_GAN * self.loss_G_GAN
            + self.opt.lambda_rec * self.loss_G_rec
            + self.opt.lambda_idt * self.loss_G_idt
            + self.opt.lambda_kl * self.loss_G_kl
            + self.opt.lambda_struct * self.loss_G_struct_smooth
            + self.opt.lambda_kin * self.loss_G_kin
        )
        return self.loss_G

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    @torch.no_grad()
    def single_forward(self):
        latent = self.netG(self.real_A, [], "encode")
        latents_fake, _, _ = self.inference(latent, apply_deformation=self.opt.use_struct_flow)
        _ = self._decode(latents_fake, domain_value=1.0)

    @torch.no_grad()
    def translate(self, x, *_unused):
        was_training = self.netG.training
        self.netG.eval()
        self.netStruct.eval()
        self.netGen.eval()

        latent = self.netG(x, [], "encode")
        latents_fake, _, _ = self.inference(latent, apply_deformation=self.opt.use_struct_flow)
        out = self._decode(latents_fake, domain_value=1.0)

        if was_training:
            self.netG.train()
            self.netStruct.train()
            self.netGen.train()
        return out

    @torch.no_grad()
    def interpolation(self, x_a, x_b):
        if self.opt.direction == "AtoB":
            x = x_a
        else:
            x = x_b

        self.netG.eval()
        self.netStruct.eval()
        self.netGen.eval()

        interps = []
        for i in range(min(x.size(0), 2)):
            latent = self.netG(x[i].unsqueeze(0), [], "encode")
            _, _, _, path_states = self.inference(
                latent,
                apply_deformation=self.opt.use_struct_flow,
                return_path=True,
            )

            picks = torch.linspace(0, len(path_states) - 1, steps=6).long().tolist()
            local = [self._decode(path_states[idx], domain_value=1.0) for idx in picks]
            local = torch.cat(local, dim=0)
            interps.append(local)

        self.netG.train()
        self.netStruct.train()
        self.netGen.train()
        return interps

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        self.netStruct.eval()
        self.netGen.eval()

        if self.opt.direction == "BtoA":
            x_a, x_b = x_b, x_a

        x_a_recon = []
        x_b_recon = []
        x_ab = []
        for i in range(x_a.size(0)):
            h_a = self.netG(x_a[i].unsqueeze(0), [], "encode")
            h_b = self.netG(x_b[i].unsqueeze(0), [], "encode")

            x_a_recon.append(self._decode(h_a, domain_value=0.0))
            x_b_recon.append(self._decode(h_b, domain_value=1.0))

            h_ab, _, _ = self.inference(h_a, apply_deformation=self.opt.use_struct_flow)
            x_ab.append(self._decode(h_ab, domain_value=1.0))

        x_a_recon = torch.cat(x_a_recon)
        x_b_recon = torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)

        self.netG.train()
        self.netStruct.train()
        self.netGen.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon
