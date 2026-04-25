import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from models.dino_attention import DinoAttentionExtractor


class Pix2PixAttentionModel(BaseModel):
    """pix2pix-style paired translation from DINO patch features to attention maps.

    This model follows the pix2pix recipe:
    - generator
    - 70x70 PatchGAN discriminator
    - cGAN loss + L1 reconstruction loss
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            model="pix2pix_attention",
            dataset_mode="aligned",
            input_nc=768,
            output_nc=1,
            ngf=16,
            ndf=16,
            netG="resnet_6blocks",
            netD="basic",
            normG="instance",
            normD="instance",
            no_dropout=False,
            gan_mode="vanilla",
            lambda_L1=100.0,
            batch_size=1,
            load_size=224,
            crop_size=224,
            preprocess="resize_and_crop",
            controlled_pairing=True,
            paired_ratio=0.1,
            serial_batches=True,
            dino_image_size=224,
            pix2pix_attention_size=28,
        )
        parser.add_argument("--lambda_GAN", type=float, default=1.0, help="weight for conditional GAN loss")
        parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 reconstruction loss")
        parser.add_argument("--dino_image_size", type=int, default=224,
                            help="input image size used by the frozen DINO attention extractor")
        parser.add_argument("--pix2pix_attention_size", type=int, default=28,
                            help="spatial size of the attention maps used by the generator/discriminator")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        self.visual_names = ["real_A_attn", "fake_B", "real_B"]
        self.model_names = ["G", "D"]

        self.attention_size = int(getattr(opt, "pix2pix_attention_size", opt.crop_size))
        self.dino_extractor = DinoAttentionExtractor(
            model_name="dino_vitb8",
            image_size=int(getattr(opt, "dino_image_size", opt.crop_size)),
        ).to(self.device).eval()
        for param in self.dino_extractor.parameters():
            param.requires_grad = False

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            norm=opt.normG,
            use_dropout=not opt.no_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            no_antialias=opt.no_antialias,
            no_antialias_up=opt.no_antialias_up,
            gpu_ids=self.gpu_ids,
            opt=opt,
        )
        self.netD = networks.define_D(
            opt.input_nc + opt.output_nc,
            opt.ndf,
            opt.netD,
            n_layers_D=opt.n_layers_D,
            norm=opt.normD,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            no_antialias=opt.no_antialias,
            gpu_ids=self.gpu_ids,
            opt=opt,
        )

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])

    def _extract_attention(self, images):
        with torch.no_grad():
            attn = self.dino_extractor(images)
        if attn.shape[-2:] != (self.attention_size, self.attention_size):
            attn = F.interpolate(attn, size=(self.attention_size, self.attention_size), mode="bilinear", align_corners=False)
        return attn

    def _extract_feature(self, images):
        with torch.no_grad():
            _, feat = self.dino_extractor(images, return_patch_feat=True)
        if feat.shape[-2:] != (self.attention_size, self.attention_size):
            feat = F.interpolate(feat, size=(self.attention_size, self.attention_size), mode="bilinear", align_corners=False)
        return feat

    @staticmethod
    def _normalize_feature_map(feat, eps=1e-6):
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        return (feat - mean) / std

    @staticmethod
    def _to_pix2pix_range(attn):
        # DINO attention is already in [0, 1]; pix2pix expects normalized tensors.
        return attn.mul(2.0).sub(1.0)

    @staticmethod
    def _from_pix2pix_range(attn):
        return attn.add(1.0).div(2.0).clamp_(0.0, 1.0)

    def _forward_attention_pair(self, src_img, tgt_img):
        real_A_attn = self._to_pix2pix_range(self._extract_attention(src_img))
        real_A_feat = self._normalize_feature_map(self._extract_feature(src_img))
        real_B_attn = self._to_pix2pix_range(self._extract_attention(tgt_img))
        fake_B = self.netG(real_A_feat)
        return real_A_attn, real_A_feat, fake_B, real_B_attn

    def set_input(self, input):
        self.real_A_img = input["A"].to(self.device)
        self.real_B_img = input["B"].to(self.device)
        self.image_paths = input["A_paths"]
        self.image_paths_B = input["B_paths"]
        is_paired = input.get("is_paired", None)
        if is_paired is not None:
            if isinstance(is_paired, bool):
                is_paired = torch.tensor([is_paired], device=self.device)
            else:
                is_paired = is_paired.to(self.device).bool().view(-1)
            self.is_paired = is_paired
        else:
            self.is_paired = torch.ones(self.real_A_img.size(0), device=self.device, dtype=torch.bool)

        if self.isTrain and not bool(self.is_paired.all()):
            raise RuntimeError(
                "Pix2PixAttentionModel requires fully paired samples. "
                "Use --dataset_mode aligned for a strict paired dataset."
            )

    def forward(self):
        self.real_A_attn, self.real_A_feat, self.fake_B, self.real_B = self._forward_attention_pair(self.real_A_img, self.real_B_img)

    def backward_D(self):
        fake_AB = torch.cat([self.real_A_feat, self.fake_B.detach()], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat([self.real_A_feat, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat([self.real_A_feat, self.fake_B], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = float(getattr(self.opt, "lambda_GAN", 1.0)) * self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = float(getattr(self.opt, "lambda_L1", 100.0)) * self.criterionL1(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizer_G.step()

    def compute_visuals(self):
        # Keep the pix2pix-attention maps in the visual slots.
        with torch.no_grad():
            self.real_A_attn, self.real_A_feat, self.fake_B, self.real_B = self._forward_attention_pair(self.real_A_img, self.real_B_img)

    @torch.no_grad()
    def translate(self, src_img):
        real_A_feat = self._normalize_feature_map(self._extract_feature(src_img.to(self.device)))
        return self.netG(real_A_feat)

    @torch.no_grad()
    def extract_attention(self, images):
        return self._to_pix2pix_range(self._extract_attention(images.to(self.device)))

    @torch.no_grad()
    def sample(self, images_A, images_B):
        real_A_attn, _, fake_B, real_B = self._forward_attention_pair(images_A.to(self.device), images_B.to(self.device))
        return [real_A_attn, fake_B, real_B]

    @torch.no_grad()
    def interpolation(self, images_A, images_B):
        return self.sample(images_A, images_B)
