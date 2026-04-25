import os
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import util.util as util


class AlignedDataset(BaseDataset):
    """Paired dataset that loads A/B samples by the same sorted index.

    Expected directory layout:
    - /path/to/dataroot/trainA, trainB
    - /path/to/dataroot/valA, valB
    - /path/to/dataroot/testA, testB

    When controlled_pairing is enabled during training, only the front
    paired_ratio slice of the sorted paired list is used, mirroring the
    controlled pairing protocol used by the FM baseline.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        if opt.phase == "test" and not os.path.exists(self.dir_A) and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.controlled_pairing = bool(getattr(opt, "controlled_pairing", False) and opt.phase == "train")
        self.paired_ratio = float(getattr(opt, "paired_ratio", 0.1))
        pair_base_size = min(self.A_size, self.B_size)
        self.paired_cutoff = max(0, min(pair_base_size, int(self.paired_ratio * pair_base_size)))
        self.transform = get_transform(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __getitem__(self, index):
        if self.controlled_pairing and self.paired_cutoff > 0:
            index = index % self.paired_cutoff
        else:
            index = index % min(self.A_size, self.B_size)

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        A = self.transform(A_img)
        B = self.transform(B_img)

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "is_paired": True,
        }

    def __len__(self):
        if self.controlled_pairing and self.paired_cutoff > 0:
            return max(self.paired_cutoff, int(getattr(self.opt, "batch_size", 1)))
        return min(self.A_size, self.B_size)
