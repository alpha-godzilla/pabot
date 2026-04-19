import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import os
import re
import util.util as util


_SLICE_RE = re.compile(r"^(?P<patient>.+?)_slice(?P<slice>\d+)$", re.IGNORECASE)


def _parse_patient_and_slice(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _SLICE_RE.match(stem)
    if match is None:
        return stem, -1
    return match.group("patient"), int(match.group("slice"))


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.controlled_pairing = bool(getattr(opt, "controlled_pairing", False) and opt.phase == "train")
        self.paired_ratio = float(getattr(opt, "paired_ratio", 0.1))
        self.pair_seed = int(getattr(opt, "pair_seed", 3407))
        pair_base_size = min(self.A_size, self.B_size)
        self.paired_cutoff = max(0, min(pair_base_size, int(self.paired_ratio * pair_base_size)))

    def _is_phi_pretrain_stage(self):
        if not bool(getattr(self.opt, "isTrain", False)):
            return False
        if not self.controlled_pairing:
            return False

        epoch_count = int(getattr(self.opt, "epoch_count", 1))
        stage_epoch = max(0, int(self.current_epoch) - epoch_count)

        phi_pretrain_epochs = int(getattr(self.opt, "phi_pretrain_epochs", 0))
        phi_pretrain_max_epochs = getattr(self.opt, "phi_pretrain_max_epochs", None)
        if phi_pretrain_max_epochs is None:
            max_phi_epochs = phi_pretrain_epochs
        else:
            max_phi_epochs = int(phi_pretrain_max_epochs)
        max_phi_epochs = max(0, max_phi_epochs)

        return stage_epoch < max_phi_epochs

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        in_phi_stage = self._is_phi_pretrain_stage()

        if in_phi_stage and self.paired_cutoff > 0:
            # During phi pretraining, sample only from the paired prefix.
            paired_index = index % self.paired_cutoff
            A_path = self.A_paths[paired_index]
            index_B = paired_index % self.B_size
            is_paired = True
        elif self.controlled_pairing:
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            # Front paired_ratio samples are strictly paired; the rest are reproducibly unpaired.
            is_paired = (index % self.A_size) < self.paired_cutoff and (index % self.A_size) < self.B_size
            if is_paired:
                index_B = index % self.B_size
            else:
                unpaired_low = min(self.paired_cutoff, self.B_size - 1)
                rng = random.Random(self.pair_seed + self.current_epoch * 1000003 + index)
                index_B = rng.randint(unpaired_low, self.B_size - 1)
        elif self.opt.serial_batches:   # make sure index is within then range
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            index_B = index % self.B_size
            is_paired = True
        else:   # randomize the index for domain B to avoid fixed pairs.
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            index_B = random.randint(0, self.B_size - 1)
            is_paired = False
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        patient_id, slice_idx = _parse_patient_and_slice(A_path)
        

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path,
            'is_paired': is_paired,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self._is_phi_pretrain_stage() and self.paired_cutoff > 0:
            # Keep at least one full batch to avoid zero-iteration epochs with drop_last=True.
            return max(self.paired_cutoff, int(getattr(self.opt, "batch_size", 1)))
        return max(self.A_size, self.B_size)
