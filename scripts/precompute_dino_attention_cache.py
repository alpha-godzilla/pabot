import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.dino_attention import DinoAttentionExtractor


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.to_tensor(image) * 2.0 - 1.0
        return tensor, image_path


def collect_images(dataroot, splits):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = []
    for split in splits:
        split_dir = Path(dataroot) / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                paths.append(path)
    return sorted(paths)


def cache_file_path(cache_dir, rel_root, image_path):
    rel = os.path.relpath(str(image_path), rel_root)
    if rel.startswith(".."):
        rel = os.path.basename(str(image_path))
    return os.path.join(cache_dir, rel + ".pt")


def main():
    parser = argparse.ArgumentParser(description="Precompute DINO attention cache (.pt per image)")
    parser.add_argument("--dataroot", required=True, help="dataset root containing trainA/trainB/valA/valB/testA/testB")
    parser.add_argument("--cache_dir", required=True, help="output cache directory")
    parser.add_argument("--splits", default="trainA,trainB,valA,valB,testA,testB", help="comma-separated splits to process")
    parser.add_argument("--dino_model_name", default="dino_vitb8")
    parser.add_argument("--dino_image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing cache files")
    args = parser.parse_args()

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    image_paths = collect_images(args.dataroot, splits)
    if len(image_paths) == 0:
        print("No images found for splits:", splits)
        return

    os.makedirs(args.cache_dir, exist_ok=True)
    rel_root = os.path.abspath(args.dataroot)

    dataset = ImagePathDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    extractor = DinoAttentionExtractor(
        model_name=args.dino_model_name,
        image_size=args.dino_image_size,
    ).to(device)
    extractor.eval()

    total = len(image_paths)
    done = 0
    with torch.no_grad():
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            attn_map, cls_attn = extractor(images, return_cls_attn=True)

            for idx, image_path in enumerate(batch_paths):
                out_path = cache_file_path(args.cache_dir, rel_root, image_path)
                if (not args.overwrite) and os.path.exists(out_path):
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(
                    {
                        "attn_map": attn_map[idx].detach().cpu(),
                        "cls_attn": cls_attn[idx].detach().cpu(),
                        "dino_model_name": args.dino_model_name,
                        "dino_image_size": int(args.dino_image_size),
                        "source_image": str(image_path),
                    },
                    out_path,
                )
            done += len(batch_paths)
            if done % 500 == 0 or done >= total:
                print(f"Processed {done}/{total}")

    print(f"Done. Cache saved to: {args.cache_dir}")


if __name__ == "__main__":
    main()
