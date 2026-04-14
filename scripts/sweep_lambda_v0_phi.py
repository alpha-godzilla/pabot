#!/usr/bin/env python3
import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TextIO


@dataclass
class Job:
    name: str
    gpu_id: str
    command: List[str]
    log_path: Path
    process: Optional[subprocess.Popen] = None
    log_handle: Optional[TextIO] = None


def parse_float_list(raw_values: List[str]) -> List[float]:
    if len(raw_values) == 1 and "," in raw_values[0]:
        raw_values = [part.strip() for part in raw_values[0].split(",") if part.strip()]
    return [float(value) for value in raw_values]


def build_common_train_args(args: argparse.Namespace) -> List[str]:
    common = [
        "--dataroot",
        args.dataroot,
        "--dataset_mode",
        args.dataset_mode,
        "--direction",
        args.direction,
        "--model",
        args.model,
        "--name",
        args.base_name,
        "--checkpoints_dir",
        args.checkpoints_dir,
        "--gpu_ids",
        "{gpu_id}",
        "--batch_size",
        str(args.batch_size),
        "--n_epochs",
        str(args.n_epochs),
        "--n_epochs_decay",
        str(args.n_epochs_decay),
        "--epoch_count",
        str(args.epoch_count),
        "--eval_epoch_freq",
        str(args.eval_epoch_freq),
        "--save_epoch_freq",
        str(args.save_epoch_freq),
        "--controlled_pairing",
        str(args.controlled_pairing).lower(),
        "--paired_ratio",
        str(args.paired_ratio),
        "--pair_seed",
        str(args.pair_seed),
        "--a_backbone",
        args.a_backbone,
        "--a_vit_depth",
        str(args.a_vit_depth),
        "--a_vit_dim",
        str(args.a_vit_dim),
        "--a_vit_heads",
        str(args.a_vit_heads),
        "--a_vit_patch",
        str(args.a_vit_patch),
        "--dino_model_name",
        args.dino_model_name,
        "--dino_image_size",
        str(args.dino_image_size),
        "--input_nc",
        str(args.input_nc),
        "--output_nc",
        str(args.output_nc),
        "--ngf",
        str(args.ngf),
        "--ndf",
        str(args.ndf),
        "--normG",
        args.normG,
        "--normD",
        args.normD,
        "--init_type",
        args.init_type,
        "--init_gain",
        str(args.init_gain),
        "--load_size",
        str(args.load_size),
        "--crop_size",
        str(args.crop_size),
        "--preprocess",
        args.preprocess,
        "--random_scale_max",
        str(args.random_scale_max),
        "--max_dataset_size",
        args.max_dataset_size,
        "--style_dim",
        str(args.style_dim),
        "--stylegan2_G_num_downsampling",
        str(args.stylegan2_G_num_downsampling),
        "--ode_steps",
        str(args.ode_steps),
        "--struct_channels",
        str(args.struct_channels),
        "--phi_hidden_channels",
        str(args.phi_hidden_channels),
        "--struct_velocity_mode",
        args.struct_velocity_mode,
        "--use_structure_attention",
        str(args.use_structure_attention).lower(),
        "--structure_attention_source",
        args.structure_attention_source,
        "--noise_std",
        str(args.noise_std),
        "--struct_grad_scale",
        str(args.struct_grad_scale),
        "--log_attention_map",
        str(args.log_attention_map).lower(),
        "--lambda_pair",
        str(args.lambda_pair),
        "--lambda_path",
        str(args.lambda_path),
        "--lambda_vs",
        str(args.lambda_vs),
        "--lambda_ortho",
        str(args.lambda_ortho),
        "--lambda_GAN",
        str(args.lambda_GAN),
        "--lambda_idt",
        str(args.lambda_idt),
        "--lambda_kl",
        str(args.lambda_kl),
        "--lambda_rec",
        str(args.lambda_rec),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--lambda_v0_match",
        "{lambda_v0_match}",
        "--lambda_phi_pair",
        "{lambda_phi_pair}",
        "--lambda_phi_attn",
        str(args.lambda_phi_attn),
        "--v0_stopgrad_phi",
        str(args.v0_stopgrad_phi).lower(),
        "--tag",
        args.tag,
        "--no_html",
    ]

    if args.use_wandb:
        common.extend(
            [
                "--use_wandb",
                "--wandb_project",
                args.wandb_project,
                "--wandb_mode",
                args.wandb_mode,
            ]
        )
        if args.wandb_entity:
            common.extend(["--wandb_entity", args.wandb_entity])

    if args.continue_train:
        common.append("--continue_train")

    if args.no_flip:
        common.append("--no_flip")
    if args.no_dropout:
        common.append("--no_dropout")
    if args.use_cam_weight:
        common.append("--use_cam_weight")

    return common


def make_job_name(base_name: str, lambda_v0_match: float, lambda_phi_pair: float) -> str:
    return f"{base_name}_v0m{lambda_v0_match:g}_phi{lambda_phi_pair:g}"


def build_command(
    python_exe: str,
    train_py: Path,
    common_args: List[str],
    train_gpu_id: str,
    lambda_v0_match: float,
    lambda_phi_pair: float,
) -> List[str]:
    cmd: List[str] = [python_exe, str(train_py)]
    for item in common_args:
        if item == "{gpu_id}":
            cmd.append(train_gpu_id)
        elif item == "{lambda_v0_match}":
            cmd.append(f"{lambda_v0_match}")
        elif item == "{lambda_phi_pair}":
            cmd.append(f"{lambda_phi_pair}")
        else:
            cmd.append(item)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep lambda_v0_match and lambda_phi_pair with GPU-aware parallel scheduling.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to launch train.py")
    parser.add_argument("--train_py", default="train.py", help="Path to train.py")
    parser.add_argument("--checkpoints_dir", default="./checkpoints")
    parser.add_argument("--dataroot", default="/home/ljc/code/PaBoT-main/datasets")
    parser.add_argument("--base_name", default="dual_dino_phi_fresh_sweep", help="Base experiment name; suffixes are appended automatically")
    parser.add_argument("--tag", default="dual_dino_phi_sweep")
    parser.add_argument("--gpu_ids", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids used as a scheduling pool")
    parser.add_argument(
        "--train_gpu_ids",
        default="0",
        help="GPU ids passed to train.py inside each child process. Keep as '0' when using CUDA_VISIBLE_DEVICES mapping.",
    )
    parser.add_argument("--lambda_v0_match", nargs="+", default=["0.25", "0.5", "1.0"], help="Values to sweep")
    parser.add_argument("--lambda_phi_pair", nargs="+", default=["0.25", "0.5", "1.0"], help="Values to sweep")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_epochs_decay", type=int, default=50)
    parser.add_argument("--epoch_count", type=int, default=1)
    parser.add_argument("--eval_epoch_freq", type=int, default=1)
    parser.add_argument("--save_epoch_freq", type=int, default=1)
    parser.add_argument("--model", default="dual_velocity_struct")
    parser.add_argument("--dataset_mode", default="unaligned")
    parser.add_argument("--direction", default="BtoA")
    parser.add_argument("--controlled_pairing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paired_ratio", type=float, default=0.1)
    parser.add_argument("--pair_seed", type=int, default=3407)
    parser.add_argument("--a_backbone", default="vit")
    parser.add_argument("--a_vit_depth", type=int, default=4)
    parser.add_argument("--a_vit_dim", type=int, default=256)
    parser.add_argument("--a_vit_heads", type=int, default=8)
    parser.add_argument("--a_vit_patch", type=int, default=2)
    parser.add_argument("--dino_model_name", default="dino_vitb8")
    parser.add_argument("--dino_image_size", type=int, default=224)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--normG", default="instance")
    parser.add_argument("--normD", default="instance")
    parser.add_argument("--init_type", default="xavier")
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--preprocess", default="resize_and_crop")
    parser.add_argument("--random_scale_max", type=float, default=3.0)
    parser.add_argument("--max_dataset_size", default="inf")
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--stylegan2_G_num_downsampling", type=int, default=1)
    parser.add_argument("--ode_steps", type=int, default=4)
    parser.add_argument("--struct_channels", type=int, default=64)
    parser.add_argument("--phi_hidden_channels", type=int, default=32)
    parser.add_argument("--struct_velocity_mode", default="learned")
    parser.add_argument("--use_structure_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structure_attention_source", default="rollout")
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--struct_grad_scale", type=float, default=0.1)
    parser.add_argument("--log_attention_map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--lambda_path", type=float, default=0.1)
    parser.add_argument("--lambda_vs", type=float, default=0.01)
    parser.add_argument("--lambda_ortho", type=float, default=0.01)
    parser.add_argument("--lambda_GAN", type=float, default=1.0)
    parser.add_argument("--lambda_idt", type=float, default=5.0)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_rec", type=float, default=5.0)
    parser.add_argument("--lambda_phi_attn", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--v0_stopgrad_phi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", default="PaBoT")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--no_flip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_dropout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cam_weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue_train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    gpu_pool = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_pool:
        print("[ERROR] --gpu_ids is empty")
        return 2

    sweep_v0 = parse_float_list(args.lambda_v0_match)
    sweep_phi = parse_float_list(args.lambda_phi_pair)
    grid = list(itertools.product(sweep_v0, sweep_phi))

    if not grid:
        print("[ERROR] no sweep combinations generated")
        return 2

    train_py = Path(args.train_py).resolve()
    if not train_py.exists():
        print(f"[ERROR] train.py not found: {train_py}")
        return 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.checkpoints_dir) / f"{args.base_name}_{timestamp}"
    log_dir = Path("logs") / f"{args.base_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    common_args = build_common_train_args(args)
    common_args = [str(run_root) if item == args.checkpoints_dir else item for item in common_args]

    jobs: List[Job] = []
    for idx, (lambda_v0_match, lambda_phi_pair) in enumerate(grid):
        gpu_id = gpu_pool[idx % len(gpu_pool)]
        name = make_job_name(args.base_name, lambda_v0_match, lambda_phi_pair)
        log_path = log_dir / f"{name}.log"

        cmd = build_command(
            python_exe=args.python,
            train_py=train_py,
            common_args=[item.replace(args.base_name, name) if item == args.base_name else item for item in common_args],
            train_gpu_id=args.train_gpu_ids,
            lambda_v0_match=lambda_v0_match,
            lambda_phi_pair=lambda_phi_pair,
        )

        # Replace the base name placeholder while keeping the rest of the command intact.
        for i, token in enumerate(cmd):
            if token == args.base_name:
                cmd[i] = name
                break

        jobs.append(Job(name=name, gpu_id=gpu_id, command=cmd, log_path=log_path))

    print("========== Sweep Plan ==========")
    print(f"Grid size: {len(grid)}")
    print(f"GPUs: {', '.join(gpu_pool)}")
    print(f"Logs: {log_dir}")
    print(f"Checkpoints root: {run_root}")
    for job in jobs:
        print(f"- {job.name} -> GPU {job.gpu_id} -> {job.log_path.name}")
    print("===============================")

    if args.dry_run:
        for job in jobs:
            print("\n".join(job.command))
        return 0

    pending = jobs[:]
    running: Dict[str, Job] = {}
    completed = 0
    failed = 0

    while pending or running:
        launched = False
        for gpu_id in gpu_pool:
            slot = gpu_id
            if slot in running:
                proc = running[slot].process
                if proc is not None and proc.poll() is not None:
                    job = running.pop(slot)
                    return_code = proc.returncode
                    if job.log_handle is not None and not job.log_handle.closed:
                        job.log_handle.close()
                    if return_code == 0:
                        completed += 1
                        print(f"[DONE] {job.name} on GPU {job.gpu_id}")
                    else:
                        failed += 1
                        print(f"[FAIL] {job.name} on GPU {job.gpu_id} (code {return_code})")
                    launched = True
                continue

            next_job_index = None
            for i, job in enumerate(pending):
                if job.gpu_id == gpu_id:
                    next_job_index = i
                    break
            if next_job_index is None:
                continue

            job = pending.pop(next_job_index)
            log_handle = job.log_path.open("w", encoding="utf-8")
            log_handle.write("COMMAND:\n")
            log_handle.write(" ".join(job.command) + "\n\n")
            log_handle.flush()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            proc = subprocess.Popen(job.command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            job.process = proc
            job.log_handle = log_handle
            running[slot] = job
            print(f"[RUN ] {job.name} on GPU {gpu_id} -> {job.log_path}")
            launched = True

        if not launched:
            time.sleep(10)

    print("========== Sweep Done ==========")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Logs: {log_dir}")
    print(f"Checkpoints root: {run_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
