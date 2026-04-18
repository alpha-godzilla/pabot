#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINE_RE = re.compile(r"\(epoch:\s*(\d+),")
KV_RE = re.compile(r"([A-Za-z0-9_]+):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def mean(values):
    return sum(values) / len(values) if values else 0.0


def mse(values):
    if not values:
        return 0.0
    return sum(v * v for v in values) / len(values)


def ema(values, alpha):
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def parse_log(path, loss_key, ignore_zeros=False):
    all_values = []
    by_epoch = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_epoch = LINE_RE.search(line)
            if not m_epoch:
                continue
            epoch = int(m_epoch.group(1))
            kv = dict(KV_RE.findall(line))
            if loss_key not in kv:
                continue

            value = float(kv[loss_key])
            if ignore_zeros and value == 0.0:
                continue

            all_values.append(value)
            by_epoch[epoch].append(value)

    return all_values, by_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-epoch average and MSE, and plot avg/EMA curves"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="checkpoints/dual_dino_phi_only/loss_log.txt",
        help="Path to loss_log.txt",
    )
    parser.add_argument(
        "--loss_key",
        type=str,
        default="G_phi_pair",
        help="Loss key to track (e.g. G_phi_pair / G_v0_match / G_pair)",
    )
    parser.add_argument(
        "--ignore_zeros",
        action="store_true",
        help="Ignore zero-valued samples in statistics",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.2,
        help="EMA smoothing alpha in (0, 1]",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output image path. Default: <log_dir>/<loss_key>_avg_ema_curve.png",
    )
    args = parser.parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema_alpha must be in (0, 1]")

    all_values, by_epoch = parse_log(args.log, args.loss_key, args.ignore_zeros)

    if not all_values:
        print(
            f"No samples found for loss_key={args.loss_key} in {args.log}. "
            "Please check the key name."
        )
        return

    global_avg = mean(all_values)
    global_mse = mse(all_values)

    print("=" * 72)
    print(f"Log: {args.log}")
    print(f"Loss key: {args.loss_key}")
    print(f"Samples: {len(all_values)}")
    print(f"Ignore zeros: {args.ignore_zeros}")
    print("-" * 72)
    print(f"Global average: {global_avg:.6f}")
    print(f"Global MSE: {global_mse:.6f}")
    print("=" * 72)
    print("Per-epoch stats:")
    print("epoch\tn\taverage\tmse")

    sorted_epochs = sorted(by_epoch.keys())
    epoch_avgs = []
    for epoch in sorted_epochs:
        vals = by_epoch[epoch]
        avg = mean(vals)
        mse_val = mse(vals)
        epoch_avgs.append(avg)
        print(f"{epoch}\t{len(vals)}\t{avg:.6f}\t{mse_val:.6f}")

    epoch_ema = ema(epoch_avgs, args.ema_alpha)

    min_avg_idx = min(range(len(epoch_avgs)), key=lambda i: epoch_avgs[i])
    min_ema_idx = min(range(len(epoch_ema)), key=lambda i: epoch_ema[i])
    min_avg_epoch = sorted_epochs[min_avg_idx]
    min_ema_epoch = sorted_epochs[min_ema_idx]

    print("=" * 72)
    print(f"Min AVG: epoch={min_avg_epoch}, value={epoch_avgs[min_avg_idx]:.6f}")
    print(f"Min EMA: epoch={min_ema_epoch}, value={epoch_ema[min_ema_idx]:.6f}")

    out_path = args.out
    if not out_path:
        log_dir = os.path.dirname(args.log) or "."
        out_path = os.path.join(log_dir, f"{args.loss_key}_avg_ema_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_epochs, epoch_avgs, label="Average Loss", color="tab:blue", linewidth=2)
    plt.plot(
        sorted_epochs,
        epoch_ema,
        label=f"EMA Loss (alpha={args.ema_alpha})",
        color="tab:orange",
        linewidth=2,
    )

    plt.scatter([min_avg_epoch], [epoch_avgs[min_avg_idx]], color="tab:blue", s=70, zorder=5)
    plt.scatter([min_ema_epoch], [epoch_ema[min_ema_idx]], color="tab:orange", s=70, zorder=5)

    plt.annotate(
        f"Min AVG\nE{min_avg_epoch}: {epoch_avgs[min_avg_idx]:.4f}",
        xy=(min_avg_epoch, epoch_avgs[min_avg_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        color="tab:blue",
    )
    plt.annotate(
        f"Min EMA\nE{min_ema_epoch}: {epoch_ema[min_ema_idx]:.4f}",
        xy=(min_ema_epoch, epoch_ema[min_ema_idx]),
        xytext=(10, -35),
        textcoords="offset points",
        color="tab:orange",
    )

    plt.title(f"{args.loss_key}: Average vs EMA Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved plot: {out_path}")

 
if __name__ == "__main__":
    main()
