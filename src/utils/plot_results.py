import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.generator import Generator

def smooth_curve(scalars, weight=0.8):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_losses_combined(
    g_csv:    str = "./outputs/epoch_loss_g_fastgan.csv",
    d_csv:    str = "./outputs/epoch_loss_d_fastgan.csv",
    save_path: str = "./outputs/loss_curves.png",
    weight:   float = 0.85,
):
    df_g = pd.read_csv(g_csv)
    df_d = pd.read_csv(d_csv)

    configs = [
        (df_g, "Generator Loss",     "#ff7f0e"),
        (df_d, "Discriminator Loss", "#0e7eff"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=300)

    for ax, (df, title, color) in zip(axes, configs):
        steps  = df["Step"]
        raw    = df["Value"].tolist()
        smooth = smooth_curve(raw, weight)

        ax.plot(steps, raw,    color=color, alpha=0.2, linewidth=1,   label="Raw")
        ax.plot(steps, smooth, color=color, alpha=1.0, linewidth=2,   label="Smoothed")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss",   fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=11)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, format="png", bbox_inches="tight")
    print(f"Loss curves saved → {save_path}")
    plt.show()

def plot_generated_grid(
    weights_path: str = "./outputs/weights/generator_ema.pth",
    config_path:  str = "./config/default.yaml",
    save_path:    str = "./outputs/generated_grid.png",
    seed:         int = 42,
):
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    z_dim      = cfg["training"]["z_dim"]
    nfc_base   = cfg["training"]["nfc_base"]
    image_size = cfg["training"]["image_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(z_dim=z_dim, nfc_base=nfc_base, image_size=image_size).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()
    print(f"Loaded weights from {weights_path}")

    torch.manual_seed(seed)
    z = torch.randn(9, z_dim, device=device)
    with torch.no_grad():
        imgs = generator(z)                          # (9, 3, H, W), range [-1, 1]

    imgs = (imgs.clamp(-1, 1) + 1) * 0.5            # → [0, 1]
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()   # (9, H, W, 3)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8), dpi=300,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img)
        ax.axis("off")

    fig.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0)
    print(f"Generated grid saved → {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_losses_combined(
        g_csv     = "./outputs/epoch_loss_g_fastgan.csv",
        d_csv     = "./outputs/epoch_loss_d_fastgan.csv",
        save_path = "./outputs/loss_curves.png",
        weight    = 0.85,
    )

    plot_generated_grid(
        weights_path = "./outputs/weights/generator_ema.pth",
        config_path  = "./config/default.yaml",
        save_path    = "./outputs/generated_grid.png",
        seed         = 42,
    )