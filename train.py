import os
import torch
import yaml
from rich.console import Console

console = Console()

from src.models.generator     import Generator
from src.models.discriminator import Discriminator
from src.dataset.dataset      import get_image_dataloader
from src.engine.trainer       import FastGANTrainer

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open("config/default.yaml", "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Using device:[/] [cyan]{device}[/]")

    # ------------------------------------------------------------------ Config
    DATA_DIR      = cfg["data"]["data_dir"]
    NUM_WORKERS   = cfg["data"]["num_workers"]

    IMAGE_SIZE    = cfg["training"]["image_size"]
    BATCH_SIZE    = cfg["training"]["batch_size"]
    Z_DIM         = cfg["training"]["z_dim"]
    NFC_BASE      = cfg["training"]["nfc_base"]
    NUM_EPOCHS    = cfg["training"]["num_epochs"]

    LR            = cfg["training"]["lr"]
    BETAS         = tuple(cfg["training"]["betas"])
    RECON_LAMBDA  = cfg["training"].get("recon_lambda",   1.0)
    DIFFAUG       = cfg["training"].get("diffaug_policy", "color,translation,cutout")

    USE_AMP           = cfg["training"].get("use_amp",           False)
    USE_EMA           = cfg["training"].get("use_ema",           True)
    EMA_DECAY         = cfg["training"].get("ema_decay",         0.999)
    SAVE_EVERY        = cfg["training"].get("save_every",        20)
    CHECKPOINT_EVERY  = cfg["training"].get("checkpoint_every",  50)

    IMAGES_DIR    = cfg["outputs"]["images_dir"]
    WEIGHTS_PATH  = cfg["outputs"]["weights_path"]
    LOG_DIR       = cfg["outputs"]["log_dir"]

    # ------------------------------------------------------------------ Data
    console.print("\n[bold]Loading data...[/]")
    dataloader = get_image_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        train=True,
    )
    console.print(f"  Dataset size : [cyan]{len(dataloader.dataset):,}[/] images")
    console.print(f"  Image size   : [cyan]{IMAGE_SIZE}×{IMAGE_SIZE}[/]")
    console.print(f"  Batch size   : [cyan]{BATCH_SIZE}[/]  →  [cyan]{len(dataloader):,}[/] batches/epoch")

    # ------------------------------------------------------------------ Models
    console.print("\n[bold]Initialising FastGAN models...[/]")
    generator = Generator(
        z_dim=Z_DIM,
        nfc_base=NFC_BASE,
        image_size=IMAGE_SIZE,
    )
    discriminator = Discriminator(
        nfc_base=NFC_BASE,
        image_size=IMAGE_SIZE,
    )

    g_params = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    console.print(f"  Generator     : [cyan]{g_params:,}[/] params")
    console.print(f"  Discriminator : [cyan]{d_params:,}[/] params")

    # ------------------------------------------------------------------ Trainer
    trainer = FastGANTrainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        z_dim=Z_DIM,
        lr=LR,
        betas=BETAS,
        recon_lambda=RECON_LAMBDA,
        diffaug_policy=DIFFAUG,
        use_amp=USE_AMP,
        use_ema=USE_EMA,
        ema_decay=EMA_DECAY,
        save_every=SAVE_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    # ------------------------------------------------------------------ Train
    ckpt_dir = os.path.join(os.path.dirname(WEIGHTS_PATH), "checkpoints")
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir=os.path.join(IMAGES_DIR, "fastgan"),
        log_dir=LOG_DIR,
        checkpoint_dir=ckpt_dir,
    )

    # ------------------------------------------------------------------ Save
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    torch.save(trainer.generator.state_dict(), WEIGHTS_PATH)
    console.print(f"[bold green]Generator weights saved[/] → [cyan]{WEIGHTS_PATH}[/]")

    if USE_EMA and getattr(trainer, "ema_generator", None) is not None:
        ema_path = WEIGHTS_PATH.replace(".pth", "_ema.pth")
        torch.save(trainer.ema_generator.state_dict(), ema_path)
        console.print(f"[bold green]EMA generator weights saved[/] → [cyan]{ema_path}[/]")

if __name__ == "__main__":
    main()
