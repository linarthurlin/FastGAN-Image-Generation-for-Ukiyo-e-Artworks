import os
import json
import yaml
import torch
import PIL.Image
import torchvision.transforms as T
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore

from src.models.generator  import Generator
from src.dataset.dataset   import get_image_dataloader


def interpolate(batch_01: torch.Tensor) -> torch.Tensor:
    to_pil    = T.ToPILImage()
    to_tensor = T.ToTensor()
    return torch.stack([
        to_tensor(to_pil(img).resize((299, 299), PIL.Image.BILINEAR))
        for img in batch_01
    ])


def main():
    with open("config/default.yaml", "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_DIR     = cfg["data"]["data_dir"]
    BATCH_SIZE   = cfg["training"]["batch_size"]
    IMAGE_SIZE   = cfg["training"]["image_size"]
    Z_DIM        = cfg["training"]["z_dim"]
    NFC_BASE     = cfg["training"]["nfc_base"]
    NUM_WORKERS  = cfg["data"]["num_workers"]
    WEIGHTS_PATH = cfg["outputs"]["weights_path"]

    # Prefer EMA weights when available
    ema_path  = WEIGHTS_PATH.replace(".pth", "_ema.pth")
    load_path = ema_path if os.path.exists(ema_path) else WEIGHTS_PATH
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Model weights not found at {load_path}. "
        )
    print(f"Loading weights from: {load_path}")

    # ---- Data ----------------------------------------------------------------
    print("Loading validation data...")
    dataloader = get_image_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        train=False,
    )
    print(f"  {len(dataloader.dataset):,} images  /  {len(dataloader):,} batches")

    # ---- Model ---------------------------------------------------------------
    generator = Generator(
        z_dim=Z_DIM,
        nfc_base=NFC_BASE,
        image_size=IMAGE_SIZE,
    ).to(device)
    generator.load_state_dict(torch.load(load_path, map_location=device))
    generator.eval()
    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")

    # ---- Evaluation step (mirrors the blog's evaluation_step) ----------------
    def evaluation_step(engine, batch):
        real_imgs = batch
        if isinstance(real_imgs, (list, tuple)):
            real_imgs = real_imgs[-1]
        real_imgs = real_imgs.to(device)
        B = real_imgs.size(0)

        z = torch.randn(B, Z_DIM, device=device)
        with torch.no_grad():
            generator.eval()
            fake_imgs = generator(z)

        # Denorm [-1, 1] → [0, 1] then move to CPU for PIL resize
        real_01 = (real_imgs.clamp(-1, 1) + 1) * 0.5
        fake_01 = (fake_imgs.clamp(-1, 1) + 1) * 0.5

        # Resize to 299×299 via PIL BILINEAR (required by InceptionV3)
        real_299 = interpolate(real_01.cpu())
        fake_299 = interpolate(fake_01.cpu())

        return fake_299, real_299

    # ---- Attach metrics (same pattern as the blog) ---------------------------
    evaluator  = Engine(evaluation_step)
    fid_metric = FID(device=device)
    is_metric  = InceptionScore(device=device, output_transform=lambda x: x[0])

    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator,  "is")

    print("\nCalculating FID & Inception Score …")
    evaluator.run(dataloader, max_epochs=1)

    metrics = evaluator.state.metrics
    fid_score = metrics["fid"]
    is_score  = metrics["is"]

    print("-" * 40)
    print(f"FID              : {fid_score:.4f}")
    print(f"Inception Score  : {is_score:.4f}")
    print("-" * 40)

    # ---- Save scores ---------------------------------------------------------
    results = {
        "weights": load_path,
        "fid":     fid_score,
        "is":      is_score,
    }
    out_path = cfg["outputs"].get("eval_results", "outputs/eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
