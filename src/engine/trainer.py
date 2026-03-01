import os
import copy

import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, MofNCompleteColumn,
    TextColumn, TimeRemainingColumn, SpinnerColumn,
)

console = Console()

from src.utils.losses      import d_logistic_loss, g_logistic_loss, recon_loss
from src.utils.diffaugment import diffaugment


class FastGANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        dataloader,
        device,
        z_dim: int = 256,
        lr: float = 2e-4,
        betas: tuple = (0.0, 0.99),
        recon_lambda: float = 1.0,
        use_amp: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        save_every: int = 20,
        checkpoint_every: int = 50,
        diffaug_policy: str = "color,translation,cutout",
    ):
        self.generator     = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader    = dataloader
        self.device        = device

        self.z_dim         = z_dim
        self.recon_lambda  = recon_lambda
        self.use_amp       = use_amp
        self.use_ema           = use_ema
        self.ema_decay         = ema_decay
        self.save_every        = save_every
        self.checkpoint_every  = checkpoint_every
        self.diffaug_policy    = diffaug_policy

        torch.backends.cudnn.benchmark = True

        self.opt_gen  = optim.Adam(self.generator.parameters(),     lr=lr, betas=betas)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # EMA generator for stable evaluation samples
        if self.use_ema:
            self.ema_generator = copy.deepcopy(self.generator).to(device)
            self.ema_generator.eval()
            for p in self.ema_generator.parameters():
                p.requires_grad_(False)
        else:
            self.ema_generator = None

        self._fixed_z = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_ema(self):
        if not (self.use_ema and self.ema_generator is not None):
            return
        with torch.no_grad():
            for p, e in zip(self.generator.parameters(),
                            self.ema_generator.parameters()):
                e.data.lerp_(p.data, 1.0 - self.ema_decay)
            for b, eb in zip(self.generator.buffers(),
                             self.ema_generator.buffers()):
                eb.data.copy_(b.data)

    def _sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=self.device)

    def _save_checkpoint(self, ckpt_dir: str, epoch: int):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            "epoch":           epoch,
            "generator":       self.generator.state_dict(),
            "discriminator":   self.discriminator.state_dict(),
            "opt_gen":         self.opt_gen.state_dict(),
            "opt_disc":        self.opt_disc.state_dict(),
        }
        if self.use_ema and self.ema_generator is not None:
            ckpt["ema_generator"] = self.ema_generator.state_dict()
        path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pth")
        torch.save(ckpt, path)
        console.print(f"  [bold green][Checkpoint][/] saved → [cyan]{path}[/]")

    def _save_samples(self, save_dir: str, writer: SummaryWriter, epoch: int):
        model = self.ema_generator if self.use_ema else self.generator
        model.eval()
        with torch.no_grad():
            fake = model(self._fixed_z)
        model.train()
        vis  = (fake.clamp(-1, 1) + 1) * 0.5
        n    = min(16, vis.shape[0])
        grid = torchvision.utils.make_grid(vis[:n], nrow=4, normalize=False)
        torchvision.utils.save_image(grid, f"{save_dir}/epoch_{epoch:04d}.png")
        writer.add_image("FastGAN/generated", grid, epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int, save_dir: str, log_dir: str,
              checkpoint_dir: str = "outputs/weights/checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        writer      = SummaryWriter(log_dir=os.path.join(log_dir, "fastgan"))
        global_step = 0

        console.print(f"\n[bold cyan]=== FastGAN Training ({num_epochs} epochs) ===[/]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[metrics]}"),
            console=console,
            refresh_per_second=5,
        ) as progress:
            epoch_task = progress.add_task(
                "[bold cyan]Total epochs", total=num_epochs, metrics=""
            )

            for epoch in range(num_epochs):
                self.generator.train()
                self.discriminator.train()

                batch_task = progress.add_task(
                    f"[green]Epoch {epoch + 1:>4}/{num_epochs}",
                    total=len(self.dataloader),
                    metrics="",
                )

                loss_d_avg = loss_g_avg = 0.0

                for real_imgs in self.dataloader:
                    if isinstance(real_imgs, (list, tuple)):
                        real_imgs = real_imgs[-1]
                    real_imgs = real_imgs.to(self.device, non_blocking=True)
                    B = real_imgs.size(0)

                    if self._fixed_z is None:
                        self._fixed_z = self._sample_z(16)

                    # ==== Train Discriminator ==============================
                    z         = self._sample_z(B)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        fake_imgs = self.generator(z).detach()
                        # DiffAugment: augment both real and fake before D sees them
                        real_aug = diffaugment(real_imgs, self.diffaug_policy)
                        fake_aug = diffaugment(fake_imgs, self.diffaug_policy)
                        real_logit, real_part, real_recon = self.discriminator(real_aug)
                        fake_logit, fake_part, fake_recon = self.discriminator(fake_aug)
                        loss_adv_d  = d_logistic_loss(real_logit, fake_logit)
                        loss_part_d = d_logistic_loss(real_part,  fake_part)
                        loss_recon  = (recon_loss(real_recon, real_imgs)
                                       + recon_loss(fake_recon, fake_imgs))
                        loss_d = loss_adv_d + loss_part_d + self.recon_lambda * loss_recon

                    self.opt_disc.zero_grad(set_to_none=True)
                    self.scaler.scale(loss_d).backward()
                    self.scaler.step(self.opt_disc)
                    self.scaler.update()

                    # ==== Train Generator ==================================
                    z = self._sample_z(B)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        fake_imgs                = self.generator(z)
                        fake_aug                 = diffaugment(fake_imgs, self.diffaug_policy)
                        fake_logit, fake_part, _ = self.discriminator(fake_aug)
                        loss_g = g_logistic_loss(fake_logit) + g_logistic_loss(fake_part)

                    self.opt_gen.zero_grad(set_to_none=True)
                    self.scaler.scale(loss_g).backward()
                    self.scaler.step(self.opt_gen)
                    self.scaler.update()

                    self._update_ema()

                    # ==== Logging ==========================================
                    ld, lg = loss_d.item(), loss_g.item()
                    loss_d_avg += ld
                    loss_g_avg += lg

                    writer.add_scalar("FastGAN/Loss_D",       ld,                         global_step)
                    writer.add_scalar("FastGAN/Loss_G",       lg,                         global_step)
                    writer.add_scalar("FastGAN/Loss_D_adv",   loss_adv_d.item(),          global_step)
                    writer.add_scalar("FastGAN/Loss_D_part",  loss_part_d.item(),         global_step)
                    writer.add_scalar("FastGAN/Loss_D_recon", loss_recon.item(),          global_step)
                    writer.add_scalar("FastGAN/D_real",       real_logit.mean().item(),   global_step)
                    writer.add_scalar("FastGAN/D_fake",       fake_logit.mean().item(),   global_step)
                    global_step += 1

                    progress.update(batch_task, advance=1,
                                    metrics=f"D={ld:.3f}  G={lg:.3f}")

                n_b = max(len(self.dataloader), 1)
                writer.add_scalar("FastGAN/Epoch_Loss_D", loss_d_avg / n_b, epoch + 1)
                writer.add_scalar("FastGAN/Epoch_Loss_G", loss_g_avg / n_b, epoch + 1)

                progress.update(
                    epoch_task, advance=1,
                    metrics=f"D={loss_d_avg / n_b:.3f}  G={loss_g_avg / n_b:.3f}",
                )
                progress.remove_task(batch_task)

                if (epoch + 1) % self.save_every == 0 or epoch == 0:
                    self._save_samples(save_dir, writer, epoch + 1)

                if self.checkpoint_every > 0 and (epoch + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_dir, epoch + 1)

        writer.close()
        console.print("[bold green]FastGAN training complete.[/]\n")

