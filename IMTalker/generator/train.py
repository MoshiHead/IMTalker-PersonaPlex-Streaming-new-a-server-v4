import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torch import pi
from torch.nn import Module
from torch.utils import data
from torch import nn, optim
from einops import rearrange, repeat
from generator.dataset import AudioMotionSmirkGazeDataset
from FM import FMGenerator
from generate import DataProcessor
from options.base_options import BaseOptions
from pytorch_lightning.loggers import TensorBoardLogger
from renderer.models import IMTRenderer

# ==========================================
# 1. New EMA Class Helper
# ==========================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def cosmap(t):
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

class L1loss(Module):
    def forward(self, pred, target, **kwargs):
        return F.l1_loss(pred, target)

class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = FMGenerator(opt)
        self.opt = opt
        self.loss_fn = L1loss()
        
        self.ema = EMA(self.model, decay=0.9999) 

    def forward(self, x):
        return self.model(x)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    def on_validation_epoch_start(self):
        self.ema.apply_shadow()

    def on_validation_epoch_end(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state_dict"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.shadow = checkpoint["ema_state_dict"]

    def training_step(self, batch, batch_idx):
        m_now = batch["m_now"]

        noise = torch.randn_like(m_now)
        times = torch.rand(m_now.size(0), device=self.device)
        t = append_dims(times, m_now.ndim - 1)
        noised_motion = t * m_now + (1 - t) * noise
        gt_flow = m_now - noise

        batch["m_now"] = noised_motion

        pred_flow_anchor = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow_anchor, gt_flow)
        velocity_loss = self.loss_fn(pred_flow_anchor[:, 1:] - pred_flow_anchor[:, :-1], 
                                     gt_flow[:, 1:] - gt_flow[:, :-1])

        train_loss = fm_loss + velocity_loss

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("fm_loss", fm_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        m_now = batch["m_now"]
        noise = torch.randn_like(m_now); times = torch.rand(m_now.size(0), device=self.device); t = append_dims(times, m_now.ndim - 1)
        noised_motion = t * m_now + (1 - t) * noise; gt_flow = m_now - noise
        batch["m_now"] = noised_motion
        pred_flow_anchor = self.model(batch, t=times)

        fm_loss = self.loss_fn(pred_flow_anchor, gt_flow)
        velocity_loss = self.loss_fn(pred_flow_anchor[:, 1:] - pred_flow_anchor[:, :-1], 
                                     gt_flow[:, 1:] - gt_flow[:, :-1])

        val_loss = fm_loss + velocity_loss

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_fm_loss", fm_loss, prog_bar=True)
    
    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        if "ema_state_dict" in ckpt:
            print("[INFO] Found EMA weights in checkpoint. Loading EMA weights for better stability.")
            state_dict = ckpt["ema_state_dict"]
        else:
            print("[INFO] EMA weights not found. Loading standard state_dict.")
            state_dict = ckpt.get("state_dict", ckpt)

        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model_state_dict = self.model.state_dict()
        loadable_params = {}
        unmatched_keys = []

        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                loadable_params[k] = v
            else:
                unmatched_keys.append(k)

        missing_keys, unexpected_keys = self.model.load_state_dict(loadable_params, strict=False)

        self.ema.register()

        print(f"[INFO] Loaded {len(loadable_params)} params from checkpoint.")
        if missing_keys:
            print(f"[WARNING] Missing keys: {missing_keys}")
        if unmatched_keys:
            print(f"[WARNING] {len(unmatched_keys)} keys skipped.")

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.opt.iter, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": scheduler}


class PreviewVideoCallback(pl.Callback):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.processor = None
        self.renderer = None
        self.last_rendered_step = -1

    def _load_renderer(self, device):
        if self.renderer is not None:
            return

        self.renderer = IMTRenderer(self.opt).to(device).eval()
        renderer_ckpt = torch.load(self.opt.infer_renderer_path, map_location="cpu")["state_dict"]
        ae_state_dict = {k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")}
        self.renderer.load_state_dict(ae_state_dict, strict=False)

    def _get_processor(self):
        if self.processor is None:
            self.processor = DataProcessor(self.opt)
        return self.processor

    @torch.no_grad()
    def _render_preview(self, trainer, pl_module):
        if self.opt.infer_every_n_steps <= 0:
            return

        step = trainer.global_step
        if step <= 0 or step == self.last_rendered_step or step % self.opt.infer_every_n_steps != 0:
            return

        if not (self.opt.infer_ref_path and self.opt.infer_aud_path and self.opt.infer_renderer_path):
            return

        if not (
            os.path.exists(self.opt.infer_ref_path)
            and os.path.exists(self.opt.infer_aud_path)
            and os.path.exists(self.opt.infer_renderer_path)
        ):
            if trainer.is_global_zero:
                print("[Preview] Skipping preview render because one or more inference paths do not exist.")
            return

        trainer.strategy.barrier("preview_start")

        try:
            if trainer.is_global_zero:
                was_training = pl_module.model.training
                pl_module.model.eval()
                pl_module.ema.apply_shadow()

                try:
                    device = pl_module.device
                    self._load_renderer(device)
                    processor = self._get_processor()
                    data = processor.preprocess(
                        self.opt.infer_ref_path,
                        self.opt.infer_aud_path,
                        crop=self.opt.infer_crop,
                    )
                    source = data["s"].to(device)
                    audio = data["a"].to(device)

                    f_r, g_r = self.renderer.dense_feature_encoder(source)
                    t_r = self.renderer.latent_token_encoder(source)
                    sample = pl_module.model.sample(
                        {"a": audio, "ref_x": t_r},
                        a_cfg_scale=self.opt.infer_a_cfg_scale,
                        nfe=self.opt.infer_nfe,
                        seed=self.opt.infer_seed,
                    )

                    ta_r = self.renderer.adapt(t_r, g_r)
                    m_r = self.renderer.latent_token_decoder(ta_r)
                    frames = []
                    for frame_idx in range(sample.shape[1]):
                        ta_c = self.renderer.adapt(sample[:, frame_idx, ...], g_r)
                        m_c = self.renderer.latent_token_decoder(ta_c)
                        frames.append(self.renderer.decode(m_c, m_r, f_r))
                    video = torch.stack(frames, dim=1).squeeze(0)

                    out_dir = Path(self.opt.exp_path) / self.opt.exp_name / self.opt.infer_out_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"step_{step:07d}.mp4"

                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        vid = video.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
                        vid = (vid * 255).type(torch.uint8)
                        torchvision.io.write_video(str(tmp_path), vid, fps=self.opt.fps)
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(tmp_path),
                            "-i",
                            self.opt.infer_aud_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            str(out_path),
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink()

                    self.last_rendered_step = step
                    print(f"[Preview] Saved preview video to {out_path}")
                finally:
                    pl_module.ema.restore()
                    if was_training:
                        pl_module.model.train()
        finally:
            trainer.strategy.barrier("preview_end")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._render_preview(trainer, pl_module)
    
class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--dataset_path", "--dataset_pat", dest="dataset_path", default=None, type=str)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--iter', default=5000000, type=int)
        parser.add_argument("--exp_path", type=str, default='./exps')
        parser.add_argument("--exp_name", type=str, default='debug')
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--val_num_workers", type=int, default=0)
        parser.add_argument("--train_list_path", type=str, default=None)
        parser.add_argument("--val_list_path", type=str, default=None)
        parser.add_argument("--log_every_n_steps", type=int, default=10)
        parser.add_argument("--infer_every_n_steps", type=int, default=0)
        parser.add_argument("--infer_ref_path", type=str, default="./assets/source_5.png")
        parser.add_argument("--infer_aud_path", type=str, default="./assets/audio_3.wav")
        parser.add_argument("--infer_renderer_path", type=str, default="./checkpoints/renderer.ckpt")
        parser.add_argument("--infer_out_dir", type=str, default="preview_videos")
        parser.add_argument("--infer_seed", type=int, default=25)
        parser.add_argument("--infer_nfe", type=int, default=10)
        parser.add_argument("--infer_a_cfg_scale", type=float, default=3.0)
        parser.add_argument("--infer_crop", action="store_true")
        
        return parser

class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage):
        train_list_path = self.opt.train_list_path
        val_list_path = self.opt.val_list_path
        if train_list_path is None and self.opt.dataset_path:
            default_train = Path(self.opt.dataset_path) / "train.txt"
            if default_train.exists():
                train_list_path = str(default_train)
        if val_list_path is None and self.opt.dataset_path:
            default_val = Path(self.opt.dataset_path) / "test.txt"
            if default_val.exists():
                val_list_path = str(default_val)

        self.train_dataset = AudioMotionSmirkGazeDataset(
            opt=self.opt,
            start=0,
            end=-100,
            split="train",
            list_path=train_list_path,
        )
        self.val_dataset = AudioMotionSmirkGazeDataset(
            opt=self.opt,
            start=-100,
            end=-1,
            split="val",
            list_path=val_list_path,
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            num_workers=self.opt.num_workers,
            batch_size=self.opt.batch_size,
            shuffle=True,
            persistent_workers=self.opt.num_workers > 0,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            num_workers=self.opt.val_num_workers,
            batch_size=8,
            shuffle=False,
            persistent_workers=self.opt.val_num_workers > 0,
        )

if __name__ == '__main__':
    opt = TrainOptions().parse()
    system = System(opt)
    dm = DataModule(opt)

    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True
    )
    if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
        system.load_ckpt(opt.resume_ckpt)
        
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=opt.iter,
        log_every_n_steps=opt.log_every_n_steps,
        val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[checkpoint_callback, PreviewVideoCallback(opt)],
        enable_progress_bar=True,
    )

    trainer.fit(system, dm)


