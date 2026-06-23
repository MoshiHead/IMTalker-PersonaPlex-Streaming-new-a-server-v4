"""One-shot SMIRK encoder on the preprocessed reference tensor → cam (3,).

Used when --static_pose_zero --static_ref_cam: broadcast first-frame cam for all T.
Encoder is cached by (smirk_root, ckpt, device).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp

_ENC: Optional[torch.nn.Module] = None
_ENC_KEY: Optional[tuple[str, str, str]] = None


def crop_face(frame: np.ndarray, landmarks: np.ndarray, scale: float = 1.4, image_size: int = 224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, dst_pts)


def _load_encoder(smirk_root: Path, checkpoint_path: Path, device: torch.device) -> "torch.nn.Module":
    global _ENC, _ENC_KEY
    key = (str(smirk_root.resolve()), str(checkpoint_path.resolve()), str(device))
    if _ENC is not None and _ENC_KEY == key:
        return _ENC

    root_s = str(smirk_root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    from src.smirk_encoder import SmirkEncoder

    model = SmirkEncoder().to(device)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    state = {
        k.replace("smirk_encoder.", "", 1): v for k, v in ckpt.items() if "smirk_encoder." in k
    }
    if not state:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    _ENC = model
    _ENC_KEY = key
    return model


def _frame_to_tensor(
    frame_bgr: np.ndarray,
    mediapipe_fn,
    image_size: int,
    crop: bool,
    device: torch.device,
) -> Optional[torch.Tensor]:
    landmarks = mediapipe_fn(frame_bgr)
    if landmarks is None:
        return None
    landmarks = landmarks[..., :2]

    if crop:
        tform = crop_face(frame_bgr, landmarks, scale=1.4, image_size=image_size)
        frame_bgr = warp(
            frame_bgr, tform.inverse, output_shape=(image_size, image_size), preserve_range=True
        ).astype(np.uint8)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def _ref_tensor_to_bgr(s_1_3_hw: torch.Tensor) -> np.ndarray:
    """s: [1,3,H,W] float in [0,1] → BGR uint8 HxW."""
    x = (s_1_3_hw.detach().float().cpu().clamp(0, 1) * 255).to(torch.uint8)
    rgb = x.squeeze(0).permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def first_frame_cam_from_ref_tensor(
    s_bchw: torch.Tensor,
    *,
    smirk_root: str,
    ckpt_path: Optional[str],
    device: torch.device,
    image_size: int = 224,
    mediapipe_crop: bool = True,
) -> torch.Tensor:
    """Return cam (3,) float32 on `device` from SMIRK encoder on reference image tensor."""
    root = Path(smirk_root).resolve()
    ck = Path(ckpt_path) if ckpt_path else root / "pretrained_models" / "SMIRK_em1.pt"
    if not ck.is_file():
        raise FileNotFoundError(f"SMIRK checkpoint not found: {ck}")

    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    frame_bgr = _ref_tensor_to_bgr(s_bchw)
    model = _load_encoder(root, ck, device)
    tensor: Optional[torch.Tensor] = None
    cwd_prev = os.getcwd()
    try:
        os.chdir(root)
        from utils.mediapipe_utils import run_mediapipe

        tensor = _frame_to_tensor(frame_bgr, run_mediapipe, image_size, mediapipe_crop, device)
    finally:
        os.chdir(cwd_prev)

    if tensor is None:
        raise RuntimeError(
            "[SMIRK] MediaPipe found no face on preprocessed reference; "
            "try --crop or a clearer reference."
        )
    out = model(tensor)
    cam = out["cam"].squeeze(0).detach().float().view(-1)
    if cam.numel() != 3:
        raise RuntimeError(f"[SMIRK] unexpected cam shape: {tuple(cam.shape)}")
    return cam.to(device)
