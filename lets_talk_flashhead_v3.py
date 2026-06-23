"""
lets_talk_flashhead_v3.py — FlashTalk v3: Moshi Helium + SoulX-FlashHead Unified Streaming Server
===================================================================================================
updated-2024-06-20 again
Integrates Moshi Helium (full-duplex S2S) with SoulX-FlashHead (talking head avatar)
by directly feeding Moshi's transformer_out tokens into FlashHead's pipeline,
bypassing Wav2Vec2 entirely.

Token Bridge Architecture:
  Moshi (12.5Hz, 4096-dim) → Sliding 100-token deque (8s)
  → Interpolate 100→200 (maps 12.5Hz → 25fps)
  → MoshiToWav2VecAdapter (12-layer transformer, ~93M params)
  → Stack 12 hidden states → (200, 12, 768) — EXACT Wav2Vec2 format
  → Center-index gather [167:200] ±2 → (1, 33, 5, 12, 768)
  → AudioProjModel (FROZEN) → DiT (FROZEN) → VAE (FROZEN)

A/V Sync:
  Each chunk bundles N Moshi decoded audio chunks + N×2 video frames.
  Lite: 12 tokens = 960ms audio ↔ 24 frames @ 25fps = 960ms video
  Pro:  14 tokens = 1120ms audio ↔ 28 frames @ 25fps = 1120ms video

Frozen Components (unchanged from standard SoulX-FlashHead):
  - AudioProjModel: (1, 33, 5, 12, 768) → (1, 9, 32, 1536)
  - DiT (WanModelAudioProject): 4 denoising steps with audio cross-attention
  - VAE Decoder: latent → 33 pixel frames (512×512)
  - VAE Encoder: motion frame carry-over between chunks
  - Color correction, flow matching — all standard

Trainable Component:
  - MoshiToWav2VecAdapter (~93M params)
  - Checkpoint: ./checkpoints/moshi_to_flashhead_adapter.pt

Usage:
  python lets_talk_flashhead_v3.py --flash-model-type lite --ref-image <path>
  python lets_talk_flashhead_v3.py --flash-model-type pro --ref-image <path>
"""

import argparse
import asyncio
import base64
import json
import math
import os
import queue
import socket
import sys
import threading
import time
from collections import namedtuple
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketState
import uvicorn


# ────────────────────────────────────────────────────────────────────────────
#  Environment
# ────────────────────────────────────────────────────────────────────────────
os.environ["NO_CUDA_GRAPH"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = os.path.dirname(os.path.abspath(__file__))
SOULX_ROOT = os.path.join(ROOT, "SoulX-FlashHead")
STATIC_DIR = os.path.join(ROOT, "unitalk", "static")
STATIC_INDEX = os.path.join(STATIC_DIR, "flashtalk_head_saved.html")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────────────────────
MOSHI_SR = 24000
MOSHI_TOKEN_RATE = 12.5                                     # tokens/sec
MOSHI_TOKEN_DURATION_MS = 80                                # ms per token
MOSHI_FRAME_SAMPLES = int(MOSHI_SR / MOSHI_TOKEN_RATE)      # 1920 samples
MOSHI_DIM = 4096
SILENCE_THRESHOLD = 0.01

# FlashHead audio context
DEQUE_DURATION_S = 8                                        # 8 second context
DEQUE_SIZE = int(DEQUE_DURATION_S * MOSHI_TOKEN_RATE)       # 100 tokens
FLASHHEAD_FPS = 25
FLASHHEAD_FRAME_MS = 1000 / FLASHHEAD_FPS                   # 40ms
INTERP_TARGET = int(DEQUE_DURATION_S * FLASHHEAD_FPS)       # 200 frame-tokens
FRAME_NUM = 33                                              # FlashHead total frames per chunk

# Wav2Vec2 output format that FlashHead expects
WAV2VEC_LAYERS = 12
WAV2VEC_DIM = 768

# Queue sizes
TOKEN_QUEUE_MAXSIZE = 200
DISPATCH_QUEUE_MAXSIZE = 4
MIC_QUEUE_MAXSIZE = 48

# Default model paths
DEFAULT_FLASH_CKPT = os.path.join(SOULX_ROOT, "models", "SoulX-FlashHead-1_3B")
DEFAULT_FLASH_WAV2VEC = os.path.join(SOULX_ROOT, "models", "wav2vec2-base-960h")
DEFAULT_REF_IMAGE = os.path.join(SOULX_ROOT, "examples", "1.jpeg")
DEFAULT_ADAPTER_CKPT_DIR = os.path.join(ROOT, "checkpoints")
ADAPTER_FILENAME = "adapter_phase2_latest_ep4.pt"

MOSHI_PRESETS = {
    "q8":   {"repo": "kyutai/moshiko-pytorch-q8",   "dtype": torch.bfloat16},
    "bf16": {"repo": "kyutai/moshiko-pytorch-bf16",  "dtype": torch.bfloat16},
    "fp32": {"repo": "kyutai/moshiko-pytorch-bf16",  "dtype": torch.float32},
}

# Buffer latency constants (ms) — how long we wait before starting playback
# so the client has enough frames+audio to maintain smooth streaming
BUFFER_LATENCY = {
    "lite": 1250,   # 960ms accumulation + ~350ms gen + 90ms safety
    "pro":  1800,   # 1120ms accumulation + ~600ms gen + 80ms safety
}


# ────────────────────────────────────────────────────────────────────────────
#  Lazy Imports (avoid circular / heavy startup)
# ────────────────────────────────────────────────────────────────────────────
LMGen = None
CheckpointInfo = None
_flashhead_imports = None


def _ensure_moshi_imports():
    global LMGen, CheckpointInfo
    if LMGen is not None:
        return
    moshi_pkg = os.path.join(ROOT, "moshi", "moshi")
    if moshi_pkg not in sys.path:
        sys.path.insert(0, moshi_pkg)
    from moshi.models import LMGen as _LMGen
    from moshi.models.loaders import CheckpointInfo as _CI
    LMGen = _LMGen
    CheckpointInfo = _CI
    print("[Moshi] Imports OK.")


def _ensure_flashhead_imports():
    global _flashhead_imports
    if _flashhead_imports is not None:
        return _flashhead_imports

    if SOULX_ROOT not in sys.path:
        sys.path.insert(0, SOULX_ROOT)

    prev_cwd = os.getcwd()
    try:
        os.chdir(SOULX_ROOT)
        from flash_head.inference import (
            get_pipeline,
            get_base_data,
            get_infer_params,
            run_pipeline,
        )
    finally:
        os.chdir(prev_cwd)

    _flashhead_imports = {
        "get_pipeline": get_pipeline,
        "get_base_data": get_base_data,
        "get_infer_params": get_infer_params,
        "run_pipeline": run_pipeline,
    }
    print("[FlashHead] Imports OK.")
    return _flashhead_imports


# ────────────────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────────────────

def _queue_put_latest(q: queue.Queue, item):
    """Bounded insert — drops oldest when full."""
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def _resample_audio_np(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float audio with soxr if available, else linear interp."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or src_sr == dst_sr:
        return audio
    try:
        import soxr
        return np.asarray(soxr.resample(audio, src_sr, dst_sr), dtype=np.float32)
    except Exception:
        dst_len = max(1, int(round(audio.size * float(dst_sr) / float(src_sr))))
        x_old = np.linspace(0, 1, num=audio.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0, 1, num=dst_len, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
#  MoshiTokenDeque — 8-second sliding buffer of Moshi transformer_out tokens
# ────────────────────────────────────────────────────────────────────────────

class MoshiTokenDeque:
    """
    Thread-safe sliding window of Moshi tokens [DEQUE_SIZE, MOSHI_DIM].
    Initialized with zeros (silence). Moshi thread pushes; FlashHead reads.

    The deque holds exactly 100 tokens (8s at 12.5Hz).
    After linear interpolation 100→200, this maps to 200 frame-aligned
    tokens at 25fps, matching FlashHead's expected audio context.

    This mirrors the standard FlashHead audio deque:
      Standard: deque([0.0] * 128000, maxlen=128000)  → 8s of raw audio
      Ours:     deque([zeros] * 100, maxlen=100)       → 8s of Moshi tokens
    """

    def __init__(self, size: int = DEQUE_SIZE, dim: int = MOSHI_DIM):
        self._lock = threading.Lock()
        self._size = size
        self._dim = dim
        self._buffer = torch.zeros(size, dim)  # CPU
        self._total_pushed = 0

    def push(self, token: torch.Tensor):
        """Push one token. token shape: [1, 1, D] or [D]."""
        tok = token.detach().cpu().reshape(1, self._dim)
        with self._lock:
            self._buffer = torch.cat([self._buffer[1:], tok], dim=0)
            self._total_pushed += 1

    def push_batch(self, tokens: torch.Tensor):
        """Push N tokens at once. tokens shape: [N, D]."""
        tokens = tokens.detach().cpu()
        n = tokens.shape[0]
        with self._lock:
            self._buffer = torch.cat([self._buffer[n:], tokens], dim=0)
            self._total_pushed += n

    def snapshot(self) -> torch.Tensor:
        """Return a copy of the full buffer [DEQUE_SIZE, MOSHI_DIM]."""
        with self._lock:
            return self._buffer.clone()

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed


# ────────────────────────────────────────────────────────────────────────────
#  MoshiToWav2VecAdapter — 12-layer transformer adapter replacing Wav2Vec2
# ────────────────────────────────────────────────────────────────────────────

class MoshiToWav2VecAdapter(nn.Module):
    """
    Transformer-based adapter that replicates the Wav2Vec2-base encoder
    architecture to produce semantically equivalent multi-layer hidden states.

    ━━━ WHY THIS ARCHITECTURE? ━━━

    FlashHead's AudioProjModel was trained on Wav2Vec2 hidden states stacked
    from 12 transformer layers. Each layer captures a different level of
    audio abstraction:
      - Layers 1–3:  Low-level acoustic features (phonemes, pitch, energy)
      - Layers 4–8:  Mid-level features (syllables, prosody, rhythm)
      - Layers 9–12: High-level features (semantic, linguistic, speaker)

    A simple Linear projection cannot replicate this multi-level structure.
    This adapter uses the same 12-layer transformer architecture so that
    each layer learns to produce outputs at the corresponding abstraction
    level, exactly matching what AudioProjModel expects.

    ━━━ WHAT IT REPLACES (from standard FlashHead) ━━━

    Standard Wav2Vec2 pipeline:
      Raw audio → CNN feature extractor (7 conv layers, → 512-dim)
                → Linear interpolation (→ 200 tokens)
                → Feature projection (512 → 768)
                → 12 transformer layers → 12 × (1, 200, 768)
                → Stack → (200, 12, 768)

    Our adapter:
      Moshi tokens → Linear interpolation (100 → 200, done BEFORE adapter)
                   → Feature projection (4096 → 768)
                   → 12 transformer layers → 12 × (1, 200, 768)
                   → Stack → (200, 12, 768)  ← SAME OUTPUT FORMAT

    ━━━ PARAMETER COUNT ━━━

    - Feature projection:  4096 × 768 + norms  ≈   3.2M
    - Conv position:       768 × 48 × 128      ≈   4.7M
    - 12 Transformer layers: 12 × ~7.1M         ≈  85.0M
    - Total:                                    ≈  93.0M  (vs Wav2Vec2-base: 95M)

    ━━━ CHECKPOINT ━━━

    Weight Loading:
      1. If <adapter_ckpt_dir>/moshi_to_flashhead_adapter.pt exists → load
      2. Otherwise → random init (train before use for meaningful lip-sync)

    Input:  [N, 4096]  or  [B, N, 4096]
    Output: [N, 12, 768]  or  [B, N, 12, 768]
    """

    def __init__(
        self,
        moshi_dim: int = MOSHI_DIM,
        hidden_dim: int = WAV2VEC_DIM,     # 768 (same as Wav2Vec2-base)
        num_layers: int = WAV2VEC_LAYERS,  # 12  (same as Wav2Vec2-base)
        num_heads: int = 12,               # Wav2Vec2-base: 12 heads
        ffn_dim: int = 3072,               # Wav2Vec2-base: 4× hidden = 3072
        dropout: float = 0.0,              # no dropout at inference
        conv_pos_kernel: int = 128,        # Wav2Vec2: num_conv_pos_embeddings
        conv_pos_groups: int = 16,         # Wav2Vec2: num_conv_pos_embedding_groups
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # ── Feature Projection ──────────────────────────────────────────
        # Replaces Wav2Vec2's 7-layer CNN feature extractor (audio → 512)
        # + feature_projection (512 → 768).
        # We project from Moshi's 4096-dim directly to 768.
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(moshi_dim),
            nn.Linear(moshi_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Convolutional Position Encoding ─────────────────────────────
        # Wav2Vec2 uses Conv1d for position encoding instead of sinusoidal.
        # This captures local temporal structure with a large receptive field.
        self.conv_pos = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=conv_pos_kernel,
            padding=conv_pos_kernel // 2,
            groups=conv_pos_groups,
        )
        self.conv_pos_gelu = nn.GELU()

        # ── Pre-Transformer LayerNorm ───────────────────────────────────
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # ── 12 Transformer Encoder Layers ───────────────────────────────
        # Each layer produces a distinct hidden state at a different
        # abstraction level, matching Wav2Vec2's 12-layer transformer.
        # We DON'T use nn.TransformerEncoder because we need to collect
        # the intermediate hidden states from EACH layer.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-norm, same as Wav2Vec2
            )
            for _ in range(num_layers)
        ])

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init matching typical transformer initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking Wav2Vec2-base encoder.

        Args:
            x: [N, 4096] or [B, N, 4096] — Moshi tokens after interpolation

        Returns:
            [N, 12, 768] or [B, N, 12, 768] — stacked hidden states from
            all 12 transformer layers, matching Wav2Vec2 output format.

        Data flow:
            Moshi tokens [N, 4096]
              → Feature projection → [N, 768]
              → Convolutional position encoding (adds temporal info)
              → LayerNorm + Dropout
              → Transformer Layer 1  → hidden_state_1 [N, 768]
              → Transformer Layer 2  → hidden_state_2 [N, 768]
              → ...                    (each layer: different abstraction)
              → Transformer Layer 12 → hidden_state_12 [N, 768]
              → Stack [h1..h12] → [N, 12, 768]
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, 4096]
            squeeze = True

        B, N, _ = x.shape

        # Step 1: Feature projection  4096 → 768
        x = self.feature_projection(x)  # [B, N, 768]

        # Step 2: Convolutional position encoding
        x_conv = x.transpose(1, 2)                  # [B, 768, N]
        x_conv = self.conv_pos(x_conv)               # [B, 768, N + k//2]
        x_conv = x_conv[:, :, :N]                    # Trim to [B, 768, N]
        x_conv = self.conv_pos_gelu(x_conv)
        x = x + x_conv.transpose(1, 2)              # [B, N, 768]

        # Step 3: LayerNorm + Dropout before transformer
        x = self.layer_norm(x)
        x = self.input_dropout(x)

        # Step 4: Forward through 12 transformer layers
        # Collect hidden state from EACH layer (the whole point of this design)
        hidden_states = []
        for layer in self.transformer_layers:
            x = layer(x)                             # [B, N, 768]
            hidden_states.append(x)

        # Step 5: Stack all 12 layer outputs → [B, N, 12, 768]
        output = torch.stack(hidden_states, dim=2)   # [B, N, 12, 768]

        if squeeze:
            output = output.squeeze(0)               # [N, 12, 768]

        return output

    def save_checkpoint(self, ckpt_dir: str):
        """Save adapter weights to checkpoint directory."""
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, ADAPTER_FILENAME)
        torch.save(self.state_dict(), path)
        print(f"[Adapter] Saved checkpoint → {path}")
        return path

    def load_checkpoint(self, ckpt_dir: str) -> bool:
        """
        Load adapter weights from checkpoint directory.
        Returns True if checkpoint was found and loaded, False otherwise.
        """
        path = os.path.join(ckpt_dir, ADAPTER_FILENAME)
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        print(f"[Adapter] Loaded fine-tuned checkpoint ← {path}")
        return True


# ────────────────────────────────────────────────────────────────────────────
#  Audio embedding from tokens (replaces standard FlashHead's Wav2Vec2 path)
# ────────────────────────────────────────────────────────────────────────────

def get_audio_embedding_from_tokens(
    adapter: MoshiToWav2VecAdapter,
    deque_snapshot: torch.Tensor,
    frame_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Replaces FlashHead's standard get_audio_embedding() for the token path.

    Mirrors the exact data flow of the standard Wav2Vec2 pipeline:

      Standard FlashHead (preprocess_audio + get_audio_embedding):
        Raw Audio (128000,) → Wav2Vec2FeatureExtractor (normalize)
        → CNN feature_extractor → (1, ~399, 512)
        → linear_interpolation(seq_len=200) → (1, 200, 512)
        → feature_projection(512→768) → (1, 200, 768)
        → 12 transformer encoder layers (with output_hidden_states=True)
        → torch.stack(hidden_states[1:], dim=1).squeeze(0) → (12, 200, 768)
        → rearrange("b s d -> s b d") → (200, 12, 768)
        → center-index [167:200] ±2 → (1, 33, 5, 12, 768)

      Our Token Path:
        Moshi tokens (100, 4096) → F.interpolate(→200, 4096)
        → adapter.feature_projection(4096→768) → (1, 200, 768)
        → 12 transformer layers → stack → (200, 12, 768)  ← SHAPE MATCH ✓
        → center-index [167:200] ±2 → (1, 33, 5, 12, 768) ← IDENTICAL ✓

    Then inside DiT.forward() (both paths feed into the same frozen code):
      context (1, 33, 5, 12, 768) → AudioProjModel → (1, 9, 32, 1536)
      → DiT cross-attention → denoised latent → VAE decode → video frames

    Args:
        adapter: The trained MoshiToWav2VecAdapter
        deque_snapshot: [100, 4096] token buffer snapshot (CPU)
        frame_num: 33 (total frames per FlashHead chunk)
        device: target device (cuda)
        dtype: target dtype (bfloat16)

    Returns:
        [1, 33, 5, 12, 768] — exact same format as standard FlashHead
    """
    # deque_snapshot: [100, 4096], on CPU
    tokens = deque_snapshot.unsqueeze(0)  # [1, 100, 4096]

    # ─── Step 1: Linear interpolation 100 → 200 ───────────────────────
    # Matches FlashHead's linear_interpolation(extract_features, seq_len=200)
    # Standard FlashHead: (1, ~399, 512) → interp → (1, 200, 512)
    # Our path:           (1,  100, 4096) → interp → (1, 200, 4096)
    tokens = tokens.transpose(1, 2)  # [1, 4096, 100]
    tokens = F.interpolate(
        tokens.float(), size=INTERP_TARGET, mode="linear", align_corners=True
    )  # [1, 4096, 200]
    tokens = tokens.transpose(1, 2)  # [1, 200, 4096]

    # ─── Step 2: Adapter forward (replaces Wav2Vec2 encoder) ──────────
    # Standard: feature_projection(512→768) → 12 transformer layers → stack
    # Ours:     feature_projection(4096→768) → 12 transformer layers → stack
    tokens = tokens.to(device)
    audio_emb = adapter(tokens)  # [1, 200, 12, 768]

    # Ensure [200, 12, 768] shape (adapter auto-squeezes if batch=1)
    if audio_emb.dim() == 4:
        audio_emb = audio_emb.squeeze(0)  # [200, 12, 768]

    assert audio_emb.shape == (INTERP_TARGET, WAV2VEC_LAYERS, WAV2VEC_DIM), \
        f"Expected ({INTERP_TARGET}, {WAV2VEC_LAYERS}, {WAV2VEC_DIM}), got {audio_emb.shape}"

    # ─── Step 3: Center-index gathering (IDENTICAL to standard FlashHead) ─
    # From flash_head/inference.py get_audio_embedding():
    #   indices = (torch.arange(2*2+1) - 2) * 1 = [-2, -1, 0, 1, 2]
    #   center_indices = torch.arange(start, end).unsqueeze(1) + indices.unsqueeze(0)
    #   center_indices = torch.clamp(center_indices, min=0, max=end-1)
    #   audio_embedding = audio_emb[center_indices][None,...].contiguous()
    audio_start_idx = INTERP_TARGET - frame_num  # 200 - 33 = 167
    audio_end_idx = INTERP_TARGET                # 200

    indices = (torch.arange(5, device=device) - 2)  # [-2, -1, 0, 1, 2]
    center_indices = (
        torch.arange(audio_start_idx, audio_end_idx, device=device).unsqueeze(1)
        + indices.unsqueeze(0)
    )  # [33, 5]
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)

    audio_embedding = audio_emb[center_indices]  # [33, 5, 12, 768]
    audio_embedding = audio_embedding[None, ...].contiguous()  # [1, 33, 5, 12, 768]

    assert audio_embedding.shape == (1, frame_num, 5, WAV2VEC_LAYERS, WAV2VEC_DIM), \
        f"Expected (1, {frame_num}, 5, {WAV2VEC_LAYERS}, {WAV2VEC_DIM}), got {audio_embedding.shape}"

    return audio_embedding.to(dtype=dtype)


# ────────────────────────────────────────────────────────────────────────────
#  MoshiEngine — Moshi Helium S2S model
# ────────────────────────────────────────────────────────────────────────────

class MoshiEngine:
    def __init__(self, precision="bf16", repo_override=None, device=DEVICE):
        preset = MOSHI_PRESETS.get(precision, MOSHI_PRESETS["bf16"])
        self.precision = precision
        self.hf_repo = repo_override or preset["repo"]
        self.dtype = preset["dtype"]
        self.device = device

        self.mimi = None
        self.lm = None
        self.lm_gen = None
        self.text_tokenizer = None
        self.loaded = False

        self._text_lock = threading.Lock()
        self._latest_text = ""

    def get_latest_text(self) -> str:
        with self._text_lock:
            return self._latest_text

    def _set_latest_text(self, text: str):
        if not text:
            return
        with self._text_lock:
            self._latest_text = text

    def _decode_text_piece(self, tokens: torch.Tensor) -> str:
        """Best-effort text token decoding from Moshi output."""
        tok = self.text_tokenizer
        if tok is None or tokens is None:
            return ""
        try:
            token_id = int(tokens[0, 0, 0].item())
        except Exception:
            return ""
        if token_id <= 0:
            return ""

        for fn in [
            lambda: tok.decode([token_id]),
            lambda: tok.decode(token_id),
            lambda: tok.id_to_piece(token_id),
            lambda: tok.convert_ids_to_tokens([token_id])[0],
        ]:
            try:
                piece = fn()
                if piece is not None:
                    piece = str(piece).strip()
                    if piece and piece.lower() != "<pad>":
                        return piece
            except Exception:
                continue
        return ""

    def load(self):
        if self.loaded:
            return
        _ensure_moshi_imports()

        print("[Moshi] Loading checkpoint info...")
        info = CheckpointInfo.from_hf_repo(self.hf_repo)

        print("[Moshi] Loading Mimi codec...")
        self.mimi = info.get_mimi(device=self.device)

        print(f"[Moshi] Loading LM ({self.precision}, {self.dtype})...")
        self.lm = info.get_moshi(device=self.device, dtype=self.dtype)
        self.lm.eval()

        if self.precision == "q8":
            fixed = 0
            for module in self.lm.modules():
                if hasattr(module, "weight_scb") and module.weight_scb.dtype != torch.float32:
                    module.weight_scb.data = module.weight_scb.data.float()
                    fixed += 1
            if fixed:
                print(f"[Moshi] Fixed {fixed} QLinear scale buffers → float32")

        self.lm_gen = LMGen(self.lm)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        try:
            self.text_tokenizer = info.get_text_tokenizer()
            print("[Moshi] Text tokenizer loaded.")
        except Exception as e:
            print(f"[Moshi] Text tokenizer not available: {e}")
            self.text_tokenizer = None

        self.loaded = True
        print(
            f"[Moshi] Ready. dim={self.lm.dim}, dep_q={self.lm.dep_q}, "
            f"frame_size={self.frame_size}"
        )

    @torch.no_grad()
    def run_streaming(
        self,
        token_queue: queue.Queue,
        mic_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        """
        Main Moshi streaming loop. Runs in a dedicated thread.

        Produces (token_id, transformer_out_cpu, audio_pcm_np, arrival_ts) tuples
        and pushes them to token_queue at 12.5 Hz.

        Each token corresponds to 80ms of audio and will map to exactly
        2 video frames after the 100→200 interpolation.
        """
        assert self.loaded, "Call load() first"
        print("[MoshiThread] Starting streaming...")

        # Reset streaming states — walk lm_gen (NOT lm_gen.lm_model) so the
        # parent StreamingContainer's own _streaming_state also gets cleared.
        # Without this, the 2nd session trips "is already streaming!" assert.
        for root in (self.mimi, self.lm_gen):
            for _, mod in root.named_modules():
                if getattr(mod, "_streaming_state", None) is not None:
                    mod._streaming_state = None

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        cuda_stream = torch.cuda.Stream() if self.device == "cuda" else None
        step = 0
        first_frame = True
        token_id = 0

        try:
            while not stop_event.is_set():
                try:
                    mic_chunk = mic_queue.get(timeout=0.2)
                except queue.Empty:
                    mic_chunk = np.zeros(self.frame_size, dtype=np.float32)

                pcm = (
                    torch.from_numpy(mic_chunk)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                )

                ctx = torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext()
                with ctx:
                    codes = self.mimi.encode(pcm)

                    if first_frame:
                        self.lm_gen._step(codes)
                        first_frame = False
                        step += 1
                        del codes, pcm
                        continue

                    result = self.lm_gen._step(codes)
                    if result is None:
                        step += 1
                        del codes, pcm
                        continue

                    tokens, transformer_out = result

                    # Extract transformer_out for the token bridge
                    # This is the 4096-dim hidden state from Moshi's main transformer
                    token_id += 1
                    t_out_cpu = transformer_out.detach().cpu()  # [1, 1, 4096]
                    arrival_ts = time.perf_counter()

                    # Decode audio through Mimi
                    audio_pcm = None
                    if self.lm.dep_q > 0:
                        out_pcm = self.mimi.decode(tokens[:, 1:])
                        audio_pcm = out_pcm[0, 0].detach().cpu().numpy()
                        del out_pcm

                    # Best-effort text decoding
                    text_piece = self._decode_text_piece(tokens)
                    if text_piece:
                        self._set_latest_text(text_piece)

                    # Push (token_id, transformer_out, audio_pcm, arrival_ts)
                    _queue_put_latest(
                        token_queue,
                        (token_id, t_out_cpu, audio_pcm, arrival_ts),
                    )

                    del tokens, transformer_out, codes, pcm, t_out_cpu

                step += 1
                if step % 500 == 0:
                    print(
                        f"  [MoshiThread] step={step}, token_id={token_id}, "
                        f"token_q={token_queue.qsize()}"
                    )

        except Exception as e:
            print(f"[MoshiThread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[MoshiThread] Exiting.")


# ────────────────────────────────────────────────────────────────────────────
#  FlashHeadTokenEngine — Generates video chunks from Moshi tokens
# ────────────────────────────────────────────────────────────────────────────

class FlashHeadTokenEngine:
    """
    Runs the frozen SoulX-FlashHead pipeline with our adapter replacing Wav2Vec2.

    The pipeline is:
      Adapter output (1, 33, 5, 12, 768)
      → AudioProjModel (FROZEN): (1, 33, 5, 12, 768) → (1, 9, 32, 1536)
      → DiT denoising (FROZEN): 4 steps with audio cross-attention
      → VAE Decode (FROZEN): latent → 33 pixel frames (512×512)
      → Motion frame carry-over (FROZEN): last N frames → encode → next chunk
      → Discard overlap → 24/28 new frames

    All frozen components are loaded from standard SoulX-FlashHead checkpoints.
    Only the adapter is loaded from ./checkpoints/
    """

    def __init__(
        self,
        ckpt_dir: str,
        wav2vec_dir: str,
        model_type: str,
        ref_image: str,
        adapter_ckpt_dir: str = DEFAULT_ADAPTER_CKPT_DIR,
        base_seed: int = 42,
        device: str = DEVICE,
    ):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.wav2vec_dir = os.path.abspath(wav2vec_dir)
        self.model_type = model_type
        self.ref_image = os.path.abspath(ref_image)
        self.adapter_ckpt_dir = os.path.abspath(adapter_ckpt_dir)
        self.base_seed = int(base_seed)
        self.device = device

        self.pipeline = None
        self.adapter = None
        self.infer_params = None
        self.frame_num = None
        self.motion_frames_num = None
        self.slice_len = None
        self.tokens_per_chunk = None  # Moshi tokens needed per chunk
        self.tgt_fps = None
        self._chunk_idx = 0

    def load(self):
        if self.pipeline is not None:
            return

        fh = _ensure_flashhead_imports()

        print("[FlashHead] Loading pipeline...")
        self.pipeline = fh["get_pipeline"](
            world_size=1,
            ckpt_dir=self.ckpt_dir,
            wav2vec_dir=self.wav2vec_dir,
            model_type=self.model_type,
        )
        self.infer_params = fh["get_infer_params"]()
        self.frame_num = int(self.infer_params["frame_num"])          # 33
        self.motion_frames_num = int(self.infer_params["motion_frames_num"])
        self.tgt_fps = int(self.infer_params["tgt_fps"])             # 25
        self.slice_len = self.frame_num - self.motion_frames_num

        # Moshi tokens needed per chunk:
        #   slice_len frames × 40ms / 80ms per token = slice_len / 2
        # Lite (vae_stride[0]=8):  motion=9, slice=24 → 12 tokens (960ms)
        # Pro  (vae_stride[0]=4):  motion=5, slice=28 → 14 tokens (1120ms)
        self.tokens_per_chunk = self.slice_len // 2

        vae_stride_0 = self.pipeline.config.vae_stride[0]

        print(
            f"[FlashHead] model={self.model_type}, vae_stride[0]={vae_stride_0}, "
            f"frame_num={self.frame_num}, motion={self.motion_frames_num}, "
            f"slice_len={self.slice_len}, tokens_per_chunk={self.tokens_per_chunk}"
        )

        # ── Create the adapter (the ONLY trainable component) ────────
        self.adapter = MoshiToWav2VecAdapter(
            moshi_dim=MOSHI_DIM,
            hidden_dim=WAV2VEC_DIM,      # 768
            num_layers=WAV2VEC_LAYERS,   # 12
            num_heads=12,
            ffn_dim=3072,
        )
        loaded = self.adapter.load_checkpoint(self.adapter_ckpt_dir)
        if not loaded:
            print(
                f"[Adapter] No checkpoint found at {self.adapter_ckpt_dir}/"
                f"{ADAPTER_FILENAME}. Using random initialization "
                f"(neutral motion until fine-tuned)."
            )
        # self.adapter = self.adapter.to(self.device).to(self.pipeline.param_dtype).eval()
        self.adapter = self.adapter.to(self.device).eval()
        

        # Set reference image
        self.set_reference(self.ref_image)
        self._chunk_idx = 0

        # Print parameter summary
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        total_pipe_params = sum(p.numel() for p in self.pipeline.model.parameters())
        print(
            f"[FlashHead] Ready. tokens_per_chunk={self.tokens_per_chunk}\n"
            f"  Adapter params (trainable):    {adapter_params:,} "
            f"({adapter_params * 4 / 1e6:.1f} MB @ fp32)\n"
            f"  DiT+AudioProj params (frozen): {total_pipe_params:,} "
            f"({total_pipe_params * 2 / 1e9:.2f} GB @ bf16)"
        )
        print(f"  Adapter checkpoint: {'LOADED ✓' if loaded else 'NOT FOUND (random init)'}")

    def set_reference(self, ref_image_path: str):
        fh = _ensure_flashhead_imports()
        ref_image_path = os.path.abspath(ref_image_path)
        if not os.path.exists(ref_image_path):
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
        self.ref_image = ref_image_path
        fh["get_base_data"](
            self.pipeline,
            cond_image_path_or_dir=self.ref_image,
            base_seed=self.base_seed,
            use_face_crop=False,
        )
        self.pipeline.reset_person_name(self.pipeline.person_name)
        self._chunk_idx = 0

    @torch.inference_mode()
    def warmup(self, n_chunks: int = 2):
        """
        Run warmup chunks with silence to trigger torch.compile JIT compilation.

        FlashHead uses torch.compile for the DiT model and VAE, which means
        the first forward pass is significantly slower due to compilation.
        Running warmup chunks ensures the real-time generation starts fast.
        """
        print(f"[FlashHead] Warmup: {n_chunks} chunks...")
        fh = _ensure_flashhead_imports()
        silence_deque = torch.zeros(DEQUE_SIZE, MOSHI_DIM)

        for i in range(n_chunks):
            t0 = time.perf_counter()
            audio_emb = get_audio_embedding_from_tokens(
                self.adapter,
                silence_deque,
                self.frame_num,
                device=torch.device(self.device),
                dtype=self.pipeline.param_dtype,
            )
            _ = fh["run_pipeline"](self.pipeline, audio_emb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [FlashHead] warmup chunk {i + 1}/{n_chunks} done ({elapsed:.0f}ms)")

        # Reset pipeline state after warmup
        self.pipeline.reset_person_name(self.pipeline.person_name)
        self._chunk_idx = 0
        print("[FlashHead] Warmup complete.")

    @torch.inference_mode()
    def generate_chunk(self, deque_snapshot: torch.Tensor) -> np.ndarray:
        """
        Generate a video chunk from the current deque snapshot.

        Args:
            deque_snapshot: [100, 4096] token buffer snapshot

        Returns:
            np.ndarray of uint8 frames [N, H, W, 3]
            N = slice_len (24 lite / 28 pro) for stream mode
        """
        fh = _ensure_flashhead_imports()

        audio_emb = get_audio_embedding_from_tokens(
            self.adapter,
            deque_snapshot,
            self.frame_num,
            device=torch.device(self.device),
            dtype=self.pipeline.param_dtype,
        )

        video = fh["run_pipeline"](self.pipeline, audio_emb)

        # Stream mode: always discard motion overlap frames
        # (standard FlashHead stream mode does the same)
        video = video[self.motion_frames_num:]

        self._chunk_idx += 1
        return video.detach().cpu().numpy().astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
#  ChunkBundle — groups audio + video for A/V sync
# ────────────────────────────────────────────────────────────────────────────

ChunkBundle = namedtuple(
    "ChunkBundle",
    ["chunk_id", "audio_pcm_list", "video_frames", "gen_time_ms",
     "start_token_id", "end_token_id",
     "first_token_ts", "last_token_ts", "frames_ready_ts"],
)


# ────────────────────────────────────────────────────────────────────────────
#  ConversationSession — WebSocket session managing all threads
# ────────────────────────────────────────────────────────────────────────────

class ConversationSession:
    """
    Manages a single conversation session between browser client and server.

    Thread architecture:
      [MoshiThread]  → token_queue → [FlashHeadThread] → dispatch_queue → [Dispatcher async]
                                                                          ↓
      [Receiver async] ← WebSocket ←───────────────────────────────────── ↓

    A/V Sync Strategy:
      1. FlashHead thread accumulates tokens_per_chunk (12/14) Moshi tokens
      2. These tokens represent exactly 960ms/1120ms of audio
      3. Generate video frames for that exact duration (24/28 frames @ 25fps)
      4. Bundle audio PCM + video frames into a ChunkBundle
      5. Dispatch both together — client plays them in sync
      6. Initial buffer latency (1300-1800ms) ensures smooth startup

    K-th Moshi token sync:
      Token k produces 80ms audio. After interpolation, token k maps to
      frames (2k-1) and (2k) in the 25fps video. The bundled dispatch
      ensures audio from tokens [start..end] plays alongside the
      corresponding video frames.
    """

    def __init__(self, websocket: WebSocket, moshi: MoshiEngine,
                 flash: FlashHeadTokenEngine, args):
        self.ws = websocket
        self.moshi = moshi
        self.flash = flash
        self.args = args
        self.show_sync = getattr(args, 'show_sync', False)

        self.stop_event = threading.Event()
        self.token_queue = queue.Queue(maxsize=TOKEN_QUEUE_MAXSIZE)
        self.mic_queue = queue.Queue(maxsize=MIC_QUEUE_MAXSIZE)
        self.dispatch_queue = queue.Queue(maxsize=DISPATCH_QUEUE_MAXSIZE)

        self.token_deque = MoshiTokenDeque(DEQUE_SIZE, MOSHI_DIM)

        self.client_sr = 48000
        self._mic_resample_buf = np.zeros(0, dtype=np.float32)

        self.moshi_thread = None
        self.flashhead_thread = None

        # Telemetry
        self._total_chunks = 0
        self._session_t0 = 0.0
        self._total_frames_sent = 0
        self._avg_gen_ms = 0.0

        # A/V sync gap tracking:
        #   gap = audio_token_arrival_time - frames_ready_time
        #   positive → audio was ready first (good)
        #   negative → frames arrived first
        #   zero → both arrived at same time
        #
        # "first_gap": gap for the first token of the chunk vs frames
        # "last_gap":  gap for the last token of the chunk vs frames
        # (all frames arrive at same time since they're generated as batch)
        self._sync_first_gap_sum = 0.0
        self._sync_last_gap_sum = 0.0
        self._sync_count = 0

    @property
    def avg_first_gap_ms(self) -> float:
        """Average gap (ms) between first token arrival and frames ready."""
        return (self._sync_first_gap_sum / max(1, self._sync_count)) * 1000

    @property
    def avg_last_gap_ms(self) -> float:
        """Average gap (ms) between last token arrival and frames ready."""
        return (self._sync_last_gap_sum / max(1, self._sync_count)) * 1000

    # ── Microphone Ingest ──────────────────────────────────────────────

    def _push_client_audio(self, payload: bytes):
        """Convert browser PCM Int16 to Moshi-sized float32 chunks."""
        if not payload:
            return
        audio_i16 = np.frombuffer(payload, dtype=np.int16)
        if audio_i16.size == 0:
            return
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        audio_24k = _resample_audio_np(audio_f32, int(self.client_sr), MOSHI_SR)

        if self._mic_resample_buf.size:
            audio_24k = np.concatenate([self._mic_resample_buf, audio_24k])

        while audio_24k.size >= MOSHI_FRAME_SAMPLES:
            chunk = audio_24k[:MOSHI_FRAME_SAMPLES]
            _queue_put_latest(self.mic_queue, chunk)
            audio_24k = audio_24k[MOSHI_FRAME_SAMPLES:]

        self._mic_resample_buf = audio_24k

    # ── Encoding Helpers ───────────────────────────────────────────────

    @staticmethod
    def _encode_jpeg_b64(frame_rgb: np.ndarray) -> str | None:
        """Encode RGB frame to JPEG base64 for WebSocket transport."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    @staticmethod
    def _encode_audio_b64(audio_24k: np.ndarray) -> str:
        """Encode float32 audio to PCM Int16 base64."""
        pcm = np.clip(audio_24k, -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        return base64.b64encode(pcm_i16.tobytes()).decode("ascii")

    # ── FlashHead Video Generation Thread ──────────────────────────────

    def _flashhead_loop(self):
        """
        Accumulates Moshi tokens, generates video chunks, bundles with audio.
        Runs in a dedicated thread.

        Flow per chunk:
          1. Wait for tokens_per_chunk (12/14) tokens from Moshi
          2. Push all to the 100-token sliding deque
          3. Take a snapshot → [100, 4096]
          4. Generate video via adapter → FlashHead → 24/28 frames
          5. Create ChunkBundle with audio PCM + video frames + timestamps
          6. Push to dispatch queue for WebSocket sending
        """
        tokens_needed = self.flash.tokens_per_chunk
        chunk_id = 0

        print(
            f"[FlashHeadThread] Starting. tokens_per_chunk={tokens_needed}, "
            f"model={self.flash.model_type}"
        )

        try:
            while not self.stop_event.is_set():
                # ── Accumulate tokens_per_chunk tokens from Moshi ──────
                token_batch = []     # list of [4096] tensors
                audio_batch = []     # list of (1920,) numpy arrays
                start_tid = None
                end_tid = None
                first_token_ts = None
                last_token_ts = None

                for i in range(tokens_needed):
                    if self.stop_event.is_set():
                        return
                    try:
                        token_id, t_out, audio_pcm, arrival_ts = self.token_queue.get(timeout=2.0)
                    except queue.Empty:
                        continue

                    token_batch.append(t_out.reshape(MOSHI_DIM))
                    if audio_pcm is not None:
                        audio_batch.append(audio_pcm)
                    if start_tid is None:
                        start_tid = token_id
                        first_token_ts = arrival_ts
                    end_tid = token_id
                    last_token_ts = arrival_ts

                if len(token_batch) < tokens_needed:
                    continue  # Not enough tokens, retry

                # ── Push accumulated tokens to the deque ──────────────
                batch_tensor = torch.stack(token_batch, dim=0)  # [N, 4096]
                self.token_deque.push_batch(batch_tensor)

                # ── Take snapshot and generate ────────────────────────
                t0 = time.perf_counter()
                snapshot = self.token_deque.snapshot()
                frames = self.flash.generate_chunk(snapshot)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_ms = (time.perf_counter() - t0) * 1000
                frames_ready_ts = time.perf_counter()

                # Update rolling average gen time
                if self._avg_gen_ms == 0:
                    self._avg_gen_ms = gen_ms
                else:
                    self._avg_gen_ms = 0.8 * self._avg_gen_ms + 0.2 * gen_ms

                # ── Track A/V sync gap ────────────────────────────────
                # gap = audio_arrived - frames_ready
                # positive → audio was ready before frames (expected)
                # negative → frames arrived before that token's audio
                if first_token_ts is not None and last_token_ts is not None:
                    first_gap = first_token_ts - frames_ready_ts
                    last_gap = last_token_ts - frames_ready_ts
                    self._sync_first_gap_sum += first_gap
                    self._sync_last_gap_sum += last_gap
                    self._sync_count += 1

                # ── Bundle audio + video for A/V sync ─────────────────
                bundle = ChunkBundle(
                    chunk_id=chunk_id,
                    audio_pcm_list=audio_batch,
                    video_frames=frames,
                    gen_time_ms=gen_ms,
                    start_token_id=start_tid or 0,
                    end_token_id=end_tid or 0,
                    first_token_ts=first_token_ts or 0.0,
                    last_token_ts=last_token_ts or 0.0,
                    frames_ready_ts=frames_ready_ts,
                )
                _queue_put_latest(self.dispatch_queue, bundle)

                chunk_id += 1
                self._total_chunks = chunk_id

                # ── Terminal logging (minimal, periodic) ──────────────
                if chunk_id <= 3 or chunk_id % 10 == 0:
                    sync_info = ""
                    if self.show_sync and self._sync_count > 0:
                        sync_info = (
                            f" sync_1st={self.avg_first_gap_ms:+.0f}ms"
                            f" sync_last={self.avg_last_gap_ms:+.0f}ms"
                        )
                    print(
                        f"  [Chunk {chunk_id:>4d}] gen={gen_ms:.0f}ms "
                        f"avg={self._avg_gen_ms:.0f}ms "
                        f"frames={frames.shape[0]} "
                        f"tokens={start_tid}-{end_tid} "
                        f"deque={self.token_deque.total_pushed}"
                        f"{sync_info}"
                    )

        except Exception as e:
            print(f"[FlashHeadThread] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[FlashHeadThread] Exiting.")

    # ── Async WebSocket Tasks ──────────────────────────────────────────

    async def _receiver(self):
        """
        Receives mic audio and control messages from the browser.
        Binary messages are raw PCM Int16 audio.
        JSON messages are control commands (start, stop).
        """
        while not self.stop_event.is_set():
            msg = await self.ws.receive()

            text = msg.get("text")
            data = msg.get("bytes")
            mtype = msg.get("type")

            if mtype == "websocket.disconnect":
                raise WebSocketDisconnect()

            if text is not None:
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                t = obj.get("type")
                if t == "start":
                    self.client_sr = int(obj.get("sample_rate", self.client_sr))
                    print(f"[Session] Client sample_rate={self.client_sr}")
                elif t == "stop":
                    self.stop_event.set()
                    return
                continue

            if data is not None:
                self._push_client_audio(data)

    async def _dispatcher(self):
        """
        Dispatches ChunkBundles to the WebSocket.

        A/V Sync Protocol:
          1. Send 'chunk_audio' message with all audio PCM for the chunk
          2. Send 'chunk_frame' messages for each video frame
          3. Client buffers initially (BUFFER_LATENCY ms) then starts
             playing audio and video together at 25fps

        The chunk_audio message includes:
          - chunk_id: for A/V sync tracking
          - n_frames: how many video frames will follow
          - sample_rate: 24000 (Moshi output)
          - buffer_ms: suggested initial buffer latency
          - Sync gap averages for frontend display

        The chunk_frame messages include:
          - chunk_id + frame_idx: for precise sync
          - Telemetry data for the UI dashboard
        """
        total_frames_sent = 0
        buffer_ms = BUFFER_LATENCY.get(self.flash.model_type, 1400)

        while not self.stop_event.is_set():
            try:
                bundle: ChunkBundle = self.dispatch_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue

            # ── Send chunk audio first (one message for entire chunk) ──
            if bundle.audio_pcm_list:
                audio_concat = np.concatenate(bundle.audio_pcm_list)
                await self.ws.send_json({
                    "type": "chunk_audio",
                    "chunk_id": bundle.chunk_id,
                    "sample_rate": MOSHI_SR,
                    "n_frames": len(bundle.video_frames),
                    "pcm_s16le_b64": self._encode_audio_b64(audio_concat),
                    "start_token_id": bundle.start_token_id,
                    "end_token_id": bundle.end_token_id,
                    "gen_ms": round(bundle.gen_time_ms, 1),
                    "buffer_ms": buffer_ms,
                    "avg_first_gap_ms": round(self.avg_first_gap_ms, 1),
                    "avg_last_gap_ms": round(self.avg_last_gap_ms, 1),
                })

            # ── Send video frames ──────────────────────────────────────
            for i, frame in enumerate(bundle.video_frames):
                if self.stop_event.is_set():
                    break

                total_frames_sent += 1
                self._total_frames_sent = total_frames_sent

                jpeg_b64 = self._encode_jpeg_b64(frame)
                if jpeg_b64 is not None:
                    elapsed = max(1e-6, time.perf_counter() - self._session_t0)
                    server_fps = total_frames_sent / elapsed

                    msg = {
                        "type": "chunk_frame",
                        "chunk_id": bundle.chunk_id,
                        "frame_idx": i,
                        "total_frames": len(bundle.video_frames),
                        "jpeg_b64": jpeg_b64,
                        "server_fps": round(server_fps, 1),
                        "gen_ms": round(bundle.gen_time_ms, 1),
                        "avg_gen_ms": round(self._avg_gen_ms, 1),
                        "deque_total": self.token_deque.total_pushed,
                        "chunks_done": self._total_chunks,
                        "moshi_text": self.moshi.get_latest_text() or "",
                    }
                    await self.ws.send_json(msg)

                # Brief yield to keep async loop responsive
                # (actual frame pacing is done client-side)
                await asyncio.sleep(0.001)

    # ── Session Lifecycle ──────────────────────────────────────────────

    def _reset_moshi_streaming(self):
        # Walk lm_gen (NOT lm_gen.lm_model) so the parent StreamingContainer's
        # own _streaming_state is cleared too — otherwise reconnect fails with
        # "is already streaming!".
        try:
            for root in (self.moshi.mimi, self.moshi.lm_gen):
                if root is None:
                    continue
                for _, mod in root.named_modules():
                    if getattr(mod, "_streaming_state", None) is not None:
                        mod._streaming_state = None
        except Exception:
            pass

    async def run(self):
        self._session_t0 = time.perf_counter()

        # Start Moshi streaming thread
        self.moshi_thread = threading.Thread(
            target=self.moshi.run_streaming,
            args=(self.token_queue, self.mic_queue, self.stop_event),
            daemon=True,
            name="MoshiThread",
        )
        self.moshi_thread.start()

        # Start FlashHead generation thread
        self.flashhead_thread = threading.Thread(
            target=self._flashhead_loop,
            daemon=True,
            name="FlashHeadThread",
        )
        self.flashhead_thread.start()

        # Notify client with all sync parameters
        buffer_ms = BUFFER_LATENCY.get(self.flash.model_type, 1400)
        await self.ws.send_json({
            "type": "server_ready",
            "sample_rate": MOSHI_SR,
            "fps": FLASHHEAD_FPS,
            "model_type": self.flash.model_type,
            "tokens_per_chunk": self.flash.tokens_per_chunk,
            "slice_len": self.flash.slice_len,
            "frame_ms": FLASHHEAD_FRAME_MS,
            "buffer_ms": buffer_ms,
        })

        # Run async tasks
        recv_task = asyncio.create_task(self._receiver())
        disp_task = asyncio.create_task(self._dispatcher())

        try:
            done, pending = await asyncio.wait(
                {recv_task, disp_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            self.stop_event.set()
            for task in (recv_task, disp_task):
                if not task.done():
                    task.cancel()
            if self.moshi_thread:
                self.moshi_thread.join(timeout=3.0)
            if self.flashhead_thread:
                self.flashhead_thread.join(timeout=3.0)
            self._reset_moshi_streaming()
            # Reset FlashHead for next session
            self.flash._chunk_idx = 0
            self.flash.pipeline.reset_person_name(self.flash.pipeline.person_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.perf_counter() - self._session_t0
            # Session summary
            summary = (
                f"[Session] Closed. duration={elapsed:.1f}s "
                f"chunks={self._total_chunks} frames={self._total_frames_sent}"
            )
            if self._sync_count > 0:
                summary += (
                    f"\n  A/V Sync: avg_first_gap={self.avg_first_gap_ms:+.1f}ms "
                    f"avg_last_gap={self.avg_last_gap_ms:+.1f}ms "
                    f"(+ve=audio first, -ve=frames first)"
                )
            if elapsed > 0:
                summary += f"\n  Avg throughput: {self._total_frames_sent/elapsed:.1f} fps"
            print(summary)


# ────────────────────────────────────────────────────────────────────────────
#  FastAPI Application
# ────────────────────────────────────────────────────────────────────────────

def create_app(args) -> FastAPI:
    app = FastAPI(title="FlashTalk v3 — Moshi + SoulX-FlashHead", version="3.0.0")

    # Load engines
    moshi_engine = MoshiEngine(
        precision=args.moshi_precision,
        repo_override=args.moshi_repo,
        device=args.moshi_device,
    )
    flash_engine = FlashHeadTokenEngine(
        ckpt_dir=args.flash_ckpt_dir,
        wav2vec_dir=args.flash_wav2vec_dir,
        model_type=args.flash_model_type,
        ref_image=args.ref_image,
        adapter_ckpt_dir=args.adapter_ckpt_dir,
        base_seed=args.base_seed,
        device=args.flashhead_device,
    )

    print("\n" + "=" * 70)
    print("  Loading Models...")
    print("=" * 70)

    flash_engine.load()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] After FlashHead: {used:.1f} GB / {total:.1f} GB")

    moshi_engine.load()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] After Moshi: {used:.1f} GB / {total:.1f} GB")

    # Warmup FlashHead to trigger torch.compile
    flash_engine.warmup(n_chunks=args.warmup_chunks)

    app.state.moshi = moshi_engine
    app.state.flash = flash_engine
    app.state.args = args

    # ── Routes ──

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "device": DEVICE,
            "moshi_precision": args.moshi_precision,
            "flash_model_type": args.flash_model_type,
            "tokens_per_chunk": flash_engine.tokens_per_chunk,
            "slice_len": flash_engine.slice_len,
            "buffer_ms": BUFFER_LATENCY.get(args.flash_model_type, 1400),
        })

    @app.get("/")
    async def root():
        if os.path.exists(STATIC_INDEX):
            return FileResponse(STATIC_INDEX)
        return JSONResponse(
            {"error": "Frontend HTML not found", "expected": STATIC_INDEX},
            status_code=404,
        )

    @app.websocket("/ws/conversation")
    async def ws_conversation(websocket: WebSocket):
        await websocket.accept()
        print("[WS] Client connected.")
        session = ConversationSession(
            websocket,
            app.state.moshi,
            app.state.flash,
            app.state.args,
        )
        try:
            await session.run()
        except WebSocketDisconnect:
            print("[WS] Client disconnected.")
        except Exception as e:
            print(f"[WS] Error: {e}")
            import traceback
            traceback.print_exc()
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()

    return app


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FlashTalk v3 — Moshi + SoulX-FlashHead Unified Streaming Server"
    )
    p.add_argument("--host", default="0.0.0.0", help="Server host")
    p.add_argument("--port", type=int, default=7860, help="Server port")

    p.add_argument("--ref-image", default=DEFAULT_REF_IMAGE, help="Reference image")
    p.add_argument("--base-seed", type=int, default=42)

    p.add_argument("--flash-ckpt-dir", default=DEFAULT_FLASH_CKPT)
    p.add_argument("--flash-wav2vec-dir", default=DEFAULT_FLASH_WAV2VEC)
    p.add_argument(
        "--flash-model-type",
        default="lite",
        choices=["lite", "pro"],
        help="FlashHead model: lite (faster, ~350ms/chunk) or pro (better quality, ~600ms/chunk)"
    )

    p.add_argument("--warmup-chunks", type=int, default=2, help="Warmup chunks for torch.compile JIT")

    p.add_argument(
        "--adapter-ckpt-dir",
        default=DEFAULT_ADAPTER_CKPT_DIR,
        help=(
            "Directory for adapter checkpoint. If "
            f"{ADAPTER_FILENAME} exists here, fine-tuned weights are loaded. "
            "Otherwise random init is used."
        ),
    )

    p.add_argument(
        "--moshi-precision",
        default="bf16",
        choices=["q8", "bf16", "fp32"],
    )
    p.add_argument("--moshi-repo", default=None, help="HF repo override")

    p.add_argument("--moshi-device", default=DEVICE, help="Device for Moshi")
    p.add_argument("--flashhead-device", default=DEVICE, help="Device for FlashHead")

    p.add_argument(
        "--show-sync",
        action="store_true",
        help="Show A/V sync gap telemetry in terminal and frontend"
    )

    return p.parse_args()


def _ensure_port_available(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
    except OSError as e:
        raise RuntimeError(
            f"Port {host}:{port} unavailable ({e}). Use another --port."
        ) from e
    finally:
        sock.close()


def main():
    args = parse_args()
    _ensure_port_available(args.host, args.port)

    # Compute expected latencies
    if args.flash_model_type == "lite":
        tokens_per_chunk = 12
        accumulation_ms = 960
        est_gen_ms = 350
    else:
        tokens_per_chunk = 14
        accumulation_ms = 1120
        est_gen_ms = 600
    est_latency = accumulation_ms + est_gen_ms
    buffer_ms = BUFFER_LATENCY.get(args.flash_model_type, 1400)

    print("\n" + "=" * 70)
    print("  FlashTalk v3 — Moshi Helium + SoulX-FlashHead")
    print("=" * 70)
    print(f"  Host:Port         : {args.host}:{args.port}")
    print(f"  Flash Model       : {args.flash_model_type}")
    print(f"  Moshi Precision   : {args.moshi_precision}")
    print(f"  Moshi Device      : {args.moshi_device}")
    print(f"  FlashHead Device  : {args.flashhead_device}")
    print(f"  Tokens/Chunk      : {tokens_per_chunk}")
    print(f"  Accumulation      : {accumulation_ms}ms")
    print(f"  Est. Gen Time     : ~{est_gen_ms}ms")
    print(f"  Est. 1st Chunk    : ~{est_latency}ms")
    print(f"  Buffer Latency    : {buffer_ms}ms")
    print(f"  Warmup Chunks     : {args.warmup_chunks}")
    print(f"  Ref Image         : {args.ref_image}")
    print(f"  Adapter Ckpt Dir  : {args.adapter_ckpt_dir}")
    print(f"  Show Sync Info    : {args.show_sync}")
    print("=" * 70 + "\n")

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
