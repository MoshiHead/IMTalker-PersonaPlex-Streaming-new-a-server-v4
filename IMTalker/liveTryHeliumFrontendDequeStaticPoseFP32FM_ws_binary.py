"""liveTryHeliumFM_ws_binary.py — same as liveTryHeliumFM, but Moshi reply A/V uses binary WS.

Use this entrypoint + static/index_v3_binary.html to avoid huge JSON/base64 av_frame payloads.
Original liveTryHeliumFM.py + static/index_v3.html are unchanged.

Architecture (reply mode): unchanged; only av_frame wire format differs (see ws_av_binary_codec.py).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import queue

import ws_av_binary_codec as _wsbin
import sys
import threading
import time
import types
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.FM import FMGenerator
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
from liveTry import MoshiOnlyEngine
from renderer.models import IMTRenderer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24_000          # Mimi sample rate (24kHz)
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
MAIN_CODEBOOKS = 8          # codebooks used for Helium input embeddings
PREBUFFER_CHUNKS = 0        # produce this many chunks before sender starts pacing
WAV2VEC_SR = 16_000


class PlasticityProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(4096)
        self.net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_ln(x))


class PlasticityUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(768, 768, kernel_size=4, stride=4)

    def forward(self, low: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.up(low.transpose(1, 2).contiguous())
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()


class PlasticityCausalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.15,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(768)
        self.ff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class PlasticityCausalTransformer(nn.Module):
    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PlasticityCausalBlock() for _ in range(8)])
        self.norm = nn.LayerNorm(768)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class StudioNativeLiveAdapter(nn.Module):
    """Frontend fp32 adapter live wrapper.

    Training contract:
      raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768]
      live contract:
      projected frontend -> frozen Wav2Vec2 encoder -> final hidden -> IMTalker audio_projection.
    """

    def __init__(self, wav2vec_model_path: str, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = HeliumToWav2VecFrontendAdapter(num_layers=int(num_layers), dropout=float(dropout))
        self.wav2vec = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).eval().float()
        for param in self.wav2vec.parameters():
            param.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward_single(self, source: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = source.unsqueeze(0).contiguous()
        target_len = int(target_len)
        frontend_len = max(1, target_len * 2)
        frontend50 = self.model(src.float(), target_len=frontend_len).float()
        final50 = self.wav2vec.encode_from_projected_frontend(frontend50).last_hidden_state.float()
        final25 = F.interpolate(
            final50.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)[0].float().contiguous()
        return frontend50[0].float().contiguous(), final50[0].float().contiguous(), final25


class MoshiOnlyEngineWithHidden(MoshiOnlyEngine):
    """Moshi reply engine that also returns the main LM hidden for each generated step.

    Layer[-2] is exposed as a native LMGen output so Moshi can keep CUDA graph
    replay enabled. We do not use Python forward hooks in this path.
    """

    def __init__(self, *args, capture_layer: int = -2, **kwargs) -> None:
        self.tf_capture_layer = int(capture_layer)
        super().__init__(*args, **kwargs)
        self._install_graph_hidden_capture()

    def _install_graph_hidden_capture(self) -> None:
        lm_model = self.lm
        lm_gen = self.lm_gen
        if hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output"):
            @torch.no_grad()
            def personaplex_step_with_hidden(
                self_gen,
                input_tokens: torch.Tensor = None,
                moshi_tokens: torch.Tensor = None,
                text_token: torch.Tensor = None,
                depformer_replace_tokens: torch.Tensor | None = None,
            ):
                prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
                if prepared is None:
                    return None
                input_, provided_, target_, model_input_position, target_position = prepared
                state = self_gen._streaming_state
                transformer_out, text_logits = state.graphed_main(input_)
                output = self_gen.process_transformer_output(
                    transformer_out,
                    text_logits,
                    provided_,
                    target_,
                    model_input_position,
                    target_position,
                )
                return output, transformer_out, transformer_out

            lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)
            lm_gen.streaming_forever(1)
            self._warmup_runtime()
            print("[liveTryPlasticity] installed PersonaPlex graphed hidden capture", flush=True)
            return

        from moshi.models.lm import scatter_with_mask_
        from moshi.modules.transformer import create_sin_embedding
        from moshi.utils.sampling import sample_token

        capture_layer = int(self.tf_capture_layer) % len(lm_model.transformer.layers)

        old_state = getattr(lm_gen, "_streaming_state", None)
        if old_state is not None:
            with contextlib.suppress(Exception):
                old_state.__exit__(None, None, None)
            with contextlib.suppress(Exception):
                lm_gen._stop_streaming()

        def forward_text_with_layer(self_lm, sequence, sum_condition=None, cross_attention_src=None):
            B, K, S = sequence.shape
            assert K == self_lm.num_codebooks, (K, self_lm.num_codebooks)
            input_sequence = sequence
            input_ = None
            for cb_index in range(self_lm.num_audio_codebooks):
                audio_emb = self_lm.emb[cb_index](input_sequence[:, cb_index + self_lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = self_lm.text_emb(input_sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb
            if sum_condition is not None:
                input_ = input_ + sum_condition.to(input_)
            if cross_attention_src is not None:
                cross_attention_src = cross_attention_src.to(input_)

            transformer = self_lm.transformer
            _, T, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)

            transformer_out = x.to(dtype_input)
            layer_hidden = captured.to(dtype_input)
            if self_lm.out_norm:
                transformer_out = self_lm.out_norm(transformer_out)
            text_logits = self_lm.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits, layer_hidden

        @torch.no_grad()
        def step_with_layer(self_gen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None):
            state = self_gen._streaming_state
            if state is None:
                raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
            lm_model_local = self_gen.lm_model

            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
            assert S == 1, "Only support being given steps one by one."
            needed_tokens = lm_model_local.num_codebooks - lm_model_local.dep_q - 1
            assert Ki >= needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            if Ki > needed_tokens:
                input_tokens = input_tokens[:, :needed_tokens, :]

            CT = state.cache.shape[2]
            delays = self_gen.delays_cuda[lm_model_local.dep_q + 1:]
            write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
            scatter_with_mask_(state.cache[:, lm_model_local.dep_q + 1:], -1, write_positions, input_tokens, state.exec_mask[:, None, None])

            is_init = state.offsets[:, None, None] <= self_gen.delays_cuda[:, None]
            is_init |= ~state.exec_mask[:, None, None]
            positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
            input_ = state.cache.gather(dim=2, index=positions)
            input_ = torch.where(is_init, state.initial, input_)

            if self_gen.check:
                assert not (input_ == lm_model_local.ungenerated_token_id).any(), (state.offsets, input_)
                assert (input_[:, lm_model_local.audio_offset:] <= lm_model_local.card).all(), input_
                assert (input_[:, :1] <= lm_model_local.text_card).all()

            zero = torch.full((1,), lm_model_local.zero_token_id, dtype=torch.long, device=input_.device)
            if self_gen.cfg_coef != 1.:
                if state.cfg_is_masked_until is not None:
                    limit = self_gen.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                    is_zeroed = state.offsets[:, None, None] <= limit
                    masked = torch.where(is_zeroed & ~is_init, zero, input_)
                    input_ = torch.cat([input_, masked], dim=0)
                else:
                    input_ = input_.repeat(2, 1, 1)
                if self_gen.cfg_is_no_text:
                    input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

            transformer_out, text_logits, layer_hidden = state.graphed_main(input_, state.condition_sum, state.condition_cross)
            if self_gen.cfg_coef != 1.:
                logits, logits_null = text_logits.chunk(2)
                if self_gen.cfg_is_no_text:
                    text_logits = logits
                    layer_hidden = layer_hidden[:B]
                else:
                    text_logits = logits_null + (logits - logits_null) * self_gen.cfg_coef
                    layer_hidden = layer_hidden[:B]

            if self_gen.on_text_logits_hook:
                self_gen.on_text_logits_hook(text_logits)
            text_token = sample_token(text_logits.float(), self_gen.use_sampling, self_gen.temp_text, self_gen.top_k_text)
            assert text_token.dim() == 3, text_token.shape
            assert text_token.shape[2] == 1
            assert text_token.shape[1] == 1, "Only one text stream supported."
            text_token = text_token[:, 0, 0]
            if self_gen.on_text_hook is not None:
                self_gen.on_text_hook(text_token)

            if state.graphed_depth is None:
                audio_tokens = None
            else:
                if depformer_replace_tokens is None:
                    audio_tokens = state.graphed_depth(text_token, transformer_out)
                else:
                    assert depformer_replace_tokens.dim() == 3
                    audio_tokens = depformer_replace_tokens.squeeze(-1)
                if self_gen.on_audio_hook is not None:
                    self_gen.on_audio_hook(audio_tokens)

            state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
            state.offset_cpu += 1
            positions = (state.offsets % CT)[:, None, None]
            scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])
            if audio_tokens is not None:
                audio_tokens = audio_tokens[:, :, None]
                scatter_with_mask_(state.cache[:, 1: lm_model_local.dep_q + 1, :], -1, positions.expand_as(audio_tokens), audio_tokens, state.exec_mask[:, None, None])

            if not self_gen.support_out_of_sync and state.offset_cpu <= self_gen.max_delay:
                return None
            gen_delays_cuda = self_gen.delays_cuda[: lm_model_local.dep_q + 1]
            index = (state.offsets[:, None, None] - self_gen.max_delay + gen_delays_cuda[:, None]) % CT
            out = state.cache.gather(dim=2, index=index)
            mask = (state.offsets <= self_gen.max_delay) | ~state.exec_mask
            out[mask, :, :] = lm_model_local.ungenerated_token_id
            return out, transformer_out, layer_hidden

        lm_model.forward_text = types.MethodType(forward_text_with_layer, lm_model)
        lm_gen._step = types.MethodType(step_with_layer, lm_gen)
        lm_gen.streaming_forever(1)
        self._warmup_runtime()
        print(f"[liveTryPlasticity] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        t_lm0 = time.perf_counter()
        lm_out = self.lm_gen._step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"Moshi graph layer[-2] contract failure: got {type(lm_out)} len={len(lm_out) if isinstance(lm_out, tuple) else 'n/a'}")
            tokens, _transformer_out, layer_hidden = lm_out
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        print(
            "[liveTryStudio] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"hidden={helium_hidden is not None} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms",
            flush=True,
        )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames.

        The base MoshiOnlyEngine drains the entire input buffer before returning.
        In live mode that is dangerous: mic audio can accumulate while Moshi is
        loading, then avatar frames do not reach the sender until the backlog is
        fully processed. Bounded draining keeps the producer/sender interleaved.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _pcm_f32_to_i16_b64(pcm: np.ndarray) -> str:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((arr * 32767.0).astype(np.int16).tobytes()).decode("ascii")


def _pcm_f32_to_i16_bytes(pcm: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def split_audio_into_frame_slices(pcm: np.ndarray, fps: float) -> list[np.ndarray]:
    frame_samples = int(round(TARGET_SR / float(fps)))
    arr = np.asarray(pcm, dtype=np.float32)
    n_frames = max(0, int(round(arr.shape[0] / frame_samples)))
    if n_frames == 0:
        return []
    total = n_frames * frame_samples
    if arr.shape[0] < total:
        arr = np.pad(arr, (0, total - arr.shape[0]))
    elif arr.shape[0] > total:
        arr = arr[:total]
    return [arr[i * frame_samples:(i + 1) * frame_samples].copy() for i in range(n_frames)]


def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).float().numpy()


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FM + Renderer weight loading (identical to liveTryFM.py)
# ---------------------------------------------------------------------------

def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
    t_total = time.perf_counter()
    fm = FMGenerator(args).to(device).eval()
    ckpt = torch.load(args.generator_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}
    ema = ckpt.get("ema_state_dict")
    if isinstance(ema, dict):
        cleaned.update(ema)
    missing, unexpected = fm.load_state_dict(cleaned, strict=False)
    _sync_cuda()
    print(
        f"[liveTryHeliumFM][FM] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return fm


def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
    t_total = time.perf_counter()
    renderer = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
    missing, unexpected = renderer.load_state_dict(cleaned, strict=False)
    renderer = renderer.to(dtype=dtype)
    _sync_cuda()
    if getattr(args, "compile_renderer", False):
        @torch.no_grad()
        def _fused_render(motion_latent, g_r, m_r, f_r):
            ta_c = renderer.adapt(motion_latent, g_r)
            m_c = renderer.latent_token_decoder(ta_c)
            frames = renderer.decode(m_c, m_r, f_r)
            return frames
        renderer._fused_render = torch.compile(_fused_render)
    print(
        f"[liveTryHeliumFM][renderer] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return renderer


# ---------------------------------------------------------------------------
# Helium extractor (chunk-local, batch LM)
# ---------------------------------------------------------------------------

class HeliumExtractor:
    """Prefix-growing raw Helium + global interpolation.

    This matches the best Stage 3 diagnostic:
      audio prefix -> raw Helium -> append only new raw steps
      -> one global interpolation over accumulated raw Helium
      -> emit last target_frames
    """

    def __init__(
        self,
        helium_mimi: "MimiModel",
        helium_lm: "LMModel",
        device: torch.device,
    ) -> None:
        self.helium_mimi = helium_mimi
        self.helium_lm = helium_lm
        self.device = device
        self._lock = threading.Lock()
        self._prefix_pcm = np.empty(0, dtype=np.float32)
        self._raw_parts: list[torch.Tensor] = []
        self._prev_raw_len = 0
        self._emitted_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._prefix_pcm = np.empty(0, dtype=np.float32)
            self._raw_parts = []
            self._prev_raw_len = 0
            self._emitted_frames = 0

    def _extract_raw(self, pcm_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).to(self.device, dtype=torch.float32)[None, None]

        codes = self.helium_mimi.encode(wav)
        codes = codes[:, :MAIN_CODEBOOKS, :].detach()
        batch_size, n_q, total_steps = codes.shape

        dtype = next(self.helium_lm.parameters()).dtype
        input_emb = torch.zeros(
            batch_size, total_steps, self.helium_lm.dim, device=self.device, dtype=dtype
        )
        for q in range(n_q):
            input_emb = input_emb + self.helium_lm.emb[q](codes[:, q].long())

        padding_ids = torch.full(
            (batch_size, total_steps),
            self.helium_lm.existing_text_padding_id,
            dtype=torch.long,
            device=self.device,
        )
        input_emb = input_emb + self.helium_lm.text_emb(padding_ids)

        if getattr(self.helium_lm.transformer, "_streaming_state", None) is not None:
            raise RuntimeError("helium_lm must stay in batch mode (non-streaming)")

        captured: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            captured.append(out.detach())

        handle = self.helium_lm.transformer.layers[-2].register_forward_hook(_hook)
        try:
            self.helium_lm.transformer(input_emb)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Helium hook captured {len(captured)} tensors; expected 1")
        return captured[0].squeeze(0).float().contiguous()  # [T_raw, 4096]

    @torch.no_grad()
    def extract_raw_chunk(self, pcm_np: np.ndarray) -> torch.Tensor:
        """Return only the new raw 12.5Hz Helium steps for one new audio chunk."""
        pcm = np.asarray(pcm_np, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_raw_chunk expects non-empty 1D PCM")

        with self._lock:
            self._prefix_pcm = np.concatenate([self._prefix_pcm, pcm], axis=0)
            raw_prefix = self._extract_raw(self._prefix_pcm)
            new_raw = raw_prefix[self._prev_raw_len:]
            if int(new_raw.shape[0]) == 0 and int(raw_prefix.shape[0]) > 0:
                new_raw = raw_prefix[-1:]
            self._raw_parts.append(new_raw.cpu())
            self._prev_raw_len = int(raw_prefix.shape[0])
            return new_raw.contiguous()

    @torch.no_grad()
    def extract_exact_chunk_from_prefix(
        self,
        pcm_prefix: np.ndarray,
        chunk_start_frame: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Return the raw 12.5Hz Helium slice for a video-frame window.

        This path is used by file-mode lookahead. It must return raw Helium
        steps so the studio adapter remains the only temporal upsampler.
        """
        pcm = np.asarray(pcm_prefix, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_exact_chunk_from_prefix expects non-empty 1D PCM")
        with self._lock:
            raw_prefix = self._extract_raw(pcm)
        start_frame = int(chunk_start_frame)
        end_frame = start_frame + int(target_frames)
        start_raw = int(round(start_frame * 0.5))
        end_raw = int(round(end_frame * 0.5))
        start_raw = max(0, min(start_raw, int(raw_prefix.shape[0])))
        end_raw = max(start_raw + 1, min(end_raw, int(raw_prefix.shape[0])))
        if end_raw > int(raw_prefix.shape[0]):
            raise RuntimeError(
                f"Requested raw slice [{start_raw}, {end_raw}) exceeds prefix Helium length {raw_prefix.shape[0]}"
            )
        return raw_prefix[start_raw:end_raw].contiguous()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveHeliumFMEngine:
    """Helium extraction + FM + renderer, session-stateful."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.args = args
        self.device = torch.device(args.device)
        self.dtype = torch.float32 if args.fp32 else torch.bfloat16
        self.fps = float(args.fps)
        self.audio_chunk_sec = float(getattr(args, "audio_chunk_sec", 0.96))
        self.audio_chunk_samples = int(round(self.audio_chunk_sec * TARGET_SR))
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)
        trained_window = int(round(float(args.wav2vec_sec) * self.fps))
        if self.fm_chunk_frames != trained_window:
            print(
                f"[liveTryHeliumFM] WARNING fm_chunk_frames={self.fm_chunk_frames} "
                f"but wav2vec_sec*fps={trained_window}",
                flush=True,
            )

        t_total = time.perf_counter()

        # FM + renderer
        self.fm = _load_fm(args, self.device)
        self.renderer = _load_renderer(args, self.device, self.dtype)

        # Frontend fp32 adapter: raw 12.5Hz Helium -> Wav2Vec2 frontend -> frozen Wav2Vec2 final hidden.
        t_adapter = time.perf_counter()
        self.studio_adapter = StudioNativeLiveAdapter(
            args.wav2vec_model_path,
            args.adapter_num_layers,
            args.adapter_dropout,
        ).to(self.device).float().eval()
        payload = torch.load(args.adapter_path, map_location="cpu")
        if isinstance(payload, dict):
            saved_args = payload.get("args", {})
            if saved_args and int(saved_args.get("num_layers", args.adapter_num_layers)) != int(args.adapter_num_layers):
                print(
                    f"[liveTryHeliumFrontendFM] WARNING checkpoint num_layers={saved_args.get('num_layers')} "
                    f"but CLI adapter_num_layers={args.adapter_num_layers}",
                    flush=True,
                )
            state = payload.get("adapter", payload.get("model", payload))
        else:
            state = payload
        self.studio_adapter.load_state_dict(state, strict=True)
        _sync_cuda()
        print(
            f"[liveTryHeliumFrontendFM][adapter] frontend-fp32 loaded in {_ms(t_adapter):.0f}ms "
            f"path={args.adapter_path} layers={args.adapter_num_layers}",
            flush=True,
        )

        # Raw HF Wav2Vec2 target path used during Helium adapter training.
        t_w2v = time.perf_counter()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_model_path,
            local_files_only=True,
        )
        self.wav2vec_model = self.studio_adapter.wav2vec
        _sync_cuda()
        print(
            f"[liveTryHeliumStudioFM][wav2vec] loaded in {_ms(t_w2v):.0f}ms "
            f"path={args.wav2vec_model_path}",
            flush=True,
        )

        # Reference image: pre-compute identity + motion-ref features once
        ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
        with torch.no_grad():
            self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
            self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
            ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
            self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()

        # Moshi models for Helium extraction
        self._init_moshi(args)

        # Optional local audio file (simulate-live mode)
        self.audio_pcm: np.ndarray | None = None
        if getattr(args, "audio_path", "") and Path(args.audio_path).is_file():
            self.audio_pcm = load_audio_24k(args.audio_path)
            print(
                f"[liveTryHeliumFM] audio_path loaded: {self.audio_pcm.shape[0]/TARGET_SR:.2f}s "
                f"chunk={self.audio_chunk_sec:.3f}s/{self.audio_chunk_samples} samples "
                f"fm_chunk={self.fm_chunk_frames}f",
                flush=True,
            )

        # Shared noise tensor (pre-generated, indexed by absolute frame position)
        self.noise_buf: torch.Tensor | None = None
        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(
                1, max_frames, int(args.dim_w), device=self.device, generator=gen
            )
            print(f"[liveTryHeliumFM] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        # Per-session state (reset on each new client)
        self.stream_state: dict | None = None
        self.abs_frame: int = 0
        self.helium_context_tail: torch.Tensor | None = None
        self.helium_deque_size: int = 100
        self.helium_deque: torch.Tensor | None = None
        self.helium_deque_filled: int = 0
        self._pcm_accum: np.ndarray = np.empty(0, dtype=np.float32)
        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_50_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_projected_audio_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_reply_events: list[dict] = []
        self._session_started_wall: float = time.time()

        # JPEG encoding thread pool (CPU-only work, parallelizable)
        self._jpeg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="jpeg"
        )

        # Warmup
        self._warmup()

        print(
            f"[liveTryHeliumFM] ready — total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} render_sub={self.render_sub_batch} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def _init_moshi(self, args: argparse.Namespace) -> None:
        if bool(getattr(args, "direct_reply_hidden", False)) and bool(getattr(args, "enable_moshi_reply", False)):
            self.extractor = None
            print("[liveTryHeliumStudioFM] using direct Moshi reply hidden; batch Helium extractor skipped", flush=True)
            return

        from generate_helium import load_mimi_and_lm

        t0 = time.perf_counter()
        helium_mimi, helium_lm, _ = load_mimi_and_lm(args)
        helium_mimi.eval()
        helium_lm.eval()
        print(
            f"[liveTryHeliumFM] Moshi loaded in {_ms(t0):.0f}ms "
            f"dtype={next(helium_lm.parameters()).dtype}",
            flush=True,
        )

        self.extractor = HeliumExtractor(helium_mimi, helium_lm, self.device)

    def reset_session(self) -> None:
        """Call when a new WebSocket client connects or sends 'start'."""
        self.stream_state = None
        self.abs_frame = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0
        self._pcm_accum = np.empty(0, dtype=np.float32)
        if self.extractor is not None:
            self.extractor.reset()
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_50_parts = []
        self._session_adapter_25_parts = []
        self._session_projected_audio_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_reply_events = []
        self._session_started_wall = time.time()

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_pcm = np.zeros(self.audio_chunk_samples, dtype=np.float32)

        t0 = time.perf_counter()
        if self.extractor is None:
            raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / float(self.fps))))
            dummy_helium = torch.zeros(raw_steps, 4096, device=self.device, dtype=torch.float32)
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium=skipped direct_hidden raw_steps={raw_steps}", flush=True)
        else:
            dummy_helium = self.extractor.extract_raw_chunk(dummy_pcm)
            _sync_cuda()
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium={_ms(t0):.0f}ms", flush=True)
            self.extractor.reset()

        t0 = time.perf_counter()
        motion, _info = self._sample_motion_from_helium(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] fm={_ms(t0):.0f}ms motion={tuple(motion.shape)}", flush=True)
        self.stream_state = None
        self.abs_frame = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0

        t0 = time.perf_counter()
        dummy_motion = torch.zeros(self.render_sub_batch, 32, device=self.device, dtype=self.dtype)
        _frames, _timings = self._render_motion(dummy_motion)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] renderer={_ms(t0):.0f}ms", flush=True)

        # Warmup JPEG pool
        t0 = time.perf_counter()
        dummy_np = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = encode_jpeg_b64(dummy_np, self.jpeg_quality)
        print(f"[liveTryHeliumFM][warmup] jpeg={_ms(t0):.0f}ms", flush=True)

        self.stream_state = None

    def feed_pcm(self, pcm_s16le_bytes: bytes) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_pcm_f32(pcm)

    def feed_pcm_f32(self, pcm_f32: np.ndarray) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        self._pcm_accum = np.concatenate([self._pcm_accum, pcm])
        if self._pcm_accum.shape[0] < self.audio_chunk_samples:
            return None
        chunk = self._pcm_accum[:self.audio_chunk_samples].copy()
        self._pcm_accum = self._pcm_accum[self.audio_chunk_samples:]
        motion, info = self._process_pcm_chunk(chunk, self.fm_chunk_frames)
        self._record_session_chunk(chunk, motion, info)
        return motion, info, chunk

    @torch.no_grad()
    def _process_pcm_chunk(self, pcm_chunk: np.ndarray, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        target_frames = max(1, min(int(target_frames), self.fm_chunk_frames))
        if self.extractor is None:
            raise RuntimeError("direct_reply_hidden mode cannot process raw browser audio directly")

        t0 = time.perf_counter()
        helium = self.extractor.extract_raw_chunk(pcm_chunk)
        timings["helium_ms"] = _ms(t0)
        motion, fm_info = self._sample_motion_from_helium(helium, target_frames)
        timings.update(fm_info)
        return motion, timings

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        t_adapter = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        current_steps = int(helium.shape[0])
        deque_size = int(getattr(self, "helium_deque_size", 100))
        if self.helium_deque is None:
            self.helium_deque = torch.zeros(deque_size, helium.shape[1], device=self.device, dtype=torch.float32)
            self.helium_deque_filled = 0
        if current_steps >= deque_size:
            self.helium_deque = helium[-deque_size:].detach().clone()
            self.helium_deque_filled = deque_size
        else:
            self.helium_deque = torch.cat([self.helium_deque[current_steps:], helium], dim=0).contiguous()
            self.helium_deque_filled = min(deque_size, int(self.helium_deque_filled) + current_steps)

        # Training contract: adapter sees a full 100-token / 8s Helium window.
        # For early live chunks the left side is zero-padded, matching the initialized deque.
        target_len_25_full = deque_size * 2
        _baseline, _cnn, feat_25_full = self.studio_adapter.forward_single(self.helium_deque, target_len_25_full)
        fresh_frames = max(1, current_steps * 2)
        if int(feat_25_full.shape[0]) < fresh_frames:
            raise RuntimeError(f"Deque adapter output too short: got {feat_25_full.shape[0]}, need {fresh_frames}")
        feat_25 = feat_25_full[-fresh_frames:].contiguous()
        if int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T.contiguous()
        projected_a = self.fm._project_audio(feat_25.unsqueeze(0).float())
        timings["adapter_ms"] = _ms(t_adapter)
        timings["helium_ms"] = timings["adapter_ms"]
        timings["helium_deque_filled"] = int(self.helium_deque_filled)

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        static_pose_values = getattr(self.args, "static_pose_values", None)
        if static_pose_values is not None:
            pose = torch.tensor(static_pose_values, device=self.device, dtype=torch.float32).view(1, 3)
            data["pose"] = pose.expand(target_frames, 3).contiguous()
        elif bool(getattr(self.args, "static_pose_zero", False)):
            data["pose"] = torch.zeros(target_frames, 3, device=self.device, dtype=torch.float32)
        if self.noise_buf is not None:
            end_frame = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, self.abs_frame:end_frame]
        t_fm = time.perf_counter()
        motion, self.stream_state = self.fm.sample(
            data,
            a_cfg_scale=float(self.args.a_cfg_scale),
            nfe=int(self.args.nfe),
            stream_state=self.stream_state,
            return_state=True,
        )
        timings["fm_ms"] = _ms(t_fm)

        motion = motion.squeeze(0)[:target_frames].detach()
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = feat_25.detach().cpu()  # motionfield adapter is already 25fps; kept key for dump compatibility
        timings["adapter_feat_25"] = feat_25.detach().cpu()
        timings["projected_audio"] = projected_a.squeeze(0).detach().cpu()
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        return motion, timings

    def _record_session_chunk(self, pcm_chunk: np.ndarray, motion: torch.Tensor, info: dict) -> None:
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        helium_feat = info.get("helium_feat")
        if isinstance(helium_feat, torch.Tensor):
            self._session_helium_parts.append(helium_feat.float().cpu().clone())
        adapter_feat_50 = info.get("adapter_feat_50")
        if isinstance(adapter_feat_50, torch.Tensor):
            self._session_adapter_50_parts.append(adapter_feat_50.float().cpu().clone())
        adapter_feat_25 = info.get("adapter_feat_25")
        if isinstance(adapter_feat_25, torch.Tensor):
            self._session_adapter_25_parts.append(adapter_feat_25.float().cpu().clone())
        projected_audio = info.get("projected_audio")
        if isinstance(projected_audio, torch.Tensor):
            self._session_projected_audio_parts.append(projected_audio.float().cpu().clone())
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "abs_start": int(info.get("abs_start", 0)),
            "frames": int(info.get("frames", int(motion.shape[0]))),
            "samples": int(len(pcm_chunk)),
            "helium_ms": float(info.get("helium_ms", 0.0)),
            "fm_ms": float(info.get("fm_ms", 0.0)),
        })

    @torch.no_grad()
    def _extract_wav2vec_raw_50hz(self, audio_24k: np.ndarray) -> torch.Tensor:
        arr = np.asarray(audio_24k, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return torch.empty((0, 768), dtype=torch.float32)
        wav = torch.from_numpy(arr).view(1, -1)
        wav16 = torchaudio.functional.resample(wav, TARGET_SR, WAV2VEC_SR).squeeze(0).contiguous().numpy()
        inputs = self.wav2vec_feature_extractor(
            wav16,
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            "input_values": inputs.input_values.to(self.device),
        }
        if getattr(inputs, "attention_mask", None) is not None:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        frontend = self.wav2vec_model.extract_projected_frontend(**kwargs)
        feat = self.wav2vec_model.encode_from_projected_frontend(
            frontend
        ).last_hidden_state.detach().float().cpu()[0].contiguous()
        return feat

    def dump_last_session(self, *, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = np.concatenate(self._session_audio_parts, axis=0) if self._session_audio_parts else np.empty(0, dtype=np.float32)

        motion_path = session_dir / "full_motion.pt"
        helium_path = session_dir / "full_helium_raw.pt"
        adapter_50_path = session_dir / "full_adapter_w2v_50hz.pt"
        adapter_25_path = session_dir / "full_adapter_w2v_25fps.pt"
        wav2vec_50_path = session_dir / "full_wav2vec_50hz.pt"
        projected_audio_path = session_dir / "full_projected_audio_32.pt"
        audio_path = session_dir / "full_moshi_reply_24k.wav"
        live_tokens_path = session_dir / "live_mimi_tokens.pt"
        reply_events_path = session_dir / "reply_events.jsonl"
        reply_text_path = session_dir / "reply_text.txt"
        meta_path = session_dir / "meta.json"
        helium = None
        adapter_50 = None
        adapter_25 = None
        wav2vec_50 = None
        projected_audio = None
        live_tokens = None
        if self._session_helium_parts:
            helium = torch.cat(self._session_helium_parts, dim=0).contiguous()
        if self._session_adapter_50_parts:
            adapter_50 = torch.cat(self._session_adapter_50_parts, dim=0).contiguous()
        if self._session_adapter_25_parts:
            adapter_25 = torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
        if self._session_projected_audio_parts:
            projected_audio = torch.cat(self._session_projected_audio_parts, dim=0).contiguous()
        if self._session_live_token_parts:
            live_tokens = torch.cat(self._session_live_token_parts, dim=2).contiguous()
        if audio.size > 0:
            wav2vec_50 = self._extract_wav2vec_raw_50hz(audio)

        torch.save({
            "motion": motion,
            "chunks": self._session_chunk_rows,
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 768)),
            "audio_adapter_dim": int(getattr(self.args, "audio_adapter_dim", 512)),
            "wav2vec_sec": float(self.args.wav2vec_sec),
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
            "source": source,
        }, motion_path)
        if helium is not None:
            torch.save({
                "helium": helium,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "audio_chunk_sec": float(self.audio_chunk_sec),
                "fm_chunk_frames": int(self.fm_chunk_frames),
                "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 4096)),
                "source": source,
            }, helium_path)
        if adapter_50 is not None:
            torch.save({
                "adapter_feat_50": adapter_50,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, adapter_50_path)
        if adapter_25 is not None:
            torch.save({
                "adapter_feat_25": adapter_25,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, adapter_25_path)
        if wav2vec_50 is not None:
            torch.save({
                "wav2vec_50hz": wav2vec_50,
                "chunks": self._session_chunk_rows,
                "sample_rate": int(WAV2VEC_SR),
                "source": source,
            }, wav2vec_50_path)
        if projected_audio is not None:
            torch.save({
                "projected_audio": projected_audio,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, projected_audio_path)
        if live_tokens is not None:
            torch.save({
                "live_mimi_tokens": live_tokens,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, live_tokens_path)
        if audio.size > 0:
            torchaudio.save(str(audio_path), torch.from_numpy(audio).view(1, -1), TARGET_SR)
        if self._session_reply_events:
            with reply_events_path.open("w", encoding="utf-8") as f:
                for row in self._session_reply_events:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            reply_text = "".join(str(row.get("piece", "")) for row in self._session_reply_events)
            reply_text_path.write_text(reply_text, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "source": source,
            "session_started_wall": float(self._session_started_wall),
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "motion_frames": int(motion.shape[0]),
            "helium_frames": int(helium.shape[0]) if helium is not None else 0,
            "adapter_50_frames": int(adapter_50.shape[0]) if adapter_50 is not None else 0,
            "adapter_25_frames": int(adapter_25.shape[0]) if adapter_25 is not None else 0,
            "wav2vec_50_frames": int(wav2vec_50.shape[0]) if wav2vec_50 is not None else 0,
            "projected_audio_frames": int(projected_audio.shape[0]) if projected_audio is not None else 0,
            "audio_samples": int(audio.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size > 0 else 0.0,
            "reply_text_chars": int(sum(len(str(row.get("piece", ""))) for row in self._session_reply_events)),
            "reply_events": int(len(self._session_reply_events)),
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }, indent=2), encoding="utf-8")
        print(f"[liveTryHeliumFM] dumped last session -> {session_dir}", flush=True)
        return session_dir

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])

        g_r_sub = self.g_r.expand(n, -1)
        m_r_sub = tuple(m.expand(n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(n, -1, -1, -1) for f in self.f_r]

        fused = getattr(self.renderer, '_fused_render', None)
        if fused is not None:
            frames = fused(motion, g_r_sub, m_r_sub, f_r_sub)
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    def render_and_encode_subbatch(
        self,
        motion_sub: torch.Tensor,
        audio_slices: list[np.ndarray],
        abs_start: int,
        text_payload: str,
        avatar_chunk_id: int,
        total_gen_ms: float,
    ) -> list[dict]:
        """Render a sub-batch of frames, JPEG-encode in parallel, return packet dicts."""
        frames_np, _render_info = self._render_motion(motion_sub)

        jpeg_futures = []
        for frame_rgb in frames_np:
            jpeg_futures.append(
                self._jpeg_pool.submit(encode_jpeg_bytes, frame_rgb, self.jpeg_quality)
            )

        packets = []
        gen_ms_i = int(round(float(total_gen_ms)))
        sr_i = int(round(float(TARGET_SR)))
        for j, fut in enumerate(jpeg_futures):
            idx = abs_start + j
            audio_slice = audio_slices[j] if j < len(audio_slices) else np.zeros(
                int(round(TARGET_SR / self.fps)), dtype=np.float32
            )
            jpeg_bytes = fut.result()
            pcm_b = _pcm_f32_to_i16_bytes(audio_slice)
            blob = _wsbin.pack_av_frame(
                idx,
                idx + 1,
                gen_ms_i,
                sr_i,
                jpeg_bytes,
                pcm_b,
                text_payload,
                int(avatar_chunk_id),
            )
            packets.append(
                {
                    "frame_number": idx,
                    "ws_kind": "bytes",
                    "data": blob,
                    "t_ready": time.perf_counter(),
                }
            )
        return packets

    def audio_slice(self, frame_idx: int) -> np.ndarray:
        if self.audio_pcm is None:
            frame_samples = int(round(TARGET_SR / self.fps))
            return np.zeros(frame_samples, dtype=np.float32)
        frame_samples = int(round(TARGET_SR / self.fps))
        start = frame_idx * frame_samples
        chunk = self.audio_pcm[start:start + frame_samples]
        if chunk.shape[0] < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.shape[0]))
        return chunk


# ---------------------------------------------------------------------------
# WebSocket streaming coroutine (file-driven mode, unchanged)
# ---------------------------------------------------------------------------

async def stream_from_file(ws: WebSocket, engine: LiveHeliumFMEngine) -> None:
    """Simulate live streaming using --audio_path, sending frames back over WS."""
    if engine.audio_pcm is None:
        await ws.send_json({"type": "error", "msg": "No --audio_path given for file-streaming mode"})
        return

    audio = engine.audio_pcm
    lookahead_chunks = max(0, int(getattr(engine.args, "file_chunk_lookahead", 0)))
    total_chunks = int(np.ceil(len(audio) / engine.audio_chunk_samples))
    start_wall = time.perf_counter()
    emitted = 0

    print(
        f"[liveTryHeliumFM] stream_from_file: {total_chunks} chunks "
        f"lookahead={lookahead_chunks}",
        flush=True,
    )

    async def _emit_motion_chunk(
        motion: torch.Tensor,
        fm_info: dict,
        chunk_label: int,
        emitted_so_far: int,
    ) -> int:
        helium_ms = float(fm_info["helium_ms"])
        fm_ms = float(fm_info["fm_ms"])
        n_frames = int(motion.shape[0])
        all_frames_np: list[np.ndarray] = []
        render_ms = 0.0
        for sb_start in range(0, n_frames, engine.render_sub_batch):
            sub = motion[sb_start:sb_start + engine.render_sub_batch].to(
                engine.device, dtype=engine.dtype
            )
            frames_np, render_info = engine._render_motion(sub)
            render_ms += float(render_info["total_ms"])
            all_frames_np.extend(frames_np)

        print(
            f"[liveTryHeliumFM][chunk#{chunk_label}] "
            f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
            f"render={render_ms:.0f}ms frames={n_frames} "
            f"abs_start={fm_info['abs_start']}",
            flush=True,
        )

        for j, frame_rgb in enumerate(all_frames_np):
            idx = emitted_so_far + j
            chunk_id = idx + 1
            audio_b64 = _pcm_f32_to_i16_b64(engine.audio_slice(idx))
            jpeg_b64 = encode_jpeg_b64(frame_rgb, engine.jpeg_quality)

            await ws.send_json({
                "type": "chunk_audio",
                "chunk_id": chunk_id,
                "sample_rate": TARGET_SR,
                "pcm_s16le_b64": audio_b64,
            })
            await ws.send_json({
                "type": "chunk_frame",
                "chunk_id": chunk_id,
                "frame_idx": 0,
                "jpeg_b64": jpeg_b64,
                "moshi_text": (
                    f"Helium+FM | chunk#{chunk_label} "
                    f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
                    f"render={render_ms:.0f}ms"
                ),
                "server_fps": round(float(engine.fps), 1),
                "chunks_done": chunk_label,
            })
            target_t = start_wall + (idx + 1) / engine.fps
            await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
        return emitted_so_far + len(all_frames_np)

    if lookahead_chunks <= 0:
        for chunk_idx in range(total_chunks):
            pcm_chunk = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ]
            pcm_real = pcm_chunk.copy()
            target_frames = int(round(pcm_chunk.shape[0] * engine.fps / TARGET_SR))
            if pcm_chunk.shape[0] < engine.audio_chunk_samples:
                pcm_chunk = np.pad(pcm_chunk, (0, engine.audio_chunk_samples - pcm_chunk.shape[0]))

            motion, fm_info = engine._process_pcm_chunk(pcm_chunk, target_frames)
            engine._record_session_chunk(pcm_real, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_idx + 1, emitted)
    else:
        pending_real: list[np.ndarray] = []
        prefix_audio = np.empty(0, dtype=np.float32)
        chunk_counter = 0

        def _process_exact_pending(pcm_real: np.ndarray) -> tuple[torch.Tensor, dict]:
            nonlocal prefix_audio
            target_frames = int(round(pcm_real.shape[0] * engine.fps / TARGET_SR))
            t0 = time.perf_counter()
            helium = engine.extractor.extract_exact_chunk_from_prefix(
                prefix_audio,
                engine.abs_frame,
                target_frames,
            )
            _sync_cuda()
            fm_info: dict = {"helium_ms": _ms(t0)}
            motion, sample_info = engine._sample_motion_from_helium(helium, target_frames)
            fm_info.update(sample_info)
            return motion, fm_info

        for chunk_idx in range(total_chunks):
            pcm_real = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ].copy()
            prefix_audio = np.concatenate([prefix_audio, pcm_real], axis=0)
            pending_real.append(pcm_real)

            while len(pending_real) > lookahead_chunks:
                chunk_counter += 1
                oldest = pending_real.pop(0)
                motion, fm_info = _process_exact_pending(oldest)
                engine._record_session_chunk(oldest, motion, fm_info)
                emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

        while pending_real:
            chunk_counter += 1
            oldest = pending_real.pop(0)
            motion, fm_info = _process_exact_pending(oldest)
            engine._record_session_chunk(oldest, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

    await ws.send_json({"type": "stream_end", "total_frames": emitted})
    engine.dump_last_session(source=str(engine.args.audio_path))
    print(f"[liveTryHeliumFM] stream done: {emitted} frames", flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LiveHeliumFMOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Frontend fp32 Helium->Wav2Vec2 projected-frontend adapter checkpoint")
        parser.add_argument("--adapter_num_layers", type=int, default=6, help="Transformer layers in the frontend adapter checkpoint")
        parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout value used when constructing the frontend adapter")
        parser.add_argument("--stats_path", default="", help="Unused for frontend adapter mode; accepted for compatibility")
        parser.add_argument("--ref_path", required=True)
        parser.add_argument("--audio_path", default="", help="WAV to stream in fixed chunks (simulate-live mode)")
        # Moshi
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="", help="Optional local Mimi checkpoint")
        parser.add_argument("--tokenizer", default="", help="Optional local sentencepiece tokenizer")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex/Moshi LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8, help="PersonaPlex/Moshi audio codebooks")
        parser.add_argument("--moshi_context", type=int, default=0, help="Optional PersonaPlex/Moshi KV context length")
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="", help="Optional PersonaPlex voice prompt directory")
        parser.add_argument("--text_prompt", default="", help="Optional PersonaPlex system text prompt")
        parser.add_argument("--moshi_reply_device", default=None, help="Optional separate CUDA device for Moshi reply generation")
        parser.add_argument("--enable_moshi_reply", action="store_true", help="Mic -> Moshi reply audio -> Helium/FM avatar")
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, help="Use Moshi generation hidden directly instead of re-encoding reply audio")
        parser.add_argument("--no_direct_reply_hidden", dest="direct_reply_hidden", action="store_false")
        # FM
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec×fps")
        parser.add_argument("--reply_hidden_steps_per_chunk", type=int, default=0, help="Raw Moshi 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps")
        parser.add_argument("--prebuffer_chunks", type=int, default=3, help="Avatar chunks queued before sender starts pacing")
        parser.add_argument("--frame_q_backpressure", type=int, default=160)
        parser.add_argument("--static_pose_zero", action="store_true", help="Pass pose=(0,0,0) for every generated frame")
        parser.add_argument("--static_pose_values", type=float, nargs=3, default=None, help="Pass a fixed SMIRK pose=(pitch,yaw,roll) for every generated frame")
        parser.add_argument(
            "--file_chunk_lookahead",
            type=int,
            default=0,
            help="For --audio_path mode, wait this many future chunks before emitting the oldest chunk",
        )
        parser.add_argument("--render_sub_batch", type=int, default=8)
        parser.add_argument("--jpeg_quality", type=int, default=86)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--buffer_ms", type=int, default=80)
        parser.add_argument("--dump_motion", action="store_true", help="Dump last session motion/audio to disk")
        parser.add_argument("--dump_dir", default=str(ROOT / "live_try_dumps"))
        # Shared noise
        parser.add_argument("--shared_noise", action="store_true")
        parser.add_argument("--noise_seed", type=int, default=1234)
        parser.add_argument("--noise_max_frames", type=int, default=5000)
        # Precision
        parser.add_argument("--fp32", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--compile_renderer", action="store_true")
        return parser


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Helium MotionField Deque FM liveTry")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)
    moshi_engine: MoshiOnlyEngine | None = None

    def get_engine() -> LiveHeliumFMEngine:
        nonlocal engine
        if engine is None:
            engine = LiveHeliumFMEngine(args)
        return engine

    def get_moshi_engine() -> MoshiOnlyEngine:
        nonlocal moshi_engine
        if moshi_engine is None:
            moshi_engine = MoshiOnlyEngineWithHidden(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=getattr(args, "moshi_reply_device", None) or args.device,
                cfg_coef=float(args.moshi_cfg_coef),
                placeholder_jpeg_b64="",
                moshi_weight=getattr(args, "moshi_weight", ""),
                mimi_weight=getattr(args, "mimi_weight", ""),
                tokenizer=getattr(args, "tokenizer", ""),
                quantize_4bit=bool(getattr(args, "quantize_4bit", False)),
                num_codebooks=int(getattr(args, "num_codebooks", 8)),
                context=(int(args.moshi_context) if int(getattr(args, "moshi_context", 0)) > 0 else None),
                voice_prompt=getattr(args, "voice_prompt", ""),
                voice_prompt_dir=getattr(args, "voice_prompt_dir", ""),
                text_prompt=getattr(args, "text_prompt", ""),
            )
        return moshi_engine

    if bool(getattr(args, "enable_moshi_reply", False)):
        print("[liveTryHeliumFM_ws_binary] eager-loading Moshi/PersonaPlex reply engine", flush=True)
        get_moshi_engine()

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "live_moshi_reply_helium_fm" if args.enable_moshi_reply else "live_helium_fm",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "loaded": engine is not None,
        })

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        fm_engine = get_engine()
        fm_engine.reset_session()
        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None
        if reply_engine is not None:
            reply_engine.reset_session()
        browser_input_sr = 48000
        audio_packets_seen = 0

        # -- Queues for the reply pipeline --
        # mic_q: (raw_bytes, input_sr) or None to stop
        # frame_q: per-frame packet dicts or None to stop
        mic_q: queue.Queue[tuple[bytes, int] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        sender_task: asyncio.Task | None = None
        event_loop: asyncio.AbstractEventLoop | None = None
        prebuffer_ready = threading.Event()
        session_started = threading.Event()
        last_mic_level_log_wall = 0.0

        stream_task: asyncio.Task | None = None

        await ws.send_json({
            "type": "server_ready",
            "sample_rate": TARGET_SR,
            "model_type": "moshi_reply+helium_fm+renderer" if args.enable_moshi_reply else "helium_fm+renderer",
            "tokens_per_chunk": int(args.fm_chunk_frames),
            "has_audio_file": fm_engine.audio_pcm is not None,
            "buffer_ms": int(args.buffer_ms),
            "av_transport": "binary",
            "target_fps": round(float(args.fps), 2),
        })
        print("[liveTryHeliumFM] websocket connected; sent server_ready", flush=True)

        if reply_engine is not None:
            mic_q = queue.Queue(maxsize=512)
            frame_q = asyncio.Queue(maxsize=512)
            event_loop = asyncio.get_running_loop()

            def _gpu_producer_thread() -> None:
                """GPU thread: Moshi -> Helium -> FM -> render -> JPEG -> frame_q.

                Runs Moshi at maximum GPU speed. When real mic audio is not
                yet available, pads Moshi input with silence so it can keep
                generating reply audio without waiting for real-time mic
                arrival. This dramatically cuts first-reply latency.
                """
                assert mic_q is not None and frame_q is not None and event_loop is not None
                pending_reply_steps: list[dict] = []
                pending_reply_hidden: list[torch.Tensor] = []
                pending_reply_audio: list[np.ndarray] = []
                reply_avatar_chunk_idx = 0
                chunk_produce_count = 0
                stopped = False
                # Pause producer when queue is this deep (qsize() is heuristic across threads).
                FRAME_Q_BACKPRESS = max(1, int(getattr(args, "frame_q_backpressure", 96)))
                FRAME_Q_PUT_TIMEOUT_S = 120.0
                prebuffer_chunks = max(0, int(getattr(args, "prebuffer_chunks", PREBUFFER_CHUNKS)))
                hidden_steps_per_chunk = int(getattr(args, "reply_hidden_steps_per_chunk", 0))
                if hidden_steps_per_chunk <= 0:
                    hidden_steps_per_chunk = int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps)))
                hidden_steps_per_chunk = max(1, hidden_steps_per_chunk)
                max_moshi_steps_per_loop = hidden_steps_per_chunk
                silence_low_water_frames = max(
                    int(round(float(args.fps) * 2.0)),
                    prebuffer_chunks * int(args.fm_chunk_frames),
                )
                last_real_audio_wall = time.perf_counter()
                was_silent = True

                def _enqueue_frame(pkt: dict) -> None:
                    """Block until frame_q accepts pkt (real backpressure). Must run from GPU thread."""
                    fut = asyncio.run_coroutine_threadsafe(frame_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING frame_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING frame_q.put failed: {e!r}", flush=True)

                if prebuffer_chunks <= 0 and not prebuffer_ready.is_set():
                    prebuffer_ready.set()
                    print("[GPU] prebuffer=0, sender starts immediately", flush=True)

                while not stopped:
                    # --- Phase 1: drain all available mic audio (non-blocking) ---
                    drained = 0
                    while True:
                        try:
                            item = mic_q.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stopped = True
                            break
                        raw_bytes, input_sr = item
                        if raw_bytes:
                            pcm_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            rms = float(np.sqrt(np.mean(pcm_f32 ** 2))) if pcm_f32.size else 0.0
                            if rms > 0.003:
                                last_real_audio_wall = time.perf_counter()
                            no_audio_for = time.perf_counter() - last_real_audio_wall
                            if rms > 0.003 or no_audio_for < 0.25:
                                reply_engine.append_browser_pcm(
                                    np.frombuffer(raw_bytes, dtype=np.int16), input_sr
                                )
                        drained += 1

                    if stopped:
                        break

                    # --- Phase 2: yield if outbound queue is deep (sender is the bottleneck) ---
                    while frame_q.qsize() >= FRAME_Q_BACKPRESS:
                        time.sleep(0.004)

                    if not session_started.is_set():
                        time.sleep(0.003)
                        continue

                    if reply_engine.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                        q_depth = frame_q.qsize()
                        no_audio_for = time.perf_counter() - last_real_audio_wall
                        if no_audio_for >= 0.25 and q_depth < silence_low_water_frames:
                            missing = MIMI_FRAME_SIZE - int(reply_engine.input_buffer.shape[0])
                            pad = np.zeros(max(0, missing), dtype=np.float32)
                            if reply_engine.input_buffer.shape[0] > 0:
                                reply_engine.input_buffer = np.concatenate(
                                    [reply_engine.input_buffer, pad],
                                    axis=0,
                                )
                            else:
                                reply_engine.input_buffer = pad
                        else:
                            time.sleep(0.003)
                            continue

                    # --- Phase 3: run all Moshi steps and feed FM ---
                    t_recv = time.perf_counter()
                    for ev in reply_engine.process_ready_steps_limited(max_moshi_steps_per_loop):
                        pending_reply_steps.append(ev)
                        fm_engine._session_reply_events.append({
                            "step": int(ev.get("step", -1)),
                            "token": int(ev.get("token", -1)),
                            "piece": str(ev.get("piece", "")),
                            "audio_text": str(ev.get("audio_text", "")),
                            "reply_rms": float(ev.get("reply_rms", 0.0)),
                            "reply_peak": float(ev.get("reply_peak", 0.0)),
                            "input_rms": float(ev.get("input_rms", 0.0)),
                            "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                            "total_ms": float(ev.get("total_ms", 0.0)),
                        })
                        reply_pcm = (
                            np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        if bool(getattr(args, "direct_reply_hidden", False)):
                            hidden = ev.get("helium_hidden")
                            if not isinstance(hidden, torch.Tensor):
                                continue
                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())
                            pending_reply_audio.append(reply_pcm)
                            if len(pending_reply_hidden) < hidden_steps_per_chunk:
                                continue

                            used_hidden = pending_reply_hidden[:hidden_steps_per_chunk]
                            used_audio = pending_reply_audio[:hidden_steps_per_chunk]
                            used_steps = pending_reply_steps[:hidden_steps_per_chunk]
                            pending_reply_hidden = pending_reply_hidden[hidden_steps_per_chunk:]
                            pending_reply_audio = pending_reply_audio[hidden_steps_per_chunk:]
                            pending_reply_steps = pending_reply_steps[hidden_steps_per_chunk:]

                            is_speech = False
                            for s in used_steps:
                                t = s.get("token", -1)
                                rms = s.get("reply_rms", 0.0)
                                if (t not in (-1, 0, 3)) or rms > 0.005:
                                    is_speech = True
                                    break

                            if is_speech:
                                if was_silent:
                                    q_size = frame_q.qsize()
                                    print(f"[GPU] Transition from silence to speech. Clearing frame_q of size {q_size}", flush=True)
                                    def _clear():
                                        while not frame_q.empty():
                                            try:
                                                frame_q.get_nowait()
                                            except asyncio.QueueEmpty:
                                                break
                                    event_loop.call_soon_threadsafe(_clear)
                                    fm_engine.abs_frame = max(0, fm_engine.abs_frame - q_size)
                                    was_silent = False
                            else:
                                was_silent = True

                            helium_chunk = torch.cat(used_hidden, dim=0)
                            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
                            target_frames = max(1, int(round(len(pcm_chunk) * float(args.fps) / TARGET_SR)))
                            motion, fm_info = fm_engine._sample_motion_from_helium(helium_chunk, target_frames)
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)
                            fm_engine._record_session_chunk(pcm_chunk, motion, fm_info)
                        else:
                            result = fm_engine.feed_pcm_f32(reply_pcm)
                            if result is None:
                                continue

                            motion, fm_info, pcm_chunk = result
                            steps_per_avatar_chunk = max(1, int(round(len(pcm_chunk) / MIMI_FRAME_SIZE)))
                            used_steps = pending_reply_steps[:steps_per_avatar_chunk]
                            pending_reply_steps = pending_reply_steps[steps_per_avatar_chunk:]
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)

                        reply_avatar_chunk_idx += 1
                        chunk_produce_count += 1

                        moshi_total_ms = sum(float(s.get("total_ms", 0.0)) for s in used_steps)

                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        frame_audio = split_audio_into_frame_slices(pcm_chunk, args.fps)
                        n_frames = int(motion.shape[0])
                        emitted = int(fm_info["abs_start"])
                        text_payload = ev.get("audio_text") or ev.get("sampled_text") or ""
                        total_gen_ms = (
                            moshi_total_ms + float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])
                        )

                        t_chunk_start = time.perf_counter()

                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sb_end = min(sb_start + fm_engine.render_sub_batch, n_frames)
                            sub_motion = motion[sb_start:sb_end]
                            sub_audio = frame_audio[sb_start:sb_end]

                            packets = fm_engine.render_and_encode_subbatch(
                                sub_motion,
                                sub_audio,
                                abs_start=emitted + sb_start,
                                text_payload=text_payload,
                                avatar_chunk_id=avatar_chunk_id,
                                total_gen_ms=total_gen_ms,
                            )

                            for pkt in packets:
                                _enqueue_frame(pkt)

                        chunk_wall_ms = _ms(t_chunk_start)
                        produce_latency_ms = _ms(t_recv)
                        q_depth = frame_q.qsize() if frame_q is not None else -1

                        print(
                            f"[GPU][chunk#{reply_avatar_chunk_idx}] "
                            f"moshi={moshi_total_ms:.0f}ms "
                            f"helium={float(fm_info['helium_ms']):.0f}ms "
                            f"fm={float(fm_info['fm_ms']):.0f}ms "
                            f"render+jpeg={chunk_wall_ms:.0f}ms "
                            f"frames={n_frames} "
                            f"produce_latency={produce_latency_ms:.0f}ms "
                            f"frame_q={q_depth} "
                            f"abs={emitted}",
                            flush=True,
                        )

                        if (
                            chunk_produce_count == prebuffer_chunks
                            and not prebuffer_ready.is_set()
                        ):
                            prebuffer_ready.set()
                            print(
                                f"[GPU] prebuffer ready: {q_depth} frames queued "
                                f"after {chunk_produce_count} chunks",
                                flush=True,
                            )

                fut_done = asyncio.run_coroutine_threadsafe(frame_q.put(None), event_loop)
                try:
                    fut_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING sentinel put: {e!r}", flush=True)

            async def _reply_sender() -> None:
                """Wait for prebuffer, then drain frame_q at 25fps."""
                assert frame_q is not None
                send_start_wall: float | None = None
                frames_sent = 0
                starvation_events = 0
                starve_start: float | None = None
                ws_closed = False

                # Wait for GPU producer to fill the prebuffer
                while not prebuffer_ready.is_set():
                    await asyncio.sleep(0.01)
                q_depth_at_start = frame_q.qsize()
                print(
                    f"[SENDER] prebuffer filled, starting pacing with "
                    f"{q_depth_at_start} frames queued",
                    flush=True,
                )

                while True:
                    if ws_closed:
                        break

                    try:
                        packet = frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        if send_start_wall is not None and starve_start is None:
                            starve_start = time.perf_counter()
                            starvation_events += 1
                        await asyncio.sleep(0.004)
                        continue

                    if packet is None:
                        break

                    if starve_start is not None:
                        gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                        if gap_ms > 100:
                            print(
                                f"[SENDER] STARVED {gap_ms:.0f}ms "
                                f"(event #{starvation_events}) "
                                f"frame_q={frame_q.qsize()} sent={frames_sent}",
                                flush=True,
                            )
                        starve_start = None

                    idx = int(packet["frame_number"])

                    if send_start_wall is None:
                        send_start_wall = time.perf_counter()

                    try:
                        if packet.get("ws_kind") == "bytes":
                            await ws.send_bytes(packet["data"])
                        else:
                            await ws.send_json(packet["msg"])
                    except (WebSocketDisconnect, RuntimeError, Exception):
                        ws_closed = True
                        break
                    frames_sent += 1

                    target_t = send_start_wall + frames_sent / float(args.fps)
                    now = time.perf_counter()
                    sleep_s = target_t - now

                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    elif sleep_s < -0.5:
                        send_start_wall = now - (frames_sent / float(args.fps)) + 0.04
                        print(
                            f"[SENDER] RE-ANCHOR at frame {idx}, was {-sleep_s*1000:.0f}ms behind",
                            flush=True,
                        )

                    if frames_sent % 50 == 0:
                        q_depth = frame_q.qsize()
                        elapsed = time.perf_counter() - send_start_wall if send_start_wall else 0
                        print(
                            f"[SENDER] sent={frames_sent} frame={idx} "
                            f"frame_q={q_depth} "
                            f"elapsed={elapsed:.1f}s "
                            f"starve_events={starvation_events}",
                            flush=True,
                        )

            gpu_thread = threading.Thread(target=_gpu_producer_thread, daemon=True, name="gpu-producer")
            gpu_thread.start()
            sender_task = asyncio.create_task(_reply_sender())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if reply_engine is None:
                        continue
                    assert mic_q is not None
                    audio_packets_seen += 1
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    mic_rms = float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))) if pcm_i16.size else 0.0
                    mic_peak = float(np.max(np.abs(pcm_i16.astype(np.float32) / 32768.0))) if pcm_i16.size else 0.0
                    now_wall = time.perf_counter()
                    if audio_packets_seen <= 3 or now_wall - last_mic_level_log_wall >= 1.0:
                        voice = "VOICE" if mic_rms >= 0.02 else "quiet"
                        print(
                            f"[MIC] packet={audio_packets_seen} rms={mic_rms:.5f} "
                            f"peak={mic_peak:.3f} sr={browser_input_sr} {voice}",
                            flush=True,
                        )
                        last_mic_level_log_wall = now_wall
                    if audio_packets_seen <= 3:
                        print(
                            f"[liveTryHeliumFM] rx binary mic packet#{audio_packets_seen} "
                            f"bytes={len(data)} sr={browser_input_sr}",
                            flush=True,
                        )
                    if not session_started.is_set():
                        session_started.set()
                        print(
                            "[liveTryHeliumFM] auto-started session from first binary mic packet",
                            flush=True,
                        )
                    try:
                        mic_q.put_nowait((bytes(data), int(browser_input_sr)))
                    except queue.Full:
                        pass
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type", "")).lower()
                print(f"[liveTryHeliumFM] rx text type={msg_type or '<empty>'}", flush=True)

                if msg_type == "start":
                    browser_input_sr = int(payload.get("sample_rate", payload.get("sampleRate", browser_input_sr)))
                    if fm_engine.audio_pcm is not None and (stream_task is None or stream_task.done()):
                        fm_engine.reset_session()
                        stream_task = asyncio.create_task(stream_from_file(ws, fm_engine))
                    if reply_engine is not None:
                        reply_engine.reset_session()
                        session_started.set()
                    print(
                        "[liveTryHeliumFM] start → "
                        + ("streaming from file" if fm_engine.audio_pcm is not None else "live Moshi reply mode"),
                        flush=True,
                    )

                elif msg_type == "chunk_audio":
                    pcm_b64 = payload.get("pcm_s16le_b64", "")
                    if not pcm_b64:
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    result = fm_engine.feed_pcm(pcm_bytes)
                    if result is not None:
                        motion, fm_info, _pcm_chunk = result
                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        n_frames = int(motion.shape[0])
                        emitted = fm_info["abs_start"]
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sub = motion[sb_start:sb_start + fm_engine.render_sub_batch].to(
                                fm_engine.device, dtype=fm_engine.dtype
                            )
                            frames_np, _ = fm_engine._render_motion(sub)
                            for j, frame_rgb in enumerate(frames_np):
                                idx = emitted + sb_start + j
                                await ws.send_json({
                                    "type": "chunk_frame",
                                    "chunk_id": idx + 1,
                                    "frame_idx": 0,
                                    "jpeg_b64": encode_jpeg_b64(frame_rgb, fm_engine.jpeg_quality),
                                    "moshi_text": (
                                        f"live Helium+FM "
                                        f"helium={fm_info['helium_ms']:.0f}ms "
                                        f"fm={fm_info['fm_ms']:.0f}ms"
                                    ),
                                    "server_fps": round(float(args.fps), 1),
                                    "chunks_done": avatar_chunk_id,
                                })

                elif msg_type == "stop":
                    print("[liveTryHeliumFM] stop requested", flush=True)
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            if mic_q is not None:
                mic_q.put(None)
            if gpu_thread is not None:
                gpu_thread.join(timeout=10.0)
            if sender_task is not None:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await sender_task
            fm_engine.dump_last_session(source="websocket_live")
            print("[liveTryHeliumFM] websocket closed", flush=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = LiveHeliumFMOptions()
    args = parser.parse()
    args.rank = args.device
    parser.print_options()

    app = build_app(args)

    import uvicorn

    print(f"[liveTryHeliumFM_ws_binary] serving {args.html_path} (binary av_transport)")
    print(f"[liveTryHeliumFM] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
