"""personaplex_imtalker_streaming_server.py -- PersonaPlex + IMTalker Unified Streaming Server
==================================================================================================
FlashTalk-v3 architecture pattern (see ../lets_talk_flashhead_v3.py) applied to the
production PersonaPlex + IMTalker avatar pipeline.

Token Bridge Architecture (model math unchanged from production):
  PersonaPlex (12.5Hz, 4096-dim Helium hidden state)
  -> HeliumTokenDeque: 100-step sliding window (8s context)
  -> IMTalkerFrontendAdapter (trained 6-layer transformer "frontend" adapter)
  -> frozen real Wav2Vec2 encoder (encode_from_projected_frontend)
  -> interpolate to target_frames @ 25fps
  -> FMGenerator (frozen flow-matching transformer, ODE sampler) -> motion latents (32-dim)
  -> IMTRenderer (frozen GAN-style renderer) -> 512x512 RGB frames

Threading model (the architectural fix vs. the legacy single-thread server
liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py, whose one "gpu-producer" thread did
PersonaPlex stepping AND IMTalker FM-sampling/rendering/JPEG-encoding serially):
  [PersonaPlexThread] PersonaPlexEngine.run_streaming  -> helium_queue
  [IMTalkerThread]    PersonaPlexConversationSession._imtalker_loop -> dispatch_queue
  [asyncio]           PersonaPlexConversationSession._receiver / ._dispatcher

PersonaPlex generation and IMTalker rendering are now fully decoupled by helium_queue, so
avatar render time can never stall PersonaPlex's ability to keep generating at real-time
speed -- mirroring FlashTalk-v3's MoshiEngine/FlashHeadTokenEngine thread separation.

Wire protocol (frozen, do not edit): binary "AV01" frames packed by ws_av_binary_codec.py,
consumed by static/index_v3_binary_fullscreen.html. The JSON "server_ready" handshake fields
also match exactly what that HTML already parses.

This file intentionally imports NOTHING from liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
so that file -- and the offline tools that import from it directly
(offline_personaplex_imtalker_infer.py, render_saved_live_helium.py) -- remain completely
unaffected by this migration.

Usage:
  python personaplex_imtalker_streaming_server.py --generator_path ... --renderer_path ... \
      --adapter_path ... --ref_path ... --moshi_weight ... --quantize_4bit --voice_prompt NATM0.pt
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import queue
import sys
import threading
import time
import types
from collections import namedtuple
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.FM import FMGenerator
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
from renderer.models import IMTRenderer
from liveTry import MoshiOnlyEngine
import ws_av_binary_codec as _wsbin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR = 24_000          # Mimi/PersonaPlex sample rate
VIDEO_FPS_DEFAULT = 25.0    # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
HELIUM_DIM = 4096           # PersonaPlex transformer hidden width
WAV2VEC_SR = 16_000
HELIUM_DEQUE_SIZE = 100     # 8s of Helium hidden steps @ 12.5Hz

MIC_QUEUE_MAXSIZE = 64
HELIUM_QUEUE_MAXSIZE = 256
DISPATCH_QUEUE_MAXSIZE = 8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _queue_put_latest(q: "queue.Queue", item) -> None:
    """Bounded insert -- drops the oldest item when full.

    Matches FlashTalk-v3's MoshiEngine.run_streaming -> token_queue policy: under
    backpressure we prefer dropping stale generation output over blocking the realtime
    PersonaPlex stepping loop (which must keep pace with the microphone).
    """
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


def _drain_queue(q: "queue.Queue") -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


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


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# HeliumTokenDeque -- 8s sliding window of PersonaPlex Helium hidden vectors
# ---------------------------------------------------------------------------

class HeliumTokenDeque:
    """Thread-safe fixed-size ring buffer [deque_size, dim] of Helium hidden states.

    Mirrors FlashTalk-v3's MoshiTokenDeque. Initialized with zeros (silence).
    PersonaPlexThread-derived chunks are pushed by the IMTalker consumer thread;
    IMTalkerEngine reads snapshots. Lives on CPU (like MoshiTokenDeque) -- snapshots are
    moved to the compute device only for the duration of one adapter forward pass, so no
    thread holds an exclusive CUDA tensor across iterations.
    """

    def __init__(self, size: int = HELIUM_DEQUE_SIZE, dim: int = HELIUM_DIM):
        self._lock = threading.Lock()
        self._size = size
        self._dim = dim
        self._buffer = torch.zeros(size, dim)
        self._filled = 0
        self._total_pushed = 0

    def push_batch(self, tokens: torch.Tensor) -> None:
        """tokens: [N, dim] cpu/cuda float tensor."""
        tokens = tokens.detach().to("cpu", dtype=torch.float32)
        n = int(tokens.shape[0])
        with self._lock:
            if n >= self._size:
                self._buffer = tokens[-self._size:].contiguous().clone()
            else:
                self._buffer = torch.cat([self._buffer[n:], tokens], dim=0).contiguous()
            self._filled = min(self._size, self._filled + n)
            self._total_pushed += n

    def snapshot(self) -> torch.Tensor:
        with self._lock:
            return self._buffer.clone()

    @property
    def filled(self) -> int:
        with self._lock:
            return self._filled

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed


# ---------------------------------------------------------------------------
# PersonaPlexEngine -- PersonaPlex/Moshi reply engine exposing per-step Helium hidden
# ---------------------------------------------------------------------------

class PersonaPlexEngine(MoshiOnlyEngine):
    """PersonaPlex reply engine that also returns the main LM hidden state for each step.

    Ported verbatim from the production live script's MoshiOnlyEngineWithHidden
    (liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py). The hidden state is exposed
    as a native LMGen output -- via state.graphed_main(...) on the PersonaPlex path, or via a
    captured intermediate transformer layer on the generic fallback path -- so CUDA-graph
    replay stays enabled (no Python forward hooks).
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
            print("[PersonaPlexEngine] installed PersonaPlex graphed hidden capture", flush=True)
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
            _, T_, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T_, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T_, state.offsets)

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
        print(f"[PersonaPlexEngine] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        codes = self.mimi.encode(chunk)
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        lm_out = self.lm_gen._step(codes[:, :, :1])

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"PersonaPlex graph layer[-2] contract failure: got {type(lm_out)}")
            tokens, _transformer_out, layer_hidden = lm_out
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()

        token = -1
        token_piece = ""
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        total_ms = 1000.0 * (time.perf_counter() - t0)

        return {
            "step": int(self.step),
            "reply_pcm": reply_pcm,
            "reply_rms": reply_rms,
            "token": token,
            "piece": token_piece,
            "audio_text": self.audio_text,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
            "total_ms": total_ms,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process at most max_steps Mimi frames -- keeps the producer/consumer interleaved."""
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events

    def run_streaming(
        self,
        mic_queue: "queue.Queue",
        helium_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        """PersonaPlexThread main loop -- mirrors FlashTalk-v3's MoshiEngine.run_streaming.

        Does ONLY PersonaPlex stepping (Mimi encode -> LM step -> Mimi decode) and pushes
        (step_id, helium_hidden[1,4096] cpu, reply_pcm[1920] np.float32, arrival_ts,
        reply_codes, is_speech) onto helium_queue. IMTalker generation/rendering happens in
        a separate thread/consumer reading from helium_queue -- this is the core
        architectural fix vs. the legacy single-thread "gpu-producer".
        """
        print("[PersonaPlexThread] starting...", flush=True)
        last_real_audio_wall = time.perf_counter()
        try:
            while not stop_event.is_set():
                while True:
                    try:
                        raw_bytes, input_sr = mic_queue.get_nowait()
                    except queue.Empty:
                        break
                    if raw_bytes:
                        pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                        rms = (
                            float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2)))
                            if pcm_i16.size else 0.0
                        )
                        if rms > 0.003:
                            last_real_audio_wall = time.perf_counter()
                        self.append_browser_pcm(pcm_i16, input_sr)

                no_audio_for = time.perf_counter() - last_real_audio_wall
                if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                    if no_audio_for >= 0.25:
                        missing = MIMI_FRAME_SIZE - int(self.input_buffer.shape[0])
                        pad = np.zeros(max(0, missing), dtype=np.float32)
                        self.input_buffer = (
                            np.concatenate([self.input_buffer, pad])
                            if self.input_buffer.shape[0] else pad
                        )
                    else:
                        time.sleep(0.003)
                        continue

                for ev in self.process_ready_steps_limited(1):
                    hidden = ev.get("helium_hidden")
                    if not isinstance(hidden, torch.Tensor):
                        continue
                    t = ev.get("token", -1)
                    rms = ev.get("reply_rms", 0.0)
                    is_speech = (t not in (-1, 0, 3)) or rms > 0.005
                    _queue_put_latest(
                        helium_queue,
                        (
                            ev["step"],
                            hidden.squeeze(0).contiguous(),  # [1, 4096]
                            ev["reply_pcm"],
                            time.perf_counter(),
                            ev.get("reply_codes"),
                            is_speech,
                        ),
                    )
        except Exception as e:
            print(f"[PersonaPlexThread] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            print("[PersonaPlexThread] Exiting.", flush=True)


# ---------------------------------------------------------------------------
# IMTalkerFrontendAdapter -- trained Helium->Wav2Vec2-frontend + frozen real Wav2Vec2
# ---------------------------------------------------------------------------

class IMTalkerFrontendAdapter(nn.Module):
    """Frontend adapter: trained Helium->Wav2Vec2-frontend (6L transformer) + frozen Wav2Vec2.

    Ported verbatim from the production live script's StudioNativeLiveAdapter. Training
    contract: raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768] -> frozen Wav2Vec2
    encoder -> final hidden -> IMTalker audio_projection. No math changes -- same checkpoint
    contract (phase2_best_wav2vec_final_loss.pt, adapter_num_layers=6).
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


# ---------------------------------------------------------------------------
# ChunkBundle -- groups audio + video for A/V sync (FlashTalk-v3 pattern)
# ---------------------------------------------------------------------------

ChunkBundle = namedtuple(
    "ChunkBundle",
    ["chunk_id", "audio_pcm", "video_frames", "gen_time_ms",
     "start_step_id", "end_step_id", "first_token_ts", "last_token_ts", "frames_ready_ts"],
)


# ---------------------------------------------------------------------------
# IMTalkerEngine -- FM (flow-matching) + IMTRenderer, frozen checkpoints
# ---------------------------------------------------------------------------

class IMTalkerEngine:
    """Owns IMTalker's FM generator + renderer + frontend adapter + ref-image precompute.

    generate_chunk() is the FlashHeadTokenEngine.generate_chunk equivalent: takes a batch of
    fresh Helium hidden steps, pushes them into its HeliumTokenDeque, and returns rendered
    RGB frames. Runs in its own thread (IMTalkerThread), decoupled from PersonaPlexEngine.
    """

    def __init__(self, args: argparse.Namespace, device: Optional[str] = None) -> None:
        self.args = args
        self.device = torch.device(device or args.device)
        self.dtype = torch.float32 if getattr(args, "fp32", False) else torch.bfloat16
        self.fps = float(args.fps)
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)

        self.helium_deque = HeliumTokenDeque(HELIUM_DEQUE_SIZE, HELIUM_DIM)
        self.stream_state = None
        self.abs_frame = 0
        self.noise_buf: Optional[torch.Tensor] = None

        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_started_wall = time.time()

        self.jpeg_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")
        self.loaded = False

    # -- loading --------------------------------------------------------

    def load(self) -> None:
        if self.loaded:
            return
        args = self.args
        if getattr(args, "tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        t_total = time.perf_counter()
        self.fm = self._load_fm(args, self.device)
        self.renderer = self._load_renderer(args, self.device, self.dtype)

        t_adapter = time.perf_counter()
        self.adapter = IMTalkerFrontendAdapter(
            args.wav2vec_model_path, args.adapter_num_layers, args.adapter_dropout,
        ).to(self.device).float().eval()
        payload = torch.load(args.adapter_path, map_location="cpu")
        state = payload.get("adapter", payload.get("model", payload)) if isinstance(payload, dict) else payload
        self.adapter.load_state_dict(state, strict=True)
        _sync_cuda()
        print(
            f"[IMTalkerEngine][adapter] loaded in {_ms(t_adapter):.0f}ms "
            f"path={args.adapter_path} layers={args.adapter_num_layers}",
            flush=True,
        )

        ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
        with torch.no_grad():
            self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
            self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
            ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
            self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()

        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(1, max_frames, int(args.dim_w), device=self.device, generator=gen)
            print(f"[IMTalkerEngine] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        self._warmup()
        self.loaded = True
        print(
            f"[IMTalkerEngine] ready -- total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} dtype={self.dtype}",
            flush=True,
        )

    @staticmethod
    def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
        t0 = time.perf_counter()
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
            f"[IMTalkerEngine][FM] loaded in {_ms(t0):.0f}ms "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        return fm

    @staticmethod
    def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
        t0 = time.perf_counter()
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
                return renderer.decode(m_c, m_r, f_r)
            renderer._fused_render = torch.compile(_fused_render)
        print(
            f"[IMTalkerEngine][renderer] loaded in {_ms(t0):.0f}ms "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        return renderer

    # -- warmup / session lifecycle --------------------------------------

    @torch.no_grad()
    def _warmup(self) -> None:
        raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / self.fps)))
        dummy_helium = torch.zeros(raw_steps, HELIUM_DIM, device=self.device)
        t0 = time.perf_counter()
        frames_np, _motion, _abs, _feat = self.generate_chunk(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[IMTalkerEngine][warmup] fm+render={_ms(t0):.0f}ms frames={frames_np.shape}", flush=True)
        _ = encode_jpeg_bytes(np.zeros((512, 512, 3), dtype=np.uint8), self.jpeg_quality)
        self.reset_session()

    def reset_session(self) -> None:
        self.stream_state = None
        self.abs_frame = 0
        self.helium_deque = HeliumTokenDeque(HELIUM_DEQUE_SIZE, HELIUM_DIM)
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_25_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_started_wall = time.time()

    # -- generation -------------------------------------------------------

    @torch.no_grad()
    def _sample_motion(self, helium_chunk: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        helium_chunk = helium_chunk.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        self.helium_deque.push_batch(helium_chunk)
        deque_snapshot = self.helium_deque.snapshot().to(self.device)

        target_len_25_full = HELIUM_DEQUE_SIZE * 2
        _frontend50, _final50, feat_25_full = self.adapter.forward_single(deque_snapshot, target_len_25_full)
        fresh_frames = max(1, int(helium_chunk.shape[0]) * 2)
        feat_25 = feat_25_full[-fresh_frames:].contiguous()
        if int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0), size=target_frames, mode="linear", align_corners=False,
            ).squeeze(0).T.contiguous()

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        if self.noise_buf is not None:
            end_frame = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, self.abs_frame:end_frame]

        motion, self.stream_state = self.fm.sample(
            data,
            a_cfg_scale=float(self.args.a_cfg_scale),
            nfe=int(self.args.nfe),
            stream_state=self.stream_state,
            return_state=True,
        )
        motion = motion.squeeze(0)[:target_frames].detach()
        abs_start = self.abs_frame
        self.abs_frame += target_frames
        return motion, feat_25, abs_start

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> np.ndarray:
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])
        g_r_sub = self.g_r.expand(n, -1)
        m_r_sub = tuple(m.expand(n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(n, -1, -1, -1) for f in self.f_r]

        fused = getattr(self.renderer, "_fused_render", None)
        if fused is not None:
            frames = fused(motion, g_r_sub, m_r_sub, f_r_sub)
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        return frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    @torch.no_grad()
    def generate_chunk(
        self, helium_chunk: torch.Tensor, target_frames: int,
    ) -> tuple[np.ndarray, torch.Tensor, int, torch.Tensor]:
        """Returns (frames_np[n,512,512,3] uint8, motion[n,32], abs_start, feat_25[n,768])."""
        motion, feat_25, abs_start = self._sample_motion(helium_chunk, target_frames)
        frames_parts = []
        for sb in range(0, target_frames, self.render_sub_batch):
            sub = motion[sb: sb + self.render_sub_batch]
            frames_parts.append(self._render_motion(sub))
        frames_np = (
            np.concatenate(frames_parts, axis=0) if frames_parts
            else np.zeros((0, 512, 512, 3), dtype=np.uint8)
        )
        return frames_np, motion, abs_start, feat_25

    # -- session dump-to-disk (--dump_motion / --dump_dir) ----------------

    def record_chunk(
        self,
        pcm_chunk: np.ndarray,
        motion: torch.Tensor,
        helium_chunk: torch.Tensor,
        feat_25: torch.Tensor,
        live_codes: list[torch.Tensor],
    ) -> None:
        if not self.dump_motion:
            return
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        self._session_helium_parts.append(helium_chunk.detach().float().cpu().clone())
        self._session_adapter_25_parts.append(feat_25.detach().float().cpu().clone())
        if live_codes:
            self._session_live_token_parts.extend(
                c.to(dtype=torch.int16).contiguous() for c in live_codes if isinstance(c, torch.Tensor)
            )
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "frames": int(motion.shape[0]),
            "samples": int(len(pcm_chunk)),
        })

    def dump_last_session(self, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = (
            np.concatenate(self._session_audio_parts, axis=0)
            if self._session_audio_parts else np.empty(0, dtype=np.float32)
        )
        helium = torch.cat(self._session_helium_parts, dim=0).contiguous() if self._session_helium_parts else None
        adapter_25 = (
            torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
            if self._session_adapter_25_parts else None
        )
        live_tokens = (
            torch.cat(self._session_live_token_parts, dim=2).contiguous()
            if self._session_live_token_parts else None
        )

        torch.save(
            {"motion": motion, "chunks": self._session_chunk_rows, "fps": float(self.fps), "source": source},
            session_dir / "full_motion.pt",
        )
        if helium is not None:
            torch.save({"helium": helium, "source": source}, session_dir / "full_helium_raw.pt")
        if adapter_25 is not None:
            torch.save({"adapter_feat_25": adapter_25, "source": source}, session_dir / "full_adapter_w2v_25fps.pt")
        if live_tokens is not None:
            torch.save({"live_mimi_tokens": live_tokens, "source": source}, session_dir / "live_mimi_tokens.pt")
        if audio.size > 0:
            torchaudio.save(str(session_dir / "full_personaplex_reply_24k.wav"), torch.from_numpy(audio).view(1, -1), TARGET_SR)

        meta = {
            "source": source,
            "fps": float(self.fps),
            "motion_frames": int(motion.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size else 0.0,
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[IMTalkerEngine] dumped last session -> {session_dir}", flush=True)
        return session_dir


# ---------------------------------------------------------------------------
# PersonaPlexConversationSession -- WebSocket session managing all threads
# ---------------------------------------------------------------------------

class PersonaPlexConversationSession:
    """Manages a single conversation session between browser client and server.

    Thread architecture:
      [PersonaPlexThread] -> helium_queue -> [IMTalkerThread] -> dispatch_queue -> [Dispatcher async]
                                                                                    |
      [Receiver async] <- WebSocket <-------------------------------------------------+

    Wire protocol is unchanged from the legacy live script: JSON "server_ready" handshake,
    then binary "AV01" frames (ws_av_binary_codec.pack_av_frame) -- index_v3_binary_fullscreen.html
    needs zero changes.
    """

    def __init__(
        self,
        websocket: WebSocket,
        personaplex: PersonaPlexEngine,
        imtalker: IMTalkerEngine,
        args: argparse.Namespace,
    ) -> None:
        self.ws = websocket
        self.personaplex = personaplex
        self.imtalker = imtalker
        self.args = args

        self.stop_event = threading.Event()
        self.mic_queue: "queue.Queue" = queue.Queue(maxsize=MIC_QUEUE_MAXSIZE)
        self.helium_queue: "queue.Queue" = queue.Queue(maxsize=HELIUM_QUEUE_MAXSIZE)
        self.dispatch_queue: "queue.Queue" = queue.Queue(maxsize=DISPATCH_QUEUE_MAXSIZE)

        self.client_sr = 48000
        self.personaplex_thread: Optional[threading.Thread] = None
        self.imtalker_thread: Optional[threading.Thread] = None

        self._session_t0 = 0.0
        self._total_chunks = 0
        self._total_frames_sent = 0
        self._avg_gen_ms = 0.0
        self._sync_first_gap_sum = 0.0
        self._sync_last_gap_sum = 0.0
        self._sync_count = 0

    @property
    def avg_first_gap_ms(self) -> float:
        return (self._sync_first_gap_sum / max(1, self._sync_count)) * 1000

    @property
    def avg_last_gap_ms(self) -> float:
        return (self._sync_last_gap_sum / max(1, self._sync_count)) * 1000

    # -- Microphone / control ingest -------------------------------------

    async def _receiver(self) -> None:
        while not self.stop_event.is_set():
            msg = await self.ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            text = msg.get("text")
            data = msg.get("bytes")

            if text is not None:
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                t = str(obj.get("type", "")).lower()
                if t == "start":
                    self.client_sr = int(obj.get("sample_rate", obj.get("sampleRate", self.client_sr)))
                    print(f"[Session] client sample_rate={self.client_sr}", flush=True)
                elif t == "stop":
                    self.stop_event.set()
                    return
                continue

            if data is not None:
                try:
                    self.mic_queue.put_nowait((bytes(data), int(self.client_sr)))
                except queue.Full:
                    pass

    # -- IMTalker consumer thread -----------------------------------------

    def _imtalker_loop(self) -> None:
        """IMTalkerThread main loop: accumulate Helium hidden steps, generate, bundle.

        Mirrors FlashTalk-v3's _flashhead_loop. Runs independently of PersonaPlexThread --
        avatar render time here never stalls PersonaPlex generation.
        """
        hidden_steps_per_chunk = int(getattr(self.args, "reply_hidden_steps_per_chunk", 0))
        if hidden_steps_per_chunk <= 0:
            hidden_steps_per_chunk = max(
                1, int(round(float(self.args.fm_chunk_frames) * 12.5 / float(self.args.fps)))
            )

        chunk_id = 0
        pending_hidden: list[torch.Tensor] = []
        pending_audio: list[np.ndarray] = []
        pending_codes: list[torch.Tensor] = []
        start_step: Optional[int] = None
        first_ts: Optional[float] = None
        was_silent = True

        print(f"[IMTalkerThread] starting. hidden_steps_per_chunk={hidden_steps_per_chunk}", flush=True)
        try:
            while not self.stop_event.is_set():
                try:
                    step_id, hidden, pcm, ts, codes, is_speech = self.helium_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                if is_speech and was_silent:
                    # Silence -> speech transition: drop stale buffered frames so the
                    # avatar doesn't visibly lag behind a fresh utterance.
                    _drain_queue(self.dispatch_queue)
                    was_silent = False
                elif not is_speech:
                    was_silent = True

                pending_hidden.append(hidden)
                pending_audio.append(pcm)
                if codes is not None:
                    pending_codes.append(codes)
                if start_step is None:
                    start_step = step_id
                    first_ts = ts
                last_step = step_id
                last_ts = ts

                if len(pending_hidden) < hidden_steps_per_chunk:
                    continue

                helium_chunk = torch.cat(pending_hidden[:hidden_steps_per_chunk], dim=0)
                pcm_chunk = np.concatenate(pending_audio[:hidden_steps_per_chunk], axis=0).astype(np.float32, copy=False)
                used_codes = pending_codes[:hidden_steps_per_chunk]
                pending_hidden = pending_hidden[hidden_steps_per_chunk:]
                pending_audio = pending_audio[hidden_steps_per_chunk:]
                pending_codes = pending_codes[hidden_steps_per_chunk:]

                target_frames = max(1, int(round(len(pcm_chunk) * float(self.args.fps) / TARGET_SR)))

                t0 = time.perf_counter()
                frames_np, motion, _abs_start, feat_25 = self.imtalker.generate_chunk(helium_chunk, target_frames)
                _sync_cuda()
                gen_ms = _ms(t0)
                frames_ready_ts = time.perf_counter()
                self.imtalker.record_chunk(pcm_chunk, motion, helium_chunk, feat_25, used_codes)

                self._avg_gen_ms = gen_ms if self._avg_gen_ms == 0 else 0.8 * self._avg_gen_ms + 0.2 * gen_ms

                if first_ts is not None:
                    self._sync_first_gap_sum += first_ts - frames_ready_ts
                    self._sync_last_gap_sum += last_ts - frames_ready_ts
                    self._sync_count += 1

                chunk_id += 1
                self._total_chunks = chunk_id
                bundle = ChunkBundle(
                    chunk_id=chunk_id,
                    audio_pcm=pcm_chunk,
                    video_frames=frames_np,
                    gen_time_ms=gen_ms,
                    start_step_id=start_step or 0,
                    end_step_id=last_step or 0,
                    first_token_ts=first_ts or 0.0,
                    last_token_ts=last_ts or 0.0,
                    frames_ready_ts=frames_ready_ts,
                )
                _queue_put_latest(self.dispatch_queue, bundle)
                start_step = None
                first_ts = None

                if chunk_id <= 3 or chunk_id % 10 == 0:
                    print(
                        f"  [Chunk {chunk_id:>4d}] gen={gen_ms:.0f}ms avg={self._avg_gen_ms:.0f}ms "
                        f"frames={frames_np.shape[0]} deque={self.imtalker.helium_deque.total_pushed} "
                        f"sync_1st={self.avg_first_gap_ms:+.0f}ms sync_last={self.avg_last_gap_ms:+.0f}ms",
                        flush=True,
                    )
        except Exception as e:
            print(f"[IMTalkerThread] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            print("[IMTalkerThread] Exiting.", flush=True)

    # -- Dispatcher (async) ------------------------------------------------

    async def _dispatcher(self) -> None:
        """Pops ChunkBundles, JPEG-encodes in parallel, sends binary AV01 frames paced @fps."""
        fps = float(self.args.fps)
        frames_sent = 0
        send_start_wall: Optional[float] = None
        loop = asyncio.get_running_loop()

        while not self.stop_event.is_set():
            try:
                bundle: ChunkBundle = self.dispatch_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.004)
                continue

            frame_audio = split_audio_into_frame_slices(bundle.audio_pcm, fps)
            n = int(bundle.video_frames.shape[0])
            gen_ms_i = int(round(bundle.gen_time_ms))
            jpeg_futures = [
                loop.run_in_executor(self.imtalker.jpeg_pool, encode_jpeg_bytes, frame, self.imtalker.jpeg_quality)
                for frame in bundle.video_frames
            ]

            for i in range(n):
                if self.stop_event.is_set():
                    break
                jpeg_bytes = await jpeg_futures[i]
                audio_slice = (
                    frame_audio[i] if i < len(frame_audio)
                    else np.zeros(int(round(TARGET_SR / fps)), dtype=np.float32)
                )
                pcm_bytes = _pcm_f32_to_i16_bytes(audio_slice)
                frame_number = self._total_frames_sent
                blob = _wsbin.pack_av_frame(
                    frame_number, frame_number + 1, gen_ms_i, TARGET_SR,
                    jpeg_bytes, pcm_bytes, "", bundle.chunk_id,
                )
                try:
                    await self.ws.send_bytes(blob)
                except Exception:
                    self.stop_event.set()
                    return

                self._total_frames_sent += 1
                frames_sent += 1
                if send_start_wall is None:
                    send_start_wall = time.perf_counter()
                target_t = send_start_wall + frames_sent / fps
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                elif sleep_s < -0.5:
                    send_start_wall = time.perf_counter() - (frames_sent / fps) + 0.04

    # -- Session lifecycle --------------------------------------------------

    async def run(self) -> None:
        self._session_t0 = time.perf_counter()
        self.personaplex.reset_session()
        self.imtalker.reset_session()

        self.personaplex_thread = threading.Thread(
            target=self.personaplex.run_streaming,
            args=(self.mic_queue, self.helium_queue, self.stop_event),
            daemon=True,
            name="PersonaPlexThread",
        )
        self.personaplex_thread.start()

        self.imtalker_thread = threading.Thread(
            target=self._imtalker_loop, daemon=True, name="IMTalkerThread",
        )
        self.imtalker_thread.start()

        await self.ws.send_json({
            "type": "server_ready",
            "sample_rate": TARGET_SR,
            "model_type": "personaplex_reply+imtalker_fm+renderer",
            "tokens_per_chunk": int(self.args.fm_chunk_frames),
            "buffer_ms": int(getattr(self.args, "buffer_ms", 80)),
            "av_transport": "binary",
            "target_fps": round(float(self.args.fps), 2),
        })
        print("[Session] sent server_ready", flush=True)

        recv_task = asyncio.create_task(self._receiver())
        disp_task = asyncio.create_task(self._dispatcher())
        try:
            done, _pending = await asyncio.wait({recv_task, disp_task}, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        finally:
            self.stop_event.set()
            for task in (recv_task, disp_task):
                if not task.done():
                    task.cancel()
            if self.personaplex_thread:
                self.personaplex_thread.join(timeout=10.0)
            if self.imtalker_thread:
                self.imtalker_thread.join(timeout=10.0)
            self.imtalker.dump_last_session(source="websocket_live")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.perf_counter() - self._session_t0
            summary = (
                f"[Session] closed. duration={elapsed:.1f}s "
                f"chunks={self._total_chunks} frames={self._total_frames_sent}"
            )
            if self._sync_count > 0:
                summary += (
                    f"\n  A/V sync: avg_first_gap={self.avg_first_gap_ms:+.1f}ms "
                    f"avg_last_gap={self.avg_last_gap_ms:+.1f}ms"
                )
            print(summary, flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class PersonaPlexImTalkerOptions(BaseOptions):
    """CLI options for the PersonaPlex+IMTalker streaming server.

    Duplicates (does not import) the production live script's LiveHeliumFMOptions flag
    list, keeping this file fully independent of
    liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py. Drops flags that only served
    dead code paths not used by production (run_personaplex_imtalker_source5_8998.sh):
    --audio_path, --file_chunk_lookahead, --static_pose_zero, --static_pose_values,
    --stats_path, --no_direct_reply_hidden.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Helium->Wav2Vec2-frontend adapter checkpoint")
        parser.add_argument("--adapter_num_layers", type=int, default=6)
        parser.add_argument("--adapter_dropout", type=float, default=0.1)
        parser.add_argument("--ref_path", required=True)
        # PersonaPlex
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="")
        parser.add_argument("--tokenizer", default="")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8)
        parser.add_argument("--moshi_context", type=int, default=0)
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="")
        parser.add_argument("--text_prompt", default="")
        parser.add_argument("--moshi_reply_device", default=None)
        parser.add_argument(
            "--enable_moshi_reply", action="store_true", default=True,
            help="Accepted for CLI/notebook parity; this server always runs the PersonaPlex+IMTalker pipeline",
        )
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument(
            "--direct_reply_hidden", action="store_true", default=True,
            help="Accepted for CLI/notebook parity; this server always uses PersonaPlex's hidden state directly",
        )
        # FM / IMTalker
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec*fps")
        parser.add_argument(
            "--reply_hidden_steps_per_chunk", type=int, default=0,
            help="Raw PersonaPlex 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps",
        )
        parser.add_argument("--prebuffer_chunks", type=int, default=0, help="Accepted for compatibility; unused (pacing happens in the dispatcher)")
        parser.add_argument("--frame_q_backpressure", type=int, default=160, help="Accepted for compatibility")
        parser.add_argument("--render_sub_batch", type=int, default=8)
        parser.add_argument("--jpeg_quality", type=int, default=58)
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
        # FMGenerator getattr-only fields, exposed for CLI/notebook parity (harmless if unset)
        parser.add_argument("--audio_feat_dim", type=int, default=768)
        parser.add_argument("--audio_adapter_dim", type=int, default=512)
        parser.add_argument("--audio_adapter_mode", default="none")
        return parser

    def parse(self):
        opt = super().parse()
        opt.rank = opt.device
        return opt


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="PersonaPlex + IMTalker Streaming Server (FlashTalk-v3 architecture)")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)

    print("\n" + "=" * 70)
    print("  Loading Models...")
    print("=" * 70)

    imtalker_engine = IMTalkerEngine(args, device=args.device)
    imtalker_engine.load()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] After IMTalker: {used:.1f} GB / {total:.1f} GB")

    personaplex_engine = PersonaPlexEngine(
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
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] After PersonaPlex: {used:.1f} GB / {total:.1f} GB")

    app.state.personaplex = personaplex_engine
    app.state.imtalker = imtalker_engine
    app.state.args = args

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
            "stage": "personaplex_reply+imtalker_fm+renderer",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "device": args.device,
            "fm_chunk_frames": imtalker_engine.fm_chunk_frames,
            "buffer_ms": int(getattr(args, "buffer_ms", 80)),
        })

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        print("[WS] client connected.", flush=True)
        session = PersonaPlexConversationSession(ws, personaplex_engine, imtalker_engine, args)
        try:
            await session.run()
        except WebSocketDisconnect:
            print("[WS] client disconnected.", flush=True)
        except Exception as e:
            print(f"[WS] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            with contextlib.suppress(Exception):
                await ws.close()

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = PersonaPlexImTalkerOptions()
    args = parser.parse()
    parser.print_options()

    print("\n" + "=" * 70)
    print("  PersonaPlex + IMTalker Streaming Server (FlashTalk-v3 architecture)")
    print("=" * 70)
    print(f"  Host:Port        : {args.host}:{args.port}")
    print(f"  Device           : {args.device}")
    print(f"  FM chunk frames  : {args.fm_chunk_frames}")
    print(f"  Ref image        : {args.ref_path}")
    print(f"  Dump motion      : {args.dump_motion}")
    print("=" * 70 + "\n")

    app = build_app(args)

    print(f"[personaplex_imtalker_streaming_server] serving {args.html_path}")
    print(f"[personaplex_imtalker_streaming_server] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
