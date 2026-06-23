"""liveTry.py - v3 one-websocket server, Step 3: Moshi audio/text only.

What this version does:
    browser mic PCM -> /ws/conversation -> original Moshi/Mimi
    Moshi reply audio/text -> JSON chunk_audio + static JPEG chunk_frame

What this version deliberately does NOT do yet:
    no VAD
    no Helium extraction
    no FM
    no IMTalker renderer
    no WebRTC / TURN / H264 / Opus

The goal is to prove the teammate-style HTML protocol works cleanly with our
original Moshi backend before adding Helium/IMTalker.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


TARGET_SR = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz, Moshi/Mimi step size


def _ensure_moshi_importable(moshi_root: str | Path) -> None:
    root = Path(moshi_root)
    pkg = root / "moshi"
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def _clean_text_piece(piece: str) -> str:
    return piece.replace("▁", " ")


def _make_placeholder_jpeg(path: str | Path | None) -> str:
    img = None
    if path:
        p = Path(path)
        if p.is_file():
            img = cv2.imread(str(p))
    if img is None:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Moshi",
            (150, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (235, 235, 235),
            3,
            cv2.LINE_AA,
        )
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("failed to encode placeholder JPEG")
    return base64.b64encode(enc.tobytes()).decode("ascii")


class MoshiOnlyEngine:
    def __init__(
        self,
        *,
        moshi_root: str,
        mimi_hf_repo: str,
        device: str,
        cfg_coef: float,
        placeholder_jpeg_b64: str,
        moshi_weight: str = "",
        mimi_weight: str = "",
        tokenizer: str = "",
        quantize_4bit: bool = False,
        num_codebooks: int = 8,
        context: int | None = None,
        voice_prompt: str = "",
        voice_prompt_dir: str = "",
        text_prompt: str = "",
    ) -> None:
        _ensure_moshi_importable(moshi_root)
        from moshi.models import LMGen, loaders

        self.device = torch.device(device)
        self.placeholder_jpeg_b64 = placeholder_jpeg_b64
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        self.text_prompt = str(text_prompt or "")
        self.voice_prompt = str(voice_prompt or "")
        self.voice_prompt_dir = str(voice_prompt_dir or "")
        self._hf_repo = mimi_hf_repo

        print(
            "[liveTry] loading Moshi "
            f"repo={mimi_hf_repo} root={moshi_root} device={self.device} cfg={cfg_coef}"
        )
        t0 = time.perf_counter()
        if hasattr(loaders, "CheckpointInfo"):
            ckpt_info = loaders.CheckpointInfo.from_hf_repo(mimi_hf_repo)
            self.mimi = ckpt_info.get_mimi(device=self.device)
            self.lm = ckpt_info.get_moshi(device=self.device, dtype=torch.bfloat16)
            self.tokenizer = ckpt_info.get_text_tokenizer()
            model_type = getattr(ckpt_info, "model_type", "moshi")
        else:
            from huggingface_hub import hf_hub_download
            import sentencepiece

            repo = mimi_hf_repo or getattr(loaders, "DEFAULT_REPO", "nvidia/personaplex-7b-v1")
            if not mimi_weight:
                mimi_weight = hf_hub_download(repo, loaders.MIMI_NAME)
            if not moshi_weight:
                moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
            if not tokenizer:
                tokenizer = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, self.device)
            self.lm = loaders.get_moshi_lm(
                moshi_weight,
                device=self.device,
                dtype=torch.bfloat16,
                quantize_4bit=bool(quantize_4bit),
                num_codebooks=int(num_codebooks),
                context=context,
            )
            self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore
            model_type = "personaplex"
        self.mimi.eval()
        self.lm.eval()

        try:
            from moshi.run_inference import get_condition_tensors

            cond_tensors = get_condition_tensors(
                model_type,
                self.lm,
                batch_size=1,
                cfg_coef=float(cfg_coef),
            )
        except Exception:
            cond_tensors = {}

        def on_text_hook(text_tokens: torch.Tensor) -> None:
            token = int(text_tokens[0].detach().item())
            piece = self.decode_piece(token)
            if piece:
                self.sampled_text += piece

        try:
            self.lm_gen = LMGen(
                self.lm,
                cfg_coef=float(cfg_coef),
                condition_tensors=cond_tensors,
                on_text_hook=on_text_hook,
            )
        except TypeError:
            self.lm_gen = LMGen(self.lm, device=self.device)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        if self.frame_size != FRAME_SIZE:
            raise RuntimeError(f"expected Mimi frame_size={FRAME_SIZE}, got {self.frame_size}")
        self._warmup_runtime()
        self.reset_session()
        print(f"[liveTry] Moshi ready in {time.perf_counter() - t0:.1f}s")

    def _resolve_voice_prompt_path(self) -> str:
        if not self.voice_prompt:
            return ""
        if os.path.isabs(self.voice_prompt) and os.path.exists(self.voice_prompt):
            return self.voice_prompt
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            candidate = os.path.join(self.voice_prompt_dir, self.voice_prompt)
            if os.path.exists(candidate):
                return candidate
        from huggingface_hub import hf_hub_download

        voices_tgz = Path(hf_hub_download(self._hf_repo, "voices.tgz"))
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        candidate = voices_dir / self.voice_prompt
        if not candidate.exists():
            raise FileNotFoundError(f"voice prompt not found: {candidate}")
        self.voice_prompt_dir = str(voices_dir)
        return str(candidate)

    @torch.no_grad()
    def _apply_system_prompts(self) -> None:
        if not hasattr(self.lm_gen, "step_system_prompts"):
            return
        voice_path = self._resolve_voice_prompt_path()
        if voice_path:
            if voice_path.endswith(".pt") and hasattr(self.lm_gen, "load_voice_prompt_embeddings"):
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            elif hasattr(self.lm_gen, "load_voice_prompt"):
                self.lm_gen.load_voice_prompt(voice_path)
            print(f"[liveTry] voice prompt: {voice_path}", flush=True)
        if self.text_prompt and hasattr(self.tokenizer, "encode"):
            with contextlib.suppress(Exception):
                wrapped = f"<|im_start|>system\n{self.text_prompt}<|im_end|>\n"
                self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrapped)
                print(f"[liveTry] text prompt loaded: {self.text_prompt[:80]!r}", flush=True)
        self.lm_gen.step_system_prompts(self.mimi)
        with contextlib.suppress(Exception):
            self.mimi.reset_streaming()

    def reset_session(self) -> None:
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        with contextlib.suppress(Exception):
            self.mimi.reset_streaming()
        with contextlib.suppress(Exception):
            self.lm_gen.reset_streaming()
        self._apply_system_prompts()

    @torch.no_grad()
    def _warmup_runtime(self, n_steps: int = 6) -> None:
        t0 = time.perf_counter()
        silence = torch.zeros(1, 1, self.frame_size, device=self.device, dtype=torch.float32)
        for idx in range(int(n_steps)):
            codes = self.mimi.encode(silence)
            if idx == 0:
                self.mimi.reset_streaming()
            tokens = self.lm_gen.step(codes[:, :, :1])
            if tokens is not None:
                reply = self.mimi.decode(tokens[:, 1:])
                _ = reply.detach().float().mean().item()
        self.reset_session()
        _sync = getattr(torch.cuda, "synchronize", None)
        if callable(_sync) and torch.cuda.is_available():
            _sync()
        print(f"[liveTry] Moshi runtime warmup done in {1000.0 * (time.perf_counter() - t0):.0f}ms")

    def decode_piece(self, token: int) -> str:
        if token in (0, 3):
            return ""
        with contextlib.suppress(Exception):
            return _clean_text_piece(self.tokenizer.id_to_piece(int(token)))
        return ""

    def append_browser_pcm(self, pcm_i16: np.ndarray, input_sr: int) -> None:
        pcm = pcm_i16.astype(np.float32) / 32768.0
        if int(input_sr) != TARGET_SR:
            wav = torch.from_numpy(pcm).view(1, -1)
            pcm = torchaudio.functional.resample(wav, int(input_sr), TARGET_SR)[0].numpy()
        self.input_buffer = np.concatenate([self.input_buffer, pcm.astype(np.float32, copy=False)])

    @torch.no_grad()
    def process_ready_steps(self) -> list[dict]:
        events: list[dict] = []
        while self.input_buffer.shape[0] >= FRAME_SIZE:
            pcm = self.input_buffer[:FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            # Same first-frame reset used in Moshi examples/live code.
            self.mimi.reset_streaming()
            self.skip_first = False

        t_lm0 = time.perf_counter()
        tokens = self.lm_gen.step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        if tokens is None:
            reply_pcm = np.zeros(FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > FRAME_SIZE:
                reply_pcm = reply_pcm[:FRAME_SIZE]

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
            "[liveTry] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms"
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
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Moshi liveTry")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    placeholder_jpeg_b64 = _make_placeholder_jpeg(args.placeholder_path)
    engine: MoshiOnlyEngine | None = None

    def get_engine() -> MoshiOnlyEngine:
        nonlocal engine
        if engine is None:
            engine = MoshiOnlyEngine(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=args.device,
                cfg_coef=args.cfg_coef,
                placeholder_jpeg_b64=placeholder_jpeg_b64,
            )
        return engine

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
        return HTMLResponse(
            f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>",
            status_code=500,
        )

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "moshi_text_audio_only",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "moshi_loaded": engine is not None,
        })

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        input_sr = 48000
        packets = 0
        samples = 0
        t0 = time.perf_counter()
        moshi = get_engine()

        await ws.send_json({
            "type": "server_ready",
            "sample_rate": TARGET_SR,
            "model_type": "moshi-only",
            "tokens_per_chunk": 1,
            "buffer_ms": 400,
        })
        print("[liveTry] websocket connected; sent server_ready")

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"[liveTry] bad json: {text[:120]!r}")
                        continue

                    msg_type = str(payload.get("type", "")).lower()
                    if msg_type == "start":
                        input_sr = int(payload.get("sample_rate", payload.get("sampleRate", input_sr)))
                        print(f"[liveTry] start: browser_sample_rate={input_sr}")
                    elif msg_type == "stop":
                        print("[liveTry] stop requested")
                        break
                    else:
                        print(f"[liveTry] text message: {payload}")
                    continue

                data = msg.get("bytes")
                if not data:
                    continue
                pcm_i16 = np.frombuffer(data, dtype=np.int16)
                if pcm_i16.size == 0:
                    continue

                packets += 1
                samples += int(pcm_i16.size)
                if packets == 1 or packets % 50 == 0:
                    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(pcm_f32, dtype=np.float32))))
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(
                        "[liveTry] mic "
                        f"packets={packets} samples={samples} "
                        f"audio_sec={samples / max(float(input_sr), 1.0):.2f} "
                        f"wall_sec={elapsed:.2f} rms={rms:.5f}"
                    )

                moshi.append_browser_pcm(pcm_i16, input_sr)
                for ev in moshi.process_ready_steps():
                    await ws.send_json({
                        "type": "chunk_audio",
                        "chunk_id": ev["step"],
                        "sample_rate": ev["sample_rate"],
                        "pcm_s16le_b64": ev["reply_i16_b64"],
                        "gen_ms": ev["total_ms"],
                    })
                    # Two static frames per 80 ms Moshi step ~= 25 fps.
                    for frame_idx in range(2):
                        await ws.send_json({
                            "type": "chunk_frame",
                            "chunk_id": ev["step"],
                            "frame_idx": frame_idx,
                            "jpeg_b64": moshi.placeholder_jpeg_b64,
                            "server_fps": 25.0,
                            "chunks_done": ev["step"],
                            "avg_gen_ms": ev["total_ms"],
                            "moshi_text": ev["audio_text"] or ev["sampled_text"],
                        })
        except WebSocketDisconnect:
            pass
        finally:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            print(
                "[liveTry] websocket closed "
                f"packets={packets} samples={samples} "
                f"audio_sec={samples / max(float(input_sr), 1.0):.2f} wall_sec={elapsed:.2f}"
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--html_path", default=str(Path(__file__).resolve().parent / "static" / "index_v3.html"))
    parser.add_argument("--placeholder_path", default="")
    parser.add_argument("--moshi_root", default="/workspace/moshi")
    parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg_coef", type=float, default=1.0)
    args = parser.parse_args()

    app = build_app(args)

    import uvicorn

    print(f"[liveTry] serving {args.html_path}")
    print(f"[liveTry] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
