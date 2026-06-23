#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary import (  # noqa: E402
    LiveHeliumFMEngine,
    LiveHeliumFMOptions,
    MIMI_FRAME_SIZE,
    MoshiOnlyEngineWithHidden,
    TARGET_SR,
    load_audio_24k,
)


def make_live_args(output_dir: Path, ref_path: str) -> argparse.Namespace:
    argv = [
        "offline_personaplex_imtalker_infer.py",
        "--generator_path", "/workspace/IMTalker/checkpoints/generator.ckpt",
        "--renderer_path", "/workspace/IMTalker/checkpoints/renderer.ckpt",
        "--adapter_path", "/workspace/exps/personaplex_frontend_adapter/personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt",
        "--adapter_num_layers", "6",
        "--adapter_dropout", "0.1",
        "--wav2vec_model_path", "/workspace/IMTalker/checkpoints/wav2vec2-base-960h",
        "--ref_path", ref_path,
        "--host", "127.0.0.1",
        "--port", "8998",
        "--device", "cuda",
        "--enable_moshi_reply",
        "--direct_reply_hidden",
        "--moshi_root", "/workspace/personaplex_bnb4",
        "--mimi_hf_repo", "nvidia/personaplex-7b-v1",
        "--moshi_weight", "/workspace/personaplex_bnb4/model_bnb_4bit.pt",
        "--quantize_4bit",
        "--num_codebooks", "8",
        "--moshi_reply_device", "cuda",
        "--moshi_cfg_coef", "1.0",
        "--voice_prompt", "NATM0.pt",
        "--text_prompt", "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way. Talk slowly.",
        "--a_cfg_scale", "1.34",
        "--nfe", "5",
        "--wav2vec_sec", "0.96",
        "--audio_chunk_sec", "0.96",
        "--fm_chunk_frames", "24",
        "--prebuffer_chunks", "0",
        "--frame_q_backpressure", "16",
        "--render_sub_batch", "8",
        "--jpeg_quality", "58",
        "--dump_motion",
        "--dump_dir", str(output_dir / "dumps"),
        "--shared_noise",
        "--noise_seed", "42",
        "--noise_max_frames", "5000",
        "--fp32",
        "--tf32",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        parser = LiveHeliumFMOptions()
        args = parser.parse()
        args.rank = args.device
        return args
    finally:
        sys.argv = old


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    pcm = np.asarray(pcm, dtype=np.float32)
    torchaudio.save(str(path), torch.from_numpy(pcm).view(1, -1), sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_audio", default="/workspace/IMTalker/assets/audio_3.wav")
    ap.add_argument("--ref_path", default="/workspace/IMTalker/assets/source_5.png")
    ap.add_argument("--output_dir", default="/workspace/IMTalker/offline_personaplex_imtalker_audio3")
    ap.add_argument("--silence_tail_sec", type=float, default=8.0)
    ap.add_argument("--max_output_sec", type=float, default=0.0, help="0 means no cap")
    args0 = ap.parse_args()

    output_dir = Path(args0.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = make_live_args(output_dir, args0.ref_path)
    fm_engine = LiveHeliumFMEngine(args)
    fm_engine.reset_session()
    reply_engine = MoshiOnlyEngineWithHidden(
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
    reply_engine.reset_session()

    in_audio = load_audio_24k(args0.input_audio)
    if float(args0.silence_tail_sec) > 0:
        in_audio = np.concatenate([
            in_audio,
            np.zeros(int(round(float(args0.silence_tail_sec) * TARGET_SR)), dtype=np.float32),
        ])
    if float(args0.max_output_sec) > 0:
        in_audio = in_audio[: int(round(float(args0.max_output_sec) * TARGET_SR))]

    video_only = output_dir / "offline_personaplex_imtalker_audio3_video_only.mp4"
    reply_wav = output_dir / "offline_personaplex_reply_audio_24k.wav"
    final_mp4 = output_dir / "offline_personaplex_imtalker_audio3_h264.mp4"
    events_path = output_dir / "reply_events.jsonl"
    text_path = output_dir / "reply_text.txt"

    writer = cv2.VideoWriter(
        str(video_only),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (512, 512),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_only}")

    hidden_steps_per_chunk = max(1, int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps))))
    pending_steps: list[dict] = []
    pending_hidden: list[torch.Tensor] = []
    pending_audio: list[np.ndarray] = []
    all_reply_audio: list[np.ndarray] = []
    all_events: list[dict] = []
    frames_written = 0
    chunks_written = 0

    total_steps = int(np.ceil(in_audio.shape[0] / MIMI_FRAME_SIZE))
    for step_idx in range(total_steps):
        chunk = in_audio[step_idx * MIMI_FRAME_SIZE:(step_idx + 1) * MIMI_FRAME_SIZE]
        if chunk.shape[0] < MIMI_FRAME_SIZE:
            chunk = np.pad(chunk, (0, MIMI_FRAME_SIZE - chunk.shape[0]))
        chunk_i16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        reply_engine.append_browser_pcm(chunk_i16, TARGET_SR)

        for ev in reply_engine.process_ready_steps_limited(1):
            safe_event = {
                "step": int(ev.get("step", -1)),
                "token": int(ev.get("token", -1)),
                "piece": str(ev.get("piece", "")),
                "audio_text": str(ev.get("audio_text", "")),
                "reply_rms": float(ev.get("reply_rms", 0.0)),
                "reply_peak": float(ev.get("reply_peak", 0.0)),
                "input_rms": float(ev.get("input_rms", 0.0)),
                "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                "total_ms": float(ev.get("total_ms", 0.0)),
            }
            all_events.append(safe_event)
            pending_steps.append(ev)
            reply_pcm = (
                np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            hidden = ev.get("helium_hidden")
            if not isinstance(hidden, torch.Tensor):
                continue
            pending_hidden.append(hidden.squeeze(0).contiguous())
            pending_audio.append(reply_pcm)
            if len(pending_hidden) < hidden_steps_per_chunk:
                continue

            used_hidden = pending_hidden[:hidden_steps_per_chunk]
            used_audio = pending_audio[:hidden_steps_per_chunk]
            used_steps = pending_steps[:hidden_steps_per_chunk]
            pending_hidden = pending_hidden[hidden_steps_per_chunk:]
            pending_audio = pending_audio[hidden_steps_per_chunk:]
            pending_steps = pending_steps[hidden_steps_per_chunk:]

            helium_chunk = torch.cat(used_hidden, dim=0)
            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
            all_reply_audio.append(pcm_chunk)
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
            frames_np, _ = fm_engine._render_motion(motion)
            for frame_rgb in frames_np:
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frames_written += 1
            chunks_written += 1
            print(
                f"[offline] chunk={chunks_written} frames_total={frames_written} "
                f"reply_text={''.join(e['piece'] for e in all_events)[-80:]!r}",
                flush=True,
            )

    writer.release()
    reply_audio = np.concatenate(all_reply_audio, axis=0) if all_reply_audio else np.zeros(0, dtype=np.float32)
    write_wav(reply_wav, reply_audio, TARGET_SR)
    with events_path.open("w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    text_path.write_text("".join(ev["piece"] for ev in all_events), encoding="utf-8")
    dump_dir = fm_engine.dump_last_session(source=f"offline:{args0.input_audio}")

    ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode == 0:
        subprocess.run([
            ffmpeg.stdout.strip(),
            "-y",
            "-i", str(video_only),
            "-i", str(reply_wav),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(final_mp4),
        ], check=True)
    else:
        final_mp4 = video_only

    meta = {
        "input_audio": args0.input_audio,
        "ref_path": args0.ref_path,
        "video": str(final_mp4),
        "video_only": str(video_only),
        "reply_wav": str(reply_wav),
        "reply_text": str(text_path),
        "reply_events": str(events_path),
        "session_dump": str(dump_dir) if dump_dir else "",
        "frames_written": frames_written,
        "chunks_written": chunks_written,
        "reply_audio_seconds": float(reply_audio.shape[0] / TARGET_SR) if reply_audio.size else 0.0,
    }
    (output_dir / "offline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
