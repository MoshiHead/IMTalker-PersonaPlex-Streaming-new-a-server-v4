#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary import (  # noqa: E402
    LiveHeliumFMEngine,
    LiveHeliumFMOptions,
)


def make_engine_args(output_dir: Path, ref_path: str) -> argparse.Namespace:
    argv = [
        "render_saved_live_helium.py",
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
        # This combination skips raw-audio Helium extraction in engine init.
        "--enable_moshi_reply",
        "--direct_reply_hidden",
        "--a_cfg_scale", "1.34",
        "--nfe", "5",
        "--wav2vec_sec", "0.96",
        "--audio_chunk_sec", "0.96",
        "--fm_chunk_frames", "24",
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
    old_argv = sys.argv
    try:
        sys.argv = argv
        parser = LiveHeliumFMOptions()
        args = parser.parse()
        args.rank = args.device
        return args
    finally:
        sys.argv = old_argv


def load_helium(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("helium", "full_helium_raw", "hidden"):
            val = obj.get(key)
            if torch.is_tensor(val):
                return val.float().contiguous()
    if torch.is_tensor(obj):
        return obj.float().contiguous()
    raise RuntimeError(f"Could not find Helium tensor in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", default="/workspace/IMTalker/live_try_dumps_personaplex_frontend_source5_cfg134/last_session")
    ap.add_argument("--ref_path", default="/workspace/IMTalker/assets/source_5.png")
    ap.add_argument("--output_dir", default="/workspace/IMTalker/offline_from_live_helium_last_session")
    ap.add_argument("--chunk_steps", type=int, default=12)
    args0 = ap.parse_args()

    session_dir = Path(args0.session_dir)
    output_dir = Path(args0.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    helium_path = session_dir / "full_helium_raw.pt"
    reply_audio_path = session_dir / "full_moshi_reply_24k.wav"
    reply_text_path = session_dir / "reply_text.txt"
    source_meta_path = session_dir / "meta.json"

    helium = load_helium(helium_path)
    if helium.ndim != 2:
        raise RuntimeError(f"Expected Helium shape (T,D), got {tuple(helium.shape)}")

    engine_args = make_engine_args(output_dir, args0.ref_path)
    engine = LiveHeliumFMEngine(engine_args)
    engine.reset_session()

    video_only = output_dir / "live_helium_to_imtalker_video_only.mp4"
    h264_out = output_dir / "live_helium_to_imtalker_h264.mp4"
    motion_out = output_dir / "rendered_motion_from_saved_helium.pt"
    meta_out = output_dir / "meta.json"

    writer = cv2.VideoWriter(
        str(video_only),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(engine.fps),
        (512, 512),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {video_only}")

    all_motion: list[torch.Tensor] = []
    chunk_steps = max(1, int(args0.chunk_steps))
    total_chunks = int(np.ceil(int(helium.shape[0]) / chunk_steps))
    frames_written = 0

    for chunk_idx, start in enumerate(range(0, int(helium.shape[0]), chunk_steps), start=1):
        h = helium[start:start + chunk_steps]
        if h.shape[0] == 0:
            continue
        target_frames = int(round(h.shape[0] * 2))
        motion, info = engine._sample_motion_from_helium(h, target_frames)
        # Store a minimal offline session so dump_last_session still works.
        pcm_dummy = np.zeros(int(round(target_frames * 24000 / float(engine.fps))), dtype=np.float32)
        engine._record_session_chunk(pcm_dummy, motion, info)
        frames_np, _render_info = engine._render_motion(motion)
        for frame_rgb in frames_np:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frames_written += 1
        all_motion.append(motion.detach().float().cpu())
        print(
            f"[saved-helium-render] chunk {chunk_idx}/{total_chunks} "
            f"helium={tuple(h.shape)} motion={tuple(motion.shape)} frames_total={frames_written}",
            flush=True,
        )

    writer.release()
    motion = torch.cat(all_motion, dim=0).contiguous() if all_motion else torch.empty(0, 32)
    torch.save({"motion": motion, "source_helium": str(helium_path), "fps": float(engine.fps)}, motion_out)
    dump_dir = engine.dump_last_session(source=f"saved_live_helium:{helium_path}")

    final_video = video_only
    if reply_audio_path.exists():
        ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
        if ffmpeg.returncode == 0:
            subprocess.run([
                ffmpeg.stdout.strip(),
                "-y",
                "-i", str(video_only),
                "-i", str(reply_audio_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(h264_out),
            ], check=True)
            final_video = h264_out

    meta = {
        "purpose": "saved live Helium states -> adapter -> IMTalker offline render",
        "session_dir": str(session_dir),
        "helium_path": str(helium_path),
        "helium_shape": list(helium.shape),
        "reply_audio_path": str(reply_audio_path) if reply_audio_path.exists() else "",
        "reply_text_path": str(reply_text_path) if reply_text_path.exists() else "",
        "source_meta_path": str(source_meta_path) if source_meta_path.exists() else "",
        "ref_path": args0.ref_path,
        "video": str(final_video),
        "video_only": str(video_only),
        "motion_path": str(motion_out),
        "dump_dir": str(dump_dir) if dump_dir else "",
        "frames_written": int(frames_written),
        "motion_shape": list(motion.shape),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
