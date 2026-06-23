#!/usr/bin/env bash
set -euo pipefail
cd /workspace/IMTalker
source /workspace/preprocess_5090/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/IMTalker:/workspace/personaplex_bnb4/moshi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script}"
python -u /workspace/IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py \
  --generator_path /workspace/IMTalker/checkpoints/generator.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --adapter_path /workspace/exps/personaplex_frontend_adapter/personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt \
  --adapter_num_layers 6 \
  --adapter_dropout 0.1 \
  --wav2vec_model_path /workspace/IMTalker/checkpoints/wav2vec2-base-960h \
  --ref_path /workspace/IMTalker/assets/2_vid_robert.png \
  --host 0.0.0.0 \
  --port 8998 \
  --device cuda \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --moshi_root /workspace/personaplex_bnb4 \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight /workspace/personaplex_bnb4/model_bnb_4bit.pt \
  --quantize_4bit \
  --num_codebooks 8 \
  --moshi_reply_device cuda \
  --moshi_cfg_coef 1.0 \
  --voice_prompt NATM0.pt \
  --text_prompt "You work for North South University which is a university and your name is Nabeel Mohammed. Information: you are answering Computer science related questions explicitly about models and telling about how moshi and personaplex are trained to ordinary people. So in lighter terms." \
  --a_cfg_scale 1.34 \
  --nfe 5 \
  --wav2vec_sec 0.96 \
  --audio_chunk_sec 0.96 \
  --fm_chunk_frames 24 \
  --reply_hidden_steps_per_chunk 0 \
  --prebuffer_chunks 0 \
  --frame_q_backpressure 160 \
  --render_sub_batch 8 \
  --jpeg_quality 58 \
  --dump_motion \
  --dump_dir /workspace/IMTalker/live_try_dumps_personaplex_frontend_source5_cfg134 \
  --shared_noise \
  --noise_seed 42 \
  --noise_max_frames 5000 \
  --fp32 \
  --tf32
