# PersonaPlex -> IMTalker Live Handoff

Working root:

```text
/home/user/D/imtalker_personaplex
```

Current pod:

```bash
ssh root@31.24.80.34 -p 16844 -i ~/.ssh/id_ed25519
```

Remote paths:

```text
/workspace/IMTalker
/workspace/personaplex_bnb4
/workspace/preprocess_5090
```

## What This Runs

```text
Browser mic
-> PersonaPlex bnb4 live reply generation
-> capture PersonaPlex transformer hidden during generation
-> PersonaPlex Helium-to-Wav2Vec frontend adapter
-> frozen Wav2Vec2 transformer
-> frozen IMTalker generator
-> frozen IMTalker renderer
-> binary websocket HTML on port 8998
```

## Local Files To Treat As Source

These local files are now staged under `/home/user/D/imtalker_personaplex/IMTalker`:

```text
liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
liveTry.py
ws_av_binary_codec.py
generator/FM.py
generator/wav2vec2.py
generator/helium_w2v_frontend_adapter.py
static/index_v3_binary_fullscreen.html
```

The HTML has been rebranded from `FlashTalk` to `IMTalker`.

## Important Checkpoints

```text
/workspace/IMTalker/checkpoints/generator.ckpt
/workspace/IMTalker/checkpoints/renderer.ckpt
/workspace/personaplex_bnb4/model_bnb_4bit.pt
/workspace/exps/personaplex_frontend_adapter/personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt
```

PersonaPlex bnb4 source:

```text
brianmatzelle/personaplex-7b-v1-bnb-4bit
```

PersonaPlex adapter source:

```text
niloy629/hdtf_preprocess/personaplex_helium_w2v_frontend_adapter/
```

## Male Voice

Use PersonaPlex natural male voice:

```text
NATM0.pt
```

The bridge exposes this through:

```bash
--voice_prompt NATM0.pt
```

Voice prompts are downloaded from `nvidia/personaplex-7b-v1` via `voices.tgz` if needed.

## Run Command

The pod launch script is:

```bash
/workspace/IMTalker/run_personaplex_imtalker_source5_8998.sh
```

It should contain:

```bash
--voice_prompt NATM0.pt
--text_prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way. Talk slowly."
--quantize_4bit
--moshi_weight /workspace/personaplex_bnb4/model_bnb_4bit.pt
```

Restart:

```bash
ssh root@31.24.80.34 -p 16844 -i ~/.ssh/id_ed25519 '
cd /workspace/IMTalker
tmux kill-session -t live_personaplex_imtalker_source5_8998 2>/dev/null || true
tmux new-session -d -s live_personaplex_imtalker_source5_8998 \
  "/workspace/IMTalker/run_personaplex_imtalker_source5_8998.sh 2>&1 | tee /workspace/IMTalker/logs/live_personaplex_imtalker_source5_8998.log"
'
```

Monitor:

```bash
ssh root@31.24.80.34 -p 16844 -i ~/.ssh/id_ed25519 \
'tail -f /workspace/IMTalker/logs/live_personaplex_imtalker_source5_8998.log'
```

Healthy startup includes:

```text
frontend-fp32 loaded
using direct Moshi reply hidden
voice prompt: .../NATM0.pt
installed PersonaPlex graphed hidden capture
serving /workspace/IMTalker/static/index_v3_binary_fullscreen.html
Uvicorn running on http://0.0.0.0:8998
```

Local test:

```bash
curl -s -o /tmp/live.html -w "%{http_code} %{size_download}\n" http://127.0.0.1:8998/
```

Expected:

```text
200 35708
```
