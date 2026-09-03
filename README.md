# PAAS_ensemble_v3_inf

**Inference-only, deployment build** of PAAS_ensemble_v3 — the 5-detector face real/fake (liveness)
ensemble `mean( FFAA, A1_9c, A2_9c, GSD, SeLop )`, served over a FastAPI JSON API (adapted from the
PAAS_ensemble_v2 service). All training / data-prep / finetuning code has been removed; the models
are frozen and the paths fixed.

Runs on the **global** `python3.12` / `transformers==4.37.2` interpreter.

## What was stripped vs PAAS_ensemble_v3

Removed: `gsd_train.py`, `gsd_build_anchor.py`, `selop_train.py`, `train_combiner.py`,
`train_ensemble9.sh`, `run_finetuning.sh`, `make_mids_data.sh`, `configs/` (GSD train configs),
`ensemble9/mids9lib/train.py`, `ensemble9/scripts/`, and the FFAA dataset-making / LoRA-merge /
sample-selection / training scripts. Kept: the `paas` pipeline, the four inference model trees
(`ffaa/` inference, `ensemble9/mids9lib` inference, `gsd/`, `selop/`), weights, the shared CLIP
backbone, `inference.py`, `test_video_image_batch.py`, and the FastAPI app.

## Serve

```bash
bash run_server.sh                                    # default 5-detector mean (paas5_mean)
PAAS_CONFIG=config/experiments/paas4_fast.json bash run_server.sh   # fast, no 7B MLLM
```

Then:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/batcher_status
# base64 single image
curl -s -X POST localhost:8000/face_liveness_base64 \
     -H 'Content-Type: application/json' \
     -d "{\"image_base64\":\"$(base64 -w0 images/0034.jpg)\"}"
# multipart upload
curl -s -X POST localhost:8000/face_liveness -F 'file=@images/0034.jpg'
```

Swagger UI at `/docs`. Endpoints: `/face_liveness`, `/face_liveness_base64`,
`/face_liveness_base64_batch`, and the trimmed-mean `*_new` variants.

## Configurations (pick with `PAAS_CONFIG`, or override live with env)

| config | components | speed | axon AUC | default thr (real-90) |
|--------|-----------|-------|----------|-----------------------|
| `paas5_mean` (default) | FFAA + A1_9c + A2_9c + GSD + SeLop | slow (7B MLLM) | 0.9998 | 0.1982 |
| `paas4_fast` | A1_9c + A2_9c + GSD + SeLop (no MLLM) | fast (all-CLIP) | 0.9996 | 0.2381 |

Env overrides (no config edit needed):
- `PAAS_COMPONENTS="A1_9c,A2_9c,gsd,selop"` — choose the fused subset; only those models load.
- `PAAS_THRESHOLD=0.373` — move the operating point (see `paas/fusion.OPERATING_POINTS`).
- `ENS_BATCH / FFAA_BATCH / GSD_BATCH / SELOP_BATCH` — per-model GPU sub-batch.
- `DYNAMIC_BATCHING=1`, `MAX_BATCH_SIZE`, `DYNAMIC_BATCH_MAX_WAIT_MS` — request coalescing.

## Optimization notes

- **Single pipeline instance**, loaded once at startup; TF32 on; per-model bf16 autocast.
- All CLIP members share ONE `base_models/clip-vit-large-patch14-336` on disk.
- **`paas4_fast`** drops the 7B LLaVA MLLM — ~10-50x lower latency for a ~0.0002 AUC cost
  (0.9998 -> 0.9996). Use it for high-throughput / latency-sensitive deployments; use `paas5_mean`
  when maximum accuracy matters.
- Weights / base models are hardlinked from PAAS_ensemble_v3 (no extra disk; still standalone).

## Batch scoring (offline, not the service)

```bash
python3.12 test_video_image_batch.py --input-dir /path/to/tree --out-dir runs/test --devices 0,1,2,3
python3.12 inference.py face.jpg --components gsd,selop        # quick CLI
```
