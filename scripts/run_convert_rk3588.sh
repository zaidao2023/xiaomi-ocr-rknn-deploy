#!/bin/bash
# rk-vlm-deploy — Xiaomi-OCR-0 -> RK3588 conversion pipeline
#
# Run this from the Rockchip rknn-llm repo's examples/multimodal_model_demo
# directory (so that export/export_vision.py, export/export_vision_rknn.py and
# the multimodal demo runtime are reachable).
#
# Configurable via environment variables:
#   PY     python interpreter with rkllm-toolkit + rknn-toolkit2 installed (default: python3)
#   MODEL  path to the Xiaomi-OCR-0 HuggingFace model directory (required)
#   LOG    log file name (default: convert_rk3588.log)
set -e

PY=${PY:-python3}
: "${MODEL:?please set MODEL to the Xiaomi-OCR-0 model directory}"
LOG=${LOG:-convert_rk3588.log}

echo "=== [1/4] Vision encoder -> ONNX ==="
$PY export/export_vision.py --path "$MODEL" --model_name qwen3.5 \
    --height 448 --width 448 --batch_size 1 --device cpu 2>&1 | tee -a "$LOG"

echo "=== [2/4] Vision ONNX -> RKNN (rk3588) ==="
$PY export/export_vision_rknn.py --path ./onnx/qwen3.5_vision.onnx \
    --model_name qwen3.5 --height 448 --width 448 --target-platform rk3588 2>&1 | tee -a "$LOG"

echo "=== [3/4] Capture LLM input embeddings (quantization calibration set) ==="
$PY data/make_input_embeds_xiaomi_ocr.py --path "$MODEL" 2>&1 | tee -a "$LOG"

echo "=== [4/4] LLM -> RKLLM (W8A8) ==="
$PY export/export_rkllm_xiaomi_ocr.py --path "$MODEL" 2>&1 | tee -a "$LOG"

echo "=== DONE ==="
ls -la ./rknn/*.rknn ./rkllm/*.rkllm 2>/dev/null
