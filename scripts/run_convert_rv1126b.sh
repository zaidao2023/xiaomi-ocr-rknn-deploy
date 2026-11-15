#!/bin/bash
# xiaomi-ocr-rknn-deploy — Xiaomi-OCR-0 -> RV1126B conversion pipeline
#
# Run this from the Rockchip rknn-llm repo's examples/multimodal_model_demo
# directory (so that export/export_vision.py, export/export_vision_rknn.py and
# the multimodal demo runtime are reachable).
#
# Configurable via environment variables:
#   PY     python interpreter with rkllm-toolkit + rknn-toolkit2 installed (default: python3)
#   MODEL  path to the Xiaomi-OCR-0 HuggingFace model directory (required)
#   QTYPE  quantization type for RV1126B: w8a8 (default, verified) or w4a16
#   LOG    log file name (default: convert_rv1126b.log)
set -e

PY=${PY:-python3}
: "${MODEL:?please set MODEL to the Xiaomi-OCR-0 model directory}"
QTYPE=${QTYPE:-w8a8}
LOG=${LOG:-convert_rv1126b.log}

echo "=== [1/4] Vision encoder -> ONNX ==="
if [ ! -f ./onnx/qwen3.5_vision.onnx ]; then
    $PY export/export_vision.py --path "$MODEL" --model_name qwen3.5 \
        --height 448 --width 448 --batch_size 1 --device cpu 2>&1 | tee -a "$LOG"
else
    echo "onnx/qwen3.5_vision.onnx already exists, skip"
fi

echo "=== [2/4] Vision ONNX -> RKNN (rv1126b) ==="
$PY export/export_vision_rknn.py --path ./onnx/qwen3.5_vision.onnx \
    --model_name qwen3.5 --height 448 --width 448 --target-platform rv1126b 2>&1 | tee -a "$LOG"

echo "=== [3/4] Capture LLM input embeddings (quantization calibration set) ==="
$PY data/make_input_embeds_xiaomi_ocr.py --path "$MODEL" 2>&1 | tee -a "$LOG"

echo "=== [4/4] LLM -> RKLLM ($QTYPE, rv1126b) ==="
$PY export/export_rkllm_xiaomi_ocr_rv1126b.py --path "$MODEL" --quantized_dtype "$QTYPE" 2>&1 | tee -a "$LOG"

echo "=== DONE ==="
ls -la ./rknn/qwen3.5_vision_rv1126b.rknn ./rkllm/xiaomi-ocr-0_${QTYPE}_rv1126b.rkllm 2>/dev/null
