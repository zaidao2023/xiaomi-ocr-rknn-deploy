import os
from rkllm.api import RKLLM
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='/path/to/Xiaomi-OCR-0', help='model path')
argparse.add_argument('--dataset', type=str, default='data/llm_inputs_xiaomi_ocr.json', help='quantization dataset (embedding format)')
argparse.add_argument('--savepath', type=str, default='./rkllm/xiaomi-ocr-0_w8a8_rk3588.rkllm', help='output rkllm path')
argparse.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
args = argparse.parse_args()

os.makedirs(os.path.dirname(args.savepath), exist_ok=True)

llm = RKLLM()
# Load model
ret = llm.load_huggingface(model=args.path, device=args.device, dtype="float32")
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype="w8a8",
                quantized_algorithm="normal", target_platform="rk3588", num_npu_core=3,
                extra_qparams=qparams, dataset=args.dataset, hybrid_rate=0, max_context=4096)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm(args.savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)
print('Export done:', args.savepath)
