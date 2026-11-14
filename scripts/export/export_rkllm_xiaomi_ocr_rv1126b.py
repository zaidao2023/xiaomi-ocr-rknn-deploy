import os
from rkllm.api import RKLLM
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='/path/to/Xiaomi-OCR-0', help='model path')
argparse.add_argument('--dataset', type=str, default='data/llm_inputs_xiaomi_ocr.json', help='quantization dataset (embedding format)')
argparse.add_argument('--savepath', type=str, default=None, help='output rkllm path')
argparse.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
argparse.add_argument('--max_context', type=int, default=1024, help='max context length')
argparse.add_argument('--quantized_dtype', type=str, default='w8a8', help='w8a8 (verified) or w4a16 (experimental)')
args = argparse.parse_args()

if args.savepath is None:
    args.savepath = './rkllm/xiaomi-ocr-0_{}_{}.rkllm'.format(args.quantized_dtype, 'rv1126b')

os.makedirs(os.path.dirname(args.savepath), exist_ok=True)

llm = RKLLM()
# Load model
ret = llm.load_huggingface(model=args.path, device=args.device, dtype="float32")
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype=args.quantized_dtype,
                quantized_algorithm="normal", target_platform="rv1126b", num_npu_core=1,
                extra_qparams=qparams, dataset=args.dataset, hybrid_rate=0, max_context=args.max_context)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm(args.savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)
print('Export done:', args.savepath)
