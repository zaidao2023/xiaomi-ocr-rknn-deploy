import torch
import os
import json
import pickle
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import argparse

class StopForward(Exception):
    """Used to stop forward pass intentionally."""
    pass

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, help='model path', required=True)
argparse.add_argument('--datasets', type=str, default='data/datasets_xiaomi_ocr.json', help='dataset json file')
argparse.add_argument('--out_dir', type=str, default='data/llm_inputs_xiaomi_ocr', help='output pickle dir (relative to cwd)')
argparse.add_argument('--info_path', type=str, default='data/llm_inputs_xiaomi_ocr.json', help='info json output path')
args = argparse.parse_args()

print(f"Loading model: qwen3.5 from {args.path}")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.path, dtype="auto", device_map="cpu",
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(args.path)

captured = []
def model_pre_hook(module, args, kwargs):
    captured.append(kwargs)
    raise StopForward

handle = model.model.language_model.register_forward_pre_hook(
    model_pre_hook,
    with_kwargs=True
)

info = []
datasets = json.load(open(args.datasets, 'r'))
os.makedirs(args.out_dir, exist_ok=True)
def expand2square(pil_img, background_color=127):
    width, height = pil_img.size
    if width == height:
        return pil_img
    new_size = (max(width, height), max(width, height))
    new_img = Image.new("RGB", new_size, background_color)
    new_img.paste(pil_img, ((new_size[0] - width) // 2, (new_size[1] - height) // 2))
    return new_img

for idx, data in enumerate(tqdm(datasets)):
    imgp = os.path.join(data["image_path"], data["image"])
    image = Image.open(imgp).convert("RGB")
    image = expand2square(image, 127)
    image = image.resize((448, 448), Image.BILINEAR)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": data["input"]},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    try:
        with torch.no_grad():
            model(**inputs)
    except StopForward:
        pass
    temp = captured[-1]
    sample_name = "sample_{}".format(idx)
    info.append({"sample": os.path.join(args.out_dir, sample_name), "token_nums": temp["inputs_embeds"].shape[1]})
    with open(os.path.join(args.out_dir, sample_name), 'wb') as f:
        pickle.dump(temp, f)
    captured.pop()
    import resource
    print(f"  mem_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB", flush=True)

handle.remove()
with open(args.info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("Done: {} samples -> {}".format(len(info), args.info_path))
