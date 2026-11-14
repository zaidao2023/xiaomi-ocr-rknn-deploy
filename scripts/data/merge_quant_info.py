import json
import pickle
import os

out_dir = "data/llm_inputs_xiaomi_ocr"
info_path = "data/llm_inputs_xiaomi_ocr.json"

names = sorted(os.listdir(out_dir), key=lambda x: int(x.replace("sample_", "")))
info = []
for n in names:
    with open(os.path.join(out_dir, n), "rb") as f:
        temp = pickle.load(f)
    info.append({"sample": os.path.join(out_dir, n),
                 "token_nums": temp["inputs_embeds"].shape[1]})

with open(info_path, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("Merged {} samples -> {}".format(len(info), info_path))
