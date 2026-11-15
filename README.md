# xiaomi-ocr-rknn-deploy

**Xiaomi-OCR-0 多模态视觉语言模型 → Rockchip NPU 端侧部署** 的完整转换与验证流水线。

本项目实现了一个 0.8B 多模态 OCR 模型（Xiaomi-OCR-0，Qwen3.5 架构）从 HuggingFace 权重到 **RK3588 / RV1126B** 端侧 NPU 推理的端到端落地：视觉编码器转换为 RKNN 模型、文本解码器量化为 RKLLM 模型，并在真实板端完成推理验证。适用于需要**把视觉语言大模型部署到端侧 NPU**的场景，也可作为多模态大模型端侧部署的参考工程。

---

## 能力概览

| 项目 | 说明 |
| --- | --- |
| 目标模型 | Xiaomi-OCR-0（0.8B，Qwen3_5ForConditionalGeneration，视觉编码器 + 文本解码器） |
| 目标平台 | **RK3588**（6 TOPS NPU）、**RV1126B**（0.5 TOPS NPU） |
| 视觉编码器 | ViT（patch16 / temporal2 / merge2）→ ONNX → **RKNN**（448×448，196 图像 token） |
| 文本解码器 | Qwen3.5 混合线性注意力 + 全注意力 → **RKLLM**（W8A8 量化，max_context 1024） |
| 量化校准 | 基于真实图像**嵌入（embedding）**的校准集，复现部署端预处理，保证激活分布一致 |
| 验证 | RV1126B 真实板端跑通收据文档解析、手写公式识别，附实测耗时 |

## 转换流水线

```
Xiaomi-OCR-0 (HF 权重)
      │
      ├── 视觉编码器 ──► export_vision.py ──► qwen3.5_vision.onnx
      │                                              │
      │                                         export_vision_rknn.py
      │                                              ▼
      │                                    qwen3.5_vision_<platform>.rknn  (NPU)
      │
      └── 文本解码器 ──► make_input_embeds 捕获图像嵌入 ──► 校准集
                                              │
                                  export_rkllm_xiaomi_ocr*.py
                                              ▼
                                   xiaomi-ocr-0_w8a8_<platform>.rkllm  (NPU)
```

视觉编码器与文本解码器分别落到 NPU，板端由多模态 demo 串联：`imgenc` 把图像编码为视觉特征，`demo` 将视觉特征与文本提示拼接后交给 RKLLM 自回归解码。

## 关键技术点

- **多模态拆分部署**：视觉塔导出为独立 RKNN 模型，文本解码器导出为 RKLLM 模型，二者通过视觉特征向量对接，符合 Rockchip 多模态部署范式。
- **嵌入级量化校准**：不用纯文本提示做校准，而是用 hook 捕获真实图像输入在文本解码器入口的 `inputs_embeds`，并**复现板端预处理**（灰底正方形填充 → 448×448 双线性缩放），使校准激活分布与实际部署完全一致。
- **RV1126B 适配**：针对 RV1126B 的约束（仅 1 个 NPU 核、量化类型）调整参数，W8A8 量化在板端完整验证。
- **可复现脚本化**：RK3588 / RV1126B 两条流水线均为一键脚本，路径通过环境变量/参数配置，便于复现与移植。

## 板端实测结果（RV1126B）

环境：RV1126B（Cortex-A53 四核，2GB 内存），rknpu 驱动 v0.9.8，rkllm-runtime 1.3.1。

| 项目 | 耗时 |
| --- | --- |
| LLM 模型加载 | ~8.5 s |
| 视觉编码 NPU 推理（448×448） | **~1.26 s / 张** |
| 解码生成 | 流畅（256 token 内正常输出） |

**测试 1 · 收据图文档解析**（输出 Markdown，结构与阅读顺序正确）：

```
MR. D.I.Y, SDN BHD
(CO.REG :704427-1)
LOT: 85B-A & KEBANGAN - JALAN KBP6,
KAWASAN PERINDUSTRIAN BALAKONG,
4330D SERI KEMBANGAN SELANGOR
(GST ID No.:9C0TQZ2R7E)
(CHERAS LEISURE MALL)
-TAX INVOICE-
CLOTH PEG E140 ... Item(s) : Qty(s): ... Total Incl.GST@% RM:
```

**测试 2 · 手写公式识别**（输出 LaTeX）：

```
\begin{align*} ... n+4(n-k)& = k^k + (m+k)(l+m) ... \end{align*}
```

> 密集小字/数字存在 W8A8 量化噪声（如 `704427-T → 704427-1`），属正常现象；整体文档结构与公式识别准确。

## 目录结构

```
xiaomi-ocr-rknn-deploy/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── TROUBLESHOOTING.md          # 转换/部署踩坑与解决方案
└── scripts/
    ├── export/
    │   ├── export_rkllm_xiaomi_ocr.py           # RK3588 LLM → RKLLM (W8A8)
    │   └── export_rkllm_xiaomi_ocr_rv1126b.py   # RV1126B LLM → RKLLM (W8A8/w4a16)
    ├── data/
    │   ├── make_input_embeds_xiaomi_ocr.py      # 捕获图像嵌入生成量化校准集
    │   ├── merge_quant_info.py                  # 合并校准集信息
    │   └── datasets_xiaomi_ocr.json             # 40 个 OCR 校准样本（图+提示）
    ├── run_convert_rk3588.sh        # RK3588 一键转换流水线
    └── run_convert_rv1126b.sh       # RV1126B 一键转换流水线
```

## 快速开始

### 0. 依赖

本项目基于 Rockchip **rknn-llm**（RKLLM 工具链）与 **rknn-toolkit2**，需在 **Linux x86_64** 环境执行转换（工具链仅提供 Linux wheel）。

```bash
# Python 3.10 ~ 3.12 虚拟环境（建议建在非系统盘，避免权限问题）
python3.12 -m venv venv && source venv/bin/activate

# rkllm-toolkit（见 rknn-llm 仓库 rkllm-toolkit/packages）
pip install rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl
# rknn-toolkit2（见 rknn-toolkit2 仓库 packages/x86_64）
pip install --no-deps rknn_toolkit2-2.5.0-cp312-cp312-manylinux_2_17_x86_64.whl
pip install protobuf==4.25.4 numpy==1.26.4 onnx==1.18.0 onnxruntime opencv-python-headless

# 注意：transformers 需 >= 5.8.1 才能加载 Qwen3_5 架构
pip install transformers==5.8.1
```

完整依赖见 `requirements.txt`。

### 1. 获取源码

```bash
git clone https://github.com/<your-name>/xiaomi-ocr-rknn-deploy.git
git clone https://github.com/airockchip/rknn-llm.git   # 上游依赖
```

将 `xiaomi-ocr-rknn-deploy/scripts/` 下的脚本复制到 rknn-llm 的
`examples/multimodal_model_demo/` 对应子目录（`export/`、`data/`），并把
`run_convert_*.sh` 放在该目录下运行——这样脚本能直接调用上游的
`export_vision.py`、`export_vision_rknn.py` 与多模态 demo。

### 2. 转换（以 RV1126B 为例）

```bash
cd rknn-llm/examples/multimodal_model_demo
MODEL=/path/to/Xiaomi-OCR-0            # HuggingFace 模型目录
PY=/path/to/venv/bin/python
bash run_convert_rv1126b.sh
```

产出：
- `rknn/qwen3.5_vision_rv1126b.rknn`（视觉编码器）
- `rkllm/xiaomi-ocr-0_w8a8_rv1126b.rkllm`（文本解码器）

RK3588 用 `bash run_convert_rk3588.sh`，逻辑相同、目标平台不同。

### 3. 板端推理

板端编译多模态 demo（见 rknn-llm `examples/multimodal_model_demo/deploy`），
推入两个模型后运行：

```bash
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
printf '<image>请对这张图片进行文档解析，将内容以 Markdown 格式输出。\nexit\n' | \
  ./demo img/receipt.jpg ./qwen3.5_vision_rv1126b.rknn "" "" \
  ./xiaomi-ocr-0_w8a8_rv1126b.rkllm 256 1024 1 rv1126b
```

参数：`max_new_tokens max_context_len rknn_core_num platform`。RV1126B 用 `1 rv1126b`，RK3588 用 `3 rk3588`。

## 说明与局限

- **w4a16 在 RV1126B 上转换崩溃**：rkllm-toolkit 在 RV1126B 上对 w4a16 做全注意力层优化时会静默退出（工具链问题，非本工程配置问题），当前 RV1126B 使用已验证的 **W8A8**。详见 `docs/TROUBLESHOOTING.md`。
- 视觉编码器按官方流程以 **float32** 构建 RKNN；如板端存储/内存紧张，可再做 int8 量化压缩。
- 模型权重（Xiaomi-OCR-0）不在本仓库内，请从 HuggingFace 下载：`SeerRay-Lab/Xiaomi-OCR-0`。

## 参考

- [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) — 上游多模态 OCR 模型
- [rknn-llm](https://github.com/airockchip/rknn-llm) — Rockchip RKLLM 工具链与多模态 demo（Apache-2.0）
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — Rockchip RKNN 工具链（Apache-2.0）

## License

Apache-2.0。本仓库脚本基于 Rockchip rknn-llm（Apache-2.0）示例改造，在此致谢。
