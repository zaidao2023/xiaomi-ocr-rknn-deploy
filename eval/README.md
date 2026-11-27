# OCR 精度评估框架

对端侧部署的 Xiaomi-OCR-0 推理结果做**字符级精度量化评估**，用客观指标（CER / 编辑距离 / 精确匹配）替代"能跑通"的主观判断，为量化方案对比（嵌入校准 vs 文本校准、视觉 fp32 vs int8 等）提供统一基准。

## 指标

| 指标 | 说明 |
| --- | --- |
| **CER** | 字符错误率 = 编辑距离 / 真值长度，越低越好 |
| **Accuracy** | 1 − CER，字符级准确率 |
| **Edit distance** | Levenshtein 编辑距离（替换+插入+删除） |
| **Exact match** | 归一化后是否完全一致 |

文本会先做归一化（折叠空白、可选忽略大小写）并剥离模型输出的代码围栏，保证对比公平。

## 测试用例

`test_cases.json` 定义用例（图片、提示词、真值文件、预测文件）。真值取自 Xiaomi-OCR-0 自带的 `example_pics/*.md`。

## 两种模式

**from-file（离线）**：对已抓取的模型输出打分，用于本地验证指标逻辑、复算历史结果。

```bash
python ocr_eval.py --mode from-file --cases test_cases.json --fixtures fixtures --out report.json
```

**run（板端）**：SSH 驱动板子上的 demo 非交互推理，抓取输出后打分。

```bash
python ocr_eval.py --mode run --cases test_cases.json \
    --ssh root@10.62.9.139 --workdir /root/xiaomi_ocr \
    --vision models/qwen3.5_vision_rv1126b.rknn \
    --llm models/xiaomi-ocr-0_w8a8_rv1126b.rkllm --out report_board.json
```

`--lower` 可启用大小写不敏感归一化。

## 目录

- `ocr_eval.py` — 评估框架主程序
- `test_cases.json` — 测试用例配置
- `fixtures/` — 真值（`*.gt.txt`）与已抓取的预测输出（`*.pred.txt`）

## 说明

- 字符级 CER 对收据等文本类 OCR 是标准且有效的指标；对手写公式这类 LaTeX 结构化输出，CER 会受语法噪声影响，数值偏低属正常，需结合具体输出判断。
- `from-file` 模式会跳过缺失预测文件的用例，便于先对已有抓取结果复算、板端再补全。
