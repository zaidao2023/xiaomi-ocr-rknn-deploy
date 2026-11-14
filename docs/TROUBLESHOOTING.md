# 踩坑与解决方案

记录将 Xiaomi-OCR-0 部署到 Rockchip NPU 过程中遇到的关键问题及解法，供复现与移植参考。

## 环境搭建

### rkllm-toolkit / rknn-toolkit2 仅有 Linux wheel
两个工具链都只提供 `linux_x86_64` 的 wheel，**无法在 Windows 原生环境转换**。需在 Linux 环境执行（可用 WSL2、虚拟机或交叉开发机）。建议虚拟环境建在非系统盘，避免权限与空间问题。

### transformers 版本需 >= 5.8.1
Xiaomi-OCR-0 为 `Qwen3_5ForConditionalGeneration` 架构，该架构在 **transformers 5.8.0 中不存在**，加载会报 `ModuleNotFoundError`。rkllm-toolkit 的依赖元数据会把 transformers 钉在 5.8.0，安装 wheel 时可能把它回拉——需手动升级并固定到 `5.8.1`。

### numpy / protobuf 版本冲突
- 依赖链可能把 numpy 升级到 2.x，与 rkllm-toolkit（要求 `numpy<=1.26.4`）冲突，报 `module 'numpy' has no attribute 'ndarray'`。固定 `numpy==1.26.4`。
- rknn-toolkit2 要求 `protobuf<=4.25.4`，依赖链可能装到 5.x/7.x，导致 RKNN 构建报 `Unkown op target`。固定 `protobuf==4.25.4`。

### rknn-toolkit2 会拉低 torch 版本
rknn-toolkit2 2.5.0 的依赖声明为 `torch<=2.4.0`，与已有的 torch 2.6.0 冲突。用 `pip install --no-deps` 安装 rknn-toolkit2 wheel，再手动补齐其依赖（protobuf、onnxruntime、opencv、fast-histogram 等），避免 torch 被降级。

## 量化校准

### 嵌入捕获 OOM
用 hook 捕获文本解码器入口的 `inputs_embeds` 时，若每轮前向构建了 autograd 计算图，内存会随样本累积，最终 OOM 被杀（实测 RSS 冲到 15GB+）。
**解法**：用 `torch.no_grad()` 包裹前向，RSS 稳定在约 3GB。

### 校准样本需与部署预处理一致
校准样本的图像若直接喂给 processor，不同长宽比的图会产生不同的图像 token 数（实测 220~8190 token），与部署端 448×448 的固定 196 图像 token 不一致，激活分布会偏移。
**解法**：在捕获脚本中复现板端 demo 的预处理——**灰底（127）正方形填充 → 448×448 双线性缩放**，再交给 processor。校准后每个样本 token 数稳定在 220~250，与部署完全对齐。

### 校准集 sample 路径解析
RKLLM 的 `build()` 解析 dataset 中的 `sample` 路径时，**相对于数据集文件所在目录**而非工作目录。路径写成 `data/llm_inputs/...` 会导致 `data/data/...` 双重路径报错。
**解法**：sample 路径相对数据集文件目录，如数据集在 `data/` 下，则写 `llm_inputs_xiaomi_ocr/sample_N`。

## RV1126B 适配

### w4a16 转换崩溃（工具链问题）
RV1126B 上使用 `w4a16` 量化，`build()` 优化到第 11 层（全注意力层）附近时进程**静默退出**（无报错、非 OOM，疑似 rkllm-toolkit 内部崩溃）。线性注意力层（0~10）正常，问题出在全注意力 + w4a16 的组合。
**解法**：RV1126B 改用已验证的 **W8A8**（官方 benchmark 亦实测 Qwen3.5-0.8B w8a8 可跑）。若后续工具链修复，可再尝试 w4a16。

### RV1126B 约束
- `num_npu_core` 仅支持 **1**。
- 官方 benchmark 显示 RV1126B 支持 w4a16 系列与 w8a8；W8A8 实测 Qwen3.5-0.8B 约 1017MB 内存、7.23 token/s。

## 板端部署

### 交叉编译 demo 需 32 位/64 位工具链与 libgomp
RV1126B BSP 的 buildroot 工具链默认**不含 OpenMP**（缺 `libgomp.spec` 与 `libgomp.so`），demo 的 CMake 里 `find_package(OpenMP REQUIRED)` 会配置失败。
**解法**：
1. demo 源码本身不使用 OpenMP（仅 librkllmrt.so 内部用到），将 `find_package(OpenMP REQUIRED)` 改为非必需即可；
2. 运行期由板端系统库 `libgomp.so.1` 提供，链接时用 `--allow-shlib-undefined` 放行。

### demo 二进制无执行权限
跨 FAT/NTFS 拷贝会丢失可执行位，板端报 `Permission denied`。`chmod +x` 后正常。

### demo 在 stdin EOF 后空输入循环
demo 读取 stdin 的循环在 EOF 时不会退出，会反复以空输入触发推理（伴随 KV cache reuse 警告）。
**解法**：用 `printf '<prompt>\nexit\n' | ./demo ...` 显式发送 `exit` 干净退出。

### 板端磁盘被日志撑满
大文件 scp 与高负载运行会使 `/var/log/syslog`、`kern.log` 暴涨（实测各 1.4G），撑满磁盘导致后续操作失败。
**解法**：`truncate -s 0 /var/log/syslog /var/log/kern.log` 清理（系统日志，低风险）。

### WSL → 板子网络慢
WSL2 虚拟网络层到板子的 scp 速度极慢（5MB 超 2 分钟）。**解法**：从 Windows 原生网络直连传输（实测约 3.5MB/s），或用 sftp 分块传输大模型。
