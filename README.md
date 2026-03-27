# MiniMind-O（Speech + Vision + LLM）

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 与 [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v) 做最小改造，增加了**语音识别（ASR）**能力：使用约 **10GB 语音-文本对**进行训练（仅预训练对齐阶段）。由于未找到合适的中英文指令微调问答数据，且预训练效果本身一般，因此**未做指令微调**。

## 数据集来源与目录结构（CS-Dialogue short_wav）

语音数据来源：[`BAAI/CS-Dialogue`](https://huggingface.co/datasets/BAAI/CS-Dialogue) 的 **104 小时**人物对话（中英文混杂），使用其中的 **short_wav** 版本。

你将其下载解压为 `dataset/short_wav/` 后，核心结构如下（注意：本文所有路径均使用 `/`）：

```text
minimind-o/
  dataset/
    short_wav/
      SCRIPT/
        ZH-CN_U0001_S0.txt
        ZH-CN_U0002_S0.txt
        ...
      WAVE/
        C0/
          ZH-CN_U0001_S0/
            ZH-CN_U0001_S0_4.wav
            ...
          ZH-CN_U0002_S0/
            ...
```

### SCRIPT 与 WAVE 的一一对应关系

- **SCRIPT 中每个 txt**：对应 **WAVE/C0 下同名文件夹**
- **txt 中每一行**：对应该文件夹下同名 `utt_id.wav`

例如：

- `dataset/short_wav/SCRIPT/ZH-CN_U0001_S0.txt` 中一行：
  - `ZH-CN_U0001_S0_4    <MIX> 嗯，我是 Luna ，嗯，你现在。`
- 对应音频：
  - `dataset/short_wav/WAVE/C0/ZH-CN_U0001_S0/ZH-CN_U0001_S0_4.wav`

另外，数据脚本会清理转写文本中形如 `<CN>/<EN>/<MIX>/<SPK/>...` 的尖括号标签（见 `dataset/build_pretrain_parquet.py`）。

## 环境与依赖安装（清华镜像）

建议直接复用 `minimind` 的虚拟环境，然后在该环境中为本项目补装额外依赖：

```bash
cd minimind-o
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 主要包含：

- `pyarrow`（Parquet 读写）
- `pillow`（图像处理）
- `openai-whisper` / `transformers` 相关依赖（Whisper 特征/模型加载）
- `gradio`（如需界面化演示）

## 模型文件准备

### 1）语音编码器（Whisper）

语音编码器使用 `whisper-base`，按 Hugging Face 仓库方式下载到本项目目录：

```bash
cd minimind-o
git clone https://huggingface.co/openai/whisper-base model/speech_model/whisper-base
```

### 2）视觉编码器（可选）

如需测试视觉能力，下载 CLIP 到：

```bash
cd minimind-o
git clone https://huggingface.co/openai/clip-vit-base-patch16 model/vision_model/clip-vit-base-patch16
```

### 3）作为多模态基座的权重（来自 MiniMind2-V）

训练时需要一个“带视觉侧工程对齐结构”的基座权重。做法是：

```bash
cd minimind-o
git clone https://huggingface.co/jingyaogong/MiniMind2-V
```

然后**只把**其中的 `pytorch_model.bin` 放到：

```text
minimind-o/out/pytorch_model.bin
```

（训练脚本默认会在 `out/` 下查找该文件作为初始化权重。）

## 构造预训练 Parquet（speech-to-text）

训练数据以 Parquet 形式读取（两列：音频二进制、转写文本二进制），由脚本 `dataset/build_pretrain_parquet.py` 从 `dataset/short_wav/` 构建：

```bash
cd minimind-o
python dataset/build_pretrain_parquet.py \
  --short_wav_root dataset/short_wav \
  --output_parquet dataset/pretrain_s2t.parquet
```

该脚本会：

- 遍历 `dataset/short_wav/WAVE/*/*/*.wav`
- 在 `dataset/short_wav/SCRIPT/<folder>.txt` 中查找同名 `utt_id` 的转写
- 清洗转写文本中的尖括号标签（如 `<MIX>`、`<SPK/>`）
- 校验音频必须是 **16kHz、16-bit PCM**，且时长需大于指定阈值（默认 0.5s）；不满足则跳过
- 按 chunk 写入 Parquet（默认 `chunk_size=256`，对应 Parquet 的 row-group 写入粒度）

最终输出默认为：`dataset/pretrain_s2t.parquet`

## 训练：语音预训练对齐（只训投影层 + LLM 最后一层）

训练脚本：`trainer/train_pretrain_olm.py`

你给出的训练方式如下（在仓库内执行）：

```bash
cd minimind-o/trainer
python train_pretrain_olm.py --use_compile 1 --use_wandb
```

训练过程关键点（来自代码实现）：

- 会固定工作目录到 `minimind-o/trainer/`，并把默认 `../out`、`../dataset` 等路径稳定为仓库相对路径
- 初始化时会从 `out/` 中加载基座权重（默认 `out/pytorch_model.bin`）
- **冻结**大部分参数，仅开放训练：
  - `model.speech_proj`（将 Whisper encoder 特征投影到 LLM hidden space）
  - `model.model.layers.<last>`（LLM 最后一层 Transformer）
- 记录训练日志使用 `swanlab`（代码中以 `import swanlab as wandb` 的方式兼容 WandB API）
- 数据默认从 `dataset/pretrain_s2t.parquet` 读取（参数 `--data_path` 可改）

### 性能/稳定性提示（实践经验）

- 本地训练时，`num_workers > 1` 可能导致主机内存占用激增而 OOM（因此脚本默认 `--num_workers 0`）
- 云端训练时，即使 CPU 打满，GPU 利用率仍可能剧烈波动；常见原因是数据解码与特征提取（WAV 解码、mel 计算、Parquet row-group 解压）开销较大，导致 GPU 等待数据

## 推理评估（speech / vision / both）

推理脚本：`eval_olm.py`

### 语音模式（speech）

示例命令（路径与中文参数建议加引号）：

```bash
cd minimind-o
python eval_olm.py \
  --mode speech \
  --load_from "./out/pretrain_olm_768.pth" \
  --audio_path "./dataset/eval_speeches/我们那边有大海，然后天气也很好，空气也很好，就是夏天的时候特别热，北京夏天热吗？.wav" \
  --prompt "这段语音说了什么？"
```

你观测到的典型输出（节选）：

```text
Model Params: 105.02M
Trainable Params: 105.016M (same counting scope as total; excludes frozen vision_encoder / speech_encoder)
[Speech] mel 时间维 T=3000, encoder 有效帧 P=1500 → 语音 token 数 N=150
Prompt: 这段语音说了什么？
Answer:
[Speed]: 19.38 tokens/s
...
```

现象：模型能大致转录/理解语音主旨，但句末可能出现重复或不稳定。

### 视觉模式（vision，可选）

先准备好 `model/vision_model/clip-vit-base-patch16`（见上文），然后：

```bash
cd minimind-o
python eval_olm.py --mode vision --load_from "./out/pretrain_olm_768.pth" --image_path "./dataset/eval_images/xxx.jpg" --prompt "图片里有什么？"
```

### 双模态（both）

脚本支持 `--mode both` 同时输入图片与语音，但由于缺少“图像+语音联合指令数据”的微调，对齐稳定性有限：实际推理时往往更偏向输出语音内容，图像内容可能胡言乱语。

## 项目原理与数据流（训练 / 推理）

本节按代码实现说明“数据如何从磁盘流到模型、以及模型如何把语音/图像注入到 LLM token 序列中”。

### 1）整体结构：LLM +（Vision Encoder）+ Speech Encoder + Projection

核心模型类：`model/model_olm.py` 中的 `MiniMindOLM`

- **LLM 主体**：`MiniMindForCausalLM`（定义见 `model/model_minimind.py`）
- **视觉编码器（可选加载）**：CLIP（`CLIPModel`），参数全部冻结
- **语音编码器（可选加载）**：Whisper encoder（`WhisperModel` 的 `encoder` 部分），参数全部冻结
- **投影层**：
  - `VisionProj`：把 CLIP 特征线性投影到 LLM hidden size
  - `SpeechProj`：把 Whisper encoder 特征线性投影到 LLM hidden size，并按时间聚合得到“语音 token 序列”

### 2）Parquet 数据集的构造与读取

#### 构造（`dataset/build_pretrain_parquet.py`）

生成 `dataset/pretrain_s2t.parquet`，两列：

- `speech_bytes`：wav 文件原始字节
- `transcript_bytes`：utf-8 编码的转写文本

清洗策略：用正则去掉 `<...>` 形式标签并压缩空白。

#### 读取（`dataset/pretrain_dataset.py`）

`PretrainDataset` 的读取特点：

- 支持传入单个 parquet、目录（读取目录下所有 `*.parquet`）、或通配符（例如 `dataset/shards/part_*.parquet`）
- **不一次性把整表读入内存**：初始化阶段只扫描 metadata，建立“全局行号 → (文件, row_group, 行内偏移)”索引
- `__getitem__` 时只解压包含该样本的 **单个 row_group**（列裁剪只读 `speech_bytes` 与 `transcript_bytes`）

输出给训练循环的 batch（由 `pretrain_collate_fn` 堆叠）：

- `input_ids`：对话模板后的 token id（固定到 `max_length`）
- `labels`：只在 `assistant` 段计算 CE，其他位置为 `-100`
- `speech_values`：Whisper 输入特征（形状为 `[B, 1, T, 80]`）
- `speech_lengths`：mel 时间维长度（形状为 `[B, 1]`）

### 3）Whisper：从音频到 encoder 特征向量

训练侧与推理侧都使用 Hugging Face 的 `WhisperFeatureExtractor` 生成 log-mel：

- WAV 解码：读 16-bit PCM，必要时转单声道；若采样率不是 16k，会重采样到 16k
- 特征提取：`WhisperFeatureExtractor(...).input_features` 输出形状为 `[1, 80, 3000]`
- 在项目里通常转置为 `[3000, 80]`（把时间维放到第一维，便于后续统一处理）

然后送入 Whisper encoder（冻结）：

- 输入：`[B, 80, 3000]`
- 输出：`[B, 1500, 512]`（Whisper encoder 的时间降采样与 hidden size）

### 4）把语音/图像“注入”到 LLM token 序列中

注入发生在 `MiniMindOLM.forward()` 的 embedding 阶段（`start_pos == 0` 的首段计算）：

1. 先算文本 token embedding：`embed_tokens(input_ids)` 得到 `hidden_states: [B, L, D]`
2. 若有图像：
   - CLIP vision 输出 patch token（去掉 CLS 后）→ `VisionProj` → 替换 prompt 中图像占位符对应区间
3. 若有语音：
   - Whisper encoder 输出 `speech_tensors: [B*num, 1500, 512]`
   - `SpeechProj` 将其按时间聚合成 `N` 个 token，并线性投影到 `D`
   - 在 `input_ids` 中查找一段连续的“语音占位符 token”（由数据侧构造）
   - 将这段占位符对应的 `hidden_states` 切片替换为语音投影后的 token 序列

替换完成后，后续就是标准 decoder-only Transformer 的自回归计算与 `generate()` 推理。

### 5）训练时的加载与数据流（`trainer/train_pretrain_olm.py`）

1. 从 `out/` 加载基座权重并初始化 `MiniMindOLM`
2. 冻结参数，仅放开 `speech_proj` 与 LLM 最后一层
3. `PretrainDataset` 从 `dataset/pretrain_s2t.parquet` 读取样本：
   - 语音 bytes → mel → `speech_values`
   - transcript → chat_template → `input_ids/labels`
4. 前向：`model(input_ids, labels, speech_values, speech_lengths)`，loss 为 `res.loss + res.aux_loss`
5. 保存：
   - `out/pretrain_olm_*.pth`：保存时会剔除 `vision_encoder.*` 与 `speech_encoder.*`（编码器本身不入库）
   - `checkpoints/pretrain_olm_*_resume.pth`：保存训练状态用于断点续训

### 6）推理时的加载与数据流（`eval_olm.py`）

1. `--load_from` 支持传目录或直接传权重文件；若目录里无 `config.json`，脚本会从权重自动推断模型层数与 hidden size
2. 按 `--mode` 决定是否加载 vision/speech encoder
3. 语音输入：
   - `MiniMindOLM.speech2tensor(audio_path)` → mel（转成 `[T, 80]`）
   - 计算并打印语音 token 数
   - 构造 prompt：将 `<speech>` 替换为对应数量的占位符
4. `model.generate(...)` 输出文本

## 目录速览

```text
minimind-o/
  model/
    model_minimind.py
    model_olm.py
    speech_model/whisper-base/               # git clone 下载
    vision_model/clip-vit-base-patch16/      # git clone 下载（可选）
  dataset/
    short_wav/                               # CS-Dialogue short_wav 解压目录
    build_pretrain_parquet.py                # short_wav -> pretrain_s2t.parquet
    pretrain_dataset.py                      # Parquet 读取 + 语音特征提取 + prompt/labels 构造
    pretrain_s2t.parquet                     # 生成后的训练数据
    eval_speeches/                           # 推理用语音样例
    eval_images/                             # 推理用图片样例
  trainer/
    train_pretrain_olm.py                    # 语音预训练对齐入口
    trainer_utils.py                         # init_olm_model / lr / checkpoint 等
  out/
    pytorch_model.bin                        # 从 MiniMind2-V 拿到的基座权重
    pretrain_olm_*.pth                       # 训练输出权重
  eval_olm.py                                # 推理脚本
  requirements.txt
```

