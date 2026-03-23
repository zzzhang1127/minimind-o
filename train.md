# MiniMind-O 训练与使用指南

本文档基于当前仓库代码的真实行为编写，覆盖以下内容：

1. 模型如何加载
2. 数据集如何构建
3. 如何训练（预训练与 SFT）
4. 常见坑位与排错建议

你已说明复用 minimind 的虚拟环境，因此本文不再重复虚拟环境创建步骤。

---

## 1. 目录约定与关键文件

建议在 minimind-o 根目录执行大多数命令，训练命令建议先切到 trainer 目录执行（因为脚本内有相对路径默认值）。

关键文件：

- model/model_olm.py
- trainer/train_pretrain_olm.py
- trainer/train_sft_olm.py
- trainer/trainer_utils.py
- dataset/build_olm_parquet.py
- dataset/lm_dataset.py
- eval_olm.py
- scripts/web_demo.py

---

## 2. 模型加载逻辑（先理解这一段）

### 2.1 权重文件命名规则

训练与加载都遵循如下命名：

~~~text
{save_dir}/{weight}_{hidden_size}{_moe可选}.pth
~~~

例如：

~~~text
out/pretrain_olm_512.pth
out/sft_olm_512.pth
out/sft_vlm_512.pth
~~~

如果启用 MoE（use_moe=1），文件名会多一个 _moe 后缀。

### 2.2 从哪里加载

在 trainer_utils.py 的 init_olm_model 中：

1. 先构建 MiniMindOLM 模型结构（LLM + vision_proj + speech_proj）
2. 根据 train_modality 决定是否加载视觉/语音编码器
3. 再 load_state_dict(strict=False)

strict=False 的意义：

- 允许你从不完全匹配的权重继续训练
- 例如从 minimind-v 权重迁移到 minimind-o 时，speech_proj 可以新初始化

### 2.3 train_modality 参数（你刚新增的核心开关）

当前支持：

- speech
- vision
- both

行为：

1. speech
- 加载语音编码器
- 不加载视觉编码器
- 冻结视觉投影层（不更新）

2. vision
- 加载视觉编码器
- 不加载语音编码器
- 冻结语音投影层（不更新）

3. both
- 视觉和语音编码器都加载
- 两个投影层都可训练

默认值：

- 预训练脚本 train_pretrain_olm.py 默认 speech
- SFT 脚本 train_sft_olm.py 默认 both

---

## 3. 数据集构建（JSONL -> Parquet）

MiniMind-O 训练读取 Parquet，推荐用 build_olm_parquet.py 从 JSONL 构建。

**大数据集（多 GB）**：`PretrainDataset` 按 **row group** 按需读取，不会整表 `read_table`；可将 `--data_path` 指向**目录**（该目录下所有 `*.parquet`）或 **通配**（如 `data/shard_*.parquet`）。若单文件只有一个超大 row group，生成数据时请控制分片/行组大小，详见 `DATA_FLOW_GUIDE.md`。

### 3.1 JSONL 输入格式

每行一个 JSON 样本，推荐字段：

- speech_path 或 audio_path：语音文件路径（相对或绝对）
- image_path：图像路径（可选）
- question/prompt + answer/response，或 conversations（二选一）

最小语音样本示例：

~~~json
{"speech_path":"wav/0001.wav","question":"请转写并总结这段语音","answer":"这段语音主要在讲......"}
~~~

多模态样本示例：

~~~json
{"speech_path":"wav/0002.wav","image_path":"img/0002.jpg","question":"请结合语音和图像回答问题","answer":"......"}
~~~

显式 conversations 示例：

~~~json
{"conversations":[{"content":"<speech>\n请识别并总结"},{"content":"这是识别结果......"}],"audio_path":"wav/0003.wav"}
~~~

说明：

1. 构建脚本支持 force_speech_tag，会自动在首轮 user 文本前补 <speech>
2. 图像列最终保存为 image_bytes（二进制）
3. 语音列最终保存为 speech_path（路径字符串）

### 3.2 构建命令

在 minimind-o 根目录执行：

~~~powershell
python dataset/build_olm_parquet.py ^
  --input_jsonl dataset/my_train.jsonl ^
  --output_parquet dataset/pretrain_olm_speech.parquet ^
  --speech_base_dir D:/data/speech ^
  --image_base_dir D:/data/image ^
  --force_speech_tag
~~~

Linux/Mac 写法：

~~~bash
python dataset/build_olm_parquet.py \
  --input_jsonl dataset/my_train.jsonl \
  --output_parquet dataset/pretrain_olm_speech.parquet \
  --speech_base_dir /data/speech \
  --image_base_dir /data/image \
  --force_speech_tag
~~~

构建完成后会打印：

- 样本总数
- 缺失语音文件数
- 缺失图像文件数

先把缺失样本修掉再训练。

### 3.3 目前的数据能力边界

你当前决定采用单图单语音策略，这是可行且稳定的。

注意：

1. 模型前向支持多图/多语音维度
2. 但当前数据管线与训练实践建议先用单图单语音，便于稳定训练与排错

---

## 4. 训练流程推荐（从 minimind-v 权重继续）

这里给出一个实用且稳定的两阶段方案：

1. 阶段A：语音投影对齐预训练（speech）
2. 阶段B：指令微调（both）

### 4.1 阶段A：语音优先预训练

建议先进入 trainer 目录执行（与脚本内部逻辑一致：`train_pretrain_olm.py` / `train_sft_olm.py` 启动时会 **自动 `cd` 到 `trainer/`**，默认 `../out`、`../dataset/...` 均相对仓库根目录）。

~~~powershell
cd trainer
~~~

命令示例（可省略与默认相同的 `--save_dir`、`--data_path`）：

~~~powershell
python train_pretrain_olm.py ^
  --save_dir ../out ^
  --save_weight pretrain_olm ^
  --weight pytorch_model ^
  --data_path ../dataset/pretrain_olm_speech.parquet ^
  --mode speech ^
  --freeze_llm 1 ^
  --batch_size 8 ^
  --epochs 4 ^
  --learning_rate 3e-4 ^
  --dtype bfloat16
~~~

参数解释：

1. `--weight pytorch_model`（默认）
- 从 `out/pytorch_model.bin` 加载基座（建议将 `minimind-v/MiniMind2-V/pytorch_model.bin` 复制到 `minimind-o/out/`），该基座已含 **vision_proj** 与视觉能力；语音预训练阶段只训练 **speech_proj**（`freeze_llm=1` 时），**vision_proj** 冻结。

2. `--mode speech`
- 只加载 Whisper 语音编码器，不加载 CLIP；**视觉权重仍从 checkpoint 中继承**（用于后续阶段），当前前向不喂图像。

3. 若需从纯 LLM 起步（无 `vision_proj`），可改为 `--weight llm_768`。

4. freeze_llm（仅 `train_pretrain_olm.py`）
- `1`：只训练 `speech_proj`，LLM（含最后一层 Transformer）全部冻结
- `0`：训练 `speech_proj` + 最后一层 Transformer，其余 LLM 冻结（可先 1 再 0 分阶段训）

### 4.2 阶段B：SFT 指令微调（默认 both）

命令示例：

~~~powershell
python train_sft_olm.py ^
  --save_dir ../out ^
  --save_weight sft_olm ^
  --from_weight pretrain_olm ^
  --data_path ../dataset/sft_olm.parquet ^
  --mode both ^
  --hidden_size 512 ^
  --num_hidden_layers 8 ^
  --batch_size 4 ^
  --epochs 2 ^
  --learning_rate 1e-6 ^
  --dtype bfloat16
~~~

说明：

1. SFT 脚本当前默认就是 both
2. 建议显式写出 train_modality both，避免后续改代码后行为变化
3. SFT 学习率建议保守，避免能力漂移

### 4.3 如果你只想训练视觉
预训练脚本 `train_pretrain_olm.py` 当前仅支持 `--mode speech`（用于语音投影对齐）。
如果要训练视觉/多模态，请改用 SFT 脚本，例如：

~~~powershell
python train_sft_olm.py --mode vision ...
（SFT 脚本内部使用 `full_finetune=True`，整模微调，仅冻结未使用模态的投影层）
~~~

---

## 5. 断点续训与权重保存

### 5.1 普通权重

保存在：

~~~text
out/{save_weight}_{hidden_size}{_moe可选}.pth
~~~

### 5.2 续训状态

保存在：

~~~text
checkpoints/{save_weight}_{hidden_size}{_moe可选}_resume.pth
~~~

启用续训：

~~~powershell
python train_sft_olm.py --from_resume 1 ...
~~~

---

## 6. 推理与联调

### 6.1 命令行快速验证

在 minimind-o 根目录：

~~~powershell
python eval_olm.py ^
  --load_from model ^
  --save_dir out ^
  --weight sft_olm ^
  --hidden_size 512 ^
  --num_hidden_layers 8 ^
  --image_path D:/demo/test.jpg ^
  --audio_path D:/demo/test.wav ^
  --prompt "先识别语音，再结合图片回答"
~~~

### 6.2 Gradio 前端

你已新增脚本：

~~~powershell
python scripts/web_demo.py ^
  --load_from model ^
  --save_dir out ^
  --weight sft_olm ^
  --hidden_size 512 ^
  --num_hidden_layers 8 ^
  --server_port 7861
~~~

浏览器打开：

~~~text
http://127.0.0.1:7861
~~~

前端支持：

1. 文本输入
2. 上传图片或摄像头拍摄单图
3. 上传音频或麦克风录入单条语音
4. 联合推理生成回复

---

## 7. 常见问题与排查

### 7.1 报找不到权重

检查：

1. save_dir 与 from_weight 是否对应
2. hidden_size 是否一致（512/768）
3. use_moe 是否一致（文件名是否带 _moe）

### 7.2 报视觉模型不存在

检查 vision_model_path 是否存在 CLIP 模型目录，例如：

~~~text
model/vision_model/clip-vit-base-patch16
~~~

### 7.3 报 whisper 缺失

确认环境安装了 openai-whisper，并且 ffmpeg 可用。

### 7.4 显存不够

优先降低：

1. batch_size
2. max_seq_len
3. hidden_size（若你允许改模型规模）

并提高 accumulation_steps。

---

## 8. 建议的最小可复现流程

1. 准备 JSONL（先做纯语音）
2. build_olm_parquet.py 转成 parquet
3. 运行阶段A预训练（speech）
4. 准备混合 SFT 数据（图文 + 音文 + 文本）
5. 运行阶段B SFT（both）
6. 用 eval_olm.py 与 scripts/web_demo.py 验证

按这条路径，你可以从 minimind-v 平滑迁移到 minimind-o，并逐步打开完整多模态能力。
