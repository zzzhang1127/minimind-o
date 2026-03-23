# OLM 数据流完整性检查与修复指南

## 🔴 关键问题汇总

### Problem #1: 语音占位符数量须与 `N = P // 10` 一致 ⚠️ **CRITICAL**

**当前设计（minimind-o）:**
- **P**：有效 Whisper **encoder** 时间帧数（代码里 `encoder_len = mel_len // 2`）。
- **N**：语音侧 token 数，`N = P // speech_frames_per_token`（默认每 10 帧 encoder → 1 个 LLM token）。`P < 10` 时 **N=0**（该样本在预训练里会被跳过）。
- 数据侧用连续 **`#`**（tokenizer 中应对应同一 `speech_token_id`，默认 **5**）重复 **N** 次替换 `<speech>`。
- 模型侧用 **`_find_consecutive_token_spans`** 定位连续 `speech_token_id`，再用 **`SpeechProj`** 输出的 `[N, hidden]` 逐段替换，**不再**使用固定长度的 `speech_ids` 模板匹配。

**影响:**
- 若 `#` 的个数 ≠ `N`，前向会 `ValueError: speech token 数与占位符不一致`。

**快速检查:**
```bash
cd dataset
python verify_tokenization.py
```

**还须检查** 与 parquet 样本一致：
```bash
python validate_data_flow.py --parquet ./pretrain_s2t.parquet
```

### 大数据集 Parquet（约 10GB+）与内存

- **`PretrainDataset`** 已改为**不**一次性 `read_table` 载入全表：启动时只扫描元数据（行组边界）并用 **schema 列名** 校验，**不会**为校验而 `read_row_group` 解压首个行组；`__getitem__` 时对该样本所在 **row group** 调用 `read_row_group`（仅两列），用毕 `del` 释放 `Table`。
- 若单个 row group 仍极大（例如整库只有一个超大 group），一次读取仍会占很多内存；请在生成 Parquet 时控制 **row group 大小**或拆成多个 **shard** 文件，并将训练参数 `parquet_path` 设为**目录**（该目录下所有 `*.parquet`）或 **通配路径**（如 `data/part_*.parquet`）。

### Problem #2: Speech Lengths 未被使用

**现象:**
```python
def forward(self, ..., speech_lengths: Optional[torch.LongTensor] = None, ...):
    # speech_lengths 从未被使用！
```

**影响:**
- 可变长度音频被默认填充到 batch 最长长度
- 填充的 mel 频谱中是 0 值，被当作有效特征处理
- Whisper encoder 在处理这些填充时产生无效特征

**长期影响:**
- 模型难以从这些"虚假"样本学习
- 训练效率降低，收敛困难

**快速修复:**
由于 `whisper.pad_or_trim()` 会把所有音频统一到 30 秒长度，所以 mel 频谱应该已经是固定大小。但为了保险，建议：

1. **验证 speech_values 是否真的都是相同大小:**
```python
# 在 dataset/validate_data_flow.py 中检查
# 如果所有 speech_tensor.shape[1] 都相同，则无需担心
```

2. **如果长度不同，需要添加 attention mask:**
   见下面的 collate_fn 实现

### Problem #3: 数据流端到端验证

**目前状态:**
- 代码逻辑看似正确
- 但没有实际运行过完整的数据→模型前向过程

**验证方法:**
```bash
cd dataset
python validate_data_flow.py --parquet ./pretrain_s2t.parquet
```

这个脚本会检查:
- ✅ Dataset 能否正确加载样本
- ✅ 占位符是否被正确替换为特殊token
- ✅ Speech 特征是否包含 NaN/Inf
- ✅ Batch collation 是否保持形状正确性
- ✅ 模型前向过程是否完成

---

## 📋 数据流分解

### 流程图

```
Parquet 文件
    ↓
[conversations] JSON 字符串
[speech_b64] base64 编码 WAV
    ↓
OLMDataset.__getitem__
    ├─ 解析 conversations JSON
    ├─ 替换 <speech> → "####...(100个#)"
    ├─ Tokenizer 处理
    │  └─ "####..." → [35, 35, ..., 35] (假设)
    ├─ decode speech_b64
    │  ├─ base64 → WAV 字节
    │  ├─ wave.open → PCM
    │  ├─ → float32 audio
    │  ├─ whisper.pad_or_trim
    │  └─ whisper.log_mel_spectrogram → (time_steps, 128)
    └─ 返回 (input_ids, labels, image, speech_tensor, speech_lengths)
    ↓
DataLoader (batch_size=N)
    └─ Stack/Pad → (N, 512), (N, 1, T, 128), ...
    ↓
Model.forward()
    ├─ Embed tokens → (N, 512, hidden_size)
    ├─ Whisper encoder(speech_tensor)
    │  └─ 处理 (N, 1, T, 128) → (N, 1, T', encoder_hidden)
    ├─ Speech proj (降采样 k=5)
    │  └─ → (N, 1, T'/5, hidden_size)
    ├─ _count_modal_proj (替换占位符)
    │  ├─ 找到所有 [35, 35, ...] 的位置
    │  └─ 用音频特征替换
    └─ LLM 前向 → logits
```

### 各环节可能的问题点

| 环节 | 检查内容 | 可能问题 | 解决方案 |
|---|---|---|---|
| Tokenizer | `#`*100 → [35]*100 | ID 不对应 | 运行 verify_tokenization.py |
| Base64 解码 | WAV 字节有效性 | 损坏/不完整的 WAV | 检查 build_olm_parquet.py 中的 audio_bytes |
| Mel 频谱 | shape (T, 128) | Whisper 输出格式错误 | 验证 whisper.log_mel_spectrogram 返回值 |
| Speech proj | 降采样 K=5 正确 | 长度不匹配 | 在 speech_proj.forward 中打印 debug 信息 |
| 占位符替换 | _find_indices 找到 ID | 没有找到占位符 | 检查 model_olm.py 中 _count_modal_proj |

---

## ✅ 完整修复流程

### Step 1: 验证 Tokenization
```bash
python dataset/verify_tokenization.py
```

**如果失败:**
- 确认 tokenizer 下单个 `#` 的 `input_id` 与 `OLMConfig.speech_token_id`（默认 5）一致
- 确认样本里「`#` 的个数」等于 `N = (mel_len//2) // 10`（见 `num_speech_tokens_from_encoder_length`）

### Step 2: 验证数据流
```bash
python dataset/validate_data_flow.py --parquet dataset/pretrain_s2t.parquet
```

**检查清单:**
- ✅ `speech_tensor shape: (1, T, 80)` — mel 维 80
- ✅ `expected speech tokens N` 与 `input_ids` 里 `#`（`speech_token_id`）计数一致
- ✅ `Speech tensor contains NaN: False`
- ✅ `Batch collation valid`（`pretrain_collate_fn`）
- ✅ `Forward pass successful`

### Step 3: 处理可变长度音频 (可选)

如果验证中发现 `speech_tensor` 长度不一致，需要添加 collate_fn：

```python
# 在 train_pretrain_olm.py 中

from dataset.olm_collate import olm_collate_fn

# 修改 DataLoader
loader = DataLoader(
    train_ds,
    batch_sampler=batch_sampler,
    collate_fn=olm_collate_fn,  # ← 添加这行
    num_workers=args.num_workers,
    pin_memory=True
)

# 修改 train_epoch 的解包
for step, (input_ids, labels, pixel_values, speech_values, speech_lengths, speech_mask) in enumerate(loader):
    # speech_mask 是新返回的注意力mask
    # 在模型中使用 speech_mask 来避免处理填充部分
```

### Step 4: 开始训练 (在所有检查通过后)

```bash
cd trainer
python train_pretrain_olm.py \
  --save_dir ../out \
  --save_weight pretrain_olm \
  --data_path ../dataset/pretrain_s2t.parquet \
  --weight pytorch_model \
  --mode speech \
  --freeze_llm 1 \
  --batch_size 4 \
  --epochs 1 \
  --learning_rate 3e-4 \
  --dtype bfloat16
```

---

## 🔧 Debug 技巧

### 如果模型收不到音频特征

**诊断步骤:**

1. 检查 `_find_consecutive_token_spans(tokens, speech_token_id)` 是否找到连续 `#`：

```python
# 在 model_olm.py 中调试语音注入
def _inject_speech_tokens(...):
    spans = self._find_consecutive_token_spans(tokens, token_id)
    print(f"[DEBUG] speech spans: {spans}")
    
    if not speech_proj_grouped or not spans:
        print(f"[DEBUG] Returning h unchanged (no match found)")
        return h
    # ... rest of code
```

2. 检查 `create_chat_prompt` 是否正确替换占位符:

```python
# 在 lm_dataset.py 中添加 debug
def create_chat_prompt(self, conversations):
    messages = []
    for i, turn in enumerate(conversations):
        role = 'user' if i % 2 == 0 else 'assistant'
        content = turn['content'].replace('<image>', self.image_token).replace('<speech>', self.speech_token)
        print(f"[DEBUG] Turn {i} - role={role}, <speech> present={('<speech>' in turn['content'])}, "
              f"speech_token length={len(self.speech_token)}")
        messages.append({"role": role, "content": content})
    return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

3. 检查 tokenizer 后的结果:

```python
# 在 __getitem__ 中添加 debug
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
speech_token_id = self.tokenizer('#', add_special_tokens=False).input_ids[0]
speech_count = (torch.tensor(input_ids) == speech_token_id).sum().item()
print(f"[DEBUG] Total input_ids: {len(input_ids)}, "
      f"Speech token (ID={speech_token_id}) count: {speech_count}")
```

---

## 📝 关键检查清单

- [ ] 运行 `verify_tokenization.py` 确认占位符 ID 正确
- [ ] 运行 `validate_data_flow.py` 确认数据能通过整个管道
- [ ] 检查 speech_tensor 中没有 NaN/Inf
- [ ] 验证 model forward 能处理 batch 数据
- [ ] (可选) 如果长度不一致，使用 collate_fn
- [ ] 小规模测试 (1 epoch, 小 batch) 验证训练能进行
- [ ] 检查 loss 是否递减

---

## ❓ 常见问题

**Q: 为什么音频特征没有被注入到模型？**
A: 最可能是 `_count_modal_proj` 没有找到占位符。运行 `verify_tokenization.py` 检查 `#`*100 的 token ID 是否为 35。

**Q: 训练时所有 sample 的 loss 都一样，没有下降？**
A: 可能模型没有收到有效信息。检查：
1. 占位符是否被正确替换
2. Speech tensor 是否包含有效数据（不全是 0）
3. 模型架构是否正确冻结

**Q: 内存不足，mel 频谱很大？**
A: 调整参数：
- 减小 batch_size
- 减小 max_seq_len
- 使用 float16 代替 float32

---

## 参考资源

- 模型定义: `model/model_olm.py`
- Dataset: `dataset/lm_dataset.py`
- 验证脚本: `dataset/verify_tokenization.py`
- 验证脚本: `dataset/validate_data_flow.py`
- Collate 函数: `dataset/olm_collate.py`
