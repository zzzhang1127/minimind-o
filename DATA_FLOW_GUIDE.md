# OLM 数据流完整性检查与修复指南

## 🔴 关键问题汇总

### Problem #1: 占位符 Tokenization 不确定 ⚠️ **CRITICAL**

**现象:**
- `OLMConfig.speech_ids = [35] * 100` 假设每个 `#` token ID 都是 35
- 但 tokenizer 处理 `'####...(100个#)'` 时可能不会生成 100 个ID
- 导致 `_count_modal_proj` 找不到需要替换的占位符

**影响:**
- 音频特征无法正确注入
- 模型收不到音频信息
- 训练失败或特征混乱

**快速检查:**
```bash
cd dataset
python verify_tokenization.py
```

**预期输出:**
```
✅ 占位符 tokenization 正确！
   Token 数量: 100
   Token IDs: [35, 35, 35, ..., 35]
   是否全为 35: True
   是否恰好 100 个: True
```

如果 ❌ 失败，则需要**修改 OLMConfig**：

修复步骤:
1. 运行验证脚本获得实际的 token 数量和 IDs
2. 更新 `OLMConfig.speech_ids = [real_id] * real_count`
3. 确保 `speech_special_token` 长度也要调整

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
- 记下实际的 token 数量和 IDs
- 修改 `model/model_olm.py` 中的 OLMConfig

```python
# 修前
class OLMConfig(MiniMindConfig):
    speech_ids: List = [35] * 100
    
# 修后 (假设实际是 [32] * 50)
class OLMConfig(MiniMindConfig):
    speech_ids: List = [32] * 50  # ← 根据验证结果更新
    speech_special_token: str = '#' * 50  # ← 也要匹配
```

### Step 2: 验证数据流
```bash
python dataset/validate_data_flow.py --parquet dataset/pretrain_s2t.parquet
```

**检查清单:**
- ✅ `Sample [0]: speech_tensor shape: (1, T, 128)` - 维度正确
- ✅ `Speech placeholder tokens in input_ids: 100` - 占位符被正确替换
- ✅ `Speech tensor contains NaN: False` - 没有无效值
- ✅ `Batch collation valid` - Batch 处理正常
- ✅ `Forward pass successful` - 模型能处理数据

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
  --weight sft_vlm \
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

1. 检查 `_count_modal_proj` 是否找到占位符:

```python
# 在 model_olm.py 中添加 debug 代码
def _count_modal_proj(self, tokens, h, modal_tensors, modal_ids, seqlen=512):
    indices = self._find_indices(tokens, modal_ids)
    print(f"[DEBUG] Looking for modal_ids={modal_ids}")
    print(f"[DEBUG] Found indices: {indices}")
    
    if modal_tensors is None or not indices:
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
