# 🚀 数据流验证与训练快速启动

根据用户反馈的详细数据流说明，我已经完成了以下代码检查和生成。请按照此指南逐步执行。

## 📋 执行步骤

### Phase 1: 数据完整性验证 (5 分钟)

```bash
# 进入 dataset 目录
cd dataset

# Step 1: 自动检测并修复 tokenization 问题
python fix_tokenization.py

# 输出应该看起来像这样:
# ✅ ALL CONFIGURATIONS ARE CORRECT!
# 
# 如果出现 ❌ CONFIGURATION MISMATCH，按提示修改：
#    - model/model_olm.py 中的 OLMConfig
#    - dataset/lm_dataset.py 中的 OLMDataset.__init__

# Step 2: 验证完整的数据流
python validate_data_flow.py --parquet ./pretrain_s2t.parquet

# 输出应该看起来像:
# ✅ Dataset loaded: 6490 samples
# ✅ Sample valid (all 3 samples)
# ✅ Batch collation valid
# ✅ Forward pass successful
# ✅ Conversation format valid
# ✅ All validations passed!
```

### Phase 2: 数据流关键检查点

验证脚本会检查以下内容。如果看到 ❌ 错误，参考对应的修复方案：

| 检查项 | 预期结果 | 失败时的修复 |
|---|---|---|
| **Tokenization** | 100 个 `#` → 100 个 ID=35 的 token | 运行 `fix_tokenization.py` 后修改 OLMConfig |
| **Sample 加载** | speech_tensor shape (1, T, 128) | 检查 _speech_b64_to_tensor 返回值 |
| **占位符替换** | "Speech placeholder tokens: 100" | 检查 tokenizer 是否被正确初始化 |
| **Speech 特征** | 无 NaN/Inf 值 | 检查 base64 解码过程 |
| **Batch 拼接** | shape (batch_size, 1, T, 128) | 检查 DataLoader collate 逻辑 |
| **模型前向** | loss 值正常（如 5.2）| 检查 model_olm.py forward 实现 |

### Phase 3: 开始训练前的最后检查

运行以下脚本进行轻量级前向测试：

```bash
# 小规模测试 (仅测试第一个 batch，不训练)
python validate_data_flow.py --parquet ./pretrain_s2t.parquet

# 如果全部通过 ✅，则可以开始训练
```

### Phase 4: 启动训练

```bash
cd trainer

# 使用最小配置进行 1 epoch 测试
python train_pretrain_olm.py \
  --save_dir ../out \
  --save_weight pretrain_olm_test \
  --data_path ../dataset/pretrain_s2t.parquet \
  --weight llm_768 \
  --mode speech \
  --freeze_llm 1 \
  --batch_size 4 \
  --epochs 1 \
  --log_interval 50 \
  --save_interval 9999 \
  --learning_rate 3e-4 \
  --dtype float16 \
  --device cuda:0

# 监控输出：
# - loss 应该从 ~8 递减到 ~5
# - 每条日志应该显示 4 个样本被处理
# - 不应该看到 NaN/Inf 错误
```

## 🔍 关键词汇速查

### 占位符处理

| 组件 | 工作 | 预期行为 |
|---|---|---|
| **build_olm_parquet.py** | 生成 conversations | `[{"content": "<speech>\n..."}, {"content": "转录文本"}]` |
| **OLMDataset** | 替换 `<speech>` | `<speech>` → `####...(100个#)` |
| **Tokenizer** | 编码占位符 | `####...` → `[35, 35, ..., 35]` (100 个) |
| **_count_modal_proj** | 找到替换点 | 定位 100 个 ID=35 的位置，用音频特征替换 |

### 音频处理

| 组件 | 输入 | 处理 | 输出 |
|---|---|---|---|
| **speech_b64** | base64 WAV | 解码 | 字节流 |
| **wave.open** | 字节流 | 读取 PCM | 16-bit 整数 arrays |
| **whisper.pad_or_trim** | audio 数组 | 统一到 30 秒 | float32 [-1, 1] |
| **whisper.log_mel_spectrogram** | audio | 计算 mel 频谱 | (n_mels, time_steps) |
| **permute(1, 0)** | mel 频谱 | 交换维度 | (time_steps, 128) |
| **speech_proj** | mel 频谱 | 降采样 k=5 | (time_steps//5, hidden_size) |
| **_count_modal_proj** | 降采样特征 | 替换占位符 | 融合到 embedding |

## ⚠️ 常见错误与快速修复

### Error 1: `ValueError: Expected 16kHz wav, got 44100Hz`
```python
# 原因: WAV 文件采样率不对
# 修复: 在 build_olm_parquet.py 中添加重采样
# 或在 build_olm_parquet.py 的 prompt_text 中说明
# 实际上已经在代码中检查了，这意味着数据集中有错误的 WAV
```

### Error 2: `RuntimeError: speech_ids 不匹配`
```python
# 原因: tokenizer 生成的 ID 与配置不符
# 修复: 
#   1. 运行 python dataset/fix_tokenization.py
#   2. 根据输出更新 OLMConfig
#   3. 重新运行验证脚本
```

### Error 3: `Occupation: Found 0 indices for speech_ids`
```python
# 原因: _count_modal_proj 没有找到占位符
# 根本原因: tokenizer 没有生成预期的 ID 序列
# 修复: 检查占位符 tokenization (Error 2 的修复)
```

### Error 4: `OOM (Out of Memory)`
```python
# 快速修复:
batch_size = 2        # ← 从 4 改为 2
hidden_size = 256     # ← 从 512 改为 256
dtype = float16       # ← 改用半精度 (已设置)
num_workers = 0       # ← 关闭多进程（如有需要）
```

## 📊 预期的训练输出

正常的训练日志应该看起来像：

```
[Start] Epoch [1/1]: Initializing data loader...
[Phase] Model frozen parts:
  - vision_encoder: requires_grad=False
  - speech_encoder: requires_grad=False
  - vision_proj: requires_grad=False  (train_modality='speech')
  - speech_proj: requires_grad=True
  - backbone: requires_grad=False (freeze_llm=1)

Epoch:[1/1](1/163), loss: 7.3281, logits_loss: 7.1203, aux_loss: 0.2078, lr: 0.00030000, epoch_time: 24.5min
Epoch:[1/1](50/163), loss: 5.8234, logits_loss: 5.6101, aux_loss: 0.2133, lr: 0.00029814, epoch_time: 8.2min
Epoch:[1/1](100/163), loss: 5.2145, logits_loss: 5.0002, aux_loss: 0.2143, lr: 0.00029628, epoch_time: 5.1min
Epoch:[1/1](163/163), loss: 4.8923, logits_loss: 4.6780, aux_loss: 0.2143, lr: 0.00029442, epoch_time: 0.0min

✅ Training completed. Model saved.
```

关键指标：
- ✅ Loss 递减（7.3 → 4.9）
- ✅ 无 NaN/Inf
- ✅ batch 正常处理（50 steps = 200 samples）

## 🎯 最终检查清单

在开始训练前，确保已完成：

- [ ] 运行 `fix_tokenization.py` - 配置正确 ✅
- [ ] 运行 `validate_data_flow.py` - 所有验证通过 ✅
- [ ] 确认 `pretrain_s2t.parquet` 存在且有 6490+ 样本
- [ ] 确认 parquet 中有 `speech_b64` 列（非 None）
- [ ] 确认 parquet 中有 `conversations` 列，格式正确
- [ ] 小规模测试 (batch_size=2, epochs=1) 完成且 loss 递减

如果所有以上都检查完成，则可以开始完整训练！

---

## 📚 参考文件

生成的验证/修复文件：
- `dataset/verify_tokenization.py` - 快速检查占位符 tokenization
- `dataset/fix_tokenization.py` - 自动检测并生成修复代码
- `dataset/validate_data_flow.py` - 完整的数据流端到端验证
- `dataset/olm_collate.py` - 自定义 collate_fn（可选）
- `DATA_FLOW_GUIDE.md` - 详细的数据流文档

原有文件（已更新）：
- `model/model_olm.py` - 包含 speech_ids 和 image_ids 配置
- `dataset/lm_dataset.py` - 包含占位符替换和 speech_b64 解码逻辑
- `trainer/train_pretrain_olm.py` - 训练脚本
