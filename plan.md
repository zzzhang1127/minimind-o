## Plan: MiniMind-o 全模态落地计划

**User Hard Constraints (must follow)**
1. 代码风格、命名风格、目录组织需模仿 minimind-v。
1. 模型结构重点参考 minimind-v 的 model/model_vlm.py。
1. 数据目录必须为 dataset/，数据处理文件为 dataset/lm_dataset.py（参考 minimind-v 同名文件写法）。
1. 训练数据格式保持 Parquet 统一。
1. 训练最终结果保存到 out/。
1. 临时检查点保存到 checkpoint/（参数、优化器状态、断点续训信息）。
1. 训练脚本必须放到 trainer/。
1. 在 minimind-o 下新增 eval_olm.py 用于效果测试。
1. 允许按 minimind-v 代码写法直接复用改造。

目标是在现有 MiniMind2-V 权重基础上，最小侵入地加入语音输入理解能力（ASR+语音问答），并保持与 minimind-v 一致的项目结构、环境依赖、训练风格、配置方式与前端交互体验。  
已按你确认的边界收敛：中文优先、单机1-2张24G显卡、本期不做TTS语音输出。

**Steps**
1. Phase 0 方案冻结与目录基线（先决步骤）
1. 以 minimind-v 作为模板建立 minimind-o 同构工程骨架，确保训练入口、数据目录、脚本命名、README组织、Web Demo运行方式一致。
1. 明确首期只增加语音输入链路，不改动现有图像输入路径，避免影响既有VLM可用性。
1. 产物：最小可运行骨架、依赖清单、阶段性里程碑与回滚点。

1. Phase 1 模型结构扩展（依赖 Phase 0）
1. 在 VLM 模型中新增 speech encoder 与 speech projector 模块，采用 OpenOmni 的轻量模式（Whisper encoder + 下采样投影），并保持与现有视觉投影风格一致。
1. 新增语音特殊标记并接入 tokenizer，输入模板扩展为可并存文本、图像、语音三类占位。
1. 初始冻结策略：先冻结 LLM 与视觉编码器，只训练 speech projector 与少量上层参数；显存不足时维持该策略直到收敛。
1. 产物：具备 speech 前向路径的 MiniMind-o 模型定义与权重加载逻辑。

1. Phase 2 数据集选型与构建（可与 Phase 1 并行推进，最终合流）
1. 训练集（预训练对齐）建议采用开源且中小规模语音转写数据，优先中文并混入少量英文：
1. AISHELL-1（中文，约178h）
1. ST-CMDS（中文，约100h）
1. Primewords（中文，约100h）
1. LibriSpeech train-clean-100（英文，约100h）
1. 总体原始规模约 478h，按时长切片与多模板扩增后，构建 30万到60万条 speech-text 训练样本。
1. 微调集（SFT）采用“公开语料转指令”的方式构建 8万到20万条：
1. 基于上面语料的转写、纠错、摘要、关键词提取、问答模板化指令
1. 混入现有图文SFT数据防遗忘，推荐语音样本:图文样本比例为 6:4 到 7:3
1. 数据格式统一到 Parquet，字段保持与 minimind-v 对齐并扩展音频字段，避免改动训练主循环。
1. 增加数据治理步骤：许可证核验、时长分布、坏样本率、语言占比、重复样本率。

1. Phase 3 训练脚本与配置落地（依赖 Phase 1+2）
1. 复用现有两阶段流程，新增语音版脚本但保持参数风格一致：
1. 语音预训练对齐阶段：加载 MiniMind2-V 基础权重，主要训练 speech projector 与连接层
1. SFT阶段：在上阶段权重基础上进行语音指令微调，混合少量图文SFT样本保能力
1. 单机1-2张24G建议配方：
1. 混合精度 bf16/fp16 + 梯度累积 + gradient checkpointing
1. 预训练建议 2到3 epoch，SFT建议 1到2 epoch
1. 先 projector-only，再逐步解冻 LLM 最后若干层
1. 断点续训与恢复机制沿用现有 checkpoint 语义，确保训练中断可恢复。
1. 产物：可复现实验配置、可续训 checkpoint、阶段权重。

1. Phase 4 推理与前端（依赖 Phase 3）
1. 推理脚本新增音频路径参数，支持文本+图像+语音任意组合输入。
1. Web Demo 在现有图片上传基础上新增音频上传组件（文件上传优先，保留后续麦克风扩展位），调用链保持与现有 gradio 事件绑定风格一致。
1. 保证不破坏现有图像聊天路径，默认行为与 minimind-v 保持一致。
1. 产物：可上传音频并得到文本回答的前端演示。

1. Phase 5 评测、回归与发布（依赖 Phase 4）
1. 语音侧评测：CER/WER（中文优先），并做语音问答人工打分。
1. 图像侧回归：沿用 minimind-v 的图像测试集与样例，确认无明显退化。
1. 端到端验收：文本-only、图像-only、语音-only、图像+语音四种输入模式全部通过。
1. README与示例命令补齐，包含数据准备、训练、推理、Web Demo、故障排查与资源建议。
1. 产物：可交付的 minimind-o 首版（含训练脚本、权重说明、演示入口、评测记录）。

**推荐数据配比与训练节奏**
1. Stage A（语音对齐暖启动）：语音样本 100%，先训 speech projector，时长约 20万到30万样本。
1. Stage B（联合对齐）：语音:图文 = 7:3，继续 10万到20万样本，防止视觉遗忘。
1. Stage C（SFT）：语音指令:图文指令 = 6:4，规模 8万到20万对话。
1. 每阶段都做 500 step 冒烟，再放大全量训练。

**关键复用锚点（实现参考）**
- 模型与权重加载：  
[trainer_utils 初始化入口](minimind-v/trainer/trainer_utils.py#L66)  
[trainer_utils checkpoint 保存](minimind-v/trainer/trainer_utils.py#L96)  
[MiniMind2-V 权重目录](minimind-v/MiniMind2-V)
- 训练入口与参数风格：  
[预训练脚本参数](minimind-v/trainer/train_pretrain_vlm.py#L84)  
[预训练数据路径参数](minimind-v/trainer/train_pretrain_vlm.py#L98)  
[SFT脚本参数](minimind-v/trainer/train_sft_vlm.py#L84)  
[SFT数据路径参数](minimind-v/trainer/train_sft_vlm.py#L98)
- 数据管线：  
[VLMDataset 定义](minimind-v/dataset/lm_dataset.py#L16)  
[对话模板构造](minimind-v/dataset/lm_dataset.py#L32)  
[Parquet 读取字段位点](minimind-v/dataset/lm_dataset.py#L62)
- 前端上传与推理链路：  
[图像上传组件位点](minimind-v/scripts/web_demo_vlm.py#L124)  
[上传事件回调位点](minimind-v/scripts/web_demo_vlm.py#L125)  
[聊天函数入口](minimind-v/scripts/web_demo_vlm.py#L142)  
[生成调用位点](minimind-v/scripts/web_demo_vlm.py#L76)
- 语音模块参考：  
[Whisper 包装器](OpenOmni/openomni/model/speech_encoder/speech_encoder.py#L9)  
[语音投影器](OpenOmni/openomni/model/speech_projector/speech_projector.py#L8)  
[语音模块初始化](OpenOmni/openomni/model/llava_her_arch.py#L115)  
[语音编码流程](OpenOmni/openomni/model/llava_her_arch.py#L245)  
[OpenOmni 语音数据组织参考](OpenOmni/README.md#L95)

**Verification**
1. 数据验证：检查采样率统一（16k）、时长截断策略、坏音频率<1%、重复率阈值、字段完整性。
1. 冒烟训练：每阶段先跑 100到500 step，确认 loss 下降且无 NaN/梯度爆炸。
1. 全量训练：完成 Stage A/B/C 后导出权重并记录关键指标曲线。
1. 回归测试：文本与图像任务对比基线，确认能力不明显回退。
1. Demo 验收：上传音频后可稳定返回文本回答，图像上传功能保持可用。

**Decisions**
- 包含范围：语音输入理解、语音数据构建、两阶段训练、前端上传音频、文档与评测。
- 不包含范围：语音输出合成（TTS）、实时流式语音对话、复杂多轮音频会话记忆。
- 风险控制：优先最小改造与分阶段冻结，先稳定功能再追求更高性能。

**Implementation Checklist (execution order)**
1. 目录搭建：model/、dataset/、trainer/、out/、checkpoint/。
1. 模型搭建：model/model_olm.py（视觉+语音+文本统一前向）。
1. 数据搭建：dataset/lm_dataset.py（Parquet读取、<image>/<speech>占位替换、标签构造）。
1. 训练工具：trainer/trainer_utils.py（模型加载、参数统计、checkpoint保存与恢复）。
1. 训练脚本：trainer/train_pretrain_olm.py、trainer/train_sft_olm.py。
1. 评测脚本：eval_olm.py（文本/图像/语音单独或组合输入测试）。
1. 后续增强：Web Demo 增加语音上传与训练数据构建脚本（在当前骨架稳定后进行）。

该计划已写入会话计划文件，后续你可以直接让执行代理按此分阶段实施。
