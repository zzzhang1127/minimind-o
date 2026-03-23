import os
import torch
import warnings
import torch.nn.functional as F
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

warnings.filterwarnings('ignore')

_WHISPER_FEATURE_EXTRACTOR = None


def _get_whisper_feature_extractor():
    """
    Cache WhisperFeatureExtractor to avoid repeated disk IO.
    """
    global _WHISPER_FEATURE_EXTRACTOR
    if _WHISPER_FEATURE_EXTRACTOR is None:
        from transformers import WhisperFeatureExtractor
        _WHISPER_FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(
            os.path.join(os.path.dirname(__file__), 'speech_model', 'whisper-base')
        )
    return _WHISPER_FEATURE_EXTRACTOR


def num_speech_tokens_from_encoder_length(encoder_len: int, frames_per_token: int = 10) -> int:
    """
    有效 Whisper encoder 帧数 P（满 30s 约 1500）。
    语音 token 数 N = P // frames_per_token（纯向下取整）；P < frames_per_token 时 N=0。
    等价于 floor((s/30) * (1500/10))（s 为有效时长对应秒数时与 mel 对齐）。
    """
    if encoder_len <= 0:
        return 0
    return encoder_len // frames_per_token


class OLMConfig(MiniMindConfig):
    model_type = "minimind-o"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            # 占位符在数据侧按音频长度动态生成「#」*N；模型侧用连续 speech_token_id 定位整段。
            speech_special_token: str = "",
            speech_ids: List = [5] * 10,  # 遗留字段；语音注入已改为连续 speech_token_id 可变长匹配
            speech_token_id: int = 5,
            # 满 30s → 1500 帧 encoder；每 10 帧池化为 1 个语音 token → 最多 150 个。
            speech_frames_per_token: int = 10,
            # Use local whisper-base by default (see get_speech_model).
            speech_encoder: str = 'whisper-base',
            speech_encoder_hidden_size: int = 512,
            **kwargs,
    ):
        # 旧版配置字段兼容
        kwargs.pop("speech_encoder_keep_frames", None)
        kwargs.pop("speech_encoder_ds_rate", None)
        kwargs.pop("speech_encoder_output_tokens", None)
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.speech_special_token = speech_special_token
        self.speech_ids = speech_ids
        self.speech_token_id = speech_token_id
        self.speech_frames_per_token = speech_frames_per_token
        self.speech_encoder = speech_encoder
        self.speech_encoder_hidden_size = speech_encoder_hidden_size
        super().__init__(**kwargs)


class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(ve_hidden_size, hidden_size)
        )

    def forward(self, image_encoders):
        return self.vision_proj(image_encoders)


class SpeechProj(nn.Module):
    def __init__(self, speech_hidden_size=512, hidden_size=512, frames_per_token: int = 10):
        super().__init__()
        self.frames_per_token = frames_per_token
        self.linear = nn.Linear(speech_hidden_size, hidden_size, bias=True)

    def forward(self, x, lengths: Optional[torch.LongTensor] = None):
        """
        x: [B, T(=1500), 512] Whisper encoder 输出。
        lengths: 有效 encoder 帧数 P（mel 时间维 // 2）。N = P // K（K=frames_per_token），余帧丢弃。
        返回 List[Tensor]，长度 B，每项 [N_i, D]（N_i 可为 0 当 P < K）。
        """
        B, T, D = x.shape
        fpt = self.frames_per_token
        outs: List[torch.Tensor] = []

        if lengths is None:
            lens = torch.full((B,), T, device=x.device, dtype=torch.long)
        else:
            lens = lengths.to(x.device).clamp(min=0, max=T)

        for i in range(B):
            p = int(lens[i].item())
            n = num_speech_tokens_from_encoder_length(p, fpt)
            if n == 0:
                outs.append(x.new_zeros(0, self.linear.out_features))
                continue
            seg = x[i, :p, :]
            used = n * fpt
            seg = seg[:used]
            chunks = seg.view(n, fpt, D).mean(dim=1)
            outs.append(self.linear(chunks))
        return outs


class MiniMindOLM(MiniMindForCausalLM):
    config_class = OLMConfig

    def __init__(
            self,
            params: OLMConfig = None,
            vision_model_path="./model/vision_model/clip-vit-base-patch16",
            load_vision_encoder: bool = True,
            load_speech_encoder: bool = True,
    ):
        super().__init__(params)
        if not params:
            params = OLMConfig()
        self.params = params
        if load_vision_encoder:
            self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        else:
            self.vision_encoder, self.processor = None, None
        self.speech_encoder = self.__class__.get_speech_model(params.speech_encoder) if load_speech_encoder else None
        self.vision_proj = VisionProj(hidden_size=params.hidden_size)
        self.speech_proj = SpeechProj(
            speech_hidden_size=params.speech_encoder_hidden_size,
            hidden_size=params.hidden_size,
            frames_per_token=params.speech_frames_per_token,
        )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        candidate_paths = [
            model_path,
            "../model/vision_model/clip-vit-base-patch16",
        ]
        target_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if target_path is None:
            return None, None

        model = CLIPModel.from_pretrained(target_path)
        processor = CLIPImageProcessor.from_pretrained(target_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def get_speech_model(model_name: str):
        """
        Load whisper encoder from local `model/speech_model/whisper-base` (or from a provided HF id/path).
        We rely on HuggingFace's WhisperModel to match the expected output shape:
        input_features: [B, 80, 3000] -> encoder_hidden: [B, 1500, 512]
        """
        from transformers import WhisperModel

        candidate_paths = [
            model_name,
            # Local default path (minimind-o/model/speech_model/whisper-base)
            os.path.join(os.path.dirname(__file__), 'speech_model', 'whisper-base'),
            # Fallback relative paths (when cwd == trainer/)
            "../model/speech_model/whisper-base",
            "../minimind-o/model/speech_model/whisper-base",
        ]
        target_path = next((p for p in candidate_paths if isinstance(p, str) and os.path.exists(p)), None)
        if target_path is None:
            # Allow HF model id like "openai/whisper-base"
            target_path = model_name

        wm = WhisperModel.from_pretrained(target_path)
        encoder = getattr(wm, "encoder", None)
        if encoder is None and hasattr(wm, "model"):
            encoder = getattr(wm.model, "encoder", None)
        if encoder is None:
            raise RuntimeError(f"Failed to resolve whisper encoder from: {target_path}")

        for param in encoder.parameters():
            param.requires_grad = False
        return encoder.eval()

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def speech2tensor(audio_path):
        """
        Convert a wav file into whisper input_features tensor.
        Output is shaped as [T, 80] so that model forward can permute it to [B, 80, T].
        """
        import io
        import wave
        import numpy as np

        # Decode wav bytes -> float32 mono
        with open(audio_path, "rb") as f:
            wav_bytes = f.read()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM wav is supported, got sample_width={sample_width}")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        # Whisper expects 16kHz.
        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, duration, target_len, endpoint=False),
                np.linspace(0, duration, len(audio), endpoint=False),
                audio,
            )

        # WhisperFeatureExtractor will generate [80, 3000]
        extractor = _get_whisper_feature_extractor()
        features = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features  # [1, 80, 3000]
        # Return [T, 80]
        return features[0].transpose(0, 1).contiguous()

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        return outputs.last_hidden_state[:, 1:, :]

    @staticmethod
    def get_speech_embeddings(speech_tensors, speech_model):
        with torch.no_grad():
            # [B, T, M] -> [B, M, T]
            outputs = speech_model(speech_tensors.permute(0, 2, 1))
            return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs

    @staticmethod
    def _find_indices(tokens, target_ids):
        target = torch.tensor(target_ids).to(tokens.device)
        token_len = len(target_ids)
        if token_len > tokens.size(1):
            return None
        tokens_view = tokens.unfold(1, token_len, 1)
        matches = (tokens_view == target).all(dim=2)
        return {
            batch_idx: [(idx.item(), idx.item() + token_len - 1) for idx in matches[batch_idx].nonzero(as_tuple=True)[0]]
            for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
        } or None

    @staticmethod
    def _find_consecutive_token_spans(tokens: torch.Tensor, token_id: int):
        """连续相同 token_id 的区间 [start, end]（含端点），用于可变长语音占位。"""
        B, L = tokens.shape
        out = {}
        for b in range(B):
            row = tokens[b]
            spans = []
            i = 0
            while i < L:
                if row[i].item() == token_id:
                    j = i + 1
                    while j < L and row[j].item() == token_id:
                        j += 1
                    spans.append((i, j - 1))
                    i = j
                else:
                    i += 1
            if spans:
                out[b] = spans
        return out if out else None

    def _inject_speech_tokens(
        self,
        h: torch.Tensor,
        tokens: torch.Tensor,
        speech_proj_grouped: List[List[torch.Tensor]],
        seqlen: int,
        token_id: int,
    ):
        spans = self._find_consecutive_token_spans(tokens, token_id)
        if not speech_proj_grouped or not spans:
            return h
        new_h = []
        for i in range(h.size(0)):
            if i not in spans:
                new_h.append(h[i])
                continue
            h_i = h[i]
            modal_idx = 0
            row = speech_proj_grouped[i]
            for start_idx, end_idx in spans[i]:
                if modal_idx >= len(row):
                    break
                modal_hidden = row[modal_idx]
                span_len = end_idx - start_idx + 1
                if modal_hidden.size(0) != span_len:
                    raise ValueError(
                        f"speech token 数与占位符不一致: modal_hidden.size(0)={modal_hidden.size(0)} "
                        f"vs span_len={span_len}。请检查数据侧 '#' 数量与 encoder 长度公式。"
                    )
                h_i = torch.cat((h_i[:start_idx], modal_hidden, h_i[end_idx + 1:]), dim=0)[:seqlen]
                modal_idx += 1
            new_h.append(h_i)
        return torch.stack(new_h, dim=0)

    @staticmethod
    def num_speech_tokens_from_encoder_length(encoder_len: int, frames_per_token: int = 10) -> int:
        return num_speech_tokens_from_encoder_length(encoder_len, frames_per_token)

    def _count_modal_proj(self, tokens, h, modal_tensors, modal_ids, seqlen=512):
        indices = self._find_indices(tokens, modal_ids)
        if modal_tensors is None or not indices:
            return h

        new_h = []
        for i in range(h.size(0)):
            if i not in indices:
                new_h.append(h[i])
                continue

            h_i = h[i]
            modal_idx = 0
            for start_idx, end_idx in indices[i]:
                if modal_idx >= modal_tensors.size(1):
                    break
                modal_hidden = modal_tensors[i][modal_idx]
                span_len = end_idx - start_idx + 1
                if modal_hidden.size(0) != span_len:
                    # This should never happen if dataset/token placeholder construction is correct.
                    # Interpolating silently would hide data bugs and produce wrong supervision.
                    raise ValueError(
                        f"modal token num mismatch: modal_hidden.size(0)={modal_hidden.size(0)} "
                        f"vs placeholder span_len={span_len}. "
                        f"Check placeholder tokenization/count and modal encoder/projection output length."
                    )
                h_i = torch.cat((h_i[:start_idx], modal_hidden, h_i[end_idx + 1:]), dim=0)[:seqlen]
                modal_idx += 1
            new_h.append(h_i)

        return torch.stack(new_h, dim=0)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            labels: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            speech_values: Optional[torch.FloatTensor] = None,
            speech_lengths: Optional[torch.LongTensor] = None,
            **args
    ):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0 and self.vision_encoder is not None:
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            bs, num, c, im_h, im_w = pixel_values.shape
            vision_tensors = torch.stack([
                MiniMindOLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=1)
            vision_proj = self.vision_proj(vision_tensors)
            hidden_states = self._count_modal_proj(
                tokens=input_ids,
                h=hidden_states,
                modal_tensors=vision_proj,
                modal_ids=self.params.image_ids,
                seqlen=input_ids.shape[1],
            )

        if speech_values is not None and start_pos == 0 and self.speech_encoder is not None:
            if len(speech_values.shape) == 3:
                speech_values = speech_values.unsqueeze(1)
            bs, num, t, m = speech_values.shape
            # If provided, use speech_lengths to mask out padded time steps.
            # Whisper encoder doesn't take an attention mask here, so we zero out invalid frames.
            if speech_lengths is not None:
                # normalize shape to [bs, num]
                if speech_lengths.dim() == 1:
                    speech_lengths = speech_lengths.unsqueeze(1)
                if speech_lengths.dim() == 2 and speech_lengths.size(1) == 1 and num > 1:
                    speech_lengths = speech_lengths.expand(bs, num)
                if speech_lengths.dim() != 2 or speech_lengths.size(0) != bs:
                    raise ValueError(
                        f"Invalid speech_lengths shape: expected [bs] or [bs, num], got {tuple(speech_lengths.shape)} "
                        f"(bs={bs}, num={num})"
                    )

                speech_lengths = speech_lengths.to(speech_values.device).clamp(0, t)
                time_ids = torch.arange(t, device=speech_values.device).view(1, 1, t)  # [1,1,t]
                valid_mask = time_ids < speech_lengths.unsqueeze(-1)  # [bs,num,t] boolean
                speech_values = speech_values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

            speech_tensors = torch.stack([
                MiniMindOLM.get_speech_embeddings(speech_values[:, i, :, :], self.speech_encoder)
                for i in range(num)
            ], dim=1)

            encoder_lengths = None
            if speech_lengths is not None:
                # Whisper encoder downsamples length by 2: 3000 -> 1500.
                encoder_lengths = (speech_lengths // 2).clamp(min=0, max=speech_tensors.size(2))
                if encoder_lengths.dim() == 1:
                    encoder_lengths = encoder_lengths.unsqueeze(1)
                encoder_lengths = encoder_lengths.reshape(bs * num)

            speech_proj_list = self.speech_proj(
                speech_tensors.view(bs * num, speech_tensors.size(2), speech_tensors.size(3)),
                lengths=encoder_lengths,
            )
            grouped: List[List[torch.Tensor]] = []
            for b in range(bs):
                grouped.append([speech_proj_list[b * num + j] for j in range(num)])
            hidden_states = self._inject_speech_tokens(
                hidden_states,
                input_ids,
                grouped,
                seqlen=input_ids.shape[1],
                token_id=self.params.speech_token_id,
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum(
            [l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )
