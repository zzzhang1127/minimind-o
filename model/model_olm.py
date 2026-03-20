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


class OLMConfig(MiniMindConfig):
    model_type = "minimind-o"

    def __init__(
            self,
            image_special_token: str = '@' * 196,
            image_ids: List = [34] * 196,
            speech_special_token: str = '#' * 150,
            speech_ids: List = [5] * 150,
            # Use local whisper-base by default (see get_speech_model).
            speech_encoder: str = 'whisper-base',
            # whisper encoder output length is 1500; compress to 150 by ds_rate=10.
            speech_encoder_ds_rate: int = 10,
            speech_encoder_hidden_size: int = 512,
            **kwargs,
    ):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.speech_special_token = speech_special_token
        self.speech_ids = speech_ids
        self.speech_encoder = speech_encoder
        self.speech_encoder_ds_rate = speech_encoder_ds_rate
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
    def __init__(self, speech_hidden_size=512, hidden_size=512, ds_rate=5):
        super().__init__()
        self.k = ds_rate
        self.linear1 = nn.Linear(speech_hidden_size * ds_rate, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, hidden_size)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        x = x.contiguous().view(batch_size, x.size(1) // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


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
            ds_rate=params.speech_encoder_ds_rate,
        )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

        candidate_paths = [
            model_path,
            "../model/vision_model/clip-vit-base-patch16",
            "../minimind-v/model/vision_model/clip-vit-base-patch16",
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
    def speech2tensor(audio_path, n_mels=128):
        """
        Convert a wav file into whisper input_features tensor.
        Output is shaped as [T, 80] so that model forward can permute it to [B, 80, T].
        """
        from transformers import WhisperFeatureExtractor
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
        extractor = WhisperFeatureExtractor.from_pretrained(
            os.path.join(os.path.dirname(__file__), 'speech_model', 'whisper-base')
        )
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
                    modal_hidden = F.interpolate(
                        modal_hidden.transpose(0, 1).unsqueeze(0),
                        size=span_len,
                        mode='linear',
                        align_corners=False,
                    ).squeeze(0).transpose(0, 1)
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
            speech_tensors = torch.stack([
                MiniMindOLM.get_speech_embeddings(speech_values[:, i, :, :], self.speech_encoder)
                for i in range(num)
            ], dim=1)
            speech_proj = self.speech_proj(speech_tensors.view(bs * num, speech_tensors.size(2), speech_tensors.size(3)))
            speech_proj = speech_proj.view(bs, num, speech_proj.size(1), speech_proj.size(2))
            hidden_states = self._count_modal_proj(
                tokens=input_ids,
                h=hidden_states,
                modal_tensors=speech_proj,
                modal_ids=self.params.speech_ids,
                seqlen=input_ids.shape[1],
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
