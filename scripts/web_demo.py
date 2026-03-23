import argparse
import os
import sys
import warnings
from typing import List, Tuple

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_olm import MiniMindOLM, OLMConfig, num_speech_tokens_from_encoder_length  # noqa: E402
from trainer.trainer_utils import get_model_params, setup_seed  # noqa: E402

warnings.filterwarnings('ignore')

MODEL = None
TOKENIZER = None
PREPROCESS = None
ARGS = None


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindOLM(
            OLMConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
            ),
            vision_model_path=args.vision_model_path,
            load_vision_encoder=True,
            load_speech_encoder=True,
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindOLM.get_vision_model(args.vision_model_path)
        model.speech_encoder = MiniMindOLM.get_speech_model(args.speech_encoder)

    get_model_params(model, model.params)
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def build_prompt(
    model,
    text_prompt: str,
    with_image: bool = False,
    with_speech: bool = False,
    n_speech_tokens: int = 0,
) -> str:
    prompt = (text_prompt or '').strip()
    if with_image and '<image>' not in prompt:
        prompt = (prompt + '\n<image>').strip()
    if with_speech and '<speech>' not in prompt:
        prompt = (prompt + '\n<speech>').strip()

    prompt = prompt.replace('<image>', model.params.image_special_token)
    if with_speech:
        if n_speech_tokens > 0:
            prompt = prompt.replace('<speech>', '#' * n_speech_tokens)
        else:
            prompt = prompt.replace('<speech>', '')
    return prompt


def build_history_messages(history: List[Tuple[str, str]]):
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({'role': 'user', 'content': user_msg})
        if assistant_msg:
            messages.append({'role': 'assistant', 'content': assistant_msg})
    return messages


def predict(user_text, image, audio_path, history, max_new_tokens, temperature, top_p):
    history = history or []

    has_image = image is not None
    has_audio = bool(audio_path)
    if not (user_text and user_text.strip()) and not has_image and not has_audio:
        return history, history, ''

    pixel_values = None
    if has_image:
        if PREPROCESS is None:
            raise gr.Error('未加载视觉编码器或处理器，无法处理图片。')
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        pixel_values = MiniMindOLM.image2tensor(image, PREPROCESS).to(ARGS.device).unsqueeze(0)

    speech_values = None
    speech_lengths = None
    n_speech_tokens = 0
    if has_audio:
        if MODEL.speech_encoder is None:
            raise gr.Error('未加载语音编码器，无法处理音频。')
        speech_tensor = MiniMindOLM.speech2tensor(audio_path)
        enc_len = speech_tensor.size(0) // 2
        n_speech_tokens = num_speech_tokens_from_encoder_length(enc_len)
        if n_speech_tokens > 0:
            speech_values = speech_tensor.unsqueeze(0).unsqueeze(0).to(ARGS.device)
            speech_lengths = torch.LongTensor([[speech_tensor.size(0)]]).to(ARGS.device)

    raw_text = (user_text or '').strip()
    prompt = build_prompt(
        MODEL,
        raw_text,
        with_image=has_image,
        with_speech=has_audio,
        n_speech_tokens=n_speech_tokens,
    )

    messages = build_history_messages(history)
    messages.append({'role': 'user', 'content': prompt})
    inputs_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(inputs_text, return_tensors='pt', truncation=True).to(ARGS.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            top_p=float(top_p),
            temperature=float(temperature),
            pad_token_id=TOKENIZER.pad_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            pixel_values=pixel_values,
            speech_values=speech_values,
            speech_lengths=speech_lengths,
        )

    new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
    response = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

    shown_user_text = raw_text if raw_text else '[仅多模态输入]'
    history.append((shown_user_text, response))
    return history, history, ''


def clear_all():
    return [], [], None, None, ''


def build_demo():
    with gr.Blocks(title='MiniMind-O Demo') as demo:
        gr.Markdown('## MiniMind-O 多模态对话\n支持文字 + 单张图片(上传/摄像头) + 单条语音(上传/麦克风)。')

        chatbot = gr.Chatbot(label='对话', height=520)
        state = gr.State([])

        with gr.Row():
            image = gr.Image(label='图片输入', type='pil', sources=['upload', 'webcam'])
            audio = gr.Audio(label='语音输入', type='filepath', sources=['upload', 'microphone'])

        user_text = gr.Textbox(label='文字输入', placeholder='输入问题；也可以仅传图/音频', lines=3)

        with gr.Row():
            max_new_tokens = gr.Slider(minimum=16, maximum=1024, value=256, step=1, label='max_new_tokens')
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.65, step=0.01, label='temperature')
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.85, step=0.01, label='top_p')

        with gr.Row():
            submit = gr.Button('发送', variant='primary')
            clear = gr.Button('清空')

        submit.click(
            predict,
            inputs=[user_text, image, audio, state, max_new_tokens, temperature, top_p],
            outputs=[chatbot, state, user_text],
        )
        user_text.submit(
            predict,
            inputs=[user_text, image, audio, state, max_new_tokens, temperature, top_p],
            outputs=[chatbot, state, user_text],
        )
        clear.click(clear_all, outputs=[chatbot, state, image, audio, user_text])

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description='MiniMind-O Gradio Demo')
    parser.add_argument('--load_from', default='model', type=str, help='model path')
    parser.add_argument('--save_dir', default='out', type=str, help='model weight dir')
    parser.add_argument('--weight', default='sft_olm', type=str, help='weight prefix')
    parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='num hidden layers')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='use moe')
    parser.add_argument('--speech_encoder', default='base', type=str, help='whisper model size')
    parser.add_argument('--vision_model_path', default='./model/vision_model/clip-vit-base-patch16', type=str, help='clip path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--server_name', default='0.0.0.0', type=str, help='gradio server name')
    parser.add_argument('--server_port', default=7861, type=int, help='gradio server port')
    parser.add_argument('--share', action='store_true', help='gradio share')
    return parser.parse_args()


def main():
    global MODEL, TOKENIZER, PREPROCESS, ARGS

    ARGS = parse_args()
    setup_seed(2026)
    MODEL, TOKENIZER, PREPROCESS = init_model(ARGS)

    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=ARGS.server_name,
        server_port=ARGS.server_port,
        share=ARGS.share,
    )


if __name__ == '__main__':
    main()
