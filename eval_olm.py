import os
import time
import argparse
import warnings
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_olm import MiniMindOLM, OLMConfig
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


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
            vision_model_path="./model/vision_model/clip-vit-base-patch16",
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindOLM.get_vision_model("./model/vision_model/clip-vit-base-patch16")
        model.speech_encoder = MiniMindOLM.get_speech_model('base')

    get_model_params(model, model.params)
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def build_prompt(model, text_prompt, with_image=False, with_speech=False):
    prompt = text_prompt
    if with_image and '<image>' not in prompt:
        prompt = prompt + "\n<image>"
    if with_speech and '<speech>' not in prompt:
        prompt = prompt + "\n<speech>"

    prompt = prompt.replace('<image>', model.params.image_special_token)
    prompt = prompt.replace('<speech>', model.params.speech_special_token)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="MiniMind-O Eval")
    parser.add_argument('--load_from', default='model', type=str, help="model path")
    parser.add_argument('--save_dir', default='out', type=str, help="model weight dir")
    parser.add_argument('--weight', default='sft_olm', type=str, help="weight prefix")
    parser.add_argument('--hidden_size', default=512, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="num hidden layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="use moe")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="max new tokens")
    parser.add_argument('--temperature', default=0.65, type=float, help="temperature")
    parser.add_argument('--top_p', default=0.85, type=float, help="top p")
    parser.add_argument('--prompt', default='请先识别语音内容，再结合输入进行回答。', type=str, help="user prompt")
    parser.add_argument('--image_path', default='', type=str, help="optional image path")
    parser.add_argument('--audio_path', default='', type=str, help="optional audio path")
    parser.add_argument('--show_speed', default=1, type=int, help="show speed")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="device")
    args = parser.parse_args()

    setup_seed(2026)
    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pixel_values = None
    if args.image_path and os.path.exists(args.image_path):
        image = Image.open(args.image_path).convert('RGB')
        pixel_values = MiniMindOLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)

    speech_values = None
    speech_lengths = None
    if args.audio_path and os.path.exists(args.audio_path):
        speech_tensor = MiniMindOLM.speech2tensor(args.audio_path)
        speech_values = speech_tensor.unsqueeze(0).unsqueeze(0).to(args.device)
        speech_lengths = torch.LongTensor([[speech_tensor.size(0)]]).to(args.device)

    prompt = build_prompt(model, args.prompt, with_image=pixel_values is not None, with_speech=speech_values is not None)
    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

    print(f'Prompt: {args.prompt}')
    print('Answer: ', end='')

    st = time.time()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        pixel_values=pixel_values,
        speech_values=speech_values,
        speech_lengths=speech_lengths,
    )

    if args.show_speed:
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')


if __name__ == "__main__":
    main()
