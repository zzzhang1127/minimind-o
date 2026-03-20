"""
验证占位符tokenization是否正确
"""
import sys
import os

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer

def verify_placeholder_ids():
    tokenizer_path = "../model/tokenizer"  # 根据实际路径修改
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"❌ 无法加载tokenizer: {e}")
        return False
    
    # 测试 speech 占位符
    speech_placeholder = '#' * 150
    speech_tokens = tokenizer(speech_placeholder, add_special_tokens=False).input_ids
    
    print(f"📊 Speech 占位符验证:")
    print(f"   输入: '#'*150")
    print(f"   Token 数量: {len(speech_tokens)}")
    print(f"   Token IDs: {speech_tokens}")
    print(f"   是否全为 5: {all(t == 5 for t in speech_tokens)}")
    print(f"   是否恰好 150 个: {len(speech_tokens) == 150}")
    
    if len(speech_tokens) != 150:
        print(f"\n⚠️  警告: 期望 150 个 tokens，实际得到 {len(speech_tokens)} 个")
        print(f"   需要修改 OLMConfig.speech_ids = [{speech_tokens[0]}] * {len(speech_tokens)}")
    
    if not all(t == 35 for t in speech_tokens):
        unique_ids = set(speech_tokens)
        print(f"\n⚠️  警告: 占位符中的 token IDs 不统一")
        print(f"   不同的 IDs: {unique_ids}")
        print(f"   需要更新 speech_ids 的逻辑")
    
    # 测试 image 占位符
    image_placeholder = '@' * 196
    image_tokens = tokenizer(image_placeholder, add_special_tokens=False).input_ids
    
    print(f"\n📊 Image 占位符验证:")
    print(f"   输入: '@'*196")
    print(f"   Token 数量: {len(image_tokens)}")
    print(f"   Token IDs 前 20 个: {image_tokens[:20]}")
    print(f"   是否全为 34: {all(t == 34 for t in image_tokens)}")
    print(f"   是否恰好 196 个: {len(image_tokens) == 196}")
    
    if len(image_tokens) != 196:
        print(f"\n⚠️  警告: 期望 196 个 tokens，实际得到 {len(image_tokens)} 个")
        print(f"   需要修改 OLMConfig.image_ids = [{image_tokens[0]}] * {len(image_tokens)}")
    
    success = (len(speech_tokens) == 150 and all(t == 5 for t in speech_tokens) and
               len(image_tokens) == 196 and all(t == 34 for t in image_tokens))
    
    return success

if __name__ == "__main__":
    success = verify_placeholder_ids()
    if success:
        print("\n✅ 占位符 tokenization 正确！")
        sys.exit(0)
    else:
        print("\n❌ 占位符 tokenization 存在问题，需要修复！")
        sys.exit(1)
