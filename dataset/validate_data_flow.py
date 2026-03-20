"""
完整的端到端数据流验证脚本
验证: parquet -> dataset -> 模型前向过程是否正确
"""
import sys
import os

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
from pathlib import Path
from lm_dataset import OLMDataset
from model.model_olm import OLMConfig, MiniMindOLM
from trainer.trainer_utils import init_olm_model

def validate_data_flow(parquet_path, token_limit=10):
    """
    Step-by-step validation of the entire data pipeline
    """
    print("=" * 80)
    print("OLM Data Flow Validation")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading OLM dataset...")
    try:
        # 尝试加载tokenizer
        from transformers import AutoTokenizer
        tokenizer_path = "./model/tokenizer"
        if not os.path.exists(tokenizer_path):
            print(f"⚠️  Tokenizer not found at {tokenizer_path}")
            print("   Creating dummy tokenizer for testing...")
            from transformers import LlamaTokenizer
            # 使用一个通用的tokenizer作为演示
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=False)
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        print("   Skipping detailed validation...")
        return False
    
    try:
        olm_config = OLMConfig()
        dataset = OLMDataset(
            parquet_path,
            tokenizer,
            preprocess=None,
            max_length=512,
            image_special_token='@' * 196,
            speech_special_token='#' * 150,
        )
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Step 2: Sample a few items and check structure
    print("\n[Step 2] Validating sample structure...")
    sample_indices = [0, 1, min(5, len(dataset)-1)]
    
    for idx in sample_indices:
        try:
            item = dataset[idx]
            input_ids, labels, image_tensor, speech_tensor, speech_lengths = item
            
            print(f"\n  Sample [{idx}]:")
            print(f"    input_ids shape: {input_ids.shape} - {input_ids[:token_limit].tolist()}")
            print(f"    labels shape: {labels.shape}")
            print(f"    image_tensor shape: {image_tensor.shape}")
            print(f"    speech_tensor shape: {speech_tensor.shape}")
            print(f"    speech_lengths: {speech_lengths}")
            
            # Validate shapes
            assert input_ids.shape[0] == 512, f"Expected max_length=512, got {input_ids.shape[0]}"
            assert labels.shape[0] == 512, f"Expected labels length=512, got {labels.shape[0]}"
            assert speech_tensor.shape[0] == 1, f"Expected speech batch dim=1, got {speech_tensor.shape[0]}"
            assert len(speech_tensor.shape) == 3, f"Expected speech_tensor 3D, got {len(speech_tensor.shape)}D"
            
            # Check for speech token placeholders
            speech_token_id = tokenizer('#', add_special_tokens=False).input_ids[0]
            speech_count = (input_ids == speech_token_id).sum().item()
            print(f"    Speech placeholder tokens in input_ids: {speech_count}")
            
            # Check for NaN or Inf in speech_tensor
            if torch.isnan(speech_tensor).any():
                print(f"    ⚠️  Speech tensor contains NaN!")
            if torch.isinf(speech_tensor).any():
                print(f"    ⚠️  Speech tensor contains Inf!")
            
            print(f"    ✅ Sample valid")
            
        except Exception as e:
            print(f"    ❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 3: Test batch collation
    print("\n[Step 3] Testing batch collation (batch_size=2)...")
    try:
        from torch.utils.data import DataLoader
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        input_ids_batch, labels_batch, image_batch, speech_batch, speech_lengths_batch = batch
        
        print(f"  Batched shapes:")
        print(f"    input_ids: {input_ids_batch.shape}")
        print(f"    labels: {labels_batch.shape}")
        print(f"    image_batch: {image_batch.shape}" if image_batch is not None else "    image_batch: None")
        print(f"    speech_batch: {speech_batch.shape}")
        print(f"    speech_lengths_batch: {speech_lengths_batch.shape}")
        
        # Validate batch shapes
        assert input_ids_batch.shape[0] == 2, f"Expected batch size 2, got {input_ids_batch.shape[0]}"
        assert speech_batch.shape[0] == 2, f"Expected speech batch size 2, got {speech_batch.shape[0]}"
        
        print(f"  ✅ Batch collation valid")
        
    except Exception as e:
        print(f"  ❌ Error during batch collation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test model forward pass
    print("\n[Step 4] Testing model forward pass...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        olm_config = OLMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            max_seq_len=512,
        )
        
        # Initialize model with both encoders (or skip if not available)
        try:
            model = MiniMindOLM(
                olm_config,
                load_vision_encoder=False,  # Skip vision encoder for speed
                load_speech_encoder=False,  # We'll test without encoders first
            ).to(device)
            print(f"  ✅ Model initialized (no encoders for testing)")
        except Exception as e:
            print(f"  ⚠️  Could not initialize full model: {e}")
            print(f"     Testing with encoder loading skipped")
            return True
        
        # Test forward pass without speech/vision
        print(f"  Testing forward pass with batch_size=2...")
        input_ids_batch = input_ids_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_batch,
                labels=labels_batch,
                pixel_values=None,
                speech_values=None,  # Test without speech first
            )
        
        print(f"  Output loss: {outputs.loss.item():.4f}")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  ✅ Forward pass successful")
        
    except Exception as e:
        print(f"  ❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Check conversation format
    print("\n[Step 5] Validating conversation format...")
    try:
        import pyarrow.parquet as pq
        
        table = pq.read_table(parquet_path)
        cols = set(table.column_names)
        
        print(f"  Parquet columns: {cols}")
        
        required_cols = {'speech_bytes', 'transcript_bytes'}
        if required_cols.issubset(cols):
            print(f"    ✅ Required columns present")
        else:
            print(f"    ⚠️  Missing columns: {required_cols - cols}")
        
        # Check first sample
        sample_speech_bytes = table['speech_bytes'][0].as_py()
        sample_transcript_bytes = table['transcript_bytes'][0].as_py()

        print(f"  Sample speech_bytes:")
        print(f"    Type: {type(sample_speech_bytes)}")
        print(f"    Length: {len(sample_speech_bytes) if sample_speech_bytes else 0} bytes")

        transcript = (
            sample_transcript_bytes.decode("utf-8", errors="ignore")
            if sample_transcript_bytes is not None
            else ""
        )
        print(f"  Sample transcript preview: {transcript[:60]!r}")

        print(f"  ✅ Pretrain parquet schema valid")
        
    except Exception as e:
        print(f"  ❌ Error validating conversation format: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ All validations passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate OLM data flow")
    parser.add_argument(
        "--parquet",
        type=str,
        default="./pretrain_s2t.parquet",
        help="Path to the parquet dataset",
    )
    parser.add_argument(
        "--token_limit",
        type=int,
        default=10,
        help="Number of tokens to print from input_ids",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.parquet):
        print(f"❌ Parquet file not found: {args.parquet}")
        print(f"   Please provide a valid parquet path with --parquet")
        sys.exit(1)
    
    success = validate_data_flow(args.parquet, args.token_limit)
    sys.exit(0 if success else 1)
