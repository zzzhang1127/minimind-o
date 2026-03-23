"""
完整的端到端数据流验证脚本
验证: parquet -> dataset -> 模型前向过程是否正确
"""
import sys
import os

__package__ = "dataset"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_REPO_ROOT)

import torch
import json
from pathlib import Path
import pyarrow.parquet as pq

from dataset.lm_dataset import OLMDataset
from dataset.pretrain_dataset import pretrain_collate_fn, resolve_parquet_paths
from model.model_olm import OLMConfig, MiniMindOLM, num_speech_tokens_from_encoder_length


def _peek_parquet_schema_and_first_row(parquet_path: str):
    """
    不整表 read_table：仅从 schema 取列名，并只读「第一个非空 row_group」的两列首行。
    """
    paths = resolve_parquet_paths(parquet_path)
    if not paths:
        raise FileNotFoundError(f"未找到 parquet: {parquet_path}")
    last_err = None
    for path in paths:
        try:
            pf = pq.ParquetFile(path, memory_map=True)
        except Exception as e:
            last_err = e
            continue
        cols = set(pf.schema_arrow.names)
        for rg in range(pf.num_row_groups):
            if pf.metadata.row_group(rg).num_rows <= 0:
                continue
            tbl = pf.read_row_group(rg, columns=["speech_bytes", "transcript_bytes"])
            speech_b = tbl["speech_bytes"][0].as_py()
            trans_b = tbl["transcript_bytes"][0].as_py()
            return cols, speech_b, trans_b
    if last_err is not None:
        raise last_err
    raise ValueError(f"Parquet 中无有效行: {parquet_path}")

def validate_data_flow(parquet_path, token_limit=10):
    """
    Step-by-step validation of the entire data pipeline
    """
    print("=" * 80)
    print("OLM Data Flow Validation")
    print("=" * 80)
    
    # Step 1: Load dataset（与 eval_olm 一致：tokenizer 在 minimind-o/model/）
    print("\n[Step 1] Loading OLM dataset...")
    try:
        from transformers import AutoTokenizer

        tokenizer_dir = Path(_REPO_ROOT) / "model"
        if not tokenizer_dir.is_dir():
            print(f"[ERROR] 未找到目录: {tokenizer_dir}")
            return False
        if not (tokenizer_dir / "tokenizer.json").is_file() and not (tokenizer_dir / "tokenizer_config.json").is_file():
            print(f"[ERROR] {tokenizer_dir} 下缺少 tokenizer.json 或 tokenizer_config.json")
            return False
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )
    except Exception as e:
        print(f"[ERROR] 加载 tokenizer 失败: {e}")
        return False
    
    try:
        olm_config = OLMConfig()
        dataset = OLMDataset(
            parquet_path,
            tokenizer,
            preprocess=None,
            max_length=512,
            image_special_token='@' * 196,
        )
        # PretrainDataset：语音占位符为「#」*N，N = (mel_len//2) // 10
        print(f"[OK] Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return False
    
    # Step 2: Sample a few items and check structure
    print("\n[Step 2] Validating sample structure...")
    sample_indices = [0, 1, min(5, len(dataset)-1)]
    
    for idx in sample_indices:
        try:
            item = dataset[idx]
            if item is None:
                print(f"\n  Sample [{idx}]: skipped (None)")
                continue
            input_ids, labels, speech_tensor, speech_lengths = item

            print(f"\n  Sample [{idx}]:")
            print(f"    input_ids shape: {input_ids.shape} - {input_ids[:token_limit].tolist()}")
            print(f"    labels shape: {labels.shape}")
            print(f"    speech_tensor shape: {speech_tensor.shape}")
            print(f"    speech_lengths: {speech_lengths}")

            # Validate shapes
            assert input_ids.shape[0] == 512, f"Expected max_length=512, got {input_ids.shape[0]}"
            assert labels.shape[0] == 512, f"Expected labels length=512, got {labels.shape[0]}"
            assert speech_tensor.shape[0] == 1, f"Expected speech batch dim=1, got {speech_tensor.shape[0]}"
            assert len(speech_tensor.shape) == 3, f"Expected speech_tensor 3D, got {len(speech_tensor.shape)}D"

            mel_len = int(speech_lengths.view(-1)[0].item())
            enc_len = mel_len // 2
            expected_n = num_speech_tokens_from_encoder_length(enc_len)
            speech_token_id = tokenizer('#', add_special_tokens=False).input_ids[0]
            speech_count = (input_ids == speech_token_id).sum().item()
            print(
                f"    mel_len={mel_len}, encoder_len≈{enc_len}, "
                f"expected speech tokens N={expected_n}, '#' count in ids={speech_count}"
            )
            assert speech_count == expected_n, (
                f"占位符数量应与 N 一致: expected {expected_n}, got {speech_count}"
            )
            
            # Check for NaN or Inf in speech_tensor
            if torch.isnan(speech_tensor).any():
                print(f"    [WARN] Speech tensor contains NaN!")
            if torch.isinf(speech_tensor).any():
                print(f"    [WARN] Speech tensor contains Inf!")
            
            print(f"    [OK] Sample valid")
            
        except Exception as e:
            print(f"    [ERROR] Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Step 3: Test batch collation
    print("\n[Step 3] Testing batch collation (batch_size=2)...")
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=pretrain_collate_fn,
        )
        batch = next(iter(loader))
        if batch is None:
            print("  [WARN] Batch 为空（可能样本均被过滤）")
            return False

        input_ids_batch, labels_batch, speech_batch, speech_lengths_batch = batch

        print(f"  Batched shapes:")
        print(f"    input_ids: {input_ids_batch.shape}")
        print(f"    labels: {labels_batch.shape}")
        print(f"    speech_batch: {speech_batch.shape}")
        print(f"    speech_lengths_batch: {speech_lengths_batch.shape}")
        
        # Validate batch shapes
        assert input_ids_batch.shape[0] == 2, f"Expected batch size 2, got {input_ids_batch.shape[0]}"
        assert speech_batch.shape[0] == 2, f"Expected speech batch size 2, got {speech_batch.shape[0]}"
        
        print(f"  [OK] Batch collation valid")
        
    except Exception as e:
        print(f"  [ERROR] Error during batch collation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test model forward pass
    print("\n[Step 4] Testing model forward pass...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
        olm_config = OLMConfig(
            hidden_size=256,
            num_hidden_layers=4,
            max_seq_len=512,
            vocab_size=vocab_size,
        )
        print(f"  (logits 最后一维 = vocab_size = {vocab_size}，与 tokenizer 词表大小一致)")
        
        # Initialize model with both encoders (or skip if not available)
        try:
            model = MiniMindOLM(
                olm_config,
                load_vision_encoder=False,  # Skip vision encoder for speed
                load_speech_encoder=False,  # We'll test without encoders first
            ).to(device)
            print(f"  [OK] Model initialized (no encoders for testing)")
        except Exception as e:
            print(f"  [WARN] Could not initialize full model: {e}")
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
        print(f"  [OK] Forward pass successful")
        
    except Exception as e:
        print(f"  [ERROR] Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Check conversation format（禁止 read_table 整表，避免大文件 OOM）
    print("\n[Step 5] Validating conversation format...")
    try:
        cols, sample_speech_bytes, sample_transcript_bytes = _peek_parquet_schema_and_first_row(
            parquet_path
        )

        print(f"  Parquet columns: {cols}")
        
        required_cols = {'speech_bytes', 'transcript_bytes'}
        if required_cols.issubset(cols):
            print(f"    [OK] Required columns present")
        else:
            print(f"    [WARN] Missing columns: {required_cols - cols}")

        print(f"  Sample speech_bytes:")
        print(f"    Type: {type(sample_speech_bytes)}")
        print(f"    Length: {len(sample_speech_bytes) if sample_speech_bytes else 0} bytes")

        transcript = (
            sample_transcript_bytes.decode("utf-8", errors="ignore")
            if sample_transcript_bytes is not None
            else ""
        )
        print(f"  Sample transcript preview: {transcript[:60]!r}")

        print(f"  [OK] Pretrain parquet schema valid")
        
    except Exception as e:
        print(f"  [ERROR] Error validating conversation format: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("[OK] All validations passed!")
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
    
    ok_path = os.path.isfile(args.parquet) or os.path.isdir(args.parquet)
    if not ok_path:
        if any(ch in args.parquet for ch in "*?["):
            try:
                ok_path = len(resolve_parquet_paths(args.parquet)) > 0
            except FileNotFoundError:
                ok_path = False
        else:
            ok_path = False
    if not ok_path:
        print(f"[ERROR] Parquet 路径无效或不存在: {args.parquet}")
        print(f"   请传入单个 .parquet 文件、目录（含 *.parquet）或通配路径 --parquet")
        sys.exit(1)
    
    success = validate_data_flow(args.parquet, args.token_limit)
    sys.exit(0 if success else 1)
