"""
Deprecated module.

Pretrain dataset has been moved to `dataset/pretrain_dataset.py` and now expects
`pretrain_s2t.parquet` with schema:
  - speech_bytes (pa.binary)
  - transcript_bytes (pa.binary)

For new experiments, import `PretrainDataset` from `dataset.pretrain_dataset`.
"""

from dataset.pretrain_dataset import PretrainDataset as OLMDataset

__all__ = ["OLMDataset", "PretrainDataset"]
PretrainDataset = OLMDataset

