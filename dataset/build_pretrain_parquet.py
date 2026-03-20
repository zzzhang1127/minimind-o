import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def _clean_transcript_text(text: str) -> str:
    """
    Remove tags like <CN>/<EN>/<MIX>/... and collapse spaces.
    Example: "<CN> 嗯，我来自福建。" -> "嗯，我来自福建。"
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_script_map(script_txt: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with script_txt.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                utt_id, raw_text = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                utt_id, raw_text = parts

            utt_id = utt_id.strip()
            text = _clean_transcript_text(raw_text)
            if utt_id and text:
                mapping[utt_id] = text
    return mapping


def _iter_audio_folders(wave_root: Path) -> List[Path]:
    """
    Expected structure:
      WAVE/<first_level>/<utt_group>/*.wav
    In provided data, <first_level> is typically "C0" and <utt_group> is like "ZH-CN_U0001_S0".
    """
    folders: List[Path] = []
    for first_level in sorted(wave_root.iterdir()):
        if not first_level.is_dir():
            continue
        for second_level in sorted(first_level.iterdir()):
            if second_level.is_dir():
                folders.append(second_level)
    return folders


def build_pretrain_s2t_parquet(
    short_wav_root: str,
    output_parquet: str,
    chunk_size: int = 256,
) -> None:
    root = Path(short_wav_root)
    script_root = root / "SCRIPT"
    wave_root = root / "WAVE"

    if not script_root.exists() or not wave_root.exists():
        raise ValueError(f"Invalid short_wav root: {short_wav_root}. Need SCRIPT/ and WAVE/.")

    folders = _iter_audio_folders(wave_root)
    if len(folders) == 0:
        raise ValueError(f"No audio folders found under: {wave_root}")

    schema = pa.schema(
        [
            ("speech_bytes", pa.binary()),
            ("transcript_bytes", pa.binary()),
        ]
    )

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    writer = pq.ParquetWriter(str(output_path), schema=schema, compression="snappy")

    speech_buf: List[bytes] = []
    text_buf: List[bytes] = []
    total_pairs = 0
    missing_script = 0
    missing_text_line = 0

    t0 = time.time()
    try:
        for folder_idx, audio_folder in enumerate(folders, start=1):
            folder_name = audio_folder.name
            script_txt = script_root / f"{folder_name}.txt"
            if not script_txt.exists():
                missing_script += 1
                continue

            script_map = _load_script_map(script_txt)
            wav_files = sorted(audio_folder.glob("*.wav"))
            if not wav_files:
                continue

            for wav_path in wav_files:
                utt_id = wav_path.stem
                text = script_map.get(utt_id)
                if not text:
                    missing_text_line += 1
                    continue

                speech_buf.append(wav_path.read_bytes())
                text_buf.append(text.encode("utf-8"))
                total_pairs += 1

                if len(speech_buf) >= chunk_size:
                    table = pa.table(
                        {
                            "speech_bytes": pa.array(speech_buf, type=pa.binary()),
                            "transcript_bytes": pa.array(text_buf, type=pa.binary()),
                        }
                    )
                    writer.write_table(table)
                    speech_buf.clear()
                    text_buf.clear()

            if folder_idx % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"[Progress] folders={folder_idx}/{len(folders)} pairs={total_pairs} "
                    f"missing_script={missing_script} missing_text_line={missing_text_line} "
                    f"elapsed={elapsed/60:.1f}min"
                )

        if len(speech_buf) > 0:
            table = pa.table(
                {
                    "speech_bytes": pa.array(speech_buf, type=pa.binary()),
                    "transcript_bytes": pa.array(text_buf, type=pa.binary()),
                }
            )
            writer.write_table(table)
    finally:
        writer.close()

    print("\n[Done]")
    print(f"Output: {output_parquet}")
    print(f"Total pairs: {total_pairs}")
    print(f"Missing SCRIPT files: {missing_script}")
    print(f"Missing transcript lines for wavs: {missing_text_line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniMind-O pretrain_s2t.parquet from short_wav dataset")
    parser.add_argument(
        "--short_wav_root",
        type=str,
        default="",
        help="Root directory containing SCRIPT/ and WAVE/ (e.g. dataset/short_wav)",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        default="",
        help="Output parquet path. Default: dataset/pretrain_s2t.parquet",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Number of samples per row-group write",
    )
    args = parser.parse_args()

    dataset_dir = Path(__file__).resolve().parent
    if not args.short_wav_root:
        args.short_wav_root = str((dataset_dir / "short_wav").resolve())
    if not args.output_parquet:
        args.output_parquet = str((dataset_dir / "pretrain_s2t.parquet").resolve())

    build_pretrain_s2t_parquet(
        short_wav_root=args.short_wav_root,
        output_parquet=args.output_parquet,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

