import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e
    return rows


def _read_image_bytes(image_path: str) -> Optional[bytes]:
    if not image_path:
        return None
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return f.read()


def _ensure_speech_tag(text: str) -> str:
    if "<speech>" in text:
        return text
    return f"<speech>\n{text}".strip()


def _build_conversations(sample: Dict[str, Any], force_speech_tag: bool) -> List[Dict[str, str]]:
    if "conversations" in sample and isinstance(sample["conversations"], list):
        conv = sample["conversations"]
        normalized: List[Dict[str, str]] = []
        for turn in conv:
            content = str(turn.get("content", ""))
            normalized.append({"content": content})
        if force_speech_tag and normalized:
            normalized[0]["content"] = _ensure_speech_tag(normalized[0]["content"])
        return normalized

    question = str(sample.get("question", sample.get("prompt", ""))).strip()
    answer = str(sample.get("answer", sample.get("response", ""))).strip()
    if force_speech_tag:
        question = _ensure_speech_tag(question)

    return [
        {"content": question},
        {"content": answer},
    ]


def build_parquet(
    input_jsonl: str,
    output_parquet: str,
    speech_base_dir: str,
    image_base_dir: str,
    force_speech_tag: bool,
) -> None:
    rows = _read_jsonl(input_jsonl)

    conversations_col: List[str] = []
    image_bytes_col: List[Optional[List[bytes]]] = []
    speech_path_col: List[Optional[str]] = []

    missing_speech = 0
    missing_image = 0

    for sample in rows:
        conv = _build_conversations(sample, force_speech_tag)

        speech_path = sample.get("speech_path") or sample.get("audio_path")
        if speech_path:
            speech_path = os.path.join(speech_base_dir, speech_path) if not os.path.isabs(speech_path) else speech_path
            if not os.path.exists(speech_path):
                missing_speech += 1
                speech_path = None
        else:
            speech_path = None

        image_path = sample.get("image_path")
        image_list: Optional[List[bytes]] = None
        if image_path:
            image_path = os.path.join(image_base_dir, image_path) if not os.path.isabs(image_path) else image_path
            image_data = _read_image_bytes(image_path)
            if image_data is None:
                missing_image += 1
            else:
                image_list = [image_data]

        conversations_col.append(json.dumps(conv, ensure_ascii=False))
        image_bytes_col.append(image_list)
        speech_path_col.append(speech_path)

    table = pa.table(
        {
            "conversations": pa.array(conversations_col, type=pa.string()),
            "image_bytes": pa.array(image_bytes_col, type=pa.list_(pa.binary())),
            "speech_path": pa.array(speech_path_col, type=pa.string()),
        }
    )

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    pq.write_table(table, output_parquet)

    print(f"Done. Samples: {len(rows)}")
    print(f"Output: {output_parquet}")
    print(f"Missing speech files: {missing_speech}")
    print(f"Missing image files: {missing_image}")


def _clean_transcript_text(text: str) -> str:
    # Remove tags such as <CN>, <MIX>, <EN>, <SPK/>, <NON/> and collapse spaces.
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_script_map(script_txt: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with script_txt.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            utt_id, raw_text = line.split("\t", 1)
            utt_id = utt_id.strip()
            text = _clean_transcript_text(raw_text)
            if not utt_id or not text:
                continue
            mapping[utt_id] = text
    return mapping


def _iter_audio_folders(wave_root: Path) -> List[Path]:
    folders: List[Path] = []
    for first_level in sorted(wave_root.iterdir()):
        if not first_level.is_dir():
            continue
        for second_level in sorted(first_level.iterdir()):
            if second_level.is_dir():
                folders.append(second_level)
    return folders


def _resolve_default_output_parquet(input_jsonl: str, short_wav_root: str) -> str:
    if short_wav_root:
        base_dir = Path(short_wav_root).resolve().parent
        return str(base_dir / "pretrain_s2t.parquet")
    base_dir = Path(input_jsonl).resolve().parent if input_jsonl else Path.cwd()
    return str(base_dir / "pretrain_olm.parquet")


def build_parquet_from_short_wav(
    short_wav_root: str,
    output_parquet: str,
    prompt_text: str = "<speech>\nPlease transcribe the speech.",
    embed_speech_base64: bool = True,
    keep_speech_path: bool = False,
) -> None:
    root = Path(short_wav_root)
    script_root = root / "SCRIPT"
    wave_root = root / "WAVE"

    if not script_root.exists() or not wave_root.exists():
        raise ValueError(f"Invalid short_wav root: {short_wav_root}. Require SCRIPT/ and WAVE/.")

    conversations_col: List[str] = []
    image_bytes_col: List[Optional[List[bytes]]] = []
    speech_path_col: List[Optional[str]] = []
    speech_b64_col: List[Optional[str]] = []

    folders = _iter_audio_folders(wave_root)
    total_folders = len(folders)
    if total_folders == 0:
        raise ValueError(f"No audio folders found under: {wave_root}")

    missing_script = 0
    missing_text_line = 0
    missing_wav = 0
    total_pairs = 0
    total_audio_bytes = 0
    t0 = time.time()

    print(f"[Start] folders={total_folders}, script_root={script_root}, wave_root={wave_root}")
    for folder_idx, audio_folder in enumerate(folders, start=1):
        folder_name = audio_folder.name
        script_txt = script_root / f"{folder_name}.txt"
        if not script_txt.exists():
            missing_script += 1
            if folder_idx % 50 == 0 or folder_idx == total_folders:
                print(f"[Progress] folder={folder_idx}/{total_folders}, pairs={total_pairs}, missing_script={missing_script}")
            continue

        script_map = _load_script_map(script_txt)
        wav_files = sorted(audio_folder.glob("*.wav"))
        if not wav_files:
            missing_wav += 1
            if folder_idx % 50 == 0 or folder_idx == total_folders:
                print(f"[Progress] folder={folder_idx}/{total_folders}, pairs={total_pairs}, missing_wav_folder={missing_wav}")
            continue

        for wav_path in wav_files:
            utt_id = wav_path.stem
            text = script_map.get(utt_id)
            if not text:
                missing_text_line += 1
                continue

            conv = [
                {"content": prompt_text},
                {"content": text},
            ]
            conversations_col.append(json.dumps(conv, ensure_ascii=False))
            image_bytes_col.append(None)
            speech_path_col.append(str(wav_path.resolve()) if keep_speech_path else None)
            if embed_speech_base64:
                audio_bytes = wav_path.read_bytes()
                speech_b64_col.append(base64.b64encode(audio_bytes).decode("ascii"))
                total_audio_bytes += len(audio_bytes)
            else:
                speech_b64_col.append(None)
            total_pairs += 1

        if folder_idx % 50 == 0 or folder_idx == total_folders:
            elapsed = time.time() - t0
            print(
                f"[Progress] folder={folder_idx}/{total_folders}, pairs={total_pairs}, "
                f"missing_script={missing_script}, missing_line={missing_text_line}, elapsed={elapsed:.1f}s"
            )

    table = pa.table(
        {
            "conversations": pa.array(conversations_col, type=pa.string()),
            "image_bytes": pa.array(image_bytes_col, type=pa.list_(pa.binary())),
            "speech_path": pa.array(speech_path_col, type=pa.string()),
            "speech_b64": pa.array(speech_b64_col, type=pa.string()),
        }
    )

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)
    pq.write_table(table, output_parquet)

    print("\n[Done] Build completed.")
    print(f"Output: {output_parquet}")
    print(f"Total pairs: {total_pairs}")
    print(f"Missing SCRIPT files: {missing_script}")
    print(f"Missing transcript lines for wavs: {missing_text_line}")
    print(f"Audio folders without wav files: {missing_wav}")
    print(f"Embedded speech bytes: {total_audio_bytes / (1024 * 1024):.2f} MB")
    print(f"Embed speech base64: {embed_speech_base64}")
    print(f"Keep speech path: {keep_speech_path}")

    preview_count = min(5, total_pairs)
    print(f"\n[Preview] showing {preview_count} samples from generated parquet-like rows:")
    for i in range(preview_count):
        print(f"- sample[{i}] speech_path={speech_path_col[i]}")
        if speech_b64_col[i] is not None:
            print(f"  speech_b64_prefix={speech_b64_col[i][:48]}...")
        print(f"  conversations={conversations_col[i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniMind-O parquet from jsonl manifest")
    parser.add_argument("--input_jsonl", type=str, default="", help="Input manifest jsonl")
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
        help="Output parquet path. Default: pretrain_s2t.parquet(short_wav mode) or pretrain_olm.parquet(jsonl mode)",
    )
    parser.add_argument("--speech_base_dir", type=str, default="", help="Base directory for relative speech_path")
    parser.add_argument("--image_base_dir", type=str, default="", help="Base directory for relative image_path")
    parser.add_argument(
        "--force_speech_tag",
        action="store_true",
        help="Force prepend <speech> tag to the first user turn",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default="<speech>\nPlease transcribe the speech.",
        help="Prompt text used in short_wav mode",
    )
    parser.add_argument(
        "--embed_speech_base64",
        type=int,
        default=1,
        choices=[0, 1],
        help="Embed wav data into speech_b64 column in parquet (1=yes, 0=no)",
    )
    parser.add_argument(
        "--keep_speech_path",
        type=int,
        default=0,
        choices=[0, 1],
        help="Keep absolute speech_path in parquet for compatibility/debug (1=yes, 0=no)",
    )
    args = parser.parse_args()

    # Convenience fallback: if no input is provided, try local dataset/short_wav.
    if not args.input_jsonl and not args.short_wav_root:
        auto_short_wav = (Path(__file__).resolve().parent / "short_wav").resolve()
        if auto_short_wav.exists():
            args.short_wav_root = str(auto_short_wav)
            print(f"[Info] No input args provided, auto-detected short_wav_root={args.short_wav_root}")

    output_parquet = args.output_parquet or _resolve_default_output_parquet(
        input_jsonl=args.input_jsonl,
        short_wav_root=args.short_wav_root,
    )

    if args.short_wav_root:
        # Updated: short_wav pretrain parquet is now "speech_bytes"+"transcript_bytes" only.
        # Keep this script as a backward-compatible entry point.
        # dataset/ 目录不是 python package，这里同目录直接导入
        from build_pretrain_parquet import build_pretrain_s2t_parquet

        build_pretrain_s2t_parquet(
            short_wav_root=args.short_wav_root,
            output_parquet=output_parquet,
        )
        return

    if not args.input_jsonl:
        raise ValueError("Either --input_jsonl or --short_wav_root must be provided.")

    build_parquet(
        input_jsonl=args.input_jsonl,
        output_parquet=output_parquet,
        speech_base_dir=args.speech_base_dir,
        image_base_dir=args.image_base_dir,
        force_speech_tag=args.force_speech_tag,
    )


if __name__ == "__main__":
    main()
