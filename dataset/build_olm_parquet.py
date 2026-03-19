import argparse
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniMind-O parquet from jsonl manifest")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Input manifest jsonl")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--speech_base_dir", type=str, default="", help="Base directory for relative speech_path")
    parser.add_argument("--image_base_dir", type=str, default="", help="Base directory for relative image_path")
    parser.add_argument(
        "--force_speech_tag",
        action="store_true",
        help="Force prepend <speech> tag to the first user turn",
    )
    args = parser.parse_args()

    build_parquet(
        input_jsonl=args.input_jsonl,
        output_parquet=args.output_parquet,
        speech_base_dir=args.speech_base_dir,
        image_base_dir=args.image_base_dir,
        force_speech_tag=args.force_speech_tag,
    )


if __name__ == "__main__":
    main()
