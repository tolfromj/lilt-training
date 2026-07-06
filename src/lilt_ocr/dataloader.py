"""라벨된 문서 폴더(label.json) → HF Dataset (dataloader).

각 문서의 label.json 은 {image_path, image_width, image_height, annotations} 객체이며,
annotations 는 OcrBox 배열({index, bbox:[x1,y1,x2,y2] (픽셀), text, conf, cell?, tag?}) 이다.
bbox 가 픽셀 좌표라 LiLT 입력용 0~1000 정규화를 위해 JSON 의 image_width/image_height 를
쓴다(이미지 파일 불필요). 정규화·word_ids 정렬은 추론 코드(tsl-lilt-xlmroberta.py)와 동일 공식.
"""

import json
from pathlib import Path

from datasets import Dataset

from .labels import label2id


def find_docs(data_dir) -> list[Path]:
    """data_dir 재귀 탐색 → label.json 이 있는 폴더 목록."""
    root = Path(data_dir)
    return sorted(p.parent for p in root.rglob("label.json"))


def load_doc(doc_dir: Path) -> dict:
    """한 문서 → {tokens, bboxes(0~1000), ner_tags(id)}."""
    with open(doc_dir / "label.json", encoding="utf-8") as f:
        data = json.load(f)

    W, H = data["image_width"], data["image_height"]

    tokens, bboxes, ner_tags = [], [], []
    for b in data["annotations"]:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        x1, y1, x2, y2 = b["bbox"]
        nb = [
            int(1000 * x1 / W), int(1000 * y1 / H),
            int(1000 * x2 / W), int(1000 * y2 / H),
        ]
        tokens.append(text)
        bboxes.append([min(1000, max(0, v)) for v in nb])
        tag = b.get("tag") or "O"
        if tag not in label2id:
            raise ValueError(f"알 수 없는 라벨 '{tag}' ({doc_dir}). labels.py 와 tags.ts 동기 확인.")
        ner_tags.append(label2id[tag])

    return {"tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags}


def _load_split(split_dir: Path) -> Dataset:
    """한 split 폴더(train/ or val/) 아래 모든 문서 → HF Dataset."""
    records = [load_doc(d) for d in find_docs(split_dir)]
    records = [r for r in records if r["tokens"]]  # 빈 문서 제외
    return Dataset.from_list(records)


def build_dataloader(data_dir: str = "data"):
    """폴더 기준 split: data/train → train, data/val → validation 인 DatasetDict."""
    root = Path(data_dir)
    train_ds = _load_split(root / "train")
    val_ds = _load_split(root / "val")

    if len(train_ds) == 0:
        raise FileNotFoundError(
            f"'{root / 'train'}' 아래에 label.json 이 하나도 없습니다. "
            f"라벨링한 문서 폴더를 data/train/ 에 두세요."
        )
    if len(val_ds) == 0:
        # val 없음 — train 을 val 로 재사용(스모크 테스트용).
        val_ds = train_ds

    return {"train": train_ds, "validation": val_ds}


def tokenize_and_align(examples: dict, tokenizer, max_length: int = 512) -> dict:
    """plan 2단계 토큰-라벨 정렬. padding='max_length' 로 고정 길이 → 기본 collator 사용 가능.

    특수토큰·단어 비첫조각은 label=-100(손실 무시), bbox=[0,0,0,0].
    """
    enc = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    all_bbox, all_labels = [], []
    for i in range(len(examples["tokens"])):
        word_ids = enc.word_ids(i)
        boxes = examples["bboxes"][i]
        tags = examples["ner_tags"][i]
        bbox_row, label_row, prev = [], [], None
        for wid in word_ids:
            if wid is None:                 # 특수토큰/패딩
                bbox_row.append([0, 0, 0, 0])
                label_row.append(-100)
            elif wid != prev:               # 단어 첫 조각만 라벨
                bbox_row.append(boxes[wid])
                label_row.append(tags[wid])
            else:                           # 이어지는 조각 → 손실 무시
                bbox_row.append(boxes[wid])
                label_row.append(-100)
            prev = wid
        all_bbox.append(bbox_row)
        all_labels.append(label_row)

    enc["bbox"] = all_bbox
    enc["labels"] = all_labels
    return enc
