"""라벨된 문서 폴더(labeld.json + 이미지) → HF Dataset.

라벨러(ocr_labeler)가 저장하는 labeld.json 은 OcrBox 배열이며, 각 박스는
{index, bbox:[x1,y1,x2,y2] (픽셀), text, conf, cell?, tag?} 형태다.
bbox 가 픽셀 좌표라 LiLT 입력용 0~1000 정규화를 위해 같은 폴더의 이미지에서
W,H 를 읽는다. 정규화·word_ids 정렬은 추론 코드(tsl-lilt-xlmroberta.py)와 동일 공식.
"""

import json
from pathlib import Path

from datasets import Dataset
from PIL import Image

from .labels import label2id

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def find_docs(data_dir: str) -> list[Path]:
    """data_dir 재귀 탐색 → labeld.json 이 있는 폴더 목록."""
    root = Path(data_dir)
    return sorted(p.parent for p in root.rglob("labeld.json"))


def _find_image(doc_dir: Path) -> Path:
    """폴더 내 cleaned.png 우선, 없으면 첫 이미지 파일."""
    cleaned = doc_dir / "cleaned.png"
    if cleaned.exists():
        return cleaned
    for p in sorted(doc_dir.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            return p
    raise FileNotFoundError(f"이미지 없음(bbox 정규화 불가): {doc_dir}")


def load_doc(doc_dir: Path) -> dict:
    """한 문서 → {tokens, bboxes(0~1000), ner_tags(id)}."""
    with open(doc_dir / "labeld.json", encoding="utf-8") as f:
        boxes = json.load(f)

    W, H = Image.open(_find_image(doc_dir)).size

    tokens, bboxes, ner_tags = [], [], []
    for b in boxes:
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


def build_dataset(data_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """모든 문서 로드 → train/val 로 분할된 DatasetDict."""
    docs = find_docs(data_dir)
    if not docs:
        raise FileNotFoundError(
            f"'{data_dir}' 아래에 labeld.json 이 하나도 없습니다. "
            f"ocr_labeler 로 라벨링한 문서 폴더를 data/ 에 두세요."
        )

    records = [load_doc(d) for d in docs]
    records = [r for r in records if r["tokens"]]  # 빈 문서 제외
    ds = Dataset.from_list(records)

    if len(ds) < 2:
        # 분할 불가 — 같은 데이터를 train/val 로 사용(스모크 테스트용).
        return {"train": ds, "validation": ds}
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    return {"train": split["train"], "validation": split["test"]}


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
