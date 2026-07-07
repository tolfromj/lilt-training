"""파인튜닝 LiLT head 추론 — input_lilt_bbox.json(tag 비움) → 박스별 태그 예측 + 시각화 (CPU).

입력 input_lilt_bbox.json 은 학습용과 양식이 동일하되 각 annotation 의 `tag`란만 비어 있다.
모델이 그 `tag`를 채워, 문서마다 output/{이미지파일명}/ 에
  · {이미지파일명}.png  — 원본 위에 예측 bbox(주황)·index(검정)·tag(파랑) 시각화
  · predicted.json      — 입력과 동일 스키마, 비어 있던 tag 만 예측값으로 채움
을 저장한다. bbox 0~1000 정규화·word_ids 정렬은 dataloader.py(학습)와 동일 공식.

실행:
  uv run python -m lilt_ocr.predict "data/val/유일산업"          # 한 문서 폴더
  uv run python -m lilt_ocr.predict data/val                     # 폴더 전체(재귀)
  uv run python -m lilt_ocr.predict data/val --model checkpoints/best --out-dir output
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .labels import id2label

DEFAULT_MODEL = "checkpoints/best"
INPUT_NAME = "input_lilt_bbox.json"   # 추론 입력 파일명(tag 비움). 학습은 label.json 사용.


def find_docs(data_dir) -> list[Path]:
    """data_dir 재귀 탐색 → INPUT_NAME 이 있는 폴더 목록."""
    return sorted(p.parent for p in Path(data_dir).rglob(INPUT_NAME))

# index(검정) / tag(파랑) / bbox(주황)
COLOR_BOX = "orange"
COLOR_INDEX = "black"
COLOR_TAG = "blue"


def load_model(model_dir: str):
    """model_dir(파인튜닝 산출물) → (tokenizer, model) on CPU, eval."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    return tokenizer, model


def predict_doc(doc_dir: Path, tokenizer, model, max_length: int = 512) -> dict:
    """한 문서 input_lilt_bbox.json → 예측 tag 를 채운 data dict 반환(predicted.json 내용).

    입력의 `tag`는 무시하고, 비어 있던 각 annotation 의 tag 를 예측값으로 채운다.
    """
    with open(doc_dir / INPUT_NAME, encoding="utf-8") as f:
        data = json.load(f)

    annots = data["annotations"]

    # input_lilt_bbox.json 의 bbox 는 ocr.py(to_lilt_bbox)가 이미 0~1000 으로 정규화한
    # LiLT 입력값이다(학습 dataloader 와 동일 공간). 재정규화 없이 clamp 만 해서 그대로 쓴다.
    # text 있는 annotation 만 모델 입력으로. ann_idx 로 원본 위치를 되짚는다.
    tokens, bboxes, ann_idx = [], [], []
    for i, b in enumerate(annots):
        text = (b.get("text") or "").strip()
        if not text:
            b["tag"] = "O"          # 빈 text 는 예측 대상 아님
            continue
        tokens.append(text)
        bboxes.append([min(1000, max(0, int(v))) for v in b["bbox"]])
        ann_idx.append(i)

    # 예측 없는 단어(truncation 등)는 기본 "O".
    for i in ann_idx:
        annots[i]["tag"] = "O"

    if not tokens:
        return data

    enc = tokenizer(
        tokens, is_split_into_words=True,
        truncation=True, max_length=max_length, return_tensors="pt",
    )
    word_ids = enc.word_ids(0)
    bbox = [[0, 0, 0, 0] if wid is None else bboxes[wid] for wid in word_ids]
    enc["bbox"] = torch.tensor([bbox])

    with torch.no_grad():
        logits = model(**enc).logits          # [1, seq, num_labels]
    preds = logits.argmax(-1)[0].tolist()

    # 각 단어는 첫 subtoken 예측만 채택 → 원본 annotation 의 tag 채움.
    seen = set()
    for tok_pos, wid in enumerate(word_ids):
        if wid is None or wid in seen or wid >= len(ann_idx):
            continue
        seen.add(wid)
        annots[ann_idx[wid]]["tag"] = id2label[preds[tok_pos]]

    return data


def _load_font(size: int):
    for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def resolve_image(image_path: str, image_root: str | None) -> Path | None:
    """image_path 를 실제 파일로 해석. 상대경로면 cwd → image_root 순으로 시도. 없으면 None."""
    p = Path(image_path)
    if p.exists():
        return p
    if not p.is_absolute() and image_root:
        cand = Path(image_root) / p
        if cand.exists():
            return cand
    return None


def visualize(image_file: Path, data: dict, out_path: Path) -> None:
    """원본 이미지 위에 예측 bbox(주황)·index(검정)·tag(파랑)를 그려 저장.

    bbox 는 0~1000 정규화값이므로 실제 이미지 크기로 역정규화(× W/1000, × H/1000)한다.
    """
    image = Image.open(image_file).convert("RGB")
    W, H = image.size
    draw = ImageDraw.Draw(image)
    font = _load_font(max(14, image.height // 90))

    for b in data["annotations"]:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        nx1, ny1, nx2, ny2 = b["bbox"]
        x1, y1 = nx1 * W / 1000, ny1 * H / 1000
        x2, y2 = nx2 * W / 1000, ny2 * H / 1000
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_BOX, width=3)

        label_y = max(0, y1 - font.size - 2)
        idx = str(b.get("index", ""))
        draw.text((x1, label_y), idx, fill=COLOR_INDEX, font=font)
        # index 다음 칸에 tag(파랑).
        idx_w = draw.textlength(idx + " ", font=font)
        draw.text((x1 + idx_w, label_y), b.get("tag", "O"), fill=COLOR_TAG, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    ap = argparse.ArgumentParser(description="LiLT KIE 추론 + 시각화 (CPU)")
    ap.add_argument("input", help="input_lilt_bbox.json 이 있는 문서 폴더 또는 그 상위 폴더(재귀)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="파인튜닝 모델 폴더")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out-dir", default="output", help="결과 저장 루트")
    ap.add_argument("--image-root", default=None,
                    help="image_path 가 상대경로일 때 붙일 루트(예: ../tax_invoice)")
    args = ap.parse_args()

    docs = find_docs(args.input)
    if not docs:
        raise FileNotFoundError(f"'{args.input}' 아래에 {INPUT_NAME} 이 없습니다.")

    tokenizer, model = load_model(args.model)
    out_root = Path(args.out_dir)

    for doc_dir in docs:
        data = predict_doc(doc_dir, tokenizer, model, args.max_length)
        stem = Path(data["image_path"]).stem
        doc_out = out_root / stem
        doc_out.mkdir(parents=True, exist_ok=True)

        with open(doc_out / "predicted.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        image_file = resolve_image(data["image_path"], args.image_root)
        if image_file is None:
            print(f"  [경고] 이미지를 찾지 못해 시각화 생략: {data['image_path']} "
                  f"(--image-root 로 지정 가능)")
        else:
            visualize(image_file, data, doc_out / f"{stem}.png")

        print(f"\n[{stem}] → {doc_out}")
        for b in data["annotations"]:
            text = (b.get("text") or "").strip()
            if text:
                print(f"  {b.get('tag', 'O'):18s} | {text}")


if __name__ == "__main__":
    main()
