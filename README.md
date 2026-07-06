# lilt-ocr

세금계산서 KIE(Key Information Extraction)를 위한 **LiLT(XLM-RoBERTa) head 파인튜닝**.
백본은 `nielsr/lilt-xlm-roberta-base`를 유지하고, 분류 head 만 세금계산서 전용 26 플랫 라벨
(`ocr_labeler/src/tags.ts`와 동기)로 새로 학습한다. 학습은 GPU, 추론은 CPU
(`tsr/tsl-lilt-xlmroberta.py` 재사용).

## 구조

```
src/lilt_ocr/
  labels.py     # 26 플랫 라벨 + id2label/label2id
  dataloader.py # label.json 폴더들 → HF Dataset (0~1000 정규화, 토큰-라벨 정렬)
  train.py      # 모델 구성 + Trainer + 체크포인트 + metric (진입점)
data/         # 라벨된 문서 폴더들 (gitignore) — train/ val/ 하위 각 폴더에 label.json (image_width/image_height 포함, 이미지 불필요)
checkpoints/  # 학습 산출물 (gitignore) — 런별 타임스탬프 폴더
```

## 데이터 준비

라벨링한 각 문서 폴더(`label.json`)를 `data/train/` 또는 `data/val/` 아래에 둔다.
`label.json`은 `{image_path, image_width, image_height, annotations}` 형태로, 크기가 들어 있어
이미지 파일은 필요 없다. `dataloader.py`가 `data/train`·`data/val`을 재귀 탐색해 split을 나눈다.

```
data/
  train/{회사명}/label.json
  val/{회사명}/label.json
```

## 설치 · 학습

```bash
uv sync

# 학습 (data/ 사용, checkpoints/<시작시각>/ 에 저장)
uv run python -m lilt_ocr.train --epochs 40 --batch-size 4

# 에러로 끊긴 런 이어서 학습 (마지막 체크포인트에서 재개)
uv run python -m lilt_ocr.train --resume checkpoints/20260706-143022
```

## 체크포인트 / 재개

- 매 실행마다 `checkpoints/<YYYYMMDD-HHMMSS>/` 폴더가 새로 생기고, epoch 마다
  `checkpoint-{step}` 이 저장된다(`save_total_limit=None` → **덮어쓰지 않고 전부 보존**).
- 학습이 중단되면 `--resume checkpoints/<그 런 폴더>` 로 마지막 체크포인트(옵티마이저 상태
  포함)에서 이어서 학습한다.
- 학습 종료 시 best 모델은 `checkpoints/<런>/best/` 에 저장된다.

## 추론 연동

`tsr/tsl-lilt-xlmroberta.py` 의 `MODEL_NAME` 을 `checkpoints/<런>/best` 로 바꾸면
같은 추론 코드가 새 26 라벨로 동작한다(`LABEL_COLORS` 는 26종에 맞게 확장 필요).
