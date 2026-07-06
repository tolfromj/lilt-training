"""LiLT(XLM-RoBERTa) head 파인튜닝 진입점.

백본(nielsr/lilt-xlm-roberta-base)은 유지하고 분류 head 만 26 플랫 라벨로 새로 학습.
train on GPU, infer on CPU (추론은 tsr/tsl-lilt-xlmroberta.py 재사용).

실행:
  python -m lilt_ocr.train                          # data/ 로 학습
  python -m lilt_ocr.train --epochs 40 --batch-size 4
  python -m lilt_ocr.train --resume checkpoints/20260706-143022   # 중단된 런 이어서

체크포인트: checkpoints/<시작시각>/checkpoint-{step} 으로 epoch 마다 저장(덮어쓰지 않음).
에러로 끊기면 --resume 에 그 런 폴더를 주면 마지막 체크포인트에서 이어서 학습.
"""

import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

from .dataloader import build_dataloader, tokenize_and_align
from .labels import NUM_LABELS, id2label, label2id

BASE = "nielsr/lilt-xlm-roberta-base"


def compute_metrics(eval_pred):
    """플랫(셀 단위) 분류 → sklearn classification_report. -100 제외."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    y_true = [id2label[i] for i in labels[mask].tolist()]
    y_pred = [id2label[i] for i in preds[mask].tolist()]

    print("\n" + classification_report(y_true, y_pred, zero_division=0))
    return {
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser(description="LiLT head 파인튜닝 (세금계산서 KIE)")
    ap.add_argument("--data-dir", default="data", help="train/ val/ 하위에 label.json 문서 폴더들의 루트")
    ap.add_argument("--output-dir", default="checkpoints", help="체크포인트 루트")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                    help="백본 동결 후 분류 head 만 학습 (소량 데이터 권장). --no-freeze-backbone 로 풀 파인튜닝")
    ap.add_argument("--resume", default=None,
                    help="이어서 학습할 런 폴더(예: checkpoints/20260706-143022)")
    args = ap.parse_args()

    # 재개면 그 런 폴더를, 아니면 새 타임스탬프 런 폴더를 output_dir 로.
    if args.resume:
        run_dir = Path(args.resume)
        ckpt = get_last_checkpoint(str(run_dir))
        if ckpt is None:
            raise FileNotFoundError(f"재개할 체크포인트를 찾지 못함: {run_dir}")
        print(f"[resume] {ckpt} 에서 이어서 학습")
    else:
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.output_dir) / run_ts
        ckpt = None
        print(f"[run] 체크포인트 저장 위치: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id,
    )

    # 소량 데이터(≈15문서) 과적합 방지: 백본 동결, 분류 head 만 학습.
    if args.freeze_backbone:
        for p in model.base_model.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[freeze] 백본 동결 — 학습 파라미터 {trainable:,}/{total:,}")

    ds = build_dataloader(args.data_dir)
    tok_fn = partial(tokenize_and_align, tokenizer=tokenizer, max_length=args.max_length)
    ds = {
        split: d.map(tok_fn, batched=True, remove_columns=d.column_names)
        for split, d in ds.items()
    }
    print(f"[data] train={len(ds['train'])} val={len(ds['validation'])}")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,               # 중간 체크포인트 전부 보존(덮어쓰기 없음)
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=ckpt)

    # 추론에서 바로 쓸 수 있게 best 모델 + 토크나이저 저장.
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n[done] 최종 모델 저장: {best_dir}")
    print(f"추론: tsl-lilt-xlmroberta.py 의 MODEL_NAME 을 '{best_dir}' 로 교체")


if __name__ == "__main__":
    main()
