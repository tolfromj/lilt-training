"""세금계산서 KIE 플랫 라벨 26종 (tsr/plan.md 확정 스키마).

KEY 1 + 지정값 23 + VALUE 1 + O = 26. 접두사 없는 IO 태깅.
반드시 ocr_labeler/src/tags.ts 의 TAG_LABELS 와 개수·순서가 동일해야 한다
(라벨러가 붙인 tag 문자열을 그대로 id 로 매핑하기 때문).
"""

LABELS: list[str] = [
    "O",
    "KEY",
    "VALUE",
    "SUPPLIER_REGNO",
    "SUPPLIER_NAME",
    "SUPPLIER_CEO",
    "SUPPLIER_ADDR",
    "SUPPLIER_BIZTYPE",
    "SUPPLIER_BIZITEM",
    "BUYER_REGNO",
    "BUYER_NAME",
    "BUYER_CEO",
    "BUYER_ADDR",
    "BUYER_BIZTYPE",
    "BUYER_BIZITEM",
    "WRITE_DATE",
    "TOTAL_SUPPLYAMT",
    "TOTAL_TAXAMT",
    "GOODS_DATE",
    "GOODS_NAME",
    "GOODS_SPEC",
    "GOODS_QTY",
    "GOODS_UNITPRICE",
    "GOODS_SUPPLYAMT",
    "GOODS_TAXAMT",
    "TOTAL_AMOUNT",
]

label2id: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
id2label: dict[int, str] = {i: label for i, label in enumerate(LABELS)}

NUM_LABELS = len(LABELS)
