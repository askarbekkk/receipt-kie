# Receipt information extraction with safe auto-accept

Extract key fields from receipts (item names, prices, quantities, totals) and decide **which receipts can be accepted automatically** without a human check.

- **Data:** [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) — 1,000 receipts (800 train / 100 validation / 100 test), 27 field types.
- **Models:** a text-only baseline (XLM-RoBERTa) vs. **LayoutLMv3** (text + word positions + image).
- **Evaluation:** entity-level F1 with bootstrap 95% confidence intervals, per-field analysis, and receipt-level correctness (every field right).
- **Auto-accept:** a confidence threshold chosen on validation and applied unchanged to test.

## Results (test set, 100 receipts)

| Model | Entity F1 | 95% CI | Fully correct receipts |
|---|---|---|---|
| XLM-RoBERTa (text only) | 0.933 | 0.907–0.954 | 66% |
| LayoutLMv3 (text + layout + image) | **0.952** | 0.929–0.970 | **76%** |

- **Layout helps, and it is not noise:** paired bootstrap gain +0.019 F1 (95% CI +0.006 to +0.034). LayoutLMv3 fixes 13 receipts the text-only model gets wrong and breaks 3. The biggest gains are on fields defined by their position in a row: unit price (0.85 → 0.94), service charge, sub-items.
- **Auto-accept — confidence alone is not enough:**

| Target error rate (chosen on validation) | Auto-accepted (test) | Errors among accepted (test) |
|---|---|---|
| 0% | 50% | 2 / 50 (4%) |
| ≤ 2% | 65% | 5 / 65 (7.7%) |
| ≤ 5% | 74% | 6 / 74 (8.1%) |

Thresholds tuned on 100 validation receipts do not hold on the test set: the real error rate is higher than the target. Model confidence is over-optimistic on some receipts, so a production system needs an independent check (e.g. arithmetic consistency: item prices + tax = total) rather than a confidence threshold alone.

## Why receipt-level metrics

Entity F1 can look excellent while most documents still contain at least one wrong field. For an operations team the relevant question is "how many documents can skip human review, and how often will an accepted one be wrong?" — so the notebook reports both.

## Run

Open `receipt_kie.ipynb` in Google Colab with a T4 GPU and run it top to bottom. Training is in `kie.py` (about 1 hour for both models on a T4); it saves validation/test predictions to `results.pkl`, which the notebook loads for the analysis.

```bash
pip install -U transformers datasets seqeval accelerate
python kie.py              # trains XLM-R and LayoutLMv3, writes results.pkl
python kie.py layoutlmv3   # or train one model only
```

## Files

| File | Purpose |
|---|---|
| `kie.py` | data parsing, training of both models, saving predictions with per-word confidence |
| `receipt_kie.ipynb` | evaluation: F1 with bootstrap CIs, paired comparison, per-field errors, auto-accept thresholds |

## Next

An independent consistency check — item prices + tax + service charge = total — to catch confident mistakes that a confidence threshold lets through.
