"""Train text-only (XLM-R) and LayoutLMv3 KIE models on CORD-v2; save val/test predictions to results.pkl."""
import json, math, pickle, random, sys
import numpy as np, torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForTokenClassification, AutoProcessor,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
which = sys.argv[1:] or ["xlmr", "layoutlmv3"]


def parse(ex):
    gt = json.loads(ex["ground_truth"])
    W, H = ex["image"].size
    words, boxes, tags = [], [], []
    for line in gt["valid_line"]:
        cat = line["category"]
        ws = [w for w in line["words"] if w["text"].strip()]
        for i, w in enumerate(ws):
            q = w["quad"]
            x0, x1 = min(q["x1"], q["x4"]), max(q["x2"], q["x3"])
            y0, y1 = min(q["y1"], q["y2"]), max(q["y3"], q["y4"])
            box = [int(1000 * x0 / W), int(1000 * y0 / H), int(1000 * x1 / W), int(1000 * y1 / H)]
            boxes.append([min(max(v, 0), 1000) for v in box])
            words.append(w["text"])
            tags.append(("B-" if i == 0 else "I-") + cat)
    return {"words": words, "boxes": boxes, "tags": tags}


raw = load_dataset("naver-clova-ix/cord-v2")
data = {s: [parse(ex) for ex in raw[s]] for s in ["train", "validation", "test"]}   # images are loaded lazily
labels = sorted({t for s in data for d in data[s] for t in d["tags"]})
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print({s: len(v) for s, v in data.items()}, "labels:", len(labels), flush=True)


def train_and_predict(model, train_ds, val_ds, test_ds, out, epochs, lr, bs, collator=None):
    def metrics(p):
        pred = p.predictions.argmax(-1); true = p.label_ids
        T = [[id2label[t] for t in row if t != -100] for row in true]
        P = [[id2label[q] for q, t in zip(pr, row) if t != -100] for pr, row in zip(pred, true)]
        return {"f1": f1_score(T, P)}
    warmup = int(0.1 * math.ceil(len(train_ds) / bs) * epochs)   # 10% of all steps
    args = TrainingArguments(out, per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
                             learning_rate=lr, num_train_epochs=epochs, warmup_steps=warmup, weight_decay=0.01,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                             load_best_model_at_end=True, metric_for_best_model="f1",
                             fp16=torch.cuda.is_available(), logging_steps=50, report_to="none",
                             disable_tqdm=True, seed=SEED)
    tr = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                 data_collator=collator, compute_metrics=metrics)
    tr.train()
    res = {}
    for name, ds in [("val", val_ds), ("test", test_ds)]:
        p = tr.predict(ds)
        logits, true = p.predictions, p.label_ids
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1).numpy()
        T, P, C = [], [], []
        for lg, pr, row in zip(logits, probs, true):
            keep = row != -100
            T.append([id2label[t] for t in row[keep]])
            P.append([id2label[q] for q in lg.argmax(-1)[keep]])
            C.append(pr.max(-1)[keep])              # confidence of each word
        res[name] = (T, P, C)
    return res


results = {}
try:
    results = pickle.load(open("results.pkl", "rb"))
except FileNotFoundError:
    pass

if "xlmr" in which:
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def enc_text(d):
        e = tok(d["words"], is_split_into_words=True, truncation=True, max_length=512)
        prev, lab = None, []
        for w in e.word_ids():
            lab.append(-100 if w is None or w == prev else label2id[d["tags"][w]]); prev = w
        e["labels"] = lab
        return e

    txt = {s: [enc_text(d) for d in data[s]] for s in data}
    m = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels),
                                                        id2label=id2label, label2id=label2id)
    results["xlmr"] = train_and_predict(m, txt["train"], txt["validation"], txt["test"], "xlmr",
                                        epochs=15, lr=5e-5, bs=8, collator=DataCollatorForTokenClassification(tok))
    pickle.dump(results, open("results.pkl", "wb"))
    print("DONE xlmr", flush=True)

if "layoutlmv3" in which:
    proc = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    class LayoutDS(torch.utils.data.Dataset):
        def __init__(self, split): self.docs, self.imgs = data[split], raw[split]
        def __len__(self): return len(self.docs)
        def __getitem__(self, i):
            d = self.docs[i]
            e = proc(self.imgs[i]["image"].convert("RGB"), d["words"], boxes=d["boxes"],
                     word_labels=[label2id[t] for t in d["tags"]],
                     truncation=True, padding="max_length", max_length=512, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in e.items()}

    lay = {s: LayoutDS(s) for s in data}
    m = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(labels),
                                                        id2label=id2label, label2id=label2id)
    results["layoutlmv3"] = train_and_predict(m, lay["train"], lay["validation"], lay["test"], "layoutlmv3",
                                              epochs=20, lr=3e-5, bs=4)
    pickle.dump(results, open("results.pkl", "wb"))
    print("DONE layoutlmv3", flush=True)
