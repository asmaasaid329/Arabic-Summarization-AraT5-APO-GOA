# ============================================================
# 05 - Recompute Final Test Metrics for All Models
# Arabic Abstractive Text Summarization using AraT5v2
#
# This script evaluates the saved models on the same test set
# using identical preprocessing, generation settings, and
# evaluation metrics to ensure a fair comparison.
#
# Methods:
# 1. Baseline
# 2. Random Search
# 3. APO
# 4. GOA
#
# No training or hyperparameter optimization is performed.
# The script only loads saved checkpoints and recomputes
# the final test metrics.
# ============================================================
# ================================================================
# SAME-WAY EVALUATION OF ALL METHODS ON THE TEST SET
#   Random Search, GOA, APO and the Baseline
# ================================================================

# !pip install -q transformers datasets evaluate accelerate sentencepiece rouge_score nltk bert_score scipy

import os, json, re, gc, math, tarfile, shutil, time, glob
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLMa

from google.colab import drive
drive.mount("/content/drive")


DRIVE = "/content/drive/MyDrive"
OUT_ROOT = f"{DRIVE}/AraT5_CORRECTED_EVAL_v2"     

SEEDS = [42, 43, 44]

RUN = ["Baseline"]          

EXPECTED_EPOCH = 10      

EXPERIMENTS = {
    "RandomSearch": dict(
        ckpt_root=f"{DRIVE}/AraT5_RandomSearch/seed_{{seed}}/final_training",
        hp_file=f"{DRIVE}/AraT5_RandomSearch/seed_{{seed}}/seed_summary.json",
    ),
    "GOA": dict(
        ckpt_root=f"{DRIVE}/AraT5_GOA_MultiSeed/FINAL_MODELS/seed_{{seed}}",
        hp_file=f"{DRIVE}/AraT5_GOA_MultiSeed/FINAL_MODELS/seed_{{seed}}/final_report.json",
    ),
    "APO": dict(
        ckpt_root=f"{DRIVE}/AraT5_APO_MultiSeed_v2/FINAL_MODELS/seed_{{seed}}",
        hp_file=f"{DRIVE}/AraT5_APO_MultiSeed_v2/FINAL_MODELS/seed_{{seed}}/final_report.json",
    ),
    "Baseline": dict(
        ckpt_root=f"{DRIVE}/AraT5_Baseline/seed_{{seed}}/training",
        hp_file=f"{DRIVE}/AraT5_Baseline/seed_{{seed}}/seed_summary.json",
    ),
}

MODEL_NAME = "UBC-NLP/AraT5v2-base-1024"
MAX_INPUT_TOKENIZE_LEN = 1024
MAX_GENERATION_LENGTH = 128
NUM_BEAMS = 5
GEN_BATCH_SIZE = 16             

DATA_ARCHIVE = "/content/arabic_XLSum_v2.0.tar.bz2"
EXTRACT_DIR = "/content/arabic_XLSum"
DATA_ARCHIVE_CANDIDATES = [
    f"{DRIVE}/AraT5_APO_MultiSeed_v2/arabic_XLSum_v2.0.tar.bz2",
    f"{DRIVE}/AraT5_APO_MultiSeed/arabic_XLSum_v2.0.tar.bz2",
    f"{DRIVE}/AraT5_GOA_MultiSeed/arabic_XLSum_v2.0.tar.bz2",
    f"{DRIVE}/AraT5_Baseline/arabic_XLSum_v2.0.tar.bz2",
    f"{DRIVE}/AraT5_RandomSearch/arabic_XLSum_v2.0.tar.bz2",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7)


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def dump_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def find_jsonl(name):
    direct = os.path.join(EXTRACT_DIR, name)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(EXTRACT_DIR):
        hits = sorted(Path(EXTRACT_DIR).rglob(name))
        if hits:
            return str(hits[0])
    return None


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_last_checkpoint(folder):
    candidates = []
    for path in glob.glob(os.path.join(folder, "checkpoint-*")):
        if not os.path.isdir(path):
            continue
        try:
            step = int(os.path.basename(path).split("-")[-1])
        except ValueError:
            continue
        has_weights = (
            os.path.exists(os.path.join(path, "model.safetensors"))
            or os.path.exists(os.path.join(path, "pytorch_model.bin"))
        )
        if has_weights:
            candidates.append((step, path))

    if not candidates:
        return None, None

    path = max(candidates)[1]
    state = load_json(os.path.join(path, "trainer_state.json"))
    epoch = None if not state else state.get("epoch")
    return path, epoch


def holm_adjust(pvalues):
  
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adjusted = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def t_crit_95(n):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, n - 1))
    except Exception:
        return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)

if find_jsonl("arabic_test.jsonl") is None:
    if not os.path.isfile(DATA_ARCHIVE):
        src = next((p for p in DATA_ARCHIVE_CANDIDATES if os.path.isfile(p)), None)
        if src is None:
            raise FileNotFoundError("Dataset archive not found. Upload arabic_XLSum_v2.0.tar.bz2 to /content.")
        shutil.copy(src, DATA_ARCHIVE)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with tarfile.open(DATA_ARCHIVE, "r:bz2") as tar:
        tar.extractall(path=EXTRACT_DIR)

test_rows = read_jsonl(find_jsonl("arabic_test.jsonl"))
test_texts = [str(r["text"]) for r in test_rows]
test_refs = [str(r["summary"]) for r in test_rows]
print(f"Test examples: {len(test_rows)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# The default ROUGE tokenizer keeps only [a-z0-9] and deletes Arabic letters.
# Words are extracted with \w+ (Unicode aware), so punctuation is not part of a
# word. No stemming (the stemmer is English only).
#   ROUGE : \w+ words, no stemming        BLEU  : \w+ words
#   METEOR: library default (exact word match; WordNet is English)
#   BERTScore: lang="ar" (multilingual BERT default of the library)

def word_tok(s):
    return re.findall(r"\w+", s)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")


def compute_all_metrics(preds, refs):
    out = {}
    r = rouge.compute(predictions=preds, references=refs,
                      use_stemmer=False, tokenizer=word_tok)
    for k, v in r.items():
        out[f"rouge_{k}"] = float(v) * 100
    b = bleu.compute(predictions=preds, references=[[x] for x in refs],
                     tokenizer=word_tok)
    out["BLEU"] = float(b["bleu"]) * 100
    m = meteor.compute(predictions=preds, references=refs)
    out["METEOR"] = float(m["meteor"]) * 100
    bs = bertscore.compute(predictions=preds, references=refs, lang="ar")
    out["BERTScore_F1"] = float(np.mean(bs["f1"])) * 100
    return out


# sanity check: identical texts must score ~100 on every metric
_smoke = compute_all_metrics(test_refs[:16], test_refs[:16])
_low = {k: round(v, 2) for k, v in _smoke.items() if v < 95}
if _low:
    raise RuntimeError(f"Metric sanity check failed (identical texts): {_low}")
print("Metric sanity check passed:", {k: round(v, 1) for k, v in _smoke.items()})

def generate_all(model, max_input_length):
    enc = tokenizer(["summarize: " + t for t in test_texts],
                    max_length=MAX_INPUT_TOKENIZE_LEN, truncation=True)
    ids = [x[:max_input_length] for x in enc["input_ids"]]   # same cut as in training
    order = np.argsort([len(x) for x in ids])                # similar lengths together = faster
    preds = [None] * len(ids)

    amp = torch.autocast("cuda", dtype=torch.float16) if USE_FP16 else torch.autocast("cpu", enabled=False)
    model.eval()
    t0 = time.time()

    for s in range(0, len(order), GEN_BATCH_SIZE):
        idx = order[s:s + GEN_BATCH_SIZE]
        batch = tokenizer.pad(
            {"input_ids": [ids[i] for i in idx],
             "attention_mask": [[1] * len(ids[i]) for i in idx]},
            return_tensors="pt")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad(), amp:
            gen = model.generate(**batch, max_length=MAX_GENERATION_LENGTH,
                                 num_beams=NUM_BEAMS, do_sample=False)
        for i, d in zip(idx, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            preds[i] = d.strip()
        if (s // GEN_BATCH_SIZE) % 25 == 0:
            print(f"  generated {min(s + GEN_BATCH_SIZE, len(order))}/{len(order)} "
                  f"({(time.time() - t0) / 60:.1f} min)")
    return preds


def metric_values(per_seed):
    """{seed: metrics} -> {metric: [values in seed order]} (keys starting with _ are metadata)."""
    names = [k for k in next(iter(per_seed.values())).keys() if not k.startswith("_")]
    return {k: [float(per_seed[s][k]) for s in sorted(per_seed)] for k in names}


for name in RUN:
    cfg = EXPERIMENTS[name]
    print("\n" + "#" * 70 + f"\n{name}\n" + "#" * 70)
    per_seed = {}

    for seed in SEEDS:
        out_dir = os.path.join(OUT_ROOT, name, f"seed_{seed}")
        pred_file = os.path.join(out_dir, "predictions.jsonl")
        met_file = os.path.join(out_dir, "metrics.json")

        saved = load_json(met_file)
        if saved is not None:
            print(f"Seed {seed}: metrics already computed - skipping.")
            per_seed[seed] = saved
            continue

        ckpt_root = cfg["ckpt_root"].format(seed=seed)
        hp_doc = load_json(cfg["hp_file"].format(seed=seed))
        model_dir, epoch = resolve_last_checkpoint(ckpt_root)

        if model_dir is None or hp_doc is None:
            print(f"Seed {seed}: no checkpoint or hyperparameter file found - skipped.\n  {ckpt_root}")
            continue

        if epoch is None or abs(float(epoch) - EXPECTED_EPOCH) > 1e-6:
            print(f"Seed {seed}: newest checkpoint is at epoch {epoch}, not {EXPECTED_EPOCH} "
                  f"(final training not finished) - skipped.")
            continue

        max_len = int(hp_doc["best_hyperparameters"]["max_input_length"])
        print(f"Seed {seed}: model = {model_dir} (epoch {epoch}), max_input_length = {max_len}")

        # predictions (resume if already generated)
        preds = None
        if os.path.isfile(pred_file):
            rows = read_jsonl(pred_file)
            if len(rows) == len(test_rows):
                preds = [r["prediction"] for r in rows]
                print(f"Seed {seed}: loaded saved predictions.")

        if preds is None:
            print(f"Seed {seed}: generating ...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
            preds = generate_all(model, max_len)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            os.makedirs(out_dir, exist_ok=True)
            with open(pred_file, "w", encoding="utf-8") as f:
                for i, (p, r) in enumerate(zip(preds, test_refs)):
                    f.write(json.dumps({"id": i, "prediction": p, "reference": r},
                                       ensure_ascii=False) + "\n")

        m = compute_all_metrics(preds, test_refs)
        m["_model_dir"] = model_dir
        m["_epoch"] = float(epoch)
        m["_max_input_length"] = max_len
        dump_json(m, met_file)
        per_seed[seed] = m
        print(f"Seed {seed}:", {k: round(v, 2) for k, v in m.items() if not k.startswith("_")})

    if len(per_seed) > 0:
        n = len(per_seed)
        summary = {"experiment": name, "seeds": sorted(per_seed.keys()), "metrics": {}}
        print(f"\n{name}: mean ± std (95% CI), n = {n} seeds")
        for k, vals in metric_values(per_seed).items():
            vals = np.array(vals, dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            half = t_crit_95(n) * std / math.sqrt(n) if n > 1 else 0.0
            summary["metrics"][k] = {"values": vals.tolist(), "mean": mean, "std": std,
                                     "ci95_low": mean - half, "ci95_high": mean + half}
            print(f"  {k:<18}: {mean:7.3f} ± {std:6.3f}   CI95 [{mean - half:7.3f}, {mean + half:7.3f}]")
        if n < len(SEEDS):
            print(f"  WARNING: only {n}/{len(SEEDS)} seeds were available for {name}.")
        dump_json(summary, os.path.join(OUT_ROOT, name, "summary_mean_std.json"))

summaries = {}
for name in RUN:
    s = load_json(os.path.join(OUT_ROOT, name, "summary_mean_std.json"))
    if s:
        summaries[name] = s

if summaries:
    cols = ["rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "BLEU", "METEOR", "BERTScore_F1"]
    print("\n" + "=" * 110)
    print("FINAL TEST RESULTS (mean ± std over seeds, last epoch, same metric function)")
    print("=" * 110)
    print(f"{'method':<14}{'n':<4}" + "".join(f"{c:<16}" for c in cols))
    for name, s in summaries.items():
        row = f"{name:<14}{len(s['seeds']):<4}"
        for c in cols:
            mm = s["metrics"].get(c)
            row += f"{mm['mean']:6.2f} ± {mm['std']:<6.2f} " if mm else f"{'-':<16}"
        print(row)


comparable = {n: s for n, s in summaries.items() if len(s["seeds"]) >= 2}

if len(comparable) >= 2:
    try:
        from scipy import stats
    except Exception:
        stats = None

    rows = []
    for a, b in combinations(comparable.keys(), 2):
        for k in comparable[a]["metrics"]:
            vb = comparable[b]["metrics"].get(k, {}).get("values")
            if vb is None:
                continue
            va = comparable[a]["metrics"][k]["values"]
            p = None
            if stats is not None:
                p = float(stats.ttest_ind(va, vb, equal_var=False).pvalue)
                if math.isnan(p):
                    p = None
            rows.append({"a": a, "b": b, "metric": k,
                         "mean_diff_a_minus_b": float(np.mean(va) - np.mean(vb)),
                         "welch_p": p})

    adj = holm_adjust([r["welch_p"] for r in rows])
    for r, ap in zip(rows, adj):
        r["welch_p_holm"] = ap

    last = None
    for r in rows:
        if (r["a"], r["b"]) != last:
            print("\n" + "=" * 78 + f"\n{r['a']} vs {r['b']}\n" + "=" * 78)
            last = (r["a"], r["b"])
        raw = "n/a" if r["welch_p"] is None else f"{r['welch_p']:.4f}"
        hol = "n/a" if r["welch_p_holm"] is None else f"{r['welch_p_holm']:.4f}"
        print(f"  {r['metric']:<16}: diff = {r['mean_diff_a_minus_b']:+7.3f}   p = {raw}   p(Holm) = {hol}")

    dump_json(rows, os.path.join(OUT_ROOT, "pairwise_comparison.json"))

print(f"\nDone. Results: {OUT_ROOT}")
