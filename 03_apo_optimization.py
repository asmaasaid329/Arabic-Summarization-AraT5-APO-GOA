# ================================================================
# AraT5v2 + Arabic XL-Sum — APO MULTI-SEED.
# Seeds: 42, 43, 44 | 15 fitness evaluations per seed
# ================================================================
# 1. IMPORTS
# ================================================================

import os
import json
import time
import math
import random
import tarfile
import shutil
import gc
import glob
import contextlib
from pathlib import Path

import numpy as np
import torch
import evaluate

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed as hf_set_seed,
)

# ================================================================
# 2. GOOGLE DRIVE
# ================================================================

from google.colab import drive

drive.mount("/content/drive")

# ================================================================
# 3. CONFIGURATION
# ================================================================

# NEW folder (v2): the old APO folder used a simplified update rule and
# must not be mixed with this faithful version.
DRIVE_ROOT = "/content/drive/MyDrive/AraT5_APO_MultiSeed_v2"
os.makedirs(DRIVE_ROOT, exist_ok=True)

# Dataset archive: looked up in /content first, then in Drive.
DATA_ARCHIVE = "/content/arabic_XLSum_v2.0.tar.bz2"
DATA_ARCHIVE_DRIVE = os.path.join(DRIVE_ROOT, "arabic_XLSum_v2.0.tar.bz2")
DATA_ARCHIVE_DRIVE_CANDIDATES = [
    DATA_ARCHIVE_DRIVE,
    "/content/drive/MyDrive/AraT5_APO_MultiSeed/arabic_XLSum_v2.0.tar.bz2",
    "/content/drive/MyDrive/AraT5_GOA_MultiSeed/arabic_XLSum_v2.0.tar.bz2",
]
EXTRACT_DIR = "/content/arabic_XLSum"

MODEL_NAME = "UBC-NLP/AraT5v2-base-1024"

SEEDS = [42, 43, 44]

N_EVALUATIONS = 15
TRIAL_EPOCHS = 1

FINAL_EPOCHS = 10
FINAL_NUM_BEAMS = 5

MAX_INPUT_TOKENIZE_LEN = 1024
MAX_TARGET_LEN = 128
MAX_GENERATION_LENGTH = 128

POP_SIZE = 5
APO_NPAIRS = 1          
APO_PF_MAX = 0.1        
APO_ITER_MAX = 20       

BOUNDS = np.array([
    [128, 1024],      # max_input_length
    [1, 8],           # batch size
    [1e-5, 5e-4],     # learning rate
    [0.0, 0.1],       # weight decay
], dtype=float)

CLEANUP_CHECKPOINTS_AFTER_FINAL = False

# Batch size used for the evaluations of the FINAL training (beam-search
# generation over ~4.7k examples). Same value as in the GOA experiment.
EVAL_BATCH_SIZE = 8

SELECT_BEST_EPOCH = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_FP16 = bool(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 7
)

HPO_ROOT = os.path.join(DRIVE_ROOT, "HPO")
FINAL_ROOT = os.path.join(DRIVE_ROOT, "FINAL_MODELS")
REPORT_ROOT = os.path.join(DRIVE_ROOT, "REPORTS")

for folder in (HPO_ROOT, FINAL_ROOT, REPORT_ROOT):
    os.makedirs(folder, exist_ok=True)

for _s in SEEDS:
    os.makedirs(os.path.join(HPO_ROOT, f"seed_{_s}", "evaluations"), exist_ok=True)
    os.makedirs(os.path.join(FINAL_ROOT, f"seed_{_s}"), exist_ok=True)

print("=" * 75)
print("AraT5v2 + XL-Sum — APO MULTI-SEED (faithful APO, resumable)")
print("=" * 75)
print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}")
print(f"Evaluations per seed: {N_EVALUATIONS}")
print(f"Total HPO evaluations: {len(SEEDS) * N_EVALUATIONS}")
print(f"Trial epochs: {TRIAL_EPOCHS} | Final epochs: {FINAL_EPOCHS}")
print(f"Drive: {DRIVE_ROOT}")

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def atomic_json_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return default


def seed_paths(seed):
    seed_dir = os.path.join(HPO_ROOT, f"seed_{seed}")
    return {
        "seed_dir": seed_dir,
        "trials_dir": os.path.join(seed_dir, "evaluations"),
        "state_file": os.path.join(seed_dir, "apo_state.json"),
        "results_file": os.path.join(seed_dir, "apo_results.json"),
        "history_file": os.path.join(seed_dir, "apo_history.json"),
        "progress_file": os.path.join(seed_dir, "apo_progress.json"),
        "summary_file": os.path.join(seed_dir, "seed_summary.json"),
    }


HP_KEYS = [
    "max_input_length",
    "per_device_train_batch_size",
    "learning_rate",
    "weight_decay",
]


def hp_matches(a, b):
    """True if two hyperparameter dicts agree on the 4 searched values."""
    for k in HP_KEYS:
        if k not in a or k not in b:
            continue
        if k in ("max_input_length", "per_device_train_batch_size"):
            if int(a[k]) != int(b[k]):
                return False
        elif not math.isclose(float(a[k]), float(b[k]), rel_tol=1e-6, abs_tol=1e-12):
            return False
    return True
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def find_jsonl(name):
    direct = os.path.join(EXTRACT_DIR, name)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(EXTRACT_DIR):
        matches = sorted(Path(EXTRACT_DIR).rglob(name))
        if matches:
            return str(matches[0])
    return None


_names = ["arabic_train.jsonl", "arabic_val.jsonl", "arabic_test.jsonl"]

if any(find_jsonl(n) is None for n in _names):
    if not os.path.isfile(DATA_ARCHIVE):
        _src = next((p for p in DATA_ARCHIVE_DRIVE_CANDIDATES if os.path.isfile(p)), None)
        if _src is not None:
            print("Copying dataset archive from Drive...")
            shutil.copy(_src, DATA_ARCHIVE)
        else:
            raise FileNotFoundError(
                f"Dataset archive not found: {DATA_ARCHIVE}\n"
                f"Upload it to /content, or put a copy at {DATA_ARCHIVE_DRIVE}"
            )

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print("\nExtracting XL-Sum...")
    with tarfile.open(DATA_ARCHIVE, "r:bz2") as tar:
        tar.extractall(path=EXTRACT_DIR)

train_path, val_path, test_path = [find_jsonl(n) for n in _names]

dataset = DatasetDict({
    "train": load_jsonl(train_path),
    "validation": load_jsonl(val_path),
    "test": load_jsonl(test_path),
})

print("\nDataset sizes:")
for split in dataset:
    print(f"{split}: {len(dataset[split])}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def preprocess(batch):
    inputs = ["summarize: " + str(x) for x in batch["text"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_TOKENIZE_LEN,
        truncation=True
    )

    labels = tokenizer(
        text_target=[str(x) for x in batch["summary"]],
        max_length=MAX_TARGET_LEN,
        truncation=True
    )

    model_inputs["labels"] = [
        [tok if tok != tokenizer.pad_token_id else -100 for tok in ids]
        for ids in labels["input_ids"]
    ]

    return model_inputs


print("\nTokenizing dataset...")
tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

def truncate_dataset(split, max_length):
    def trunc_map(batch):
        return {
            "input_ids": [ids[:max_length] for ids in batch["input_ids"]],
            "attention_mask": [m[:max_length] for m in batch["attention_mask"]],
            "labels": batch["labels"],
        }

    return split.map(
        trunc_map,
        batched=True,
        remove_columns=split.column_names
    )


def params_to_hyperparameters(position):
    position = np.asarray(position, dtype=float)

    return {
        "max_input_length": int(np.clip(round(position[0]), 128, 1024)),
        "per_device_train_batch_size": int(np.clip(round(position[1]), 1, 8)),
        "learning_rate": float(np.clip(position[2], 1e-5, 5e-4)),
        "weight_decay": float(np.clip(position[3], 0.0, 0.1)),
        "num_train_epochs": TRIAL_EPOCHS,
    }


# (kept empty: this experiment starts from scratch, nothing to recover)
RECOVERY = {}
EXPECTED_HP = {}

REPLAY_CACHE = {}


def load_replay_cache(seed, results_file):
    saved = load_json(results_file, default=[]) or []

    recovered = [
        dict(seed=seed, num_train_epochs=TRIAL_EPOCHS, status="ok",
             error=None, source="console_output", **r)
        for r in RECOVERY.get(seed, [])
    ]

    source = saved if len(saved) >= len(recovered) else recovered

    clean = []
    for i, rec in enumerate(source):
        loss = rec.get("val_loss")
        ok = (
            rec.get("evaluation") == i + 1
            and rec.get("status") == "ok"
            and loss is not None
            and math.isfinite(float(loss))
        )
        if not ok:
            break
        clean.append(rec)

    if len(clean) < len(source):
        print(
            f"Seed {seed}: replay cache truncated at evaluation "
            f"{len(clean) + 1} (missing/failed record); it will be re-trained."
        )

    return clean

def write_progress(seed, fitness_log, results_file):
    ok = [
        r for r in fitness_log
        if r.get("status") == "ok" and math.isfinite(float(r["val_loss"]))
    ]
    best = min(ok, key=lambda r: float(r["val_loss"])) if ok else None

    atomic_json_dump(
        {
            "seed": seed,
            "evaluations_done": len(fitness_log),
            "budget": N_EVALUATIONS,
            "best_val_loss": None if best is None else float(best["val_loss"]),
            "best_hyperparameters": (
                None if best is None else {k: best[k] for k in HP_KEYS}
            ),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        os.path.join(os.path.dirname(results_file), "apo_progress.json")
    )


def run_fitness_evaluation(
    position,
    seed,
    evaluation_number,
    fitness_log,
    trials_dir,
    results_file
):
    hp = params_to_hyperparameters(position)

    print("\n" + "=" * 75)
    print(f"SEED {seed} | FITNESS EVALUATION {evaluation_number}/{N_EVALUATIONS}")
    print("=" * 75)
    print(json.dumps(hp, indent=2))

    cache = REPLAY_CACHE.get(seed, [])
    idx = evaluation_number - 1

    if idx < len(cache):
        rec = dict(cache[idx])

        if not hp_matches(rec, hp):
            raise RuntimeError(
                f"Seed {seed}, evaluation {evaluation_number}: regenerated "
                f"hyperparameters do not match the saved record.\n"
                f"  regenerated: {hp}\n  saved:       {rec}\n"
                "The APO path diverged from the saved run. Stop and check."
            )

        fitness_log.append(rec)
        atomic_json_dump(fitness_log, results_file)
        write_progress(seed, fitness_log, results_file)

        print(f" REPLAY (no training) | val_loss = {float(rec['val_loss']):.6f}")
        return float(rec["val_loss"])

    expected = EXPECTED_HP.get(seed, {}).get(evaluation_number)

    if expected is not None:
        if not hp_matches(expected, hp):
            raise RuntimeError(
                f"Seed {seed}, evaluation {evaluation_number}: hyperparameters "
                f"differ from the interrupted run.\n  now:      {hp}\n"
                f"  expected: {expected}\nStop and check before training."
            )
        print("✅ Hyperparameters match the interrupted run — path is consistent.")


    set_all_seeds(seed)

    start_time = time.time()
    model = None
    trainer = None
    val_loss = float("inf")
    error = None

    evaluation_dir = os.path.join(trials_dir, f"evaluation_{evaluation_number:02d}")
    os.makedirs(evaluation_dir, exist_ok=True)

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)

        train_tok = truncate_dataset(tokenized["train"], hp["max_input_length"])
        val_tok = truncate_dataset(tokenized["validation"], hp["max_input_length"])

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )

        args = Seq2SeqTrainingArguments(
            output_dir=evaluation_dir,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            learning_rate=hp["learning_rate"],
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            per_device_eval_batch_size=hp["per_device_train_batch_size"],
            weight_decay=hp["weight_decay"],
            num_train_epochs=TRIAL_EPOCHS,
            predict_with_generate=False,
            fp16=USE_FP16,
            dataloader_num_workers=2,
            report_to="none",
            seed=seed,
            data_seed=seed,
            disable_tqdm=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            processing_class=tokenizer,
            data_collator=collator
        )

        trainer.train()

        metrics = trainer.evaluate()
        val_loss = float(metrics.get("eval_loss", float("inf")))

        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss is not finite.")

    except Exception as exc:
        error = repr(exc)
        print("\nEvaluation failed:", error)

    elapsed_minutes = (time.time() - start_time) / 60.0

    record = {
        "seed": seed,
        "evaluation": evaluation_number,
        **hp,
        "val_loss": float(val_loss),
        "time_minutes": float(elapsed_minutes),
        "status": "ok" if np.isfinite(val_loss) else "failed",
        "error": error,
    }

    fitness_log.append(record)
    atomic_json_dump(fitness_log, results_file)
    atomic_json_dump(record, os.path.join(evaluation_dir, "evaluation_result.json"))
    write_progress(seed, fitness_log, results_file)

    print(f"\nValidation loss: {val_loss:.6f}")
    print(f"Time: {elapsed_minutes:.2f} minutes")
    print(f"💾 Saved evaluation {evaluation_number}/{N_EVALUATIONS} to Drive")

    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return float(val_loss)


def apo_optimization(
    seed,
    fitness_log,
    state_file,
    results_file,
    history_file,
    trials_dir
):
    rng = np.random.default_rng(seed)

    bounds = np.asarray(BOUNDS, dtype=float)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    span = ub - lb
    dim = len(lb)

    ps = POP_SIZE
    npairs = APO_NPAIRS
    iter_max = APO_ITER_MAX
    eps = np.finfo(float).eps

    def to_position(x):
        return lb + (np.asarray(x) + 1.0) / 2.0 * span

    protozoa = -1.0 + 2.0 * rng.random((ps, dim))
    fit = np.full(ps, np.inf)

    def evaluate(u):
        return run_fitness_evaluation(
            to_position(u), seed, len(fitness_log) + 1,
            fitness_log, trials_dir, results_file
        )

    def safe(f):
        # failed evaluations (inf) must not produce nan in the weights
        return np.where(np.isfinite(f), f, 1e6)

    best_u = None
    best_fit = float("inf")
    history = []

    def update_best():
        nonlocal best_u, best_fit
        finite = np.where(np.isfinite(fit))[0]
        if len(finite) > 0:
            idx = finite[np.argmin(fit[finite])]
            if fit[idx] < best_fit:
                best_fit = float(fit[idx])
                best_u = protozoa[idx].copy()

    def save_state(iteration, status):
        atomic_json_dump(
            {
                "seed": seed,
                "iteration": iteration,
                "population": to_position(protozoa).tolist(),
                "fitness": fit.tolist(),
                "best_position": None if best_u is None else to_position(best_u).tolist(),
                "best_fitness": None if not np.isfinite(best_fit) else float(best_fit),
                "history": history,
                "fitness_evaluations": len(fitness_log),
                "status": status,
            },
            state_file
        )

   
    for i in range(ps):
        if len(fitness_log) >= N_EVALUATIONS:
            break
        fit[i] = evaluate(protozoa[i])

    update_best()
    history.append(float(best_fit))
    save_state(1, "completed" if len(fitness_log) >= N_EVALUATIONS else "running")

    new = np.zeros((ps, dim))
    epn = np.zeros((npairs, dim))

  
    for it in range(2, iter_max + 1):

        if len(fitness_log) >= N_EVALUATIONS:
            break

        print(f"\nAPO iteration {it} (annealing schedule length {iter_max})")

        order = np.argsort(fit)
        fit = fit[order]
        protozoa = protozoa[order]

        pf = APO_PF_MAX * rng.random()
        ri = set(rng.permutation(ps)[: int(np.ceil(ps * pf))].tolist())

        for i in range(ps):

            if i in ri:
                # ---- dormancy or reproduction ----
                pdr = 0.5 * (1 + np.cos((1 - (i + 1) / ps) * np.pi))
                if rng.random() < pdr:
                    new[i] = -1.0 + 2.0 * rng.random(dim)          
                else:                                              
                    flag = rng.choice([1, -1])
                    mr = np.zeros(dim)
                    k = int(np.ceil(rng.random() * dim))
                    mr[rng.permutation(dim)[:k]] = 1
                    new[i] = protozoa[i] + flag * rng.random() * (-1.0 + 2.0 * rng.random(dim)) * mr
            else:
                # ---- foraging ----
                f = rng.random() * (1 + np.cos(it / iter_max * np.pi))
                mf = np.zeros(dim)
                k = min(int(np.ceil(dim * (i + 1) / ps)), dim)
                mf[rng.permutation(dim)[:k]] = 1

                sf = safe(fit)
                pah = 0.5 * (1 + np.cos(it / iter_max * np.pi))

                if rng.random() < pah:                             
                    j = int(rng.integers(ps))
                    for kp_ in range(npairs):
                        if i == 0:
                            km = i
                            kp = i + int(rng.integers(1, ps - i))
                        elif i == ps - 1:
                            km = int(rng.integers(0, ps - 1))
                            kp = i
                        else:
                            km = int(rng.integers(0, i))
                            kp = i + int(rng.integers(1, ps - i))
                        wa = np.exp(-abs(sf[km] / (sf[kp] + eps)))
                        epn[kp_] = wa * (protozoa[km] - protozoa[kp])
                    new[i] = protozoa[i] + f * (
                        protozoa[j] - protozoa[i] + (1 / npairs) * epn.sum(axis=0)
                    ) * mf
                else:                                             
                    for kp_ in range(npairs):
                        if i == 0:
                            imk, ipk = i, i + (kp_ + 1)
                        elif i == ps - 1:
                            imk, ipk = ps - (kp_ + 1), i
                        else:
                            imk, ipk = i - (kp_ + 1), i + (kp_ + 1)
                        imk = max(imk, 0)
                        ipk = min(ipk, ps - 1)
                        wh = np.exp(-abs(sf[imk] / (sf[ipk] + eps)))
                        epn[kp_] = wh * (protozoa[imk] - protozoa[ipk])
                    flag = rng.choice([1, -1])
                    xnear = (1 + flag * rng.random(dim) * (1 - it / iter_max)) * protozoa[i]
                    new[i] = protozoa[i] + f * (
                        xnear - protozoa[i] + (1 / npairs) * epn.sum(axis=0)
                    ) * mf

        new = np.clip(new, -1.0, 1.0)

        # greedy selection, evaluating one candidate at a time
        for i in range(ps):
            if len(fitness_log) >= N_EVALUATIONS:
                break
            cf = evaluate(new[i])
            if np.isfinite(cf) and cf < fit[i]:
                protozoa[i] = new[i]
                fit[i] = cf

        update_best()
        history.append(float(best_fit))

        done = len(fitness_log) >= N_EVALUATIONS
        save_state(it, "completed" if done else "running")
        atomic_json_dump(fitness_log, results_file)
        atomic_json_dump(history, history_file)

        print(f"Best validation loss: {best_fit:.6f}")
        print(f"Evaluations: {len(fitness_log)}/{N_EVALUATIONS}")

        if done:
            print("\nFitness budget reached.")
            break

    if len(fitness_log) >= N_EVALUATIONS:
        save_state(len(history), "completed")

    best_position = None if best_u is None else to_position(best_u)
    return best_position, best_fit, history, to_position(protozoa), fit


def is_completed(state):
    return (
        state is not None
        and state.get("status") == "completed"
        and int(state.get("fitness_evaluations", 0)) >= N_EVALUATIONS
    )


def build_best_hp(best_position):
    hp = params_to_hyperparameters(best_position)
    hp["num_beams"] = FINAL_NUM_BEAMS
    hp["num_train_epochs"] = FINAL_EPOCHS
    return hp


def write_seed_summary(seed, best_position, best_loss, fitness_log,
                       search_wall_minutes=None, resumed=False):
    summary_file = seed_paths(seed)["summary_file"]

    summary = {
        "seed": seed,
        "best_validation_loss": float(best_loss),
        "best_hyperparameters": build_best_hp(best_position),
        "fitness_evaluations": len(fitness_log) if fitness_log else N_EVALUATIONS,
        "total_trial_minutes": float(sum(
            r["time_minutes"] for r in (fitness_log or [])
            if r.get("time_minutes") is not None
        )),
        "search_wall_minutes_last_run": search_wall_minutes,
        "resumed": resumed,
        "trial_epochs": TRIAL_EPOCHS,
        "final_epochs": FINAL_EPOCHS,
    }

    atomic_json_dump(summary, summary_file)
    return summary


def show_completed_seed(seed, state, results_file):
    print(f"Best validation loss: {float(state['best_fitness']):.6f}")

    if state.get("best_position") is not None:
        hp = params_to_hyperparameters(state["best_position"])
        print("Best hyperparameters:")
        for k in HP_KEYS:
            print(f"  {k}: {hp[k]}")

    hist = state.get("history") or []
    print("Best-so-far history:", [round(float(x), 4) for x in hist])

    log = load_json(results_file)
    if log:
        print("\nAll evaluations:")
        for r in log:
            t = r.get("time_minutes")
            t_txt = "n/a" if t is None else f"{float(t):.1f} min"
            print(
                f"  eval {r.get('evaluation', '?'):>2} | "
                f"len={r.get('max_input_length')} bs={r.get('per_device_train_batch_size')} "
                f"lr={float(r.get('learning_rate', 0)):.3e} wd={float(r.get('weight_decay', 0)):.4f} | "
                f"val_loss={float(r.get('val_loss', float('nan'))):.6f} | {t_txt}"
            )


for seed in SEEDS:

    print("\n\n" + "#" * 75)
    print(f"SEED {seed}")
    print("#" * 75)

    p = seed_paths(seed)
    os.makedirs(p["trials_dir"], exist_ok=True)

    state = load_json(p["state_file"])

    # ---------------- already completed ----------------
    if is_completed(state):
        print(f"\nSeed {seed} already COMPLETED "
              f"({state['fitness_evaluations']}/{N_EVALUATIONS}) — skipping HPO.\n")

        show_completed_seed(seed, state, p["results_file"])

        if not os.path.isfile(p["summary_file"]):
            print("\nseed_summary.json missing — rebuilding it from the saved state.")
            write_seed_summary(
                seed,
                np.array(state["best_position"], dtype=float),
                float(state["best_fitness"]),
                load_json(p["results_file"], default=[]) or []
            )
        continue

    # ---------------- run or resume ----------------
    REPLAY_CACHE[seed] = load_replay_cache(seed, p["results_file"])
    n_replay = len(REPLAY_CACHE[seed])

    if n_replay > 0:
        print(f"\nSeed {seed}: replaying {n_replay} finished evaluations (no training); "
              f"training resumes from evaluation {n_replay + 1}/{N_EVALUATIONS}.")
    else:
        print(f"\nSeed {seed}: starting from scratch.")

    fitness_log = []
    search_start = time.time()

    best_position, best_loss, history, population, fitness = apo_optimization(
        seed=seed,
        fitness_log=fitness_log,
        state_file=p["state_file"],
        results_file=p["results_file"],
        history_file=p["history_file"],
        trials_dir=p["trials_dir"]
    )

    if best_position is None:
        raise RuntimeError(f"Seed {seed}: every evaluation failed; no best position.")

    atomic_json_dump(fitness_log, p["results_file"])

    write_seed_summary(
        seed,
        best_position,
        best_loss,
        fitness_log,
        search_wall_minutes=(time.time() - search_start) / 60.0,
        resumed=n_replay > 0
    )

    print(f"\n✅ Seed {seed} HPO finished | best val loss = {best_loss:.6f} "
          f"| evaluations = {len(fitness_log)}/{N_EVALUATIONS}")



print("\n" + "=" * 75)
print("MULTI-SEED APO HPO SUMMARY")
print("=" * 75)

seed_losses = []
all_seed_results = []

for seed in SEEDS:
    summary = load_json(seed_paths(seed)["summary_file"])

    if summary is None:
        print(f"Seed {seed}: not finished yet.")
        continue

    loss = float(summary["best_validation_loss"])
    seed_losses.append(loss)
    all_seed_results.append(summary)
    print(f"Seed {seed}: {loss:.6f}")

if seed_losses:
    mean_loss = float(np.mean(seed_losses))
    std_loss = float(np.std(seed_losses, ddof=1)) if len(seed_losses) > 1 else 0.0
else:
    mean_loss = None
    std_loss = None

multi_seed_summary = {
    "optimizer": "APO",
    "seeds": SEEDS,
    "evaluations_per_seed": N_EVALUATIONS,
    "total_fitness_evaluations": len(SEEDS) * N_EVALUATIONS,
    "trial_epochs": TRIAL_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "best_validation_loss_mean": mean_loss,
    "best_validation_loss_std": std_loss,
    "seed_results": all_seed_results,
}

MULTI_SEED_FILE = os.path.join(REPORT_ROOT, "APO_multi_seed_HPO_summary.json")
atomic_json_dump(multi_seed_summary, MULTI_SEED_FILE)

print(f"\nMean validation loss: {mean_loss}")
print(f"Standard deviation:   {std_loss}")
print(f"Saved to: {MULTI_SEED_FILE}")


class EpochCheckpointCallback(TrainerCallback):
    """
    Writes a small marker every time Trainer saves a checkpoint,
    and appends the validation metrics of every epoch to a .jsonl file.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_save(self, args, state, control, **kwargs):
        epoch = float(state.epoch or 0)
        atomic_json_dump(
            {
                "epoch": epoch,
                "global_step": int(state.global_step),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            os.path.join(self.save_dir, f"epoch_{int(round(epoch))}_checkpoint_saved.json")
        )
        print(f"\n💾 Epoch {int(round(epoch))} checkpoint saved.")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            row = {"epoch": float(state.epoch or 0), "global_step": int(state.global_step), **metrics}
            with open(os.path.join(self.save_dir, "epoch_metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        return control



rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Generated sequences can be padded with -100 when batches differ in
    # length; the tokenizer cannot decode -100.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = [
        str(x).strip()
        for x in tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ]
    decoded_labels = [
        str(x).strip()
        for x in tokenizer.batch_decode(labels, skip_special_tokens=True)
    ]

    results = {}

    # The default ROUGE tokenizer keeps only [a-z0-9] and silently deletes
    # Arabic letters, so whitespace tokenization is required for Arabic.
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False,
        tokenizer=lambda s: s.split()
    )
    for key, value in rouge_result.items():
        results[f"rouge_{key}"] = float(value * 100)

    # evaluate's BLEU wants plain strings; whitespace tokenization is kept
    # through the tokenizer argument (same intent as the original code).
    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[r] for r in decoded_labels],
        tokenizer=lambda s: s.split()
    )
    results["BLEU"] = float(bleu_result["bleu"] * 100)

    meteor_result = meteor.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    results["METEOR"] = float(meteor_result["meteor"] * 100)

    bert_result = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="ar"
    )
    results["BERTScore_F1"] = float(np.mean(bert_result["f1"]) * 100)

    return results


def _metrics_smoke_test(n=16):
 
    rows = tokenized["validation"].select(range(n))["labels"]
    width = max(len(r) for r in rows)
    arr = np.full((n, width), -100, dtype=np.int64)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    out = compute_metrics((arr.copy(), arr.copy()))
    bad = [k for k, v in out.items() if not math.isfinite(float(v))]
    if bad:
        raise RuntimeError(f"Metrics smoke test produced non-finite values: {bad}")

    # Predictions == references here, so every metric must be close to 100.
    low = {k: round(float(v), 2) for k, v in out.items() if float(v) < 95}
    if low:
        raise RuntimeError(
            f"Metrics smoke test: identical texts should score ~100 but got {low}. "
            "A metric is mis-tokenizing Arabic."
        )

    print("✅ Metrics smoke test passed:", {k: round(v, 2) for k, v in out.items()})


_metrics_smoke_test()



def find_valid_checkpoint(folder):
    
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
        complete = (
            has_weights
            and os.path.exists(os.path.join(path, "trainer_state.json"))
            and os.path.exists(os.path.join(path, "optimizer.pt"))
            and os.path.exists(os.path.join(path, "scheduler.pt"))
        )

        if complete:
            candidates.append((step, path))

    return max(candidates)[1] if candidates else None


def train_final_model_for_seed(seed, best_hp):

    seed_final_dir = os.path.join(FINAL_ROOT, f"seed_{seed}")
    os.makedirs(seed_final_dir, exist_ok=True)

    report_path = os.path.join(seed_final_dir, "final_report.json")
    existing_report = load_json(report_path)

    # final_report.json is written LAST, so its existence = seed finished.
    if existing_report is not None:
        print(f"\nFinal training for Seed {seed} already finished — skipping.")
        return existing_report

    print("\n" + "=" * 75)
    print(f"FINAL TRAINING — SEED {seed}")
    print("=" * 75)

    set_all_seeds(seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    train_final = truncate_dataset(tokenized["train"], best_hp["max_input_length"])
    val_final = truncate_dataset(tokenized["validation"], best_hp["max_input_length"])
    test_final = truncate_dataset(tokenized["test"], best_hp["max_input_length"])

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    args = Seq2SeqTrainingArguments(
        output_dir=seed_final_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_hp["learning_rate"],
        per_device_train_batch_size=best_hp["per_device_train_batch_size"],
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=best_hp["weight_decay"],
        num_train_epochs=FINAL_EPOCHS,
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        generation_num_beams=FINAL_NUM_BEAMS,
        generation_max_length=MAX_GENERATION_LENGTH,
        fp16=USE_FP16,
        save_total_limit=2,
        load_best_model_at_end=SELECT_BEST_EPOCH,
        metric_for_best_model="eval_rouge_rouge1",
        greater_is_better=True,
        dataloader_num_workers=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_final,
        eval_dataset=val_final,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochCheckpointCallback(seed_final_dir)]
    )

    latest_checkpoint = find_valid_checkpoint(seed_final_dir)

    if latest_checkpoint:
        print(f"\n Resuming from: {latest_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("\n Starting final training...")
        train_result = trainer.train()

    final_model_dir = os.path.join(seed_final_dir, "best_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    # A resumed run has almost no training time of its own; keep the original
    # train_results.json from the real training run untouched.
    trainer.save_metrics("train_resumed" if latest_checkpoint else "train",
                         train_result.metrics)

    validation_metrics = trainer.evaluate(eval_dataset=val_final)
    atomic_json_dump(
        validation_metrics,
        os.path.join(seed_final_dir, "validation_metrics.json")
    )

    test_metrics = trainer.evaluate(eval_dataset=test_final, metric_key_prefix="test")

    test_metrics_clean = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in test_metrics.items()
    }

    atomic_json_dump(
        test_metrics_clean,
        os.path.join(seed_final_dir, "test_metrics.json")
    )

    final_report = {
        "optimizer": "APO",
        "seed": seed,
        "model": MODEL_NAME,
        "best_hyperparameters": best_hp,
        "final_epochs": FINAL_EPOCHS,
        "fitness_evaluations": N_EVALUATIONS,
        "test_metrics": test_metrics_clean,
    }

    atomic_json_dump(final_report, report_path)   # written LAST

    print(f"\n✅ Seed {seed} final training finished.")
    print("\nTest metrics:")
    print(json.dumps(test_metrics_clean, indent=2, ensure_ascii=False))

    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if CLEANUP_CHECKPOINTS_AFTER_FINAL:
        for ckpt in glob.glob(os.path.join(seed_final_dir, "checkpoint-*")):
            shutil.rmtree(ckpt, ignore_errors=True)
        print("🧹 Removed checkpoint-* folders (best_model and reports are kept).")

    return final_report



FINAL_SEED_RESULTS = []

for seed in SEEDS:
    summary = load_json(seed_paths(seed)["summary_file"])

    if summary is None:
        print(f"Skipping seed {seed}: HPO summary not found.")
        continue

    result = train_final_model_for_seed(seed, summary["best_hyperparameters"])
    FINAL_SEED_RESULTS.append(result)



print("\n" + "=" * 75)
print("APO MULTI-SEED FINAL TEST SUMMARY")
print("=" * 75)

metric_values = {}

for result in FINAL_SEED_RESULTS:
    for key, value in result["test_metrics"].items():
        if isinstance(value, (int, float)):
            metric_values.setdefault(key, []).append(float(value))

final_mean_sd = {}

for metric, values in metric_values.items():
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    final_mean_sd[metric] = {
        "mean": mean,
        "std": std,
        "mean_plus_minus_sd": f"{mean:.4f} ± {std:.4f}",
        "values": values,
    }

    print(f"{metric}: {mean:.4f} ± {std:.4f}")
complete_report = {
    "optimizer": "APO",
    "model": MODEL_NAME,
    "seeds": SEEDS,
    "evaluations_per_seed": N_EVALUATIONS,
    "total_hpo_fitness_evaluations": len(SEEDS) * N_EVALUATIONS,
    "trial_epochs": TRIAL_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "final_num_beams": FINAL_NUM_BEAMS,
    "search_space": {
        "max_input_length": [128, 1024],
        "batch_size": [1, 8],
        "learning_rate": [1e-5, 5e-4],
        "weight_decay": [0.0, 0.1],
    },
    "apo_parameters": {
        "population_size": POP_SIZE,
        "pf_max": APO_PF_MAX,
        "neighbour_pairs": APO_NPAIRS,
        "annealing_schedule_length": APO_ITER_MAX,
        "iterations_executed_note": "iteration 1 = initial population, then iterations 2-3 (budget of 15 evaluations)",
    },
    "hpo_best_val_loss_mean": mean_loss,
    "hpo_best_val_loss_std": std_loss,
    "final_test_mean_sd": final_mean_sd,
    "seed_results": FINAL_SEED_RESULTS,
}

COMPLETE_REPORT_FILE = os.path.join(REPORT_ROOT, "APO_COMPLETE_MULTI_SEED_REPORT.json")
atomic_json_dump(complete_report, COMPLETE_REPORT_FILE)

print("\n" + "=" * 75)
print(" APO EXPERIMENT COMPLETED")
print("=" * 75)
print(f"\nComplete report:\n{COMPLETE_REPORT_FILE}")
print(f"\nGoogle Drive folder:\n{DRIVE_ROOT}")
print("\nEverything is saved to Google Drive.")

NUM_EXAMPLES = 3

example_rows = [dataset["test"][i] for i in range(NUM_EXAMPLES)]
example_originals = [str(r["text"]) for r in example_rows]
example_references = [str(r["summary"]) for r in example_rows]

example_enc = tokenizer(
    ["summarize: " + t for t in example_originals],
    max_length=MAX_INPUT_TOKENIZE_LEN,
    truncation=True
)


def generate_three_examples(model, max_input_length):
    batch = tokenizer.pad(
        {
            "input_ids": [x[:max_input_length] for x in example_enc["input_ids"]],
            "attention_mask": [x[:max_input_length] for x in example_enc["attention_mask"]],
        },
        return_tensors="pt"
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    amp = torch.autocast("cuda", dtype=torch.float16) if USE_FP16 else contextlib.nullcontext()

    model.eval()
    with torch.no_grad(), amp:
        ids = model.generate(
            **batch,
            max_length=MAX_GENERATION_LENGTH,
            num_beams=FINAL_NUM_BEAMS,
            do_sample=False
        )

    return [x.strip() for x in tokenizer.batch_decode(ids, skip_special_tokens=True)]


generated_by_seed = {}

for result in FINAL_SEED_RESULTS:
    seed = result["seed"]
    model_dir = os.path.join(FINAL_ROOT, f"seed_{seed}", "best_model")

    if not os.path.isdir(model_dir):
        print(f"Seed {seed}: final model not found — examples skipped.")
        continue

    max_len = int(result["best_hyperparameters"]["max_input_length"])

    ex_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    generated = generate_three_examples(ex_model, max_len)

    del ex_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generated_by_seed[seed] = generated

    atomic_json_dump(
        [
            {
                "example_number": i + 1,
                "original": example_originals[i],
                "reference_summary": example_references[i],
                "generated_summary": generated[i],
            }
            for i in range(NUM_EXAMPLES)
        ],
        os.path.join(FINAL_ROOT, f"seed_{seed}", "three_examples.json")
    )

combined_examples = []

for i in range(NUM_EXAMPLES):

    print("\n" + "=" * 80)
    print(f" EXAMPLE {i + 1}")
    print("=" * 80)

    print("\n ORIGINAL:\n")
    print(example_originals[i])

    print("\n REFERENCE SUMMARY:\n")
    print(example_references[i])

    for seed, generated in generated_by_seed.items():
        print(f"\n GENERATED SUMMARY (APO, seed {seed}):\n")
        print(generated[i])

    combined_examples.append({
        "example_number": i + 1,
        "original": example_originals[i],
        "reference_summary": example_references[i],
        "generated_summary_by_seed": {
            str(seed): generated[i] for seed, generated in generated_by_seed.items()
        },
    })

EXAMPLES_FILE = os.path.join(REPORT_ROOT, "APO_three_examples.json")
atomic_json_dump(combined_examples, EXAMPLES_FILE)

print(f"\n Examples saved to:\n{EXAMPLES_FILE}")
