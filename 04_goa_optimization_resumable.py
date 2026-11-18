# ================================================================
# AraT5v2 + Arabic XL-Sum — GOA MULTI-SEED
# ================================================================
#
#   - Seeds: 42, 43, 44 | 15 fitness evaluations per seed
# ================================================================
# IMPORTS
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
#  GOOGLE DRIVE
# ================================================================

from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/AraT5_GOA_MultiSeed"
os.makedirs(DRIVE_ROOT, exist_ok=True)

DATA_ARCHIVE = "/content/arabic_XLSum_v2.0.tar.bz2"
DATA_ARCHIVE_DRIVE = os.path.join(DRIVE_ROOT, "arabic_XLSum_v2.0.tar.bz2")
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

NUM_GOATS = 5
MAX_ITER = 20

ALPHA = 0.05
BETA = 0.5
JUMP_PROB = 0.1
REGENERATE_FRACTION = 0.2

BOUNDS = np.array([
    [128, 1024],      
    [1, 8],           
    [1e-5, 5e-4],     
    [0.0, 0.1],      
], dtype=float)

CLEANUP_CHECKPOINTS_AFTER_FINAL = False

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
print("AraT5v2 + XL-Sum — GOA MULTI-SEED (resumable)")
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
        "state_file": os.path.join(seed_dir, "goa_state.json"),
        "results_file": os.path.join(seed_dir, "goa_results.json"),
        "history_file": os.path.join(seed_dir, "goa_history.json"),
        "progress_file": os.path.join(seed_dir, "goa_progress.json"),
        "summary_file": os.path.join(seed_dir, "seed_summary.json"),
    }


HP_KEYS = [
    "max_input_length",
    "per_device_train_batch_size",
    "learning_rate",
    "weight_decay",
]


def hp_matches(a, b):
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
        if os.path.isfile(DATA_ARCHIVE_DRIVE):
            print("Copying dataset archive from Drive...")
            shutil.copy(DATA_ARCHIVE_DRIVE, DATA_ARCHIVE)
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
#  RECOVERY DATA + REPLAY CACHE
# Seed 43 was interrupted while evaluation 14 was running. Evaluations
# 1-13 finished, but only the first 5 were saved on Drive (goa_state.json).
# The values below were copied from the printed console output of the
# interrupted run. They are used ONLY when goa_results.json on Drive does
# not already contain at least as many evaluations.
#
# Every replayed evaluation is verified: the GOA path is regenerated from
# the seed and its hyperparameters must equal the recorded ones, otherwise
# the run stops (so Seeds 42/43/44 are guaranteed to follow the same path
# the original run would have followed).

RECOVERY = {
    43: [
        dict(evaluation=1,  max_input_length=712, per_device_train_batch_size=1, learning_rate=1.981449756836626e-05,   weight_decay=0.08392125825110301,    val_loss=2.3377466201782227, time_minutes=114.47),
        dict(evaluation=2,  max_input_length=654, per_device_train_batch_size=3, learning_rate=0.0003783782131911216,   weight_decay=0.026369219747837472,   val_loss=2.6051859855651855, time_minutes=39.69),
        dict(evaluation=3,  max_input_length=504, per_device_train_batch_size=4, learning_rate=0.00047810414381840643,  weight_decay=0.08919016691830428,    val_loss=2.7116539478302,    time_minutes=29.68),
        dict(evaluation=4,  max_input_length=378, per_device_train_batch_size=3, learning_rate=0.00021677979167515876,  weight_decay=0.00040777249859886845, val_loss=2.5118751525878906, time_minutes=39.71),
        dict(evaluation=5,  max_input_length=405, per_device_train_batch_size=8, learning_rate=0.0004207203413515549,   weight_decay=0.0535128257097604,     val_loss=2.5473058223724365, time_minutes=15.11),
        dict(evaluation=6,  max_input_length=781, per_device_train_batch_size=1, learning_rate=1e-05,                   weight_decay=0.08575233421314025,    val_loss=2.3768019676208496, time_minutes=113.45),
        dict(evaluation=7,  max_input_length=561, per_device_train_batch_size=3, learning_rate=0.00040243035666749134,  weight_decay=0.026719794901608036,   val_loss=2.6743545532226562, time_minutes=39.20),
        dict(evaluation=8,  max_input_length=505, per_device_train_batch_size=3, learning_rate=0.0004917090841145968,   weight_decay=0.09117162975453734,    val_loss=2.776052951812744,  time_minutes=39.13),
        dict(evaluation=9,  max_input_length=325, per_device_train_batch_size=4, learning_rate=0.00018679139475702325,  weight_decay=0.0,                    val_loss=2.463707447052002,  time_minutes=28.93),
        dict(evaluation=10, max_input_length=324, per_device_train_batch_size=8, learning_rate=0.0004289981869801588,   weight_decay=0.05571769526443362,    val_loss=2.6496150493621826, time_minutes=15.29),
        dict(evaluation=11, max_input_length=712, per_device_train_batch_size=1, learning_rate=1.981449756836626e-05,   weight_decay=0.08392125825110301,    val_loss=2.329533815383911,  time_minutes=115.15),
        dict(evaluation=12, max_input_length=683, per_device_train_batch_size=2, learning_rate=0.00019909635537974395,  weight_decay=0.05514523899947024,    val_loss=2.4879112243652344, time_minutes=58.16),
        dict(evaluation=13, max_input_length=608, per_device_train_batch_size=3, learning_rate=0.00024895932069338633,  weight_decay=0.08655571258470365,    val_loss=2.5330820083618164, time_minutes=39.27),
    ],
}

EXPECTED_HP = {
    43: {
        14: dict(
            max_input_length=519,
            per_device_train_batch_size=2,
            learning_rate=0.00010330294616269475,
            weight_decay=0.041960629125551506,
        ),
    },
}

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
        os.path.join(os.path.dirname(results_file), "goa_progress.json")
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
                "The GOA path diverged from the original run. Stop and check."
            )

        fitness_log.append(rec)
        atomic_json_dump(fitness_log, results_file)
        write_progress(seed, fitness_log, results_file)

        print(f"  REPLAY (no training) | val_loss = {float(rec['val_loss']):.6f}")
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
    print(f" Saved evaluation {evaluation_number}/{N_EVALUATIONS} to Drive")

    del trainer
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return float(val_loss)

def clip_population(population, lb, ub):
    return np.clip(population, lb, ub)


def goat_optimization(
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
    dimension = len(lb)

    population = lb + rng.random((NUM_GOATS, dimension)) * span
    fitness = np.full(NUM_GOATS, np.inf)

    best_position = None
    best_fitness = float("inf")
    history = []

    def evaluate(position):
        return run_fitness_evaluation(
            position, seed, len(fitness_log) + 1,
            fitness_log, trials_dir, results_file
        )

    def evaluate_batch(candidates):
        cf = np.full(NUM_GOATS, np.inf)
        for i in range(NUM_GOATS):
            if len(fitness_log) >= N_EVALUATIONS:
                break
            cf[i] = evaluate(candidates[i])
        return cf

    def update_best():
        nonlocal best_position, best_fitness
        finite = np.where(np.isfinite(fitness))[0]
        if len(finite) > 0:
            idx = finite[np.argmin(fitness[finite])]
            if fitness[idx] < best_fitness:
                best_fitness = float(fitness[idx])
                best_position = population[idx].copy()

    def accept_improvements(candidate, candidate_fitness):
        improve = np.isfinite(candidate_fitness) & (candidate_fitness < fitness)
        population[improve] = candidate[improve]
        fitness[improve] = candidate_fitness[improve]

    fitness[:] = evaluate_batch(population)
    update_best()

    history.append(float(best_fitness))

    atomic_json_dump(
        {
            "seed": seed,
            "population": population.tolist(),
            "fitness": fitness.tolist(),
            "best_position": None if best_position is None else best_position.tolist(),
            "best_fitness": None if not np.isfinite(best_fitness) else float(best_fitness),
            "history": history,
            "fitness_evaluations": len(fitness_log),
            "status": "running",
        },
        state_file
    )


    for iteration in range(MAX_ITER):

        if len(fitness_log) >= N_EVALUATIONS:
            break

        print(f"\nGOA iteration {iteration + 1}/{MAX_ITER}")

        candidate = clip_population(
            population + ALPHA * rng.normal(size=(NUM_GOATS, dimension)) * span,
            lb,
            ub
        )
        accept_improvements(candidate, evaluate_batch(candidate))
        update_best()
        if len(fitness_log) < N_EVALUATIONS and best_position is not None:
            candidate = clip_population(
                population + BETA * (best_position - population),
                lb,
                ub
            )
            accept_improvements(candidate, evaluate_batch(candidate))
        update_best()

        if len(fitness_log) < N_EVALUATIONS:
            candidate = population.copy()

            for i in range(NUM_GOATS):
                if len(fitness_log) >= N_EVALUATIONS:
                    break

                if rng.random() < JUMP_PROB:
                    choices = [j for j in range(NUM_GOATS) if j != i]
                    other = int(rng.choice(choices))
                    candidate[i] = (
                        population[i]
                        + JUMP_PROB * (population[other] - population[i])
                    )

            candidate = clip_population(candidate, lb, ub)
            accept_improvements(candidate, evaluate_batch(candidate))
        update_best()

        history.append(float(best_fitness))

        state = {
            "seed": seed,
            "iteration": iteration + 1,
            "population": population.tolist(),
            "fitness": fitness.tolist(),
            "best_position": None if best_position is None else best_position.tolist(),
            "best_fitness": float(best_fitness),
            "history": history,
            "fitness_evaluations": len(fitness_log),
            "status": "completed" if len(fitness_log) >= N_EVALUATIONS else "running",
        }

        atomic_json_dump(state, state_file)
        atomic_json_dump(fitness_log, results_file)
        atomic_json_dump(history, history_file)

        print(f"Best validation loss: {best_fitness:.6f}")
        print(f"Evaluations: {len(fitness_log)}/{N_EVALUATIONS}")

        if len(fitness_log) >= N_EVALUATIONS:
            print("\nFitness budget reached.")
            break

    return best_position, best_fitness, history, population, fitness

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

    REPLAY_CACHE[seed] = load_replay_cache(seed, p["results_file"])
    n_replay = len(REPLAY_CACHE[seed])

    if n_replay > 0:
        print(f"\nSeed {seed}: replaying {n_replay} finished evaluations (no training); "
              f"training resumes from evaluation {n_replay + 1}/{N_EVALUATIONS}.")
    else:
        print(f"\nSeed {seed}: starting from scratch.")

    fitness_log = []
    search_start = time.time()

    best_position, best_loss, history, population, fitness = goat_optimization(
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
print("MULTI-SEED GOA HPO SUMMARY")
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
    "optimizer": "GOA",
    "seeds": SEEDS,
    "evaluations_per_seed": N_EVALUATIONS,
    "total_fitness_evaluations": len(SEEDS) * N_EVALUATIONS,
    "trial_epochs": TRIAL_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "best_validation_loss_mean": mean_loss,
    "best_validation_loss_std": std_loss,
    "seed_results": all_seed_results,
}

MULTI_SEED_FILE = os.path.join(REPORT_ROOT, "GOA_multi_seed_HPO_summary.json")
atomic_json_dump(multi_seed_summary, MULTI_SEED_FILE)

print(f"\nMean validation loss: {mean_loss}")
print(f"Standard deviation:   {std_loss}")
print(f"Saved to: {MULTI_SEED_FILE}")

class EpochCheckpointCallback(TrainerCallback):
   

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
        print(f"\n Epoch {int(round(epoch))} checkpoint saved.")
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
    """Runs compute_metrics on a few validation labels (padded with -100,
    like generated outputs) so any metric error shows up in seconds."""
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
    """Latest checkpoint that is fully written (skips half-saved ones)."""
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
        print(f"\n🔄 Resuming from: {latest_checkpoint}")
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
        "optimizer": "GOA",
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

    _d = os.path.join(FINAL_ROOT, f"seed_{seed}")
    if not os.path.exists(os.path.join(_d, "final_report.json")):
        _ck = find_valid_checkpoint(_d)
        if _ck:
            _ts = load_json(os.path.join(_ck, "trainer_state.json"), {})
            print(f"\n[seed {seed}] will RESUME from {os.path.basename(_ck)} "
                  f"(epoch {_ts.get('epoch')}, step {_ts.get('global_step')})")
        else:
            print(f"\n[seed {seed}] no valid checkpoint, starting final training from scratch")

    result = train_final_model_for_seed(seed, summary["best_hyperparameters"])
    FINAL_SEED_RESULTS.append(result)

print("\n" + "=" * 75)
print("GOA MULTI-SEED FINAL TEST SUMMARY")
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
    "optimizer": "GOA",
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
    "goa_parameters": {
        "num_goats": NUM_GOATS,
        "max_iterations": MAX_ITER,
        "alpha": ALPHA,
        "beta": BETA,
        "jump_probability": JUMP_PROB,
        "regenerate_fraction": REGENERATE_FRACTION,
    },
    "hpo_best_val_loss_mean": mean_loss,
    "hpo_best_val_loss_std": std_loss,
    "final_test_mean_sd": final_mean_sd,
    "seed_results": FINAL_SEED_RESULTS,
}

COMPLETE_REPORT_FILE = os.path.join(REPORT_ROOT, "GOA_COMPLETE_MULTI_SEED_REPORT.json")
atomic_json_dump(complete_report, COMPLETE_REPORT_FILE)

print("\n" + "=" * 75)
print(" GOA EXPERIMENT COMPLETED")
print("=" * 75)
print(f"\nComplete report:\n{COMPLETE_REPORT_FILE}")
print(f"\nGoogle Drive folder:\n{DRIVE_ROOT}")
print("\nEverything is saved to Google Drive.")
