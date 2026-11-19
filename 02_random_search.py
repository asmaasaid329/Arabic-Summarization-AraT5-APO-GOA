# RANDOM SEARCH - 3 SEEDS
# Seeds: 42, 43, 44


import os
import tarfile
import json
import time
import random
import glob
import gc

import numpy as np
import torch

from datasets import Dataset, DatasetDict

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

import evaluate


# ===============================================================
# 1. GOOGLE DRIVE
# ===============================================================

from google.colab import drive

drive.mount("/content/drive")


DRIVE_ROOT = "/content/drive/MyDrive/AraT5_RandomSearch"

os.makedirs(DRIVE_ROOT, exist_ok=True)

SEEDS = [42, 43, 44]


print("=" * 80)
print(" RESUMABLE AraT5 RANDOM SEARCH - MULTI SEED")
print("=" * 80)

print(f"Drive folder:\n{DRIVE_ROOT}")

print(f"\nSeeds: {SEEDS}")


DATA_ARCHIVE = "/content/arabic_XLSum_v2.0.tar.bz2"

EXTRACT_DIR = "/content/arabic_XLSum"

MODEL_NAME = "UBC-NLP/AraT5v2-base-1024"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# RANDOM SEARCH
# ---------------------------------------------------------------

N_TRIALS = 15

TRIAL_EPOCHS = 1
FINAL_EPOCHS = 10

MAX_INPUT_TOKENIZE_LEN = 1024

MAX_TARGET_LEN = 128
FINAL_NUM_BEAMS = 5

SEARCH_SPACE = {

    "max_input_length": (128, 1024),

    "batch_size": (1, 8),

    "learning_rate": (1e-5, 5e-4),

    "weight_decay": (0.0, 0.1)
}


print("\nConfiguration:")

print(f"Device:              {DEVICE}")

print(f"Seeds:               {SEEDS}")

print(f"Trials per seed:     {N_TRIALS}")

print(
    f"Total HPO trials:    "
    f"{N_TRIALS * len(SEEDS)}"
)

print(f"Trial epochs:        {TRIAL_EPOCHS}")

print(f"Final epochs:        {FINAL_EPOCHS}")

print(f"Final num beams:     {FINAL_NUM_BEAMS}")

print(
    f"Max input tokenize:  "
    f"{MAX_INPUT_TOKENIZE_LEN}"
)

print(
    f"Max target length:   "
    f"{MAX_TARGET_LEN}"
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


if not os.path.exists(EXTRACT_DIR):

    print("\n Extracting dataset...")

    with tarfile.open(
        DATA_ARCHIVE,
        "r:bz2"
    ) as tar:

        tar.extractall(
            path=EXTRACT_DIR
        )

    print("✅ Extraction complete.")

else:

    print(
        "\n✅ Dataset already extracted."
    )


def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(
                json.loads(line)
            )

    return Dataset.from_list(data)


train_path = os.path.join(
    EXTRACT_DIR,
    "arabic_train.jsonl"
)

val_path = os.path.join(
    EXTRACT_DIR,
    "arabic_val.jsonl"
)

test_path = os.path.join(
    EXTRACT_DIR,
    "arabic_test.jsonl"
)


print("\n📂 Loading dataset...")

train_ds = load_jsonl(train_path)

val_ds = load_jsonl(val_path)

test_ds = load_jsonl(test_path)


dataset = DatasetDict({

    "train": train_ds,

    "validation": val_ds,

    "test": test_ds
})


print("\nDataset sizes:")

for split in dataset:

    print(
        f"   {split}: "
        f"{len(dataset[split])}"
    )

print("\n Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(

    MODEL_NAME,

    use_fast=True
)

print("✅ Tokenizer loaded.")

print(
    f"Tokenizer vocab size: "
    f"{len(tokenizer)}"
)

print(
    f"PAD token ID: "
    f"{tokenizer.pad_token_id}"
)

print(
    f"EOS token ID: "
    f"{tokenizer.eos_token_id}"
)


def preprocess_full(batch):

    inputs = [

        "summarize: " + str(text)

        for text in batch["text"]
    ]

    model_inputs = tokenizer(

        inputs,

        max_length=MAX_INPUT_TOKENIZE_LEN,

        truncation=True
    )

    labels = tokenizer(

        text_target=batch["summary"],

        max_length=MAX_TARGET_LEN,

        truncation=True
    )

    labels_ids = []

    for lab in labels["input_ids"]:

        labels_ids.append([

            token
            if token != tokenizer.pad_token_id
            else -100

            for token in lab
        ])

    model_inputs["labels"] = labels_ids

    return model_inputs


print("\n Tokenizing dataset...")

tokenized = dataset.map(

    preprocess_full,

    batched=True,

    remove_columns=dataset["train"].column_names
)

print("✅ Tokenization complete.")


def sample_random_hyperparameters(rng):

    max_input_length = rng.randint(

        SEARCH_SPACE["max_input_length"][0],

        SEARCH_SPACE["max_input_length"][1]
    )

    batch_size = rng.randint(

        SEARCH_SPACE["batch_size"][0],

        SEARCH_SPACE["batch_size"][1]
    )

    learning_rate = rng.uniform(

        SEARCH_SPACE["learning_rate"][0],

        SEARCH_SPACE["learning_rate"][1]
    )

    weight_decay = rng.uniform(

        SEARCH_SPACE["weight_decay"][0],

        SEARCH_SPACE["weight_decay"][1]
    )

    return {

        "max_input_length":
            int(max_input_length),

        "per_device_train_batch_size":
            int(batch_size),

        "learning_rate":
            float(learning_rate),

        "weight_decay":
            float(weight_decay),

        "num_train_epochs":
            TRIAL_EPOCHS
    }

def truncate_dataset(
    tokenized_split,
    max_length
):

    def trunc_map(batch):

        return {

            "input_ids": [

                ids[:max_length]

                for ids in batch["input_ids"]
            ],

            "attention_mask": [

                mask[:max_length]

                for mask in batch["attention_mask"]
            ],

            "labels":
                batch["labels"]
        }

    return tokenized_split.map(

        trunc_map,

        batched=True,

        remove_columns=
            tokenized_split.column_names
    )



def cleanup_model(
    model=None,
    trainer=None
):

    try:

        if trainer is not None:

            del trainer

    except Exception:

        pass

    try:

        if model is not None:

            del model

    except Exception:

        pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


print("\n Loading evaluation metrics...")

rouge = evaluate.load("rouge")

bleu = evaluate.load("bleu")

meteor = evaluate.load("meteor")

bertscore = evaluate.load("bertscore")

print("✅ Metrics loaded.")


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

  
    if isinstance(
        predictions,
        tuple
    ):

        predictions = predictions[0]

    predictions = np.asarray(
        predictions
    )

  
    if predictions.ndim == 3:

        print(
            f" Predictions are logits: "
            f"{predictions.shape}"
        )

        predictions = np.argmax(
            predictions,
            axis=-1
        )


    try:

        predictions = predictions.astype(
            np.int64,
            copy=False
        )

    except Exception as e:

        print(
            f" Prediction conversion failed: "
            f"{e}"
        )

        return {

            "rouge_rouge1": 0.0,

            "rouge_rouge2": 0.0,

            "rouge_rougeL": 0.0,

            "rouge_rougeLsum": 0.0,

            "BLEU": 0.0,

            "METEOR": 0.0,

            "BERTScore_F1": 0.0
        }

    # -----------------------------------------------------------
    # Tokenizer IDs
    # -----------------------------------------------------------

    vocab_size = len(tokenizer)

    pad_id = tokenizer.pad_token_id

    if pad_id is None:

        pad_id = 0

    invalid_mask = (

        (predictions < 0)
        |
        (predictions >= vocab_size)
    )

    invalid_count = int(
        np.sum(invalid_mask)
    )

    if invalid_count > 0:

        print(
            f" Found {invalid_count} "
            f"invalid prediction IDs."
        )

        predictions[
            invalid_mask
        ] = pad_id


    labels = np.asarray(labels)

    labels = labels.astype(
        np.int64,
        copy=False
    )

    labels = np.where(

        labels == -100,

        pad_id,

        labels
    )

   

    invalid_label_mask = (

        (labels < 0)
        |
        (labels >= vocab_size)
    )

    invalid_label_count = int(
        np.sum(invalid_label_mask)
    )

    if invalid_label_count > 0:

        print(
            f" Found {invalid_label_count} "
            f"invalid label IDs."
        )

        labels[
            invalid_label_mask
        ] = pad_id

  

    try:

        decoded_preds = tokenizer.batch_decode(

            predictions.tolist(),

            skip_special_tokens=True
        )

        decoded_labels = tokenizer.batch_decode(

            labels.tolist(),

            skip_special_tokens=True
        )

    except Exception as e:

        print(
            "\n Tokenizer decode error:"
        )

        print(
            repr(e)
        )

        return {

            "rouge_rouge1": 0.0,

            "rouge_rouge2": 0.0,

            "rouge_rougeL": 0.0,

            "rouge_rougeLsum": 0.0,

            "BLEU": 0.0,

            "METEOR": 0.0,

            "BERTScore_F1": 0.0
        }

  

    decoded_preds = [

        str(x).strip()

        for x in decoded_preds
    ]

    decoded_labels = [

        str(x).strip()

        for x in decoded_labels
    ]

    results = {}

   
    try:

        rouge_result = rouge.compute(

            predictions=decoded_preds,

            references=decoded_labels,

            use_stemmer=True
        )

        for key, value in rouge_result.items():

            results[
                f"rouge_{key}"
            ] = round(

                float(value) * 100,

                4
            )

    except Exception as e:

        print(
            f"⚠️ ROUGE error: {e}"
        )

        results[
            "rouge_rouge1"
        ] = 0.0

        results[
            "rouge_rouge2"
        ] = 0.0

        results[
            "rouge_rougeL"
        ] = 0.0

        results[
            "rouge_rougeLsum"
        ] = 0.0

   

    try:

        bleu_result = bleu.compute(

            predictions=decoded_preds,

            references=[
                [ref]
                for ref in decoded_labels
            ]
        )

        results["BLEU"] = round(

            float(
                bleu_result["bleu"]
            ) * 100,

            4
        )

    except Exception as e:

        print(
            f" BLEU error: {e}"
        )

        results["BLEU"] = 0.0

  

    try:

        meteor_result = meteor.compute(

            predictions=decoded_preds,

            references=decoded_labels
        )

        results["METEOR"] = round(

            float(
                meteor_result["meteor"]
            ) * 100,

            4
        )

    except Exception as e:

        print(
            f" METEOR error: {e}"
        )

        results["METEOR"] = 0.0

  
    try:

        bert_result = bertscore.compute(

            predictions=decoded_preds,

            references=decoded_labels,

            lang="ar"
        )

        results["BERTScore_F1"] = round(

            float(
                np.mean(
                    bert_result["f1"]
                )
            ) * 100,

            4
        )

    except Exception as e:

        print(
            f" BERTScore error: {e}"
        )

        results[
            "BERTScore_F1"
        ] = 0.0

    return results




def save_seed_state(
    seed_dir,
    state
):

    state_file = os.path.join(

        seed_dir,

        "random_search_state.json"
    )

    results_file = os.path.join(

        seed_dir,

        "random_search_results.json"
    )

    temp_file = state_file + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            state,

            f,

            ensure_ascii=False,

            indent=2
        )

    os.replace(
        temp_file,
        state_file
    )

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            state,

            f,

            ensure_ascii=False,

            indent=2
        )

    print(
        " Seed state saved."
    )



def objective_function(
    hp,
    trial_number,
    seed,
    trials_dir
):

    print("\n" + "=" * 80)

    print(
        f" RANDOM SEARCH "
        f"SEED {seed} - "
        f"TRIAL {trial_number}/{N_TRIALS}"
    )

    print("=" * 80)

    print("\nHyperparameters:")

    for key, value in hp.items():

        print(
            f"   {key}: {value}"
        )

    start_time = time.time()

    model = None

    trainer = None

    try:

        print(
            "\n Loading fresh model..."
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(

            MODEL_NAME
        )

        model.to(DEVICE)

        
        train_tok = truncate_dataset(

            tokenized["train"],

            hp["max_input_length"]
        )

        val_tok = truncate_dataset(

            tokenized["validation"],

            hp["max_input_length"]
        )

       

        data_collator = DataCollatorForSeq2Seq(

            tokenizer=tokenizer,

            model=model,

            padding=True
        )

       
        trial_seed = seed + trial_number

        set_seed(
            trial_seed
        )

       
        trial_output = os.path.join(

            trials_dir,

            f"trial_{trial_number}"
        )

        os.makedirs(

            trial_output,

            exist_ok=True
        )

        
        training_args = Seq2SeqTrainingArguments(

            output_dir=trial_output,

            eval_strategy="epoch",

            logging_strategy="epoch",

            save_strategy="no",

            learning_rate=hp["learning_rate"],

            per_device_train_batch_size=(
                hp[
                    "per_device_train_batch_size"
                ]
            ),

            per_device_eval_batch_size=(
                hp[
                    "per_device_train_batch_size"
                ]
            ),

            weight_decay=hp["weight_decay"],

            num_train_epochs=TRIAL_EPOCHS,

            predict_with_generate=False,

            fp16=(

                torch.cuda.is_available()

                and
                torch.cuda.get_device_capability(0)[0] >= 7
            ),

            dataloader_num_workers=2,

            report_to="none",

            seed=trial_seed,

            data_seed=trial_seed
        )

       
        trainer = Seq2SeqTrainer(

            model=model,

            args=training_args,

            train_dataset=train_tok,

            eval_dataset=val_tok,

            processing_class=tokenizer,

            data_collator=data_collator
        )

        
        print(
            "\n Trial training started..."
        )

        trainer.train()

      

        print(
            "\n Running validation..."
        )

        eval_metrics = trainer.evaluate()

        val_loss = float(

            eval_metrics.get(
                "eval_loss",
                np.inf
            )
        )

        elapsed = (

            time.time() - start_time

        ) / 60

        print(
            "\n✅ Trial completed."
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Time: "
            f"{elapsed:.2f} minutes"
        )

        return val_loss

    except Exception as e:

        print(
            "\n Trial failed."
        )

        print(
            repr(e)
        )

        return np.inf

    finally:

        cleanup_model(

            model=model,

            trainer=trainer
        )



def run_seed(seed):

    print("\n\n")

    print("#" * 80)

    print(
        f" STARTING SEED {seed}"
    )

    print("#" * 80)

  
    set_seed(seed)

  
    seed_dir = os.path.join(

        DRIVE_ROOT,

        f"seed_{seed}"
    )

    trials_dir = os.path.join(

        seed_dir,

        "trials"
    )

    final_output_dir = os.path.join(

        seed_dir,

        "final_training"
    )

    final_model_dir = os.path.join(

        seed_dir,

        "final_model"
    )

    os.makedirs(
        seed_dir,
        exist_ok=True
    )

    os.makedirs(
        trials_dir,
        exist_ok=True
    )

    os.makedirs(
        final_output_dir,
        exist_ok=True
    )

    os.makedirs(
        final_model_dir,
        exist_ok=True
    )

    state_file = os.path.join(

        seed_dir,

        "random_search_state.json"
    )

  
    if os.path.exists(state_file):

        print(
            f"\n Previous state found "
            f"for Seed {seed}."
        )

        with open(
            state_file,
            "r",
            encoding="utf-8"
        ) as f:

            state = json.load(f)

        results = state.get(
            "results",
            []
        )

        best_loss = state.get(
            "best_loss",
            None
        )

        if best_loss is None:

            best_loss = float("inf")

        best_hp = state.get(
            "best_hp",
            None
        )

        completed_trials = state.get(
            "completed_trials",
            []
        )

        print(
            f"Completed trials: "
            f"{completed_trials}"
        )

        print(
            f"Best loss: "
            f"{best_loss}"
        )

    else:

        print(
            f"\n No previous state "
            f"for Seed {seed}."
        )

        results = []

        best_loss = float("inf")

        best_hp = None

        completed_trials = []

   

    rng = random.Random(seed)

    all_hyperparameters = {}

    for trial_number in range(

        1,

        N_TRIALS + 1

    ):

        all_hyperparameters[
            trial_number
        ] = sample_random_hyperparameters(
            rng
        )

    
    search_config_file = os.path.join(

        seed_dir,

        "sampled_hyperparameters.json"
    )

    with open(
        search_config_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            {
                str(k): v
                for k, v
                in all_hyperparameters.items()
            },

            f,

            ensure_ascii=False,

            indent=2
        )

    

    if completed_trials:

        start_trial = (

            max(completed_trials) + 1

        )

    else:

        start_trial = 1

    print(
        f"\n Starting from Trial "
        f"{start_trial}"
    )

    print(
        f" Completed trials: "
        f"{completed_trials}"
    )

  

    for trial in range(

        start_trial,

        N_TRIALS + 1

    ):

        print("\n" + "=" * 80)

        print(
            f" SEED {seed} - "
            f"TRIAL {trial}/{N_TRIALS}"
        )

        print("=" * 80)

      

        hp = all_hyperparameters[
            trial
        ]

       
        val_loss = objective_function(

            hp,

            trial,

            seed,

            trials_dir
        )

       

        if not np.isfinite(
            val_loss
        ):

            print(
                "\n Trial failed."
            )

            print(
                "This trial was NOT "
                "marked as completed."
            )

            print(
                "The search will stop here."
            )

            break

       

        result = {

            "seed": seed,

            "trial": trial,

            **hp,

            "val_loss":
                float(val_loss)
        }

       

        results = [

            r

            for r in results

            if not (

                r.get("trial") == trial

                and
                r.get("seed") == seed
            )
        ]

        results.append(
            result
        )

      
        if trial not in completed_trials:

            completed_trials.append(
                trial
            )

        completed_trials.sort()

       
        if val_loss < best_loss:

            best_loss = val_loss

            best_hp = hp.copy()

            print(
                "\n🏆 NEW BEST "
                "CONFIGURATION"
            )

            for key, value in best_hp.items():

                print(
                    f"   {key}: {value}"
                )

            print(
                f"   val_loss: "
                f"{best_loss:.6f}"
            )

        
        state = {

            "seed": seed,

            "n_trials":
                N_TRIALS,

            "trial_epochs":
                TRIAL_EPOCHS,

            "final_epochs":
                FINAL_EPOCHS,

            "completed_trials":
                completed_trials,

            "best_loss": (

                None

                if best_loss == float("inf")

                else float(best_loss)
            ),

            "best_hp":
                best_hp,

            "results":
                results
        }

        save_seed_state(

            seed_dir,

            state
        )

        print(
            "\n💾 Trial safely saved."
        )

        print(
            f"Completed: "
            f"{len(completed_trials)}/"
            f"{N_TRIALS}"
        )

    

    if len(completed_trials) < N_TRIALS:

        print("\n" + "=" * 80)

        print(
            f" RANDOM SEARCH PAUSED "
            f"FOR SEED {seed}"
        )

        print("=" * 80)

        print(
            "Run the notebook again "
            "to resume."
        )

        return None

   

    print("\n" + "=" * 80)

    print(
        f" RANDOM SEARCH FINISHED "
        f"FOR SEED {seed}"
    )

    print("=" * 80)

    print(
        f"\nBest validation loss: "
        f"{best_loss:.6f}"
    )

    print(
        "\n BEST PARAMETERS"
    )

    for key, value in best_hp.items():

        print(
            f"   {key}: {value}"
        )

   

    print("\n" + "=" * 80)

    print(
        f" FINAL TRAINING "
        f"FOR SEED {seed}"
    )

    print("=" * 80)

    final_hp = best_hp.copy()

    final_hp[
        "num_train_epochs"
    ] = FINAL_EPOCHS

    final_hp[
        "num_beams"
    ] = FINAL_NUM_BEAMS

    print(
        "\nFinal hyperparameters:"
    )

    for key, value in final_hp.items():

        print(
            f"   {key}: {value}"
        )

   

    seed_test_metrics_file = os.path.join(

        seed_dir,

        "test_metrics.json"
    )

    examples_file = os.path.join(

        seed_dir,

        "three_examples.json"
    )

    if (

        os.path.exists(
            seed_test_metrics_file
        )

        and

        os.path.exists(
            examples_file
        )

    ):

        print(
            "\n✅ Final results "
            "already exist."
        )

        with open(
            seed_test_metrics_file,
            "r",
            encoding="utf-8"
        ) as f:

            test_metrics = json.load(f)

        return {

            "seed": seed,

            "best_hp":
                best_hp,

            "best_val_loss":
                best_loss,

            "test_metrics":
                test_metrics,

            "seed_dir":
                seed_dir
        }

    

    set_seed(seed)

    print(
        "\n Loading fresh "
        "AraT5v2 model..."
    )

    final_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )
    )

    final_model.to(
        DEVICE
    )

    train_final = truncate_dataset(

        tokenized["train"],

        final_hp[
            "max_input_length"
        ]
    )

    val_final = truncate_dataset(

        tokenized["validation"],

        final_hp[
            "max_input_length"
        ]
    )

    test_final = truncate_dataset(

        tokenized["test"],

        final_hp[
            "max_input_length"
        ]
    )


    data_collator = DataCollatorForSeq2Seq(

        tokenizer=tokenizer,

        model=final_model,

        padding=True
    )

    os.environ[
        "WANDB_DISABLED"
    ] = "true"

    training_args_final = (
        Seq2SeqTrainingArguments(

            output_dir=
                final_output_dir,

            eval_strategy="epoch",

            logging_strategy="epoch",

            save_strategy="epoch",

            learning_rate=
                final_hp[
                    "learning_rate"
                ],

            per_device_train_batch_size=
                final_hp[
                    "per_device_train_batch_size"
                ],

            per_device_eval_batch_size=
                final_hp[
                    "per_device_train_batch_size"
                ],

            weight_decay=
                final_hp[
                    "weight_decay"
                ],

            num_train_epochs=
                FINAL_EPOCHS,

            gradient_accumulation_steps=2,

            predict_with_generate=True,

            generation_num_beams=
                int(FINAL_NUM_BEAMS),

            generation_max_length=
                MAX_TARGET_LEN,

            fp16=(

                torch.cuda.is_available()

                and
                torch.cuda.get_device_capability(0)[0] >= 7
            ),

            save_total_limit=2,

            load_best_model_at_end=False,

            metric_for_best_model=
                "eval_rouge_rouge1",

            greater_is_better=True,

            report_to="none",

            seed=seed,

            data_seed=seed,

            dataloader_num_workers=2,

            remove_unused_columns=True
        )
    )

  
    trainer_final = Seq2SeqTrainer(

        model=final_model,

        args=training_args_final,

        train_dataset=train_final,

        eval_dataset=val_final,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )


    checkpoint_dirs = glob.glob(

        os.path.join(

            final_output_dir,

            "checkpoint-*"
        )
    )

    checkpoint_dirs = [

        x

        for x in checkpoint_dirs

        if os.path.isdir(x)

        and (

            os.path.exists(
                os.path.join(
                    x,
                    "model.safetensors"
                )
            )

            or

            os.path.exists(
                os.path.join(
                    x,
                    "pytorch_model.bin"
                )
            )
        )
    ]

    def checkpoint_number(path):

        try:

            return int(

                os.path.basename(
                    path
                ).split("-")[-1]

            )

        except Exception:

            return -1

    checkpoint_dirs.sort(
        key=checkpoint_number
    )

   

    if checkpoint_dirs:

        latest_checkpoint = (
            checkpoint_dirs[-1]
        )

        print(
            "\n VALID CHECKPOINT FOUND"
        )

        print(
            latest_checkpoint
        )

        print(
            "\n Resuming final training..."
        )

        final_train_result = (
            trainer_final.train(

                resume_from_checkpoint=
                    latest_checkpoint
            )
        )

    else:

        print(
            "\n🚀 STARTING FINAL TRAINING"
        )

        final_train_result = (
            trainer_final.train()
        )

   

    trainer_final.save_model(
        final_model_dir
    )

    tokenizer.save_pretrained(
        final_model_dir
    )

    trainer_final.save_metrics(

        "train",

        final_train_result.metrics
    )

    print(
        "\n✅ Final model saved to:"
    )

    print(
        final_model_dir
    )


    print("\n" + "=" * 80)

    print(
        f" VALIDATION RESULTS "
        f"- SEED {seed}"
    )

    print("=" * 80)

    validation_results = (
        trainer_final.evaluate(
            eval_dataset=val_final
        )
    )

    for key, value in validation_results.items():

        if isinstance(
            value,
            (float, int)
        ):

            print(
                f"{key}: {value}"
            )

    print("\n" + "=" * 80)

    print(
        f" FINAL TEST EVALUATION "
        f"- SEED {seed}"
    )

    print("=" * 80)

    test_prediction = (
        trainer_final.predict(

            test_final,

            max_length=
                MAX_TARGET_LEN,

            num_beams=
                FINAL_NUM_BEAMS
        )
    )

    test_metrics = compute_metrics(

        (

            test_prediction.predictions,

            test_prediction.label_ids
        )
    )

    print(
        "\n TEST RESULTS"
    )

    for key, value in test_metrics.items():

        print(
            f"{key:<25}: "
            f"{value}"
        )

    # ===========================================================
    # 3 EXAMPLES
    # ===========================================================

    print("\n" + "=" * 80)

    print(
        f"📝 3 EXAMPLES - SEED {seed}"
    )

    print("=" * 80)

    num_examples = min(
        3,
        len(dataset["test"])
    )

    example_dataset = (
        dataset["test"].select(
            range(num_examples)
        )
    )

    # -----------------------------------------------------------
    # Original
    # -----------------------------------------------------------

    original_texts = [

        str(x)

        for x in example_dataset["text"]
    ]

    # -----------------------------------------------------------
    # Reference
    # -----------------------------------------------------------

    reference_summaries = [

        str(x)

        for x in example_dataset["summary"]
    ]


    example_inputs = [

        "summarize: " + text

        for text in original_texts
    ]

    example_tokens = tokenizer(

        example_inputs,

        max_length=
            final_hp[
                "max_input_length"
            ],

        truncation=True,

        padding=True,

        return_tensors="pt"
    )

    example_tokens = {

        key: value.to(DEVICE)

        for key, value
        in example_tokens.items()
    }

    # -----------------------------------------------------------
    # Generate summaries
    # -----------------------------------------------------------

    final_model.eval()

    with torch.no_grad():

        generated_ids = (
            final_model.generate(

                input_ids=
                    example_tokens[
                        "input_ids"
                    ],

                attention_mask=
                    example_tokens[
                        "attention_mask"
                    ],

                max_length=
                    MAX_TARGET_LEN,

                num_beams=
                    FINAL_NUM_BEAMS,

                do_sample=False
            )
        )

    generated_summaries = (
        tokenizer.batch_decode(

            generated_ids,

            skip_special_tokens=True
        )
    )

    generated_summaries = [

        str(x).strip()

        for x in generated_summaries
    ]


    examples_output = []

    for i in range(
        num_examples
    ):

        example = {

            "example_number":
                i + 1,

            "original":
                original_texts[i],

            "reference_summary":
                reference_summaries[i],

            "generated_summary":
                generated_summaries[i]
        }

        examples_output.append(
            example
        )

        print("\n" + "-" * 80)

        print(
            f" EXAMPLE {i + 1}"
        )

        print("-" * 80)

        print(
            "\n ORIGINAL:\n"
        )

        print(
            original_texts[i]
        )

        print(
            "\n REFERENCE SUMMARY:\n"
        )

        print(
            reference_summaries[i]
        )

        print(
            "\n GENERATED SUMMARY:\n"
        )

        print(
            generated_summaries[i]
        )

        print(
            "\n" + "-" * 80
        )

   
    with open(
        examples_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            examples_output,

            f,

            ensure_ascii=False,

            indent=2
        )

    print(
        "\n 3 examples saved to:"
    )

    print(
        examples_file
    )

   

    with open(
        seed_test_metrics_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            test_metrics,

            f,

            ensure_ascii=False,

            indent=2
        )

    print(
        "\n💾 Test metrics saved:"
    )

    print(
        seed_test_metrics_file
    )

   
    seed_summary = {

        "seed":
            seed,

        "best_validation_loss":
            float(best_loss),

        "best_hyperparameters":
            best_hp,

        "final_epochs":
            FINAL_EPOCHS,

        "final_num_beams":
            FINAL_NUM_BEAMS,

        "test_metrics":
            test_metrics,

        "three_examples":
            examples_output
    }

    seed_summary_file = os.path.join(

        seed_dir,

        "seed_summary.json"
    )

    with open(
        seed_summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            seed_summary,

            f,

            ensure_ascii=False,

            indent=2
        )

    cleanup_model(

        model=final_model,

        trainer=trainer_final
    )


    return {

        "seed":
            seed,

        "best_hp":
            best_hp,

        "best_val_loss":
            best_loss,

        "test_metrics":
            test_metrics,

        "seed_dir":
            seed_dir
    }


print("\n\n")

print("#" * 80)

print(
    " MULTI-SEED RANDOM SEARCH"
)

print("#" * 80)

print(
    f"\nSeeds: {SEEDS}"
)

print(
    f"Trials per seed: "
    f"{N_TRIALS}"
)

print(
    f"Total HPO evaluations: "
    f"{N_TRIALS * len(SEEDS)}"
)

print(
    f"Final epochs per seed: "
    f"{FINAL_EPOCHS}"
)


all_seed_results = []

for seed in SEEDS:

    seed_result = run_seed(
        seed
    )

    if seed_result is None:

        print("\n" + "=" * 80)

        print(
            f" Seed {seed} "
            f"is not finished."
        )

        print(
            "Stopping here."
        )

        print(
            "Run the notebook again "
            "to resume."
        )

        print("=" * 80)

        break

    all_seed_results.append(
        seed_result
    )

if len(all_seed_results) < len(SEEDS):

    print("\n" + "=" * 80)

    print(
        " EXPERIMENT NOT YET FINISHED"
    )

    print("=" * 80)

    print(
        f"Finished seeds: "
        f"{len(all_seed_results)}/"
        f"{len(SEEDS)}"
    )

    print(
        "\nRun the notebook again "
        "to resume."
    )

else:

    metric_names = [

        "rouge_rouge1",

        "rouge_rouge2",

        "rouge_rougeL",

        "rouge_rougeLsum",

        "BLEU",

        "METEOR",

        "BERTScore_F1"
    ]

    aggregated_results = {}

    
    for metric in metric_names:

        values = [

            float(
                result[
                    "test_metrics"
                ].get(
                    metric,
                    np.nan
                )
            )

            for result
            in all_seed_results
        ]

        values = np.asarray(
            values,
            dtype=np.float64
        )

        mean_value = float(
            np.mean(values)
        )

        std_value = float(
            np.std(
                values,
                ddof=1
            )
        )

        aggregated_results[metric] = {

            "values":
                values.tolist(),

            "mean":
                mean_value,

            "std":
                std_value,

            "mean_plus_minus_std":
                (
                    f"{mean_value:.4f} ± "
                    f"{std_value:.4f}"
                )
        }

    print("\n" + "=" * 80)

    print(
        " RESULTS PER SEED"
    )

    print("=" * 80)

    for result in all_seed_results:

        seed = result["seed"]

        print(
            f"\n SEED {seed}"
        )

        print("-" * 60)

        print(
            "Best Validation Loss: "
            f"{result['best_val_loss']:.6f}"
        )

        print(
            "\nBest Hyperparameters:"
        )

        for key, value in result[
            "best_hp"
        ].items():

            print(
                f"   {key}: {value}"
            )

        print(
            "\nTest Metrics:"
        )

        for metric, value in result[
            "test_metrics"
        ].items():

            print(
                f"   {metric:<25}: "
                f"{value:.4f}"
            )


    print("\n\n" + "=" * 80)

    print(
        " FINAL RESULTS: MEAN ± STD"
    )

    print("=" * 80)

    for metric in metric_names:

        mean_value = (
            aggregated_results[
                metric
            ]["mean"]
        )

        std_value = (
            aggregated_results[
                metric
            ]["std"]
        )

        print(

            f"{metric:<25}: "
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )

    multi_seed_results = {

        "experiment":
            "AraT5v2 + XLSum Random Search",

        "seeds":
            SEEDS,

        "n_trials_per_seed":
            N_TRIALS,

        "total_hpo_trials":
            N_TRIALS * len(SEEDS),

        "trial_epochs":
            TRIAL_EPOCHS,

        "final_epochs":
            FINAL_EPOCHS,

        "final_num_beams":
            FINAL_NUM_BEAMS,

        "search_space":
            SEARCH_SPACE,

        "seed_results":
            all_seed_results,

        "aggregated_test_results":
            aggregated_results
    }

    multi_seed_results_file = os.path.join(

        DRIVE_ROOT,

        "multi_seed_final_results.json"
    )

    with open(
        multi_seed_results_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            multi_seed_results,

            f,

            ensure_ascii=False,

            indent=2
        )

    print(
        "\n Multi-seed results saved:"
    )

    print(
        multi_seed_results_file
    )

    summary_rows = []

    for result in all_seed_results:

        row = {

            "seed":
                result["seed"],

            "best_val_loss":
                result["best_val_loss"]
        }

        for metric in metric_names:

            row[metric] = (

                result[
                    "test_metrics"
                ].get(
                    metric,
                    None
                )
            )

        summary_rows.append(
            row
        )

    summary_file = os.path.join(

        DRIVE_ROOT,

        "per_seed_metrics.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            summary_rows,

            f,

            ensure_ascii=False,

            indent=2
        )


    print("\n\n" + "#" * 80)

    print(
        " EXPERIMENT COMPLETED"
    )

    print("#" * 80)

    print(
        "\nSeeds:"
    )

    print(
        SEEDS
    )

    print(
        "\nTrials per seed:"
    )

    print(
        N_TRIALS
    )

    print(
        "\nTotal HPO evaluations:"
    )

    print(
        N_TRIALS * len(SEEDS)
    )

    print(
        "\nTrial epochs:"
    )

    print(
        TRIAL_EPOCHS
    )

    print(
        "\nFinal epochs:"
    )

    print(
        FINAL_EPOCHS
    )

    print(
        "\nFinal num_beams:"
    )

    print(
        FINAL_NUM_BEAMS
    )

    print(
        "\n" + "=" * 80
    )

    print(
        " FINAL TEST RESULTS "
        "(MEAN ± STD)"
    )

    print("=" * 80)

    for metric in metric_names:

        mean_value = (
            aggregated_results[
                metric
            ]["mean"]
        )

        std_value = (
            aggregated_results[
                metric
            ]["std"]
        )

        print(

            f"{metric:<25}: "
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )

    
    print("\n\n" + "=" * 80)

    print(
        "📝 FINAL 3 EXAMPLES"
    )

    print("=" * 80)

    final_example_seed = 44

    final_examples_file = os.path.join(

        DRIVE_ROOT,

        f"seed_{final_example_seed}",

        "three_examples.json"
    )

    if os.path.exists(
        final_examples_file
    ):

        with open(
            final_examples_file,
            "r",
            encoding="utf-8"
        ) as f:

            final_examples = json.load(f)

        for example in final_examples:

            print(
                "\n" + "-" * 80
            )

            print(
                f" EXAMPLE "
                f"{example['example_number']}"
            )

            print(
                "-" * 80
            )

            print(
                "\n ORIGINAL:\n"
            )

            print(
                example["original"]
            )

            print(
                "\n REFERENCE SUMMARY:\n"
            )

            print(
                example["reference_summary"]
            )

            print(
                "\n GENERATED SUMMARY:\n"
            )

            print(
                example["generated_summary"]
            )

            print(
                "\n" + "-" * 80
            )

    else:

        print(
            " Saved examples file "
            "was not found."
        )

    # ===========================================================
    # 26. FINAL PATHS
    # ===========================================================

    print("\n" + "=" * 80)

    print(
        " GOOGLE DRIVE"
    )

    print("=" * 80)

    print(
        DRIVE_ROOT
    )

    print(
        "\n Complete multi-seed results:"
    )

    print(
        multi_seed_results_file
    )

    print(
        "\n Per-seed metrics:"
    )

    print(
        summary_file
    )

    print(
        "\n Final 3 examples:"
    )

    print(
        final_examples_file
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ ALL 3 SEEDS COMPLETED"
    )

    print("=" * 80)

