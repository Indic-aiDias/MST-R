import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pandas as pd
from tqdm.autonotebook import tqdm
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

from datasets import Dataset

import evaluate
from torch import nn

import mlflow

# COMMAND ----------

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RUN_NAME = "ms-marco-MiniLM-L-6-v2-3aglo-frozen-binary-classification"
model_save_path = f"regnlp/{RUN_NAME}"

BATCH_SIZE = 8
LR = 2e-5
TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
# STEPS=1
STEPS=20000
WARMUP_RATIO=0.1
RECALL_AT_K = 10

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model

# COMMAND ----------

model.num_labels = 2

# COMMAND ----------

# change classifier param with output feature of size 2
model.classifier = nn.Linear(model.classifier.in_features, 2)

# COMMAND ----------

model.config

# COMMAND ----------

model.config.id2label = {0: 'IRRELEVANT', 1: 'RELEVANT'}

# COMMAND ----------

model.config.id2label = {0: 'IRRELEVANT', 1: 'RELEVANT'}

# COMMAND ----------

# freeze model layers by layer name
for name, param in model.named_parameters():
    if name.startswith("bert") and "pooler" not in name:
        layer_num = [3, 4, 5]
        forbidden_layer = False
        for i in layer_num:
            if "encoder.layer." + str(i) in name:
                print("Not freezing: ", name)
                forbidden_layer = True
                break
        if not forbidden_layer:
            print("Freezing: ", name)
            param.requires_grad = False
    else:
        print("Not freezing: ", name)

# COMMAND ----------

train_df = pd.read_parquet(f"reg_nlp_labelled_data_train.parquet")
train_df

# COMMAND ----------

val_df = pd.read_parquet(f"reg_nlp_labelled_data_dev.parquet")
val_df

# COMMAND ----------

train_df["label"] = train_df["similar"].apply(lambda x: 1 if x else 0)
val_df["label"] = val_df["similar"].apply(lambda x: 1 if x else 0)

# COMMAND ----------

# train_df = train_df.sort_values(by=["question"]).head(10000)
# val_df = val_df.sort_values(by=["question"]).head(100)

# COMMAND ----------

# add certain duplicate rows to train_df. Basically we will upsample rows with similar=True by 10 times
train_df = train_df.append([train_df[train_df["similar"] == True]] * 20, ignore_index=True)

# COMMAND ----------

# shuffle train_df
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# COMMAND ----------

# make dataset with splits from train_df and val_df
train_dataset = Dataset.from_pandas(train_df[["question", "passage", "label"]])
val_dataset = Dataset.from_pandas(val_df[["question", "passage", "label"]])

# COMMAND ----------

# iterable datasets
train_dataset_it = train_dataset.to_iterable_dataset(num_shards=4)
val_dataset_it = val_dataset.to_iterable_dataset(num_shards=4)

# COMMAND ----------

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# COMMAND ----------

def preprocess_function(samples):
    texts = [samples["question"], samples["passage"]]
    return tokenizer(*texts, padding=True, truncation="longest_first")

# COMMAND ----------

def collate_function(samples):
    texts = [[s["question"] for s in samples], [s["passage"] for s in samples]]
    # 1 hot encoded vector for labels
    # labels = [[0, 1] if s["label"] == 1 else [1, 0] for s in samples]
    labels = [s["label"] for s in samples]
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "labels": labels,
        **data_collator(tokenizer(*texts, padding=True, truncation="longest_first", return_tensors="pt")),
    }

# COMMAND ----------

# tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
# tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)

# COMMAND ----------

tokenized_train_dataloader = torch.utils.data.DataLoader(train_dataset_it.with_format("torch"), collate_fn=collate_function, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=25, persistent_workers=True)
tokenized_val_dataloader = torch.utils.data.DataLoader(val_dataset_it.with_format("torch"), collate_fn=collate_function, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=25, persistent_workers=True)

# COMMAND ----------

# a = next(iter(tokenized_train_dataloader))
# a

# COMMAND ----------

# a["labels"].dtype == torch.int

# COMMAND ----------

training_args = TrainingArguments(
    output_dir=model_save_path,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # num_train_epochs=TRAIN_EPOCHS,
    max_steps=(TRAIN_EPOCHS*len(train_dataset)) // BATCH_SIZE,
    weight_decay=WEIGHT_DECAY,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=STEPS,
    save_steps=STEPS,
    # load_best_model_at_end=True,
    push_to_hub=False,
    report_to="mlflow",
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP_RATIO,
    fp16=True,
    bf16=False,
    greater_is_better=True,
    metric_for_best_model=f"recall@{RECALL_AT_K}",
    auto_find_batch_size=True,
    save_total_limit=3,
)

# COMMAND ----------

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
mse = evaluate.load("mse")

# COMMAND ----------

def assign_ranks(group):
    # group['internal_index'] = range(1, len(group) + 1)

    group = group.sort_values(by=\
['pred'], \
ascending=[False])
    # group['rank'] = range(1, len(group) + 1)
    
    # group = group.sort_values(by=['internal_index'], ascending=[True])
    return group

predictions_scores, labels, d = None, None, None
def compute_metrics(eval_pred):
    global predictions_scores, labels
    predictions_scores, labels = eval_pred
    predictions_scores_squeezed = predictions_scores.T[1]
    predictions = predictions_scores.argmax(axis=1)
    group_labels = val_df["question"]
    prc = precision.compute(predictions=predictions, references=labels)["precision"]
    rc = recall.compute(predictions=predictions, references=labels)["recall"]

    ranking_temp_df = pd.DataFrame({"group_label": group_labels, "pred": predictions_scores_squeezed, "label": labels})
    ranking_temp_df = ranking_temp_df.groupby('group_label', group_keys=False).apply(assign_ranks)
    # calculate recall@10
    # ranking_temp_df = ranking_temp_df.sort_values(by=['group_label', 'rank'], ascending=[True, True])
    # divide by 
    recall_at_k_df = ranking_temp_df.groupby('group_label', group_keys=True).apply(lambda x: x.head(RECALL_AT_K).label.sum()/x.label.sum())
    recall_at_k = recall_at_k_df.mean()

    d = {f"recall@{RECALL_AT_K}": recall_at_k, "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"], "f1": 2*(prc*rc)/(prc+rc), "precision": prc, "recall": rc}
    return d

# COMMAND ----------

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # processing_class=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

# COMMAND ----------

def get_train_dataloader(x = None):
    return tokenized_train_dataloader

def get_eval_dataloader(x = None):
    return tokenized_val_dataloader

trainer.get_train_dataloader = get_train_dataloader
trainer.get_eval_dataloader = get_eval_dataloader

# COMMAND ----------

# set your mlflow experiment id
# mlflow.set_experiment(experiment_id=1118191130778481)

# COMMAND ----------

with mlflow.start_run(run_name=RUN_NAME, log_system_metrics=True) as run:
    e = trainer.evaluate()
    print("Step 0 eval:", e)
    t = trainer.train()

# COMMAND ----------

trainer.save_model()

# COMMAND ----------

mlflow.log_artifact(model_save_path)


