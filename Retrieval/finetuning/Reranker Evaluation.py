# Databricks notebook source
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

from tqdm.autonotebook import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# COMMAND ----------

L1_TOP_K = 100

# COMMAND ----------

# MAGIC %md
# MAGIC # Data prep

# COMMAND ----------

# Replace with path for ObliQADataset test set
test_df = pd.read_json("regnlp/ObliQADataset/ObliQA_test.json")
test_df

# COMMAND ----------

test_df_index = test_df.set_index("QuestionID")

# COMMAND ----------

import json
import glob

def load_json_files_from_directory(directory_path):
  """Loads all JSON files from a given directory into a list of JSON objects."""
  json_files = glob.glob(directory_path + "/*.json")
  json_data_list = []
  for json_file in json_files:
    with open(json_file, 'r') as f:
      try:
        json_data = json.load(f) 
        json_data_list.append((json_file, json_data))
      except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {json_file}: {e}")
  return json_data_list

# Replace with path of structured json files provided with ObliQADataset
directory_path = "regnlp/ObliQADataset/StructuredRegulatoryDocuments"
json_data_list = load_json_files_from_directory(directory_path)

# COMMAND ----------

flattened_json_data_list = []

for json_file, json_data in json_data_list:
        flattened_json_data_list.extend(json_data)
len(flattened_json_data_list)

# COMMAND ----------

flattened_json_data_dict = {}

for item in flattened_json_data_list:
    flattened_json_data_dict[item["ID"]] = item

# COMMAND ----------

# Replace with path for trec file of L1 ranker
ret_df = pd.read_csv("regnlp/retrieval-evals/bge-en-icl-5-shot-single-5-rankings-exact-topk-100.trec", sep=" ", names=["QuestionID", "IterId", "PassageId", "RetrievalPos", "Score", "ModelType"])
ret_df

# COMMAND ----------

ret_df = ret_df[ret_df["RetrievalPos"] <= L1_TOP_K]
ret_df["Passage"] = ret_df["PassageId"].apply(lambda x: flattened_json_data_dict[x])
ret_df["Question"] = ret_df["QuestionID"].apply(lambda x: test_df_index.loc[x]["Question"])

ret_df_grouped = ret_df.groupby(["QuestionID", "Question"])["Passage"].apply(list).reset_index()
ret_df_grouped["GroundTruth"] = ret_df_grouped["QuestionID"].apply(lambda x: test_df_index.loc[x]["Passages"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Eval

# COMMAND ----------

# Replace MODEL NAME with the hugging face model name you want to use
BATCH_SIZE = 16
TOP_K = 20
MODEL_NAME = "yashmalviya/ms-marco-MiniLM-L-6-v2-3aglo-frozen-binary-classification"
RUN_NAME = "top{L1_TOP_K}-bge-en-icl-5-shot-single-5-ms-marco-MiniLM-L-12-v2"

# COMMAND ----------

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = model.to('cuda').eval()

# COMMAND ----------

trec_ranking_file = f"regnlp/reranking-evals/{RUN_NAME}.trec"
with open(trec_ranking_file, "w") as f:
    for i, row in tqdm(ret_df_grouped.iterrows(), total=len(ret_df_grouped)):
        all_docs_inst = [p["Passage"] for p in row["Passage"]]
        question_list = [row["Question"]] * all_docs_inst
        tokens = tokenizer(question_list, all_docs_inst,  padding=True, return_tensors='pt')
        if tokens['input_ids'].shape[1] > 512:
            tokens = tokenizer(question_list, all_docs_inst, padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        tokens = tokens.to('cuda')
        batch_score = (model(**tokens).logits.detach().cpu()).numpy()

        # sort all_docs_inst by batch_score
        sorted_docs_inst = [x for _, x in sorted(zip(batch_score, row["PassageId"]), key=lambda pair: pair[0], reverse=True)]
        top_k_docs = sorted_docs_inst[:TOP_K]
        for j, p_id in enumerate(top_k_docs):
            line = f"{row['QuestionID']} 0 {row['Passage'][p_id]['ID']} {j+1} 10 {RUN_NAME}"
            f.write(line + "\n")

# COMMAND ----------
# Evaluation using trec_eval
trec_result_ouput_path = f"regnlp/reranking-evals/results-{RUN_NAME}.txt"
!regnlp/trec_eval/trec_eval -m recall.10 -m recall.20 -m recall.50 -m map_cut.10 regnlp/qrels $trec_ranking_file &> $trec_result_ouput_path
!cat $trec_result_ouput_path

# COMMAND ----------


