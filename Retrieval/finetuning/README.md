# Content Index

1. L2 reranker dataset creation.ipynb
   1. Search and get top k documents for each retriever for all questions in all splits (train, test, dev) of the dataset
      1. You need to repeat the process for each split spearately
   2. Combine all the generated files for each sources and split as single files for each split
2. L2 reranker finetuning.py
   1. Loads a passage reranker model
   2. Replace regression task head with classifcation head (this task head had slightly better results in our experiments)
   3. Freeze some intial layer to avoid catastrophic forgetting and making training faster
   4. Load datasets, use standard huggingface (HF) trainer class with our metrics for observing training
   5. Send the metrics to mlflow (easily replacable to any other framework HF trainer supports like wadb) for tracking
3. Reranker Evaluation.py
   1. trec eval for L2 reranker
   2. Takes trec file of L1 reranker as input
   3. Runs inferences for L2 reranker and saves output trec file for reranker
   4. Evaluates output trec using trec eval
