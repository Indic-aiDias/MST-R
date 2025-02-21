# Content Index

1. LLM as a Judge.ipynb
   1. Evaluating the selected answer geeneration approach's answer quality using LLM as a Judge
      1. LLM used - ChatGPT 4o mini
   2. Saves answers to a file
2. RePASs Experiments.ipynb
   1. Contains all the experiments conducted for answer generation and their RePASs evaluation
   2. Contains the evaluation metric defined by us
   3. Experiments -
      1. Concatentation
         1. Ground Truth
         2. BM25 retrieved passages
         3. BGE retrieved passages
      2. Same sentence instead of concat with new line
         1. Ground Truth
      3. LLM for answer generations
         1. Some prompts from langchain prompt repo
         2. Modification to langchain prompt repo prompts
         3. RegNLP Paper prompt
      4. Brute Force Programatic optimisation of RePASs -
         1. Make all possible combinations of sentences and pick the combination with highest RePASs score
            1. Optimisation to retain sentence pair subscore matrices done so as to not compute RePASs from scratch again and again
         2. Experiments done -
            1. Passages from Ground truth
