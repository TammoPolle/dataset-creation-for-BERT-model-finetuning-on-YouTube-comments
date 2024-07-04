# Provenance of YouTube-Comments - Dataset-Creation for BERT-Model Based Authorship Attribution

This repository contains the source code used for the practical part of the same name thesis, submitted on the 4th of July, 2024.

## Code overview:
- dataset_creation contains the code used for the pre-processing of the collected raw comments data (chapter 3.2)
    - preprocessing.py: the script which is illustrated in Listing 3.1
    - dsc.py: some helper functions, including the function illustrated in Listing 3.2
      
- trainingdata_preparation contains the code used to prepare the pre-processed dataset for the finetuning of a certain BERT-model
    - testset_creation.py: the script which is illustrated in Listing 3.3
    - tdp.py: some helper functions
      
- experiments contains the jupyter notebooks for model finetuning
    - bert-base.ipynb: implementation of BERT-base finetuning (chapter 4.2.1)
    - bertweet-base.ipynb: implementation of BERTweet finetuning (chapter 4.2.2)
    - n-gram-baseliner.ipynb: implementation of a simple n-gram model for Authorship Attribution, by ChatGPT (chapter 4.3.2)
      
- evaluation contains the code used for the evaluation of the experiments
    - attention-weight_viz.py: code used to visualize the model attention weights for a single comment, by ChatGPT (chapter 5.3.3)
    - comment_analysis.py: the code for the statistical analysis of the comment texts, by ChatGPT (chapter 5.3.2)
    - ehp.py: some helper functions
    - set_comparison: code used for the comparison of the statistics of NP- and Gen-Set, by ChatGPT (chapter 5.3.2)
    - token-embedding_viz.ipynb: notebook for the visualization of the token embeddings, by ChatGPT (chapter 5.3.5)

