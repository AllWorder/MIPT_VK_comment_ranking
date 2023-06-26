# MIPT_VK_comment_ranking
Repository for MIPT VK hackaton. Main task is to rank comments.

## File structure 
- **preprocessing.ipynb** makes all necessary operations with texts and form ranking_train/test.tsv for next files
- **vector_space_models.ipynb** contains vector space models based on text vectorization and cosine similarity ranking. Embeddings from GlovVe, Tf-IDF, BERT.
- **vector_space_models_utils.py** contains necessary classes and functions for previous .ipynb file
- **LambdaMARTtest.ypynb** contains LambdaMART model for ranking comments based on GloVe vectors text representation
- **ranking_test.jsonl** test dataset with filled scores

## Current successes and challenges
- Успели поработать с vector space моделями
- Применили модель LambdaMART, при помощи которой и заполнили **ranking_test.jsonl**

