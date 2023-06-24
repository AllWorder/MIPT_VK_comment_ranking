# MIPT_VK_comment_ranking
Repository for MIPT VK hackaton. Main task is to rank comments.

## File structure 
- **preprocessing.ipynb** makes all necessary operations with texts and form ranking_train/test.tsv for next files
- **vector_space_models.ipynb** contains vector space models based on text vectorization and cosine similarity ranking. Embeddings from GlovVe, Tf-IDF, BERT.
- **vector_space_models_utils.py** contains necessary classes and functions for previous .ipynb file
- **LambdaMARTtest.ypynb** contains tests with LambdaMART model

## Current successes and challenges
- Успели поработать с vector space моделями
- Успели сделать некоторые тесты с LambdaMART
- Есть проблемы с выбором корректной метрики и скоростью работы алгоритмов. (Очень хотелось бы получить советы и комментарии по текущей метрике и что можно ещё посмотреть)
- В планах попробовать обучить нейросетевые модели

  Хотелось бы получить обратную связь с замечаниями или комментариями, чтобы можно было исправить ошибки, пока есть время
