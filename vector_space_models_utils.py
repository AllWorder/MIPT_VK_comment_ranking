import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from sklearn.metrics import ndcg_score

def rank_comments(post, comments, vectorizer, rank_by='cosine'):
    """
        post    : post text
        comments    : list of comments texts [a, b, c]
        vectorizer  : entity with .text_to_vector() method
        result  : ranked indexes. If predicted order is [b, c, a], then result = [1, 2, 0]
    """
    vec_post = vectorizer.text_to_vector(post).reshape(1,-1)
    vec_comments = []

    predicted_order = []  

    for comment in comments:
        vec_comments.append(vectorizer.text_to_vector(comment).reshape(1,-1))

    if rank_by == 'cosine':
        cosine_and_comments = []
        for vec, comment in zip(vec_comments, comments):
            cosine_and_comments.append( [cosine_similarity(vec_post, vec)[0][0], comment] )

        cosine_and_comments = sorted(cosine_and_comments, key=lambda tup: tup[0])[::-1]

        for res in cosine_and_comments:
            initial_index = comments.index(res[1])
            predicted_order.append(initial_index)

    elif rank_by == 'euclidian_distance':
        distance_and_vectors = []
        for vec, comment in zip(vec_comments, comments):
            distance_and_vectors.append( [np.linalg.norm(vec_post - vec), comment] )

        distance_and_vectors = sorted(distance_and_vectors, key=lambda tup: tup[0])

        for res in distance_and_vectors:
            initial_index = comments.index(res[1])
            predicted_order.append(initial_index)      

    else:
        raise AttributeError(f'No rank_by {rank_by}, please, use other. For emample, \'cosine\'')


    return predicted_order

class Text_to_vector():
    def text_to_vector(self, text: str):
        pass

class Embeddings_text_to_vector(Text_to_vector):
    """
    Class for converting texts to vectors with help of pretrained embeddings

    embeddings : pretrained embbedings (word2vec, GloVe)
    tokenizer : tokenizer with text preprocessing
    vector_dim : embedding vector dimmention
    """
    def __init__(self, embeddings, tokenizer, vector_dim: int):
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.vector_dim = vector_dim

    def text_to_vector(self, text: str, aggregation='mean', verbose = False):
        """
        text : text string

        return : embedding vector of whole text
        """
        tokens = self.tokenizer.tokenize(text)

        result = np.zeros(self.vector_dim)

        if verbose:
            print('Initial text:', text)
            print('Tokens:', tokens)

        if aggregation == 'mean':
            iterator = 0

            for word in tokens:
                if word in self.embeddings:
                    result += self.embeddings[word]
                    iterator += 1

            if iterator != 0:
                result = result / iterator
        else:
            raise AttributeError(f'No aggregation type {aggregation}, please, use other')

        return result
    
class Tf_idf_vectorizer(Text_to_vector):
    """
    Class for converting texts to vectors with help of Tf-IDF vectors

    IDF_dict : dictionary {'tokent': token_IDF}
    tokenizer : tokenizer with text preprocessing
    """
    def __init__(self, idf_dict, tokenizer):
        self.idf_dict = idf_dict
        self.tokenizer = tokenizer

    def text_to_vector(self, text: str):
        vector_dim = len(self.idf_dict)
        result = np.zeros(vector_dim)
        tokemized_text = self.tokenizer.tokenize(text)
        idf_list = list(self.idf_dict.keys())
        for token in tokemized_text:
            if token in self.idf_dict.keys():
                tf = tokemized_text.count(token)
                i = idf_list.index(token)
                result[i] = tf * self.idf_dict[token]

        return result
    
class BERT_vectorizer(Text_to_vector):
    """
    Class for converting texts to vectors with help of latent vectors from BERT model

    bert_model : pretrained BERT model
    bert_tokenizer : pretrained BERT tokenizer
    """
    def __init__(self, bert_model, bert_tokenizer, vectorization_type='default'):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.vectorization_type = vectorization_type

    def text_to_vector(self, text: str):
        self.bert_model.eval()
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.bert_tokenizer.tokenize(marked_text)
        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:509]
            tokenized_text.append('[SEP]')
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids]) # attention mask
        outputs = ''

        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, segments_tensor)

        if self.vectorization_type == 'default':
            tokens_vecs = outputs.hidden_states[-2][0]
            result = torch.mean(tokens_vecs, dim=0)
            return result
        elif self.vectorization_type == 'cls':
            lhs_tensor = outputs.last_hidden_state
            cls_tensor = lhs_tensor[:, 0, :]
            return cls_tensor.detach().numpy()
        else:
            raise AttributeError(f'No vectorization type {self.vectorization_type}, please, use other')
def get_ndcg(predicted_ranks, verbose=False, k=5):

    predicted_scores = []
    true_scores = []

    for predicted_rank in predicted_ranks:

        true_score = [5, 4, 3, 2, 1]
        predicted_score = []
        for i in range(5):
            relevance_for_pos_i = 5 - predicted_rank.index(i)
            predicted_score.append(relevance_for_pos_i)

        predicted_scores.append(predicted_score)
        true_scores.append(true_score)
    result = ndcg_score(true_scores, predicted_scores, k=k)
    if verbose:
        print(f'NDCG@{k}: {result}')

    return result

def get_predicted_order(train_df, vectorizer, verbose=False, rank_by='cosine'):

    predicted_order= []

    for i in range(len(train_df)):
        comments_text = []
        post_text = train_df.loc[i]['text']
        comments_text.append(train_df.loc[i]['comments_text_0'])
        comments_text.append(train_df.loc[i]['comments_text_1'])
        comments_text.append(train_df.loc[i]['comments_text_2'])
        comments_text.append(train_df.loc[i]['comments_text_3'])
        comments_text.append(train_df.loc[i]['comments_text_4'])

        predicted_rank = rank_comments(post_text, comments_text, vectorizer, rank_by=rank_by)

        predicted_order.append(predicted_rank)

    return predicted_order

def hits_score(predicted_orders, k=1):
    """
    dup_ranks: list of predicted ranks [[0, 3, 2, 4, 1], [1, 3, 4, 0, 2]]
    result: HITS@k
    """
    hits_value = 0

    for pr in predicted_orders:
        predicted_position_of_top_comment = pr.index(0) + 1 # from 1, not from 0
        if predicted_position_of_top_comment <= k:
            hits_value += 1

    hits_value = hits_value / len(predicted_orders)

    return hits_value

def dcg_score(predicted_orders, k=1):
    """
        dup_ranks: list of predicted ranks
        result: DCG@k
    """
    dcg_value = 0

    for pr in predicted_orders:
        predicted_position_of_top_comment = pr.index(0) + 1 # from 1, not from 0
        if predicted_position_of_top_comment <= k:
            dcg_value += 1 / np.log2(1+predicted_position_of_top_comment)

    dcg_value = dcg_value / len(predicted_orders)

    return dcg_value