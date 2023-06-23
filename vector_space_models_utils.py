import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

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
        for i, token in enumerate(self.idf_dict.keys()):
            tf = tokemized_text.count(token)
            result[i] = tf * self.idf_dict[token]

        return result
    
class BERT_vectorizer(Text_to_vector):
    """
    Class for converting texts to vectors with help of latent vectors from BERT model

    bert_model : pretrained BERT model
    bert_tokenizer : pretrained BERT tokenizer
    """
    def __init__(self, bert_model, bert_tokenizer):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

    def text_to_vector(self, text: str):
        self.bert_model.eval()


        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.bert_tokenizer.tokenize(marked_text)
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids]) # attention mask

        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, segments_tensor)

        tokens_vecs = outputs.hidden_states[-2][0]

        result = torch.mean(tokens_vecs, dim=0)

        return result