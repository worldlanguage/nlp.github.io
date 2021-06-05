import time
import math
import jieba

import pandas as pd
from collections import defaultdict
from string import punctuation as str_punc

add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = str_punc + add_punc


def read_csv():

    context_list = [
        "TfIdf是一种统计的方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。",
        "字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。",
        "TfIdf加权的各种形式常被搜索引擎应用，作为文件与用户查询之间相关程度的度量或评级。",
        "除了TfIdf以外，因特网上的搜索引擎还会使用基于链接分析的评级方法，以确定文件在搜寻结果中出现的顺序。",
    ]

    return context_list


def tokenization(context_list):

    token_list = []
    words_frequency = defaultdict(int)

    for context in context_list:
        cut_words = jieba.lcut(context)
        token = []
        for word in cut_words:
            if word not in all_punc:
                words_frequency[word] += 1
                token.append(word)

        token_list.append(token)

    return token_list, words_frequency


def tf_fit(words_frequency):

    words_number = sum(words_frequency.values())

    words_tf = {}

    for word, word_frequency in words_frequency.items():
        words_tf[word] = word_frequency / words_number

    return words_tf


def idf_fit(tfidf_params, token_list, words_frequency):

    words_idf = {}

    words_doc = defaultdict(int)

    doc_number = len(token_list)

    for word in words_frequency.keys():
        for token in token_list:
            if word in token:
                words_doc[word] += 1
        words_idf[word] = doc_number / math.log(words_doc[word] + 1, tfidf_params['log_c'])

    return words_idf


def TfIdfVectorizer(log_c=10):

    tfidf_params = {
        'log_c': log_c,
    }

    return tfidf_params


def fit(tfidf_params, token_list, words_frequency):

    words_tfidf = {}

    words_tf = tf_fit(words_frequency)

    words_idf = idf_fit(tfidf_params, token_list, words_frequency)

    submit = pd.DataFrame(columns=['word', 'word_tf', 'word_idf', 'word_tfidf'])

    for word in words_frequency.keys():
        words_tfidf[word] = words_tf[word] * words_idf[word]
        submit = submit.append(
            {'word': word,
             'word_tf': words_tf[word],
             'word_idf': words_idf[word],
             'word_tfidf': words_tfidf[word]
             },
             ignore_index=True)

    submit = submit.sort_values(by=['word_tfidf'], ascending=False).reset_index(drop=True)
    print(submit.head(20))

    return None


if __name__ == '__main__':
    sta_time = time.time()

    context_list = read_csv()

    token_list, words_frequency = tokenization(context_list=context_list)

    model = TfIdfVectorizer(log_c=2)

    fit(model, token_list, words_frequency)

    print("Time:", time.time() - sta_time)
