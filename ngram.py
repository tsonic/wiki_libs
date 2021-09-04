import pickle
import pandas as pd
import itertools
import ast
import json
import numpy as np
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from wiki_libs.preprocessing import read_category_links, process_title, path_decoration, read_page_data
from wiki_libs.cache import get_from_cached_file


STOP_WORDS = set([
    'of',
    'the',
    'in',
    'and',
    'a',
    'de',
    '&',
    'for',
    'to',
    'by',
])


def train_phraser(sentences, min_count=5):
    return Phraser(Phrases(sentences, min_count=min_count, delimiter='_', connector_words=ENGLISH_CONNECTOR_WORDS))

def train_wiki_ngram(n=3, min_count=5, out_file='ngram_model.pickle', text_source = 'both', category_single_word = False):
    titles, categories = get_titles_and_categories()
    sentences = titles
    if text_source == 'title' or category_single_word:
        sentences = titles
    elif text_source == 'both':
        sentences = titles + categories
    elif text_source == 'category':
        sentences = categories
    else:
        raise Exception(f'unknown text_source {text_source}')
    return train_ngram(sentences, n=n, min_count=min_count, out_file=out_file)

def train_ngram(sentences, n=3, min_count=5, out_file='ngram_model.pickle'):
    ngram_model = []
    for i in range(1, n):
        if i == 1:
            xgram = train_phraser(sentences, min_count=1)   # keep all unigram
        else:
            xgram = train_phraser(sentences, min_count=min_count)
        sentences = xgram[sentences]
        ngram_model.append(xgram)
    pickle.dump(ngram_model, open(out_file, "wb"))
    return ngram_model

def transform_ngram(sentences, ngram_model):
    for m in ngram_model:
        sentences = m[sentences]
    return list(sentences)

def get_titles_and_categories():
    df_cp = read_category_links()
    df_page = read_page_data()
    titles = df_page['page_title'].dropna().drop_duplicates().apply(process_title).tolist()
    categories = df_cp['page_category'].dropna().drop_duplicates().apply(process_title).tolist()
    return titles, categories

def load_ngram_model(model_file):
    return pickle.load(open(model_file, "rb"))

def remove_stop_words(l):
    return [w for w in l if w not in STOP_WORDS]

def generate_df_title_category_transformed(ngram_model, text_source, category_single_word, no_stop_words):

    df_cp = read_category_links()
    df_page = read_page_data()

    df_list = []

    if text_source in ('title', 'both'):
        df_title = (
            df_page[['page_title']]
                .drop_duplicates()
                .assign(page_title_transformed = lambda df: transform_ngram(
                    df['page_title'].apply(process_title).tolist(),
                    ngram_model) 
                )
            )
        if no_stop_words:
            df_title['page_title_transformed'] = df_title['page_title_transformed'].apply(remove_stop_words)

        df_list.append(
            df_page[['page_id', 'page_title']]
                    .merge(df_title, on = 'page_title')
                    .rename(columns = {'page_title_transformed': 'page_title_category_transformed'})
        )

    if text_source in ('category','both'):
        df_category = (
                df_cp[['page_category']]
                .drop_duplicates()
        )
        if category_single_word:
            df_category['page_category_transformed'] = [['_'.join(process_title(c))] for c in df_category['page_category']]

        else:
            df_category['page_category_transformed'] = transform_ngram(
                        df_category['page_category'].apply(process_title).tolist(),
                        ngram_model,
                    )
            if no_stop_words:
                df_category['page_category_transformed'] = df_category['page_category_transformed'].apply(remove_stop_words)

        df_tmp = df_page.merge(
                df_cp[['page_id', 'page_category']]
                .merge(df_category, on = 'page_category')
                .rename(columns = {'page_category_transformed': 'page_title_category_transformed'})
                .sample(frac = 1.0, random_state = 0)     # random sample category words
                , on = 'page_id', how = 'left' )[['page_id', 'page_title_category_transformed']]

        df_tmp['page_title_category_transformed'] = df_tmp['page_title_category_transformed'].fillna({i: [] for i in df_tmp.index})
        df_list.append(df_tmp)
        

    df_title_category_transformed = (
        pd.concat(df_list)
        .groupby(['page_id'])
            ['page_title_category_transformed'].apply(lambda x: sorted(set(itertools.chain.from_iterable(x))))
        .to_frame('page_title_category_transformed')
        .reset_index()
    )
#    df_title_category_transformed.to_parquet(path, index = False, compression = 'snappy')
    return df_title_category_transformed

def get_all_words_for_embedding(text_source, category_single_word, category_embedding, no_stop_words):
    if category_embedding:
        text_source_list = ['title', 'category']
        category_single_word_list = [False, category_single_word]
    else:
        text_source_list = [text_source]
        category_single_word_list = [category_single_word]
    
    return [
        get_from_cached_file({'prefix':'df_title_category_transformed',
                            'ngram_model_key_dict':{'prefix':'ngram_model','text_source':ts,'category_single_word':csw},
                            'no_stop_words':no_stop_words,
        })
     for ts, csw in zip(text_source_list, category_single_word_list)]