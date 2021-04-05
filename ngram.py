import pickle
import pandas as pd
import itertools
import ast
import json
import numpy as np
from gensim.models.phrases import Phrases, Phraser
from wiki_libs.preprocessing import read_category_links, process_title, path_decoration, read_page_data


NGRAM_MODEL_PATH_PREFIX = "wiki_data/ngram_model/"
def train_phraser(sentences, min_count=5):
    return Phraser(Phrases(sentences, min_count=min_count, delimiter=b'_'))

def train_ngram(sentences, n=3, min_count=5, out_file='ngram_model.pickle'):
    ngram_model = []
    for i in range(1, n):
        if i == 1:
            xgram = train_phraser(sentences, min_count=1)   # keep all unigram
        else:
            xgram = train_phraser(sentences, min_count=min_count)
        sentences = xgram[sentences]
        ngram_model.append(xgram)
    pickle.dump(ngram_model, open(NGRAM_MODEL_PATH_PREFIX + out_file, "wb"))
    return ngram_model

def transform_ngram(sentences, ngram_model):
    for m in ngram_model:
        sentences = m[sentences]
    return list(sentences)

def get_transformed_title_category(ngram_model):
    df_cp = read_category_links()
    df_page = read_page_data()
    titles = df_page['page_title'].dropna().drop_duplicates().apply(process_title).tolist()
    categories = df_cp['page_category'].dropna().drop_duplicates().apply(process_title).tolist()
    title_transformed = transform_ngram(titles, ngram_model)
    category_transformed = transform_ngram(categories, ngram_model)
    return title_transformed, category_transformed

def load_ngram_model(model_file):
    return pickle.load(open(model_file, "rb"))

def get_df_title_category_transformed(read_cached = True, 
                                        fname='df_title_category_transformed.parquet', 
                                        ngram_model_name = "title_category_ngram_model.pickle",
                                        title_only = False):
    path = path_decoration(f'wiki_data/{fname}', w2v_mimic=False)
    if read_cached:
        df_title_category_transformed = pd.read_parquet(path, columns = ['page_id','page_title_category_transformed'])
        return df_title_category_transformed

    ngram_model = load_ngram_model(NGRAM_MODEL_PATH_PREFIX + ngram_model_name)
    
    df_cp = read_category_links()
    df_page = read_page_data()

    df_title = (
        df_page[['page_title']]
            .drop_duplicates()
            .assign(page_title_transformed = lambda df: transform_ngram(
                df['page_title'].apply(process_title).tolist(),
                ngram_model,
            )
        )
    )

    df = (
        df_page[['page_id', 'page_title']]
                .merge(df_title, on = 'page_title')
                .rename(columns = {'page_title_transformed': 'page_title_category_transformed'})
    )
    if not title_only:
        df_category = (
            df_cp[['page_category']]
            .drop_duplicates()
            .assign(page_category_transformed = lambda df: transform_ngram(
                    df['page_category'].apply(process_title).tolist(),
                    ngram_model,
                )
            )
        )

        df = df.append(
            df_cp[['page_id', 'page_category']]
                .merge(df_category, on = 'page_category')
                .rename(columns = {'page_category_transformed': 'page_title_category_transformed'})
                .sample(frac = 1.0)     # random sample category words
        )

    df_title_category_transformed = (
        df
        .groupby(['page_id'])
            ['page_title_category_transformed'].apply(lambda x: list(set(itertools.chain.from_iterable(x))))
        .to_frame('page_title_category_transformed')
        .reset_index()
    )
    df_title_category_transformed.to_parquet(path, index = False, compression = 'snappy')

    return df_title_category_transformed
