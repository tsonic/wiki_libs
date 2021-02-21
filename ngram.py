import pickle
import pandas as pd
import itertools
import ast
import json
from gensim.models.phrases import Phrases, Phraser
from wiki_libs.preprocessing import read_category_links, process_title, path_decoration


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
    titles = df_cp['page_title'].dropna().drop_duplicates().apply(process_title).tolist()
    categories = df_cp['page_category'].dropna().drop_duplicates().apply(process_title).tolist()
    title_transformed = transform_ngram(titles, ngram_model)
    category_transformed = transform_ngram(categories, ngram_model)
    return title_transformed, category_transformed

def load_ngram_model(model_file):
    return pickle.load(open(model_file, "rb"))

def get_df_title_category_transformed(read_cached = True, 
                                        fname='df_title_category_transformed.parquet', 
                                        ngram_model_name = "title_category_ngram_model.pickle"):
    if read_cached:
        path = path_decoration('wiki_data/' + fname, w2v_mimic=False)
        df_title_category_transformed = pd.read_parquet(path, columns = ['page_id','page_title_category_transformed'])
        return df_title_category_transformed

    ngram_model = load_ngram_model(NGRAM_MODEL_PATH_PREFIX + ngram_model_name)
    df_cp = read_category_links().fillna({'page_title':''})
    df_title = (
        df_cp[['page_title']]
            .drop_duplicates()
            .assign(page_title_transformed = lambda df: transform_ngram(
                df['page_title'].apply(process_title).tolist(),
                ngram_model,
            )
        )
    )
    df_category = (
        df_cp[['page_category']]
        .drop_duplicates()
        .assign(page_category_transformed = lambda df: transform_ngram(
                df['page_category'].apply(process_title).tolist(),
                ngram_model,
            )
        )
    )

    df_title_category_transformed = (
        df_cp[['page_id', 'page_title','page_category']]
        .drop_duplicates()
        .merge(df_category, on = 'page_category', how = 'left')
        .groupby(['page_id', 'page_title'])
            ['page_category_transformed'].apply(lambda x: list(set(itertools.chain.from_iterable(x))))
        .to_frame('page_category_transformed')
        .reset_index()
        .merge(df_title, on = 'page_title', how = 'left')
        .assign(page_title_category_transformed = lambda df: 
            (df['page_category_transformed'] + df['page_title_transformed']).apply(lambda x: list(set(x))))
    )
    df_title_category_transformed.to_parquet(path_decoration(f'wiki_data/{fname}', w2v_mimic=False), index = False, compression = 'snappy')
    return df_title_category_transformed[['page_id','page_title_category_transformed']]