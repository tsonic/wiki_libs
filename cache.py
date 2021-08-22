
import os
import pandas as pd

import torch
import copy
from wiki_libs.preprocessing import is_colab, convert_to_colab_path

CACHE_PATH = 'wiki_data/cache'
if is_colab():
    CACHE_PATH = convert_to_colab_path(CACHE_PATH)

default_ngram_model_key_dict = {
    'prefix':'ngram_model',
    'title_only':False,
    'category_single_word':False,
}

default_df_title_category_transformed_key_dict = {
    'prefix':'df_title_category_transformed',
    'ngram_model_key_dict':default_ngram_model_key_dict,
}

default_page_word_stats = {
    'prefix':'page_word_stats',
    'ngram_model_key_dict':default_ngram_model_key_dict,
    'stats_column':'both',
    'w2v_mimic':False,
    'n_chunk':10,
}

default_key_dict = {
    'df_title_category_transformed':default_df_title_category_transformed_key_dict,
    'ngram_model':default_ngram_model_key_dict,
    'page_word_stats':default_page_word_stats,
    'page_emb_to_word_emb_tensor':{
        'prefix':'page_emb_to_word_emb_tensor',
        'title_category_trunc_len':30,
        'page_word_stats_key_dict':default_page_word_stats,
        'df_title_category_transformed_key_dict':default_df_title_category_transformed_key_dict,
    }
}

def is_cache_file_exist(path):
    ret = os.path.isfile(path)
    if ret:
        print(f'cache file {path} exist')
    else:
        print(f'cache file {path} NOT exist')
    return ret

def flatten_dict(d):
    d = copy.deepcopy(d)
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret.update(flatten_dict(v))
        else:
            ret[k] = v
    return ret

def get_cache_fname(key_dict, prefix):
    from wiki_libs.trainer import config2str
    key_dict = copy.deepcopy(key_dict)
    key_dict = flatten_dict(key_dict)
    if 'prefix' in key_dict:
        key_dict.pop('prefix')
    fname = prefix + config2str(key_dict)
    return fname


def get_from_cached_file(key_dict, **kwarg):
    """
    cached file dependency: 
    df_title_category_transformed -> ngram_model_name
    """
    from wiki_libs.trainer import update_config
    from wiki_libs.ngram import generate_df_title_category_transformed, load_ngram_model, train_wiki_ngram
    prefix = key_dict['prefix']
    print(f'prefix is {prefix}', flush = True)
    key_dict, update_key_dict = update_config(default_key_dict[prefix], key_dict)

    fname = get_cache_fname(update_key_dict, prefix)
    print(f'fname is {fname}',flush=True)
    if prefix == 'df_title_category_transformed':   # find get_df_title_category_transformed
        path = f'{CACHE_PATH}/{fname}.parquet'
        if is_cache_file_exist(path):
            return pd.read_parquet(path,  columns = ['page_id','page_title_category_transformed'], engine = 'pyarrow')
        else:
            ngram_model = get_from_cached_file(key_dict['ngram_model_key_dict'])
            df_title_category =  generate_df_title_category_transformed(
                                    ngram_model, 
                                    title_only = key_dict['ngram_model_key_dict']['title_only'],
                                    category_single_word = key_dict['ngram_model_key_dict']['category_single_word'],
                            )
            df_title_category.to_parquet(path, index = False, compression = 'snappy')
            return df_title_category
    elif prefix == 'ngram_model':
        path = f'{CACHE_PATH}/{fname}.pickle'
        if is_cache_file_exist(path):
            return load_ngram_model(path)
        else:
            return train_wiki_ngram(n=3, min_count=5, out_file=path, 
                                title_only = key_dict['title_only'], category_single_word = key_dict['category_single_word'])
    elif prefix == 'page_word_stats':
        path = f'{CACHE_PATH}/{fname}.json'
        from wiki_libs.stats import PageWordStats
        if is_cache_file_exist(path):
            return PageWordStats(read_path=path)
        else:
            ngram_model = get_from_cached_file(key_dict['ngram_model_key_dict'])
            return PageWordStats(read_path=None, 
                            output_path = path, 
                            ngram_model = ngram_model, 
                            w2v_mimic = key_dict['w2v_mimic'],
                            stats_column = key_dict['stats_column'],
                            title_only = key_dict['ngram_model_key_dict']['title_only'],
                            category_single_word = key_dict['ngram_model_key_dict']['category_single_word'], 
                            )
    elif prefix == 'page_emb_to_word_emb_tensor':
        path = f'{CACHE_PATH}/{fname}.pickle'
        if is_cache_file_exist(path):
            return torch.load(path, map_location = 'cpu')
        else:
            from wiki_libs.datasets import WikiDataset
            page_word_stats_key_dict = key_dict['page_word_stats_key_dict']
            page_word_stats = get_from_cached_file(page_word_stats_key_dict)
            return WikiDataset.generate_page_emb_to_word_emb_tensor(path, page_word_stats, 
                                                                title_only = page_word_stats_key_dict['ngram_model_key_dict']['title_only'], 
                                                                category_single_word = page_word_stats_key_dict['ngram_model_key_dict']['category_single_word'], 
                                                                title_category_trunc_len = key_dict['title_category_trunc_len'])
    else:
        raise Exception('Unknown prefix!')