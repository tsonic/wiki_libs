from wiki_libs.preprocessing import read_link_pairs_chunks
from wiki_libs.knn import top_k
import numpy as np
import pandas as pd

def page_id_lookup(df_page, page_title):
    df = df_page.query('page_title == @page_title')
    return df['page_id'].tolist()

def find_link_pairs(df_page, w2v_mimic, page_id):
    gen = read_link_pairs_chunks(n_chunk = 10, w2v_mimic =w2v_mimic)
    print('generating page id stats...')
    df_list = []
    for df_chunk in gen:
        df_chunk = df_chunk.query('page_id_target == @page_id or page_id_source == @page_id')
        df_list.append(df_chunk)
    df = pd.concat(df_list)
    return (
        df
        .merge(df_page[['page_id', 'page_title']], left_on='page_id_source', right_on = 'page_id')
        .drop(columns=['page_id'])
        .merge(df_page[['page_id', 'page_title']], left_on='page_id_target', right_on = 'page_id', suffixes = ['_source', '_target'])
        .drop(columns=['page_id'])
    )

def find_link_pairs_from_title(df_page, w2v_mimic, page_title):
    page_id = page_id_lookup(df_page, page_title)
    return find_link_pairs(df_page, w2v_mimic, page_id)

def find_page_emb_from_title(df_page, page_title, trained_model):
    page_id = page_id_lookup(df_page, page_title)
    return [trained_model.dataset.page2emb[pid] for pid in page_id]

def find_word_emb_from_title(word, trained_model, df_page):
    return [i for i in trained_model.dataset.page_emb_to_word_emb_tensor[find_page_emb_from_title(df_page, word, trained_model)[0]] 
            if i < len(trained_model.dataset.emb2word)]

def find_words_from_title(word, trained_model, df_page):
    return [trained_model.dataset.emb2word[i] for i in find_word_emb_from_title(word, trained_model, df_page)]

def compute_recall(df_links, df_embedding, nn, k_list, use_user_emb = True):
    all_page_ids = df_links['page_id_source'].unique().tolist()
    if len(all_page_ids) > 100_000:
        all_page_ids_chunks = np.array_split(all_page_ids, len(all_page_ids) / 100_000)
    else:
        all_page_ids_chunks = [all_page_ids]
    if isinstance(k_list, int):
        k_list = [k_list]
    k_max = max(k_list)

    res_list = []
    for chunk in all_page_ids_chunks:
        dist, ind = top_k(chunk, nn, df_embedding, k_max, input_type = 'page_id', output_type = 'page_id',
                pos_keys = None, neg_keys = None, use_user_emb = use_user_emb)
        df_nn = pd.DataFrame({'page_id_source': chunk, 'nn': list(ind)})
        df_merged = df_links.merge(df_nn, on = 'page_id_source')
        res_list.append(pd.concat([(df_merged['page_id_target'].values[:,np.newaxis] == np.vstack(df_merged['nn'])[:,:k])
                        .any(axis = 1).to_frame('recall')
                        .assign(k = k) for k in k_list])
                        )
    return pd.concat(res_list).groupby('k', as_index = False)['recall'].mean()


    
    