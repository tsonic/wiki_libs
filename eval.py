from wiki_libs.preprocessing import read_link_pairs_chunks
import pandas as pd

def page_id_lookup(df_cp, page_title):
    df = df_cp.query('page_title == @page_title')
    return df['page_id'].unique().tolist()

def find_link_pairs(df_cp, w2v_mimic, page_id):
    gen = read_link_pairs_chunks(n_chunk = 10, w2v_mimic =w2v_mimic)
    print('generating page id stats...')
    df_list = []
    for df_chunk in gen:
        df_chunk = df_chunk.query('page_id_target == @page_id or page_id_source == @page_id')
        df_list.append(df_chunk)
    df = pd.concat(df_list)
    return (
        df
        .merge(df_cp[['page_id', 'page_title']].drop_duplicates(), left_on='page_id_source', right_on = 'page_id')
        .drop(columns=['page_id'])
        .merge(df_cp[['page_id', 'page_title']].drop_duplicates(), left_on='page_id_target', right_on = 'page_id', suffixes = ['_source', '_target'])
        .drop(columns=['page_id'])
    )

def find_link_pairs_from_title(df_cp, w2v_mimic, page_title):
    page_id = page_id_lookup(df_cp, page_title)
    return find_link_pairs(df_cp, w2v_mimic, page_id)
