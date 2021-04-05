import torch
from wiki_libs.preprocessing import normalize, path_decoration
from wiki_libs.models import OneTower
from sklearn.neighbors import KDTree, NearestNeighbors
import pandas as pd
import json
import numpy as np

def build_knn(emb_file, df_page, w2v_mimic, use_user_emb = True, algorithm = 'brute', k = 10):
    emb_file = path_decoration(emb_file, w2v_mimic = w2v_mimic)
    saved_embeddings = torch.load(emb_file, map_location = 'cpu')
    USER_ID = 'page_id'
    ITEM_ID = 'page_id'

    if saved_embeddings['entity_type'] == "word":
        model = OneTower(**saved_embeddings['model_init_kwargs'])
        model.load_state_dict(saved_embeddings['model_state_dict'])

        page_emb_to_word_emb_tensor = saved_embeddings['page_emb_to_word_emb_tensor']

        user_embedding = model.forward_to_user_embedding_layer(page_emb_to_word_emb_tensor, in_chunks=True).detach().numpy()
        item_embedding = model.embedding_lookup_n_chunk(model.item_embeddings, page_emb_to_word_emb_tensor).detach().numpy()
        index = json.loads(str(saved_embeddings['emb2page']))
    elif saved_embeddings['entity_type'] == "page":
        user_embedding = saved_embeddings['user_embeddings']
        item_embedding = saved_embeddings['item_embeddings']
        index = json.loads(str(saved_embeddings['emb2page_over_threshold']))
    else:
        raise Exception(f"unknown entity_type '{saved_embeddings['entity_type']}'")

    df_embedding = (
        pd.DataFrame({
            'user_embedding':list(user_embedding), 
            'user_embedding_normalized':list(normalize(user_embedding)), 
            'item_embedding':list(item_embedding), 
            'item_embedding_normalized':list(normalize(item_embedding)), 
            }, index = index)
        .merge(
            df_page.drop_duplicates('page_id')
                .dropna(subset=['page_title'])
                [['page_id', 'page_title']], 
            left_index = True, right_on = 'page_id')
        .set_index('page_title')
    )
    
    #kdt = KDTree(np.vstack(df_embedding[f"{'user' if use_user_emb else 'item'}_embedding_normalized"]), leaf_size=100, metric='euclidean')

    nn = NearestNeighbors(n_neighbors=k, algorithm=algorithm, leaf_size=1000, n_jobs=-1, p=2)
    nn = nn.fit(np.vstack(df_embedding[f"{'user' if use_user_emb else 'item'}_embedding_normalized"]))
    return df_embedding, nn


def top_knn(nn, df_embedding, pos_keys = None, neg_keys = None, use_user_emb = True):
    if pos_keys is None:
        pos_keys = []
    if neg_keys is None:
        neg_keys = []
    emb_name = 'user_embedding' if use_user_emb else 'item_embedding'
    emb_dim = len(df_embedding[emb_name].iat[0])
    new_v = np.zeros(emb_dim)
    for key in pos_keys:
        new_v += df_embedding.loc[key,emb_name] 
    for key in neg_keys:
        new_v -= df_embedding.loc[key,emb_name] 
    new_v = normalize(new_v)
    dist, ind = nn.kneighbors([new_v], return_distance=True)
    return df_embedding.iloc[ind[0],:].assign(dist = dist[0])[['dist']].transpose()