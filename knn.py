import torch
from wiki_libs.preprocessing import normalize, path_decoration
from wiki_libs.models import OneTower
from sklearn.neighbors import KDTree, NearestNeighbors
import pandas as pd
import json
import numpy as np
import pdb
import gc

import faiss

class FaissKNeighbors:
    def __init__(self, n_neighbors=5, device = 'cpu'):
        self.index = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.device = device

    def fit(self, X):

        res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatL2(X.shape[1])
        if self.device == 'gpu':
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X, n_neighbors = None, return_distance = False):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        distances, indices = self.index.search(X.astype(np.float32), k=n_neighbors)
        if not return_distance:
            return indices
        return distances, indices

def build_knn(emb_file, df_page, w2v_mimic, emb_name = 'item_embedding', algorithm = 'brute', k = 10, device = 'cpu'):
    if isinstance(emb_file, str):
        emb_file = path_decoration(emb_file, w2v_mimic = w2v_mimic)
        saved_embeddings = torch.load(emb_file, map_location = 'cpu')
    else:
        # emb_file can be the output_dict embedding object directly
        saved_embeddings = emb_file
        
    USER_ID = 'page_id'
    ITEM_ID = 'page_id'

    if saved_embeddings['entity_type'] == "word":
        model = OneTower(**saved_embeddings['model_init_kwargs'])
        model.load_state_dict(saved_embeddings['model_state_dict'])

        page_emb_to_word_emb_tensor = saved_embeddings['page_emb_to_word_emb_tensor']

        user_embedding = model.forward_to_user_embedding_layer(page_emb_to_word_emb_tensor, in_chunks=True).detach().numpy()
        item_embedding = model.forward_to_user_embedding_layer(page_emb_to_word_emb_tensor, in_chunks=True, user_tower = True).detach().numpy()
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
            #'user_embedding_normalized':list(normalize(user_embedding)), 
            'item_embedding':list(item_embedding), 
            #'item_embedding_normalized':list(normalize(item_embedding)), 
            }, index = index)
        .merge(
            df_page.drop_duplicates('page_id')
                .dropna(subset=['page_title'])
                [['page_id', 'page_title']], 
            left_index = True, right_on = 'page_id')
        .set_index('page_title')
    )
    del user_embedding, item_embedding, model
    gc.collect()
    torch.cuda.empty_cache()

    
    #kdt = KDTree(np.vstack(df_embedding[f"{'user' if use_user_emb else 'item'}_embedding_normalized"]), leaf_size=100, metric='euclidean')
    if algorithm == "faiss":
        nn = FaissKNeighbors(n_neighbors=k, device=device)
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm=algorithm, leaf_size=100, n_jobs=-1, p=2)
    nn.fit(normalize(np.vstack(df_embedding[emb_name])))
    return df_embedding, nn

def top_k(inputs, nn, df_embedding,  k, input_type = 'index', output_type = 'index',
             pos_keys = None, neg_keys = None, use_user_emb = True):
    if pos_keys is None:
        pos_keys = []
    if neg_keys is None:
        neg_keys = []
    # the raw embedding still need vector operations, 
    # so does not directly use e.g. 'user_embedding_normalized'
    # this avoid scaling up vectors of regular words like 'the', 'and'
    emb_name = 'user_embedding' if use_user_emb else 'item_embedding'
    if isinstance(inputs, str) or isinstance(inputs, int):
        inputs = [inputs]
    df_embedding = df_embedding.reset_index()
    if input_type != 'index':
        df_embedding = df_embedding.set_index(input_type)

    new_v = np.vstack(df_embedding.loc[inputs, emb_name].values)
    for key in pos_keys:
        # np.array is for broadcast add
        new_v += np.array([df_embedding.loc[key, emb_name].values])
    for key in neg_keys:
        new_v -= np.array([df_embedding.loc[key, emb_name].values])
    # normalize again after vector operation
    new_v = normalize(new_v)
    dist, ind = nn.kneighbors(new_v, n_neighbors = k, return_distance=True)
    if output_type == 'index':
        return dist, ind
    df_embedding = df_embedding.reset_index()
    
    return dist, df_embedding[output_type].values[ind]



def top_knn(nn, df_embedding, pos_keys = None, neg_keys = None, use_user_emb = True):
    if pos_keys is None:
        pos_keys = []
    if neg_keys is None:
        neg_keys = []
    # the raw embedding still need vector operations, 
    # so does not directly use e.g. 'user_embedding_normalized'
    # this avoid scaling up vectors of regular words like 'the', 'and'
    emb_name = 'user_embedding' if use_user_emb else 'item_embedding'
    emb_dim = len(df_embedding[emb_name].iat[0])
    new_v = np.zeros(emb_dim)
    for key in pos_keys:
        new_v += df_embedding.loc[key,emb_name] 
    for key in neg_keys:
        new_v -= df_embedding.loc[key,emb_name] 
    # normalize again after vector operation
    new_v = normalize(new_v)
    dist, ind = nn.kneighbors(np.array([new_v]), return_distance=True)
    return df_embedding.iloc[ind[0],:].assign(dist = dist[0])[['dist']].transpose()