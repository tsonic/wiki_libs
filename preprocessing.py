import pandas as pd
from zipfile import ZipFile
from gensim.models.phrases import Phrases, Phraser
import itertools
import pickle
import numpy as np
import os

def read_category_links():
    return next(read_files_in_chunks('gdrive/My Drive/Projects with Wei/wiki_data/categorylinks_page_merged.zip', 
                sep = ',', compression = 'zip', n_chunk = 1))

def read_link_pairs_chunks(n_chunk = 10):
    print(f'reading link pairs in {n_chunk} chunks')
    return read_files_in_chunks('gdrive/My Drive/Projects with Wei/wiki_data/link_pairs.zip', 
                            sep = ',', n_chunk = n_chunk, compression = 'zip')

def read_files_in_chunks(path, sep = ',', compression = 'zip', n_chunk = 10):
    if os.path.isdir(path):
        base_files = [path+'/'+f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]
    else:
        base_files = [path]

    base_files.sort()

    file_handle_list = []
    for f in base_files:
        if compression is None:
            file_handle_list.append(f)
        elif compression == 'zip':
            zf = ZipFile(f)
            file_list_in_zip = [z.filename for z in zf.infolist() if not z.is_dir()]
            file_list_in_zip.sort()
            file_handle_list += [zf.open(z) for z in file_list_in_zip]
        else:
            raise Exception(f'Unkonwn compression type: {compression}')
    
    for file_handles in np.array_split(file_handle_list, n_chunk):
        yield pd.concat([pd.read_csv(fh, sep=sep) for fh in file_handles])                   

# def read_zip_files(path, sep = ',', n_chunk = 1):
#     zip_file = ZipFile(path)
#     files = [z.filename for z in zip_file.infolist() if not z.is_dir()]
#     files.sort()
#     # if n_chunk <= 1:
#     #     return pd.concat([pd.read_csv(zip_file.open(f), sep=sep) for f in files])
#     # else:
#     #     # return a generator if > 1 chunk
#     for f_list in np.array_split(files, n_chunk):
#         yield pd.concat([pd.read_csv(zip_file.open(f), sep=sep) for f in f_list])

def process_title(s):
  return s.lower().replace('(','').replace(')','').replace(',','').split(sep='_')

def train_phraser(sentences, min_count=5):
    return Phraser(Phrases(sentences, min_count=min_count, delimiter=b'_'))

def train_ngram(sentences, n=3, min_count=5, out_file='ngram_model.pickle'):
    ngram_model = []
    for i in range(1, n):
        xgram = train_phraser(sentences, min_count=min_count)
        sentences = xgram[sentences]
        ngram_model.append(xgram)
    pickle.dump(ngram_model, open(out_file, "wb"))
    return ngram_model

def transform_ngram(sentences, ngram_model):
    for m in ngram_model:
        sentences = m[sentences]
    return list(sentences)

def load_ngram_model(model_file):
    return pickle.load(open(model_file, "rb"))

def generate_vocab(sentences, min_count = 2):
    all_words = list(itertools.chain(*sentences))
    df_all_words = pd.DataFrame.from_records([{'word':w, 'ngram':len(w.split(sep='_'))} for w in all_words])
    vocab = df_all_words['word'].value_counts().to_frame('count').query('count>@min_count').index.tolist()
    return vocab

def is_ascii(s):
    return all(ord(c) < 128 for c in s)