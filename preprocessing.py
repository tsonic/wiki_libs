import pandas as pd
from zipfile import ZipFile
import zipfile
from gensim.models.phrases import Phrases, Phraser
import itertools
import pickle
import numpy as np
import os
from tqdm import tqdm

import sys

def is_colab():
    return 'google.colab' in sys.modules

def convert_to_colab_path(path):
    return '/content/gdrive/MyDrive/Projects with Wei/' + path

CATEGORY_LINKS_LOCATION = 'wiki_data/categorylinks_page_merged.zip'
PAGE_DATA_LOCATION = 'wiki_data/page_data.tsv'
LINK_PAIRS_LOCATION = 'wiki_data/link_pairs_shuffled_gz'


def read_page_data(w2v_mimic = False):
    path = path_decoration(PAGE_DATA_LOCATION, w2v_mimic)
    return pd.read_csv(path, sep = '\t', keep_default_na = False, na_values = [])

def read_category_links(w2v_mimic = False):
    path = path_decoration(CATEGORY_LINKS_LOCATION, w2v_mimic)
    return (
        next(read_files_in_chunks(path, 
                sep = ',', compression = 'zip', n_chunk = 1, progress_bar=False))
        .fillna({'page_title':'', 'page_category':''})
    )

def read_link_pairs_chunks(n_chunk = 10, w2v_mimic = False):
    path = path_decoration(LINK_PAIRS_LOCATION, w2v_mimic)
    print(f'reading link pairs in {n_chunk} chunks')
    gen = read_files_in_chunks(path, 
                                sep = ',', n_chunk = n_chunk, compression = None)
    if w2v_mimic == False:
        pages = set(read_page_data(w2v_mimic = w2v_mimic)['page_id'])
        for chunk in gen:
            yield chunk.query('page_id_source.isin(@pages) & page_id_target.isin(@pages)', engine = 'python')
    else:
        # if using w2v mimic dataset, there are no df_cp
        for chunk in gen:
            yield chunk

def get_file_handles_in_zip(f):
    zf = ZipFile(f)
    file_list_in_zip = [z.filename for z in zf.infolist() if not z.is_dir()]
    file_list_in_zip.sort()
    return [zf.open(f) for f in file_list_in_zip]

def get_files_in_dir(path):
    return [path+'/'+f for f in os.listdir(path) if not os.path.isdir(path+'/'+f)]

def read_files_in_chunks(path, sep = ',', compression = 'zip', n_chunk = 10, progress_bar = True, shuffle = False):
    file_handle_list = None
    if isinstance(path, list) or isinstance(path, np.ndarray):
        if len(path) == 0:
            return None
        if isinstance(path[0], zipfile.ZipExtFile) or isinstance(path[0], str):
            file_handle_list = path
        else:
            raise Exception('only support when path is a list of ZipExtFile (opened zipped file handle) or strings')

    elif isinstance(path, str):
        if os.path.isdir(path):
            base_files = get_files_in_dir(path)
        else:
            base_files = [path]

        base_files.sort()

        file_handle_list = []
        for f in base_files:
            if compression is None:
                file_handle_list.append(f)
            elif compression == 'zip':
                file_handle_list += get_file_handles_in_zip(f)
            else:
                raise Exception(f'Unkonwn compression type: {compression}')
    else:
        raise Exception("type %s for path is not supported!" % type(path))

    if shuffle:
        np.random.shuffle(file_handle_list)
    chunks = np.array_split(file_handle_list, min(n_chunk, len(file_handle_list)))
    if progress_bar:
        chunks = tqdm(chunks)
    for file_handles in chunks:
        df_list = []
        for fh in file_handles:
            df_list.append(pd.read_csv(fh, sep=sep))
            if isinstance(path[0], zipfile.ZipExtFile):
                fh.close()
        yield pd.concat(df_list)


def process_title(s):
  return s.lower().replace('(','').replace(')','').replace(',','').split(sep='_')

def generate_vocab(sentences, min_count = 2):
    all_words = list(itertools.chain(*sentences))
    df_all_words = pd.DataFrame.from_records([{'word':w, 'ngram':len(w.split(sep='_'))} for w in all_words])
    vocab = df_all_words['word'].value_counts().to_frame('count').query('count>@min_count').index.tolist()
    return vocab

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def convert_to_w2v_mimic_path(path):
    return append_suffix_to_fname(path, '_w2v_mimic')


def append_suffix_to_fname(path, suffix):
    path_segs = path.split('.')
    if len(path_segs) == 1 \
        or (len(path_segs) == 2 and path[:2] == './'):  #
        path_segs[-1] += suffix
    else:
        path_segs[-2] += suffix
    return '.'.join(path_segs)

def path_decoration(path, w2v_mimic):

    if is_colab():
        prefix = 'gdrive/My Drive/Projects with Wei/'
    else:
        prefix = './'
    path = prefix + path
    if w2v_mimic:
        path = convert_to_w2v_mimic_path(path)
    return path

def normalize(a):
    ndim = a.ndim
    denom = np.sqrt((a ** 2).sum(axis=ndim-1))
    if ndim > 1:
        # expand to a new axis
        denom = denom[:,np.newaxis]
    return a / denom