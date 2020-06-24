import pandas as pd
from zipfile import ZipFile
from gensim.models.phrases import Phrases, Phraser
import itertools

def read_category_links():
    return read_zip_files('gdrive/My Drive/Projects with Wei/wiki_data/categorylinks_page_merged.zip', sep = ',')

def read_zip_files(path, sep = ','):
    zip_file = ZipFile(path)
    return pd.concat([pd.read_csv(zip_file.open(z.filename), sep=sep) 
                        for z in zip_file.infolist() if not z.is_dir()])

def process_title(s):
  return s.lower().replace('(','').replace(')','').replace(',','').split(sep='_')

def train_phraser(sentences, min_count=5):
    return Phraser(Phrases(sentences, min_count=min_count, delimiter=b'_'))

def train_ngram(sentences, n=3):
    ngram_model = []
    for i in range(1, n):
        xgram = train_phraser(sentences, min_count=5)
        sentences = xgram[sentences]
        ngram_model.append(xgram)
    return ngram_model

def transform_ngram(sentences, ngram_model):
    for m in ngram_model:
        sentences = m[sentences]
    return list(sentences)

def generate_vocab(sentences, min_count = 2):
    all_words = list(itertools.chain(*sentences))
    df_all_words = pd.DataFrame.from_records([{'word':w, 'ngram':len(w.split(sep='_'))} for w in all_words])
    vocab = df_all_words['word'].value_counts().to_frame('count').query('count>@min_count').index.tolist()
    return vocab

def is_ascii(s):
    return all(ord(c) < 128 for c in s)