import pandas as pd
from zipfile import ZipFile
from gensim.models.phrases import Phrases, Phraser

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
    return sentences

def is_ascii(s):
    return all(ord(c) < 128 for c in s)