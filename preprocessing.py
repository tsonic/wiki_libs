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

def is_ascii(s):
    return all(ord(c) < 128 for c in s)