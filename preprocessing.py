import pandas as pd
from zipfile import ZipFile

def read_category_links():
    return read_zip_files('gdrive/My Drive/Projects with Wei/wiki_data/categorylinks_page_merged.zip', sep = ',')

def read_zip_files(path, sep = ','):
    zip_file = ZipFile(path)
    return pd.concat([pd.read_csv(zip_file.open(z.filename), sep=sep) 
                        for z in zip_file.infolist() if not z.is_dir()])

def process_title(s):
  return s.lower().replace('(','').replace(')','').replace(',','').split(sep='_')