import pickle
from gensim.models.phrases import Phrases, Phraser
from wiki_libs.preprocessing import read_category_links, process_title, path_decoration


NGRAM_MODEL_PATH_PREFIX = "wiki_data/ngram_model"
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

def get_transformed_title_category(ngram_model):
    df_cp = read_category_links()
    titles = df_cp['page_title'].dropna().drop_duplicates().apply(process_title).tolist()
    categories = df_cp['page_category'].dropna().drop_duplicates().apply(process_title).tolist()
    title_transformed = transform_ngram(titles, ngram_model)
    category_transformed = transform_ngram(categories, ngram_model)
    return title_transformed, category_transformed

def load_ngram_model(model_file):
    return pickle.load(open(model_file, "rb"))