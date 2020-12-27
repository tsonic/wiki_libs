from wiki_libs.preprocessing import convert_to_w2v_mimic_path, read_link_pairs_chunks
import pandas as pd
import json
from wiki_libs.ngram import load_ngram_model, NGRAM_MODEL_PATH_PREFIX, get_transformed_title_category
import itertools


class PageWordStats(object):
    def __init__(self, read_path,
                 ngram_model_file = "title_category_ngram_model.pickle",
                 output_path = "gdrive/My Drive/Projects with Wei/wiki_data/page_word_stats.json",
                 n_chunk = 10,
                 w2v_mimic = False,
                 json_str = None,
                 ):

        if read_path is not None:
            if w2v_mimic:
                read_path = convert_to_w2v_mimic_path(read_path)
            with open(read_path, 'r') as f:
                config = json.load(f)
                self.from_json(config)
        elif json_str is not None:
            config = json.loads(json_str)
            self.from_json(config)
        else:
            # recomptue the target page stats

            gen = read_link_pairs_chunks(n_chunk = n_chunk, w2v_mimic =w2v_mimic)
            print('generating page id stats...')
            s_page = []
            for df_chunk in gen:
                val_counts = df_chunk['page_id_target'].append(df_chunk['page_id_source']).value_counts()
                s_page.append(val_counts)
            df_stats = (
                pd.concat(s_page)
                .rename_axis("page_id")  #rename index
                .to_frame('count')
                .groupby("page_id")
                    .sum()
                .reset_index()
            )
            self.page_frequency = {row.page_id: row.count 
                                   for row in df_stats.itertuples(index = False)}
            # page2id is 'page_id' to pytorch embedding index mapping
            self.page2id = {p:i for i, p in enumerate(df_stats['page_id'])}
            # id2page maps pytorch embedding back to 'page_id'
            self.id2page = df_stats['page_id'].tolist()

            # recompute the word stats
            print('generating word/ngram stats from title and categories...')

            ngram_model = load_ngram_model(NGRAM_MODEL_PATH_PREFIX + ngram_model_file)
            if w2v_mimic:
                s_words = pd.Series(['dummy']).value_counts()
            else:
                title_transformed, category_transformed = get_transformed_title_category(ngram_model)
                s_words = (
                    pd.Series(itertools.chain(*(title_transformed + category_transformed)))
                    .value_counts()
                )
            # word_frequency is a list, where ith element is the word frequency of word with id i.
            self.word_frequency = s_words.tolist()
            self.word2id = {w:i for i, w in enumerate(s_words.index)}
            self.id2word = s_words.index.tolist()
            if w2v_mimic:
                output_path = convert_to_w2v_mimic_path(output_path)
            with open(output_path, 'w') as f:
                f.write(self.to_json_str())
        print('There are %d unique words/ngrams' % len(self.word2id))
        print('There are %d unique pages' % len(self.page2id))
    
    def from_json(self, config):
        self.word_frequency = config['word_frequency']
        self.word2id = config['word2id']
        self.id2word = config['id2word']
        self.page_frequency = {int(k):v for k, v in config['page_frequency'].items()}

        self.page2id = {int(k):v for k, v in config['page2id'].items()}
        self.id2page = config['id2page']

    def to_json_str(self):   
        return json.dumps({
            'word_frequency': self.word_frequency,
            'word2id': self.word2id,
            'id2word':self.id2word,
            'page_frequency': self.page_frequency,
            'page2id': self.page2id,
            'id2page': self.id2page,
        })
