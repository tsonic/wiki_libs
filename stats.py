from wiki_libs.preprocessing import convert_to_w2v_mimic_path, read_link_pairs_chunks, path_decoration
import pandas as pd
import json
from wiki_libs.ngram import load_ngram_model, NGRAM_MODEL_PATH_PREFIX, get_transformed_title_category
import itertools


class PageWordStats(object):
    def __init__(self, read_path,
                 ngram_model_file = "title_category_ngram_model.pickle",
                 output_path = "wiki_data/page_word_stats.json",
                 n_chunk = 10,
                 w2v_mimic = False,
                 json_str = None,
                 ):

        if read_path is not None:
            read_path = path_decoration(read_path, w2v_mimic)
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
            # page_frequency is the page_id to page count mapping.
            self.page_frequency = {row.page_id: row.count 
                                   for row in df_stats.itertuples(index = False)}

            # recompute the word stats
            print('generating word/ngram stats from title and categories...')

            ngram_model = load_ngram_model(path_decoration(NGRAM_MODEL_PATH_PREFIX, False) + '/' + ngram_model_file)
            if w2v_mimic:
                print('Each word is a "page" in w2v_mimic mode. Refer to page count as the actual word count, and discard the word count output below. ')
                s_words = pd.Series(['dummy']).value_counts()
            else:
                title_transformed, category_transformed = get_transformed_title_category(ngram_model)
                s_words = (
                    pd.Series(itertools.chain(*(title_transformed + category_transformed)))
                    .value_counts()
                )
            # word_frequency is a list, where ith element is the word frequency of word with embedding id i.
            self.word_frequency = {w:c for w, c in s_words.iteritems()}

            output_path = path_decoration(output_path, w2v_mimic)
            with open(output_path, 'w') as f:
                f.write(self.to_json_str())
        print('There are %d unique words/ngrams' % len(self.word_frequency))
        print('There are %d unique pages' % len(self.page_frequency))
    
    def from_json(self, config):
        self.word_frequency = {k:v for k, v in config['word_frequency'].items()}
        self.page_frequency = {int(k):v for k, v in config['page_frequency'].items()}

    def to_json_str(self):   
        return json.dumps({
            'word_frequency': self.word_frequency,
            'page_frequency': self.page_frequency,
        })
