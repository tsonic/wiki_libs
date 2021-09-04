from wiki_libs.preprocessing import convert_to_w2v_mimic_path, read_link_pairs_chunks, path_decoration, append_suffix_to_fname
import pandas as pd
import json
from wiki_libs.cache import get_from_cached_file
from wiki_libs.ngram import get_all_words_for_embedding
import itertools
import pickle


class PageWordStats(object):
    def __init__(self, read_path,
                 ngram_model=None,
                 output_path = "wiki_data/page_word_stats.json",
                 n_chunk = 10,
                 w2v_mimic = False,
                 json_str = None,
                 text_source = 'both',
                 stats_column = 'both',
                 category_single_word = False,
                 category_embedding = False, 
                 no_stop_words = False,
                 ):

        if read_path is not None:
            with open(read_path, 'r') as f:
                config = json.load(f)
                #self.from_json(config)
        elif json_str is not None:
            config = json.loads(json_str)
            #self.from_json(config)
        else:
            # recomptue the target page stats

            gen = read_link_pairs_chunks(n_chunk = n_chunk, w2v_mimic =w2v_mimic)
            print('generating page id stats...')
            s_page = []
            for df_chunk in gen:
                if stats_column == 'both':
                    val_counts = df_chunk['page_id_target'].append(df_chunk['page_id_source']).value_counts()
                elif stats_column == 'source':
                    val_counts = df_chunk['page_id_source'].value_counts()
                elif stats_column == 'target':
                    val_counts = df_chunk['page_id_target'].value_counts()
                else:
                    raise Exception(f'Unknown stats_columns: "{stats_column}"')
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

            if w2v_mimic:
                print('Each word is a "page" in w2v_mimic mode. Refer to page count as the actual word count, and discard the word count output below. ')
                s_words = pd.Series(['dummy']).value_counts()
            else:
                df_title_category_transformed_list = get_all_words_for_embedding(text_source, category_single_word, category_embedding, no_stop_words)
                
                s_words = [(
                    pd.Series(itertools.chain(*df_title_category_transformed['page_title_category_transformed'].tolist()))
                    .value_counts()
                ) for df_title_category_transformed in df_title_category_transformed_list]
            # word_frequency is a list, where ith element is the word frequency of word with embedding id i.
            self.word_frequency = [{w:c for w, c in sw.iteritems()} for sw in s_words]
            # with open(output_path, 'w') as f:
            #     f.write(self.to_json_str())
            self.write_pickle(output_path)
        print(f'There are {[len(w) for w in self.word_frequency]} unique words/ngrams')
        print('There are %d unique pages' % len(self.page_frequency))
    
    # def from_json(self, config):
    #     self.word_frequency = config['word_frequency']
    #     self.page_frequency = {int(k):v for k, v in config['page_frequency'].items()}

    # def to_json_str(self):   
    #     return json.dumps({
    #         'word_frequency': self.word_frequency,
    #         'page_frequency': self.page_frequency,
    #     })

    def write_pickle(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def from_pickle(read_path):
        with open(read_path, 'rb') as f:
            ret = pickle.load(f)
        return ret