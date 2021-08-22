from wiki_libs.preprocessing import convert_to_w2v_mimic_path, read_link_pairs_chunks, path_decoration, append_suffix_to_fname
import pandas as pd
import json
from wiki_libs.cache import get_from_cached_file
import itertools


class PageWordStats(object):
    def __init__(self, read_path,
                 ngram_model=None,
                 output_path = "wiki_data/page_word_stats.json",
                 n_chunk = 10,
                 w2v_mimic = False,
                 json_str = None,
                 title_only = False,
                 stats_column = 'both',
                 category_single_word = False,
                 ):

        if read_path is not None:
            # if stats_column == 'source':
            #     read_path = append_suffix_to_fname(read_path, '_source')
            # elif stats_column == 'target':
            #     read_path = append_suffix_to_fname(read_path, '_target')
            # read_path = path_decoration(read_path, w2v_mimic)
            # if title_only:
            #     read_path = append_suffix_to_fname(read_path, '_title_only')
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
                df_title_category_transformed = get_from_cached_file({'prefix':'df_title_category_transformed',
                                'ngram_model_key_dict':{'prefix':'ngram_model','title_only':title_only,'category_single_word':category_single_word}})
                
                s_words = (
                    pd.Series(itertools.chain(*df_title_category_transformed['page_title_category_transformed'].tolist()))
                    .value_counts()
                )
            # word_frequency is a list, where ith element is the word frequency of word with embedding id i.
            self.word_frequency = {w:c for w, c in s_words.iteritems()}
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
