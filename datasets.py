import torch
import pandas as pd
from wiki_libs.preprocessing import (
        read_files_in_chunks, read_category_links, 
        process_title, path_decoration, 
        read_page_data, append_suffix_to_fname,
        is_colab, convert_to_colab_path
    )
#from wiki_libs.ngram import NGRAM_MODEL_PATH_PREFIX, load_ngram_model, generate_df_title_category_transformed, transform_ngram
from wiki_libs.cache import get_from_cached_file
import numpy as np

NEGATIVE_TABLE_SIZE = 5e7
class WikiDataset(torch.utils.data.IterableDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_list, compression, n_chunk, num_negs, ns_exponent, w2v_mimic,
                page_min_count = 0, word_min_count = 0, entity_type='page', title_category_trunc_len = 50,stats_column = 'both',
                title_only = False, in_batch_neg = False,
                neg_sample_prob_corrected = False, category_single_word = False,
                ):
        'Initialization'
        #self.labels = labels
        # self.list_IDs = list_IDs
        # if isinstance(file_list, str):
        #     file_list = [file_list]


        # torch.use_deterministic_algorithms(True)

        self.file_list = file_list
        self.compression = compression
        self.n_chunk = n_chunk
        self.pos = 0

        self.chunk_iterator = None
        self.instance_dict = None
        self.page_min_count = page_min_count
        self.word_min_count = word_min_count


        self.negatives = []
        self.negpos = 0
        self.num_negs = num_negs

        self.w2v_mimic = w2v_mimic
        self.title_only = title_only
        self.stats_column = stats_column
        self.category_single_word = category_single_word
        page_word_stats = get_from_cached_file({'prefix':'page_word_stats', 
                                                    'w2v_mimic':self.w2v_mimic, 
                                                    'stats_column':self.stats_column,
                                                    'ngram_model_key_dict':{'prefix':'ngram_model','title_only':self.title_only, 'category_single_word':self.category_single_word},
                                                    })
        

        self.title_category_trunc_len = title_category_trunc_len

        self.entity_type = entity_type

        self.in_batch_neg = in_batch_neg
        self.neg_sample_prob_corrected = neg_sample_prob_corrected

        if self.entity_type == 'page':
            entity_frequency = page_word_stats.page_frequency
            self.min_count = self.page_min_count
            # self.title_category_transformed_dict = None
        else:
            entity_frequency = page_word_stats.word_frequency
            self.min_count = self.word_min_count

        if self.entity_type == 'page':
            self.truncate_tail_page = True
        else:
            self.truncate_tail_page = False

        page_frequency_, word_frequency, emb2page, emb2word, page2emb, word2emb = WikiDataset.generate_page_word_emb_index(page_word_stats)

        self.entity_frequency = {p:c for p, c in entity_frequency.items()}
        self.page_frequency = page_frequency_
        self.word_frequency = word_frequency

        self.emb2entity = list(self.entity_frequency.keys())
        self.emb2page = emb2page
        self.emb2word = emb2word

        self.entity_frequency_over_threshold = {p:c for p, c in entity_frequency.items() if c > self.min_count}
        self.page_frequency_over_threshold = {p:c for p, c in page_word_stats.page_frequency.items() if c > self.page_min_count}
        self.word_frequency_over_threshold = {p:c for p, c in page_word_stats.word_frequency.items() if c > self.word_min_count}

        # emb2page maps pytorch embedding back to 'page_id'
        self.emb2entity_over_threshold = list(self.entity_frequency_over_threshold.keys())
        self.emb2page_over_threshold = list(self.page_frequency_over_threshold.keys())
        self.emb2word_over_threshold = list(self.word_frequency_over_threshold.keys())


        # page2emb is 'page_id' to pytorch embedding index mapping
        self.entity2emb_over_threshold = {p:i for i, p in enumerate(self.emb2entity_over_threshold)}
        self.page2emb_over_threshold = {p:i for i, p in enumerate(self.emb2page_over_threshold)}
        self.word2emb_over_threshold = {p:i for i, p in enumerate(self.emb2word_over_threshold)}

        self.entity2emb = {p:i for i, p in enumerate(self.emb2entity)}
        self.page2emb = page2emb
        self.word2emb = word2emb

        self.cp_pages = set(read_page_data(w2v_mimic = self.w2v_mimic)['page_id'])
        
        if self.entity_type == 'word':
            self.page_emb_to_word_emb_tensor = get_from_cached_file({'prefix':'page_emb_to_word_emb_tensor', 
            'page_word_stats_key_dict': {'ngram_model_key_dict':{'prefix':'ngram_model','title_only':self.title_only,'category_single_word':self.category_single_word}},
            'title_category_trunc_len':self.title_category_trunc_len,
            })
        
        if entity_type == "page":
            page_id, i = zip(*self.page2emb_over_threshold.items())
        else:
            page_id, i = zip(*self.page2emb.items())
        self.page2emb_series_map = pd.Series(i, index=page_id, dtype=np.int64)

        print(f'Number of unique entities ({entity_type}) included is {len(self.entity_frequency_over_threshold)}')
        print(f'Number of unique pages included is {len(self.page_frequency_over_threshold)}')
        print(f'Number of unique words included is {len(self.word_frequency_over_threshold)}')

        self.initTableNegatives(ns_exponent=ns_exponent)

    @staticmethod
    def generate_page_emb_to_word_emb_tensor(output_path, page_word_stats, title_only, category_single_word, title_category_trunc_len):
        _, _, emb2page, emb2word, page2emb, word2emb = WikiDataset.generate_page_word_emb_index(page_word_stats)
        print('start generating page_emb_to_word_emb_tensor', flush = True)
        import time
        st = time.time()
        page_emb_to_word_emb_tensor = torch.LongTensor(
            get_from_cached_file({'prefix':'df_title_category_transformed',
                                'ngram_model_key_dict':{'prefix':'ngram_model','title_only':title_only,'category_single_word':category_single_word}})
            .set_index('page_id')
            ['page_title_category_transformed']
            # the last row in the embedding is padded 0 vector
            .apply(lambda x: [word2emb[x[i]] if i < len(x) else len(emb2word) for i in range(title_category_trunc_len)])
            .reindex(index = emb2page) #self.emb2page is a vector
            .tolist()
        )
        
        et = time.time()

        print(f'Finish generating page_emb_to_word_emb_tensor, took {et - st}', flush = True)
        torch.save(page_emb_to_word_emb_tensor, output_path)
        print(f'saved page_emb_to_word_emb_tensor to "{output_path}".')
        return page_emb_to_word_emb_tensor

    @staticmethod
    def generate_page_word_emb_index(page_word_stats):
        page_frequency = {p:c for p, c in page_word_stats.page_frequency.items()}
        word_frequency = {p:c for p, c in page_word_stats.word_frequency.items()}

        emb2page = list(page_frequency.keys())
        emb2word = list(word_frequency.keys())

        page2emb = {p:i for i, p in enumerate(emb2page)}
        word2emb = {p:i for i, p in enumerate(emb2word)}

        return page_frequency, word_frequency, emb2page, emb2word, page2emb, word2emb
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.chunk_iterator is None:
            self.chunk_iterator = read_files_in_chunks(self.file_list, compression=self.compression, 
                                                       n_chunk = self.n_chunk, progress_bar = True)
            
        if self.instance_dict is None or self.pos >= len(self.instance_dict):
            df = next(self.chunk_iterator)
            df = df.query('page_id_source.isin(@self.cp_pages) & page_id_target.isin(@self.cp_pages)', engine = 'python')
            # have to reset self.pos after pull next in above iterator, otherwise
            # it does not invalidate current instance_dict after iterator is exhausted.
            self.pos = 0
            
            if self.entity_type == "page":
                # remove page ids that is below the min count threshold
                # truncate tail page
                df = df.query('page_id_source.isin(@self.page_frequency_over_threshold) & page_id_target.isin(@self.page_frequency_over_threshold)', engine = 'python')
            df = df.assign(
                    page_id_source = lambda df: df['page_id_source'].map(self.page2emb_series_map),
                    page_id_target = lambda df: df['page_id_target'].map(self.page2emb_series_map),
                )
            df = df.sample(frac=1.0, replace = False)
            self.instance_dict = list(df.itertuples(index = False, name = None))
        ret = self.instance_dict[self.pos]
        self.pos += 1
        return ret

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        #regenerate negative table if negpos > total neg table size.
        neg_table_len = len(self.negatives)
        if (self.negpos + size) // neg_table_len >= 1:
            self.regenerateNegTable()
        self.negpos = (self.negpos + size) % neg_table_len
        if len(response) != size:
            response = np.concatenate((response, self.negatives[0:self.negpos]))
        # below using random choice with weight is too slow
        # np.random.choice(len(self.neg_sample_prob), size = shape, replace = True, p = self.neg_sample_prob)

        return response

    def initTableNegatives(self, ns_exponent):
        print('Initializing negative samples', flush=True)
        # embedding id to page counts mapping
        use_page_negative = True
        if self.entity_type == 'word' and use_page_negative:
            self.neg_sample_prob = np.array([self.page_frequency[e] for e in self.emb2page]).astype(np.float32)
            self.neg_sample_prob = self.neg_sample_prob ** ns_exponent
            self.neg_sample_prob = self.neg_sample_prob / self.neg_sample_prob.sum()
        else:
            self.neg_sample_prob = np.array([self.entity_frequency_over_threshold[e] for e in self.emb2entity_over_threshold]).astype(np.float32)
            self.neg_sample_prob = self.neg_sample_prob ** ns_exponent
            self.neg_sample_prob = self.neg_sample_prob / self.neg_sample_prob.sum()
        
        self.regenerateNegTable()

    def regenerateNegTable(self):
        self.negatives = np.repeat(range(len(self.neg_sample_prob)), self.prob_round(self.neg_sample_prob * NEGATIVE_TABLE_SIZE))
        np.random.shuffle(self.negatives)

    def prob_round(self, a):
        return np.floor(a + np.random.rand(len(a))).astype(int)

    def collate(self,batches):
        
        id_list, positive_list = zip(*batches)
        ####### input are page embedding index, instead of page id.
        pos_u = torch.LongTensor(id_list)
        pos_v = torch.LongTensor(positive_list)
        neg_v = None
        if not self.in_batch_neg:
            negs = self.getNegatives(None, self.num_negs * len(batches)).reshape((len(batches), self.num_negs))
            neg_v = torch.from_numpy(negs)
        pos_v_page = None
        neg_v_page = None
        if self.neg_sample_prob_corrected:
            pos_v_page = pos_v
            neg_v_page = neg_v
        if self.entity_type != 'page':
            pos_u = self.page_emb_to_word_emb_tensor[pos_u]
            pos_v = self.page_emb_to_word_emb_tensor[pos_v]
            if not self.in_batch_neg:
                neg_v = self.page_emb_to_word_emb_tensor[neg_v]


        return pos_u, pos_v, neg_v, pos_v_page, neg_v_page

    @staticmethod
    def worker_init_fn(worker_id, file_handle_lists):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        worker_id = worker_info.id
        # the new_seed are the same after calling method for multiple time, e.g. in different epoches.
        new_seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(new_seed) 
        np.random.shuffle(dataset.negatives)
        dataset.file_list = file_handle_lists[worker_id]