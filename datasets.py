import torch
import pandas as pd
from wiki_libs.preprocessing import read_files_in_chunks, read_category_links, process_title, path_decoration, read_page_data, append_suffix_to_fname 
from wiki_libs.ngram import NGRAM_MODEL_PATH_PREFIX, load_ngram_model, get_df_title_category_transformed, transform_ngram
import numpy as np


NEGATIVE_TABLE_SIZE = 1e8
class WikiDataset(torch.utils.data.IterableDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_list, compression, n_chunk, num_negs, page_word_stats, ns_exponent, w2v_mimic,
                page_min_count = 0, word_min_count = 0, entity_type='page', title_category_trunc_len = 50,
                ngram_model_name = "title_category_ngram_model.pickle", 
                page_emb_to_word_emb_tensor_fname = None, title_only = False,
                ):
        'Initialization'
        #self.labels = labels
        # self.list_IDs = list_IDs
        # if isinstance(file_list, str):
        #     file_list = [file_list]
        self.file_list = file_list
        self.compression = compression
        self.n_chunk = n_chunk
        self.pos = 0

        self.chunk_iterator = None
        self.instance_dict = None
        self.page_min_count = page_min_count
        self.word_min_count = word_min_count
        self.ngram_model_name = ngram_model_name
        self.build_transformed_title_category = page_emb_to_word_emb_tensor_fname is None

        self.negatives = []
        self.negpos = 0
        self.num_negs = num_negs
        self.page_word_stats = page_word_stats

        self.title_category_trunc_len = title_category_trunc_len

        self.entity_type = entity_type
        self.title_only = title_only

        if self.entity_type == 'page':
            entity_frequency = page_word_stats.page_frequency
            self.min_count = self.page_min_count
            # self.title_category_transformed_dict = None
        else:
            entity_frequency = page_word_stats.word_frequency
            self.min_count = self.word_min_count
            # self.title_category_transformed_dict = (
            #     get_df_title_category_transformed(
            #         read_cached=not self.build_transformed_title_category,
            #         ngram_model_name=ngram_model_name)
            #     .assign(page_title_category_transformed = lambda df: 
            #             df['page_title_category_transformed'].apply(lambda x: [self.entity2emb[e] for e in x]))
            #     .set_index('page_id')
            #     ['page_title_category_transformed']
            #     .to_dict()
            # )

            
            #print(f'finish loading title category transformed, size {len(self.title_category_transformed_dict)}', flush = True)

        if self.entity_type == 'page':
            self.truncate_tail_page = True

        else:
            self.truncate_tail_page = False

        self.entity_frequency = {p:c for p, c in entity_frequency.items()}
        self.page_frequency = {p:c for p, c in page_word_stats.page_frequency.items()}
        self.word_frequency = {p:c for p, c in page_word_stats.word_frequency.items()}
        self.entity_frequency_over_threshold = {p:c for p, c in entity_frequency.items() if c > self.min_count}
        self.page_frequency_over_threshold = {p:c for p, c in page_word_stats.page_frequency.items() if c > self.page_min_count}
        self.word_frequency_over_threshold = {p:c for p, c in page_word_stats.word_frequency.items() if c > self.word_min_count}

        # emb2page maps pytorch embedding back to 'page_id'
        self.emb2entity_over_threshold = list(self.entity_frequency_over_threshold.keys())
        self.emb2page_over_threshold = list(self.page_frequency_over_threshold.keys())
        self.emb2word_over_threshold = list(self.word_frequency_over_threshold.keys())

        self.emb2entity = list(self.entity_frequency.keys())
        self.emb2page = list(self.page_frequency.keys())
        self.emb2word = list(self.word_frequency.keys())
        # page2emb is 'page_id' to pytorch embedding index mapping
        self.entity2emb_over_threshold = {p:i for i, p in enumerate(self.emb2entity_over_threshold)}
        self.page2emb_over_threshold = {p:i for i, p in enumerate(self.emb2page_over_threshold)}
        self.word2emb_over_threshold = {p:i for i, p in enumerate(self.emb2word_over_threshold)}

        self.entity2emb = {p:i for i, p in enumerate(self.emb2entity)}
        self.page2emb = {p:i for i, p in enumerate(self.emb2page)}
        self.word2emb = {p:i for i, p in enumerate(self.emb2word)}

        self.w2v_mimic = w2v_mimic
        self.cp_pages = set(read_page_data(w2v_mimic = self.w2v_mimic)['page_id'])
        
        if self.entity_type == 'word':

            if page_emb_to_word_emb_tensor_fname is not None:
                # load tensor from cached
                path = f'wiki_data/{page_emb_to_word_emb_tensor_fname}'
                if self.title_only:
                    path = append_suffix_to_fname(path, '_title_only')
                self.page_emb_to_word_emb_tensor = torch.load(path, map_location = 'cpu')
                print(f'read page_emb_to_word_emb_tensor from "{path}".')
            if page_emb_to_word_emb_tensor_fname is None or self.page_emb_to_word_emb_tensor.shape[1] != self.title_category_trunc_len:
                # regenerate tensor if the cached version width disagrees with pass in value
                if page_emb_to_word_emb_tensor_fname is not None:
                    print(f'the cached page_emb_to_word_emb_tensor has width {self.page_emb_to_word_emb_tensor.shape[1]}, '
                    f'but title_category_trunc_len is {title_category_trunc_len}')
                print('start generating page_emb_to_word_emb_tensor', flush = True)
                import time
                st = time.time()
                self.page_emb_to_word_emb_tensor = (
                    get_df_title_category_transformed(
                        read_cached=not self.build_transformed_title_category,
                        ngram_model_name=ngram_model_name,
                        title_only = self.title_only,
                        )
                    .set_index('page_id')
                    ['page_title_category_transformed']
                    # the last row in the embedding is padded 0 vector
                    .apply(lambda x: [self.word2emb[x[i]] if i < len(x) else len(self.emb2word) for i in range(self.title_category_trunc_len)])
                )
                self.page_emb_to_word_emb_tensor = torch.LongTensor(
                    self.page_emb_to_word_emb_tensor
                    .reindex(index = self.emb2page) #self.emb2page is a vector
                    .tolist()
                )
                
                et = time.time()
                path = f'wiki_data/page_emb_to_word_emb_tensor.npz'
                if self.title_only:
                    path = append_suffix_to_fname(path, '_title_only')
                print(f'Finish generating page_emb_to_word_emb_tensor, took {et - st}', flush = True)
                torch.save(self.page_emb_to_word_emb_tensor, path)
                print(f'saved page_emb_to_word_emb_tensor to "{path}".')
            
        
        if entity_type == "page":
            page_id, i = zip(*self.page2emb_over_threshold.items())
        else:
            page_id, i = zip(*self.page2emb.items())
        self.page2emb_series_map = pd.Series(i, index=page_id, dtype=np.int64)

        print(f'Number of unique entities ({entity_type}) included is {len(self.entity_frequency_over_threshold)}')
        print(f'Number of unique pages included is {len(self.page_frequency_over_threshold)}')
        print(f'Number of unique words included is {len(self.word_frequency_over_threshold)}')

        self.initTableNegatives(ns_exponent=ns_exponent)

    
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
        # reshuffle negative table if negpos > total neg table size.
        if (self.negpos + size) // len(self.negatives) >= 1:
            print('reshuffle negative table...')
            np.random.shuffle(self.negatives)
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def initTableNegatives(self, ns_exponent):
        print('Initializing negative samples', flush=True)
        # embedding id to page counts mapping
        use_page_negative = True
        if self.entity_type == 'word' and use_page_negative:
            #################### May need to convert to page embedding
            page_id, page_counts = zip(*self.page_frequency.items())
            ratio = np.array(page_counts).astype(np.float64) ** ns_exponent / sum(page_counts)
            sampled_count = np.round(ratio * NEGATIVE_TABLE_SIZE).astype(np.int64)
            page_emb = [self.page2emb[p] for p in page_id]
            self.negatives = np.repeat(page_emb, sampled_count)
        else:
            ## entity_count already uses embedding index
            entity_counts = [self.entity_frequency_over_threshold[entity_id] for entity_id in self.emb2entity_over_threshold]
            ratio = np.array(entity_counts).astype(np.float64) ** ns_exponent / sum(entity_counts)
            sampled_count = np.round(ratio * NEGATIVE_TABLE_SIZE).astype(np.int64)
            self.negatives = np.repeat(range(len(sampled_count)), sampled_count)

        np.random.shuffle(self.negatives)

    def collate(self,batches):
        negs = self.getNegatives(None, self.num_negs * len(batches)).reshape((len(batches), self.num_negs))
        id_list, positive_list = zip(*batches)
        ####### input are page embedding index, instead of page id.
        pos_u = torch.LongTensor(id_list)
        pos_v = torch.LongTensor(positive_list)
        neg_v = torch.from_numpy(negs)
        if self.entity_type != 'page':
            pos_u = self.page_emb_to_word_emb_tensor[pos_u]
            pos_v = self.page_emb_to_word_emb_tensor[pos_v]
            neg_v = self.page_emb_to_word_emb_tensor[neg_v]
        return pos_u, pos_v, neg_v

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