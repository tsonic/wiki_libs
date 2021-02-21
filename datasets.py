import torch
import pandas as pd
from wiki_libs.preprocessing import read_files_in_chunks, read_category_links, process_title
from wiki_libs.ngram import NGRAM_MODEL_PATH_PREFIX, load_ngram_model, get_df_title_category_transformed, transform_ngram
import numpy as np


NEGATIVE_TABLE_SIZE = 1e8
class WikiDataset(torch.utils.data.IterableDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_list, compression, n_chunk, num_negs, page_word_stats, ns_exponent, 
                page_min_count = 0, word_min_count = 0, entity_type='page',
                ngram_model_name = "title_category_ngram_model.pickle",
                build_transformed_title_category = False,
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
        self.build_transformed_title_category = build_transformed_title_category

        self.negatives = []
        self.negpos = 0
        self.num_negs = num_negs
        self.page_word_stats = page_word_stats

        self.entity_type = entity_type

        if self.entity_type == 'page':
            entity_frequency = page_word_stats.page_frequency
            self.min_count = self.page_min_count
            self.df_title_category_transformed = None
        else:
            entity_frequency = page_word_stats.word_frequency
            self.min_count = self.word_min_count
            self.df_title_category_transformed = get_df_title_category_transformed(
                    read_cached=not self.build_transformed_title_category,
                    ngram_model_name=False,
                )

        self.entity_frequency_over_threshold = {p:c for p, c in entity_frequency.items() if c > self.min_count}
        self.page_frequency_over_threshold = {p:c for p, c in page_word_stats.page_frequency.items() if c > self.page_min_count}
        self.word_frequency_over_threshold = {p:c for p, c in page_word_stats.word_frequency.items() if c > self.word_min_count}


        # emb2page maps pytorch embedding back to 'page_id'
        self.emb2entity = list(self.entity_frequency_over_threshold.keys())
        # page2emb is 'page_id' to pytorch embedding index mapping
        self.entity2emb = {p:i for i, p in enumerate(self.emb2entity)}

        entity, i = zip(*self.entity2emb.items())
        self.entity2emb_series_map = pd.Series(i, index=entity, dtype=np.int64)

        print(f'Number of unique entities ({entity_type}) included is {len(self.entity_frequency_over_threshold)}')
        print(f'Number of unique pages included is {len(self.page_frequency_over_threshold)}')
        print(f'Number of unique words included is {len(self.word_frequency_over_threshold)}')

        self.initTableNegatives(ns_exponent=ns_exponent)

    # Iterable may not know the length of the stream before hand
    # def __len__(self):
    #     'Denotes the total number of samples'
    #     return len(self.list_IDs)

    # def __getitem__(self, index):
    #     'Generates one sample of data'
    #     # Select sample
    #     ID = self.list_IDs[index]

    #     # Load data and get label
    #     X = torch.load('data/' + ID + '.pt')
    #     y = self.labels[ID]

    #     return X, y
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.chunk_iterator is None:
            self.chunk_iterator = read_files_in_chunks(self.file_list, compression=self.compression, 
                                                       n_chunk = self.n_chunk, progress_bar = True)
            
        if self.instance_dict is None or self.pos >= len(self.instance_dict):
            df = next(self.chunk_iterator)
            # have to reset self.pos after pull next in above iterator, otherwise
            # it does not invalidate current instance_dict after iterator is exhausted.
            self.pos = 0
            # remove page ids that is below the min count threshold
            df = df.query('page_id_source.isin(@self.page_frequency_over_threshold) & page_id_target.isin(@self.page_frequency_over_threshold)', engine = 'python')
            if self.entity_type == "page":
                df = df.assign(
                        page_id_source = lambda df: df['page_id_source'].map(self.entity2emb_series_map),
                        page_id_target = lambda df: df['page_id_target'].map(self.entity2emb_series_map),
                    )
            elif self.entity_type == "word":
                df_map = self.df_title_category_transformed.assign(
                    page_title_category_transformed = lambda df: df['page_title_category_transformed'].apply(lambda x: [self.entity2emb[e] for e in x])
                )
                df = (
                    df
                    .merge(df_map, left_on = 'page_id_source', right_on = 'page_id')
                    .merge(df_map, left_on = 'page_id_target', right_on = 'page_id', suffixes = ['_source', '_target'])
                    [['page_title_category_transformed_source', 'page_title_category_transformed_target']]
                )
            else:
                raise Exception(f"Unknown entity type {self.entity_type}.")
            df = df.sample(frac=1.0, replace = False)
            self.instance_dict = list(df.itertuples(index = False, name = None))
        ret = self.instance_dict[self.pos]
        self.pos += 1
        return ret

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        # reshuffle negative table if negpos > total neg table size.
        if (self.negpos + size) // len(self.negatives) >= 1:
            # print('reshuffle negative table...')
            np.random.shuffle(self.negatives)
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def initTableNegatives(self, ns_exponent):
        
        print('Initializing negative samples', flush=True)
        # embedding id to page counts mapping
        entity_counts = [self.entity_frequency_over_threshold[entity_id] for entity_id in self.emb2entity]
        ratio = np.array(entity_counts).astype(np.float64) ** ns_exponent / sum(entity_counts)
        sampled_count = np.round(ratio * NEGATIVE_TABLE_SIZE).astype(np.int64)

        self.negatives = np.repeat(range(len(sampled_count)), sampled_count)
        np.random.shuffle(self.negatives)

    def collate(self,batches):
        negs = self.getNegatives(None, self.num_negs * len(batches)).reshape((len(batches), self.num_negs))
        id_list, positive_list = zip(*batches)
        return torch.LongTensor(id_list), torch.LongTensor(positive_list), torch.from_numpy(negs)

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