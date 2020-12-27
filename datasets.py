import torch
import pandas as pd
from wiki_libs.preprocessing import read_files_in_chunks
import numpy as np


NEGATIVE_TABLE_SIZE = 1e8
class WikiDataset(torch.utils.data.IterableDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_list, compression, n_chunk, num_negs, page_word_stats, ns_exponent):
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


        self.negatives = []
        self.negpos = 0
        self.num_negs = num_negs
        self.page_word_stats = page_word_stats

        self.word_frequency = page_word_stats.word_frequency
        self.word2id = page_word_stats.word2id
        self.id2word = page_word_stats.id2word
        self.page2id = page_word_stats.page2id
        p,i = zip(*self.page2id.items())
        self.page2id_series_map = pd.Series(i, index = p, dtype = np.int64)
        self.id2page = page_word_stats.id2page
        self.page_frequency = page_word_stats.page_frequency
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
            #print('read_new_chunk', flush = True)
            df = next(self.chunk_iterator)
            # have to reset self.pos after pull next in above iterator, otherwise 
            # it does not invalidate current instance_dict after iterator is exhausted.
            self.pos = 0

            df = df.assign(
                    page_id_source = lambda df: df['page_id_source'].map(self.page2id_series_map),
                    page_id_target = lambda df: df['page_id_target'].map(self.page2id_series_map),
                )
            df = df.sample(frac=1.0, replace = False)
            # print('source: %d, target: %d' % (df['page_id_source'].iat[0], df['page_id_target'].iat[0]))
            
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
        page_id, page_counts = zip(*self.page_frequency.items())
        ratio = np.array(page_counts).astype(np.float64) ** ns_exponent / sum(page_counts)
        sampled_count = np.round(ratio * NEGATIVE_TABLE_SIZE)

        df = pd.DataFrame.from_records(enumerate(sampled_count))
        # the column 0 is the page id, column 1 is the count of the page
        self.negatives = np.repeat(df[0].astype(np.int64).values, df[1].astype(np.int64).values)
        np.random.shuffle(self.negatives)

    def collate(self,batches):
        negs = self.getNegatives(None, self.num_negs * len(batches)).reshape((len(batches), self.num_negs))
        id_list, positive_list = zip(*batches)
        return torch.LongTensor(id_list), torch.LongTensor(positive_list), torch.from_numpy(negs)

    @staticmethod
    def worker_init_fn(worker_id, file_handle_lists):
        print('worker_init')
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        worker_id = worker_info.id
        # the new_seed are the same after calling method for multiple time, e.g. in different epoches.
        new_seed = np.random.get_state()[1][0] + worker_id

        np.random.seed(new_seed) 
        np.random.shuffle(dataset.negatives)
        dataset.file_list = file_handle_lists[worker_id]