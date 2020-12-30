import gc 
import torch
from wiki_libs.stats import PageWordStats
from wiki_libs.datasets import WikiDataset
from wiki_libs.models import OneTower
import torch.optim as optim
from wiki_libs.preprocessing import convert_to_w2v_mimic_path, get_files_in_dir, path_decoration, LINK_PAIRS_LOCATION
from torch.utils.data import DataLoader
from functools import partial
import numpy as np


class WikiTrainer:

    def __init__(self, hidden_dim1, item_embedding_dim, use_cuda, page_word_stats_path = None, input_embedding_dim=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, page_min_count=0, word_min_count=0, num_workers=0, collate_fn='custom', iprint=500, t=1e-3, ns_exponent=0.75, 
                 optimizer='adam', optimizer_kwargs=None, warm_start_model=None, lr_schedule=False, timeout=60, n_chunk=20,
                 sparse=False, single_layer=False, test=False, save_embedding=True, w2v_mimic=False, num_negs=5):
        
        self.w2v_mimic = w2v_mimic
        if self.w2v_mimic:
            print('Using w2v mimic files for training...', flush=True)

        page_word_stats = PageWordStats(read_path=page_word_stats_path, w2v_mimic=self.w2v_mimic)

        self.timeout = timeout
        self.test = test
        self.num_workers = num_workers
        self.page_min_count = page_min_count
        self.word_min_count = word_min_count
        if test:
            self.num_workers = 0
            n_chunk = 1

        # Initialize dataset, file_list set to None for now. Will update later.
        self.dataset = WikiDataset(file_list = None, compression = 'zip', n_chunk = n_chunk, 
                              page_word_stats = page_word_stats, num_negs=num_negs, 
                              ns_exponent=ns_exponent, page_min_count=page_min_count, word_min_count=word_min_count)
        if collate_fn == 'custom':
            self.collate_fn = self.dataset.collate
        else:
            self.collate_fn = None

        # self.output_file_name = output_file
        self.corpus_size = len(self.dataset.page_frequency_over_threshold)
        self.input_embedding_dim = input_embedding_dim
        self.save_embedding = save_embedding
        self.iprint = iprint
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.model = OneTower(self.corpus_size, self.input_embedding_dim, 
 #                             context_size = dataset.num_neg, 
                              hidden_dim1 = hidden_dim1, 
                              item_embedding_dim = item_embedding_dim, sparse=sparse,
                              single_layer = single_layer,
                              )
        
        print(f"total parameters is: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        if warm_start_model is not None:
            self.model.load_state_dict(torch.load(warm_start_model), strict=False)
        self.optimizer = optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_schedule = lr_schedule
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        # clearn GPU memory cache
        gc.collect()
        torch.cuda.empty_cache()

        # self.final_lr = self.initial_lr

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'sparse_adam':
            optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'sparse_dense_adam':
            opti_sparse = optim.SparseAdam([self.model.input_embeddings.weight, self.model.item_embeddings.weight], lr=self.initial_lr, **self.optimizer_kwargs)
            opti_dense = optim.Adam([self.model.linear1.weight, self.model.linear2.weight], lr=self.initial_lr, **self.optimizer_kwargs)
            optimizer = MultipleOptimizer(opti_sparse, opti_dense)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'asgd':
            optimizer = optim.ASGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        else:
            raise Exception('Unknown optimizer!')

        if self.lr_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.iterations)
        else:
            scheduler = None
        running_loss = 0.0

        iprint = self.iprint #len(self.dataloader) // 20

        for iteration in range(self.iterations):

            print("\nIteration: " + str(iteration + 1))
            

            # Initialize file handle list

            file_handle_lists = get_files_in_dir(path_decoration(LINK_PAIRS_LOCATION, self.w2v_mimic))
            if self.test:
                file_handle_lists = file_handle_lists[:2]

            # shuffle order of input for each epoch
            np.random.shuffle(file_handle_lists)
            if self.num_workers > 0:
                file_handle_lists = np.array_split(file_handle_lists, self.num_workers)
            else:
                self.timeout = 0
                self.dataset.file_list = file_handle_lists
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers, 
                                     collate_fn=self.collate_fn, 
                                     worker_init_fn=partial(self.dataset.worker_init_fn, file_handle_lists=file_handle_lists),
                                     timeout = self.timeout,
                                     drop_last = True,
                                     pin_memory=True
                                    )
            
            for i, sample_batched in enumerate(dataloader):
                # ipdb.set_trace()
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()


                    running_loss = running_loss * (1 - 5/iprint) + loss.item() * (5/iprint)
                    if i > 0 and i % iprint == 0:
                        
                        print(" Loss: " + str(running_loss) + ' lr: ' 
                            + str([param_group['lr'] for param_group in optimizer.param_groups]))
                        #print(" Loss: " + str(running_loss))
            print(i)
            if self.lr_schedule:
                scheduler.step()
            print(" Loss: " + str(running_loss))

        #self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
        if self.save_embedding:
            path = path_decoration('wiki_data/wiki_embedding/embedding.npz', self.w2v_mimic)
            self.do_save_embedding(path)

    def do_save_embedding(self, path):
        print('Saving embeddings...', flush=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'emb2page':self.dataset.emb2page,
            'user_embeddings':self.model.input_embeddings.weight.cpu().data.numpy(),
            'item_embeddings':self.model.item_embeddings.weight.cpu().data.numpy(),
            }, path)

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
