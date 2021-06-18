import gc
import pickle 
import torch
from wiki_libs.stats import PageWordStats
from wiki_libs.datasets import WikiDataset
from wiki_libs.models import OneTower
import torch.optim as optim
from wiki_libs.preprocessing import convert_to_w2v_mimic_path, get_files_in_dir, path_decoration, LINK_PAIRS_LOCATION
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
import time


class WikiTrainer:

    def __init__(self, hidden_dim1, item_embedding_dim, use_cuda, page_word_stats_path = None, input_embedding_dim=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, page_min_count=0, word_min_count=0, num_workers=0, collate_fn='custom', iprint=500, t=1e-3, ns_exponent=0.75, 
                 optimizer='adam', optimizer_kwargs=None, warm_start_model=None, lr_schedule=False, timeout=60, n_chunk=20,
                 sparse=False, single_layer=False, test=False, save_embedding=True, save_item_embedding = True, w2v_mimic=False, num_negs=5, 
                 testset_ratio = 0.1, entity_type = 'page', amp = False, page_emb_to_word_emb_tensor_fname = None, title_category_trunc_len = 30,
                 dataload_only = False, title_only = False, normalize = False, temperature = 1, two_tower = False):
        
        self.w2v_mimic = w2v_mimic
        if self.w2v_mimic:
            print('Using w2v mimic files for training...', flush=True)

        page_word_stats = PageWordStats(read_path=page_word_stats_path, w2v_mimic=self.w2v_mimic, title_only = title_only)

        self.timeout = timeout
        self.test = test
        self.num_workers = num_workers
        self.page_min_count = page_min_count
        self.word_min_count = word_min_count

        self.entity_type = entity_type
        self.testset_ratio = testset_ratio
        self.amp = amp
        self.dataload_only = dataload_only
        if test:
            self.num_workers = 0
            n_chunk = 1
            self.save_embedding = False

        # Initialize dataset, file_list set to None for now. Will update later.
        self.dataset = WikiDataset(file_list = None, compression = None, n_chunk = n_chunk, 
                              page_word_stats = page_word_stats, num_negs=num_negs, w2v_mimic = w2v_mimic,
                              ns_exponent=ns_exponent, page_min_count=page_min_count, word_min_count=word_min_count, 
                              entity_type=entity_type, page_emb_to_word_emb_tensor_fname=page_emb_to_word_emb_tensor_fname,
                              title_category_trunc_len = title_category_trunc_len, title_only = title_only
                              )
        if collate_fn == 'custom':
            self.collate_fn = self.dataset.collate
        else:
            self.collate_fn = None

        # self.output_file_name = output_file
        self.entity_type = entity_type
        if self.entity_type == 'page':
            self.corpus_size = len(self.dataset.page_frequency_over_threshold)
        else:
            self.corpus_size = len(self.dataset.word_frequency)
        self.input_embedding_dim = input_embedding_dim
        self.save_embedding = save_embedding
        self.save_item_embedding = save_item_embedding
        self.iprint = iprint
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.normalize = normalize
        self.temperature = temperature
        self.two_tower = two_tower

        self.model_init_kwargs = {
            'corpus_size':self.corpus_size,
            'input_embedding_dim':self.input_embedding_dim,
            'hidden_dim1':hidden_dim1,
            'item_embedding_dim':item_embedding_dim,
            'sparse':sparse,
            'single_layer':single_layer,
            'entity_type':self.entity_type,
            'normalize':self.normalize,
            'temperature':self.temperature,
            'two_tower':self.two_tower,
        }
        
        self.model = OneTower(**self.model_init_kwargs)
        
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
            if self.model.two_tower:
                opti_sparse = optim.SparseAdam([self.model.input_embeddings.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                opti_dense = optim.Adam([self.model.linear1.weight, self.model.linear2.weight, self.model.linear1_item.weight, self.model.linear2_item.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                optimizer = MultipleOptimizer(opti_sparse, opti_dense)                
            else:
                # opti_sparse = optim.SparseAdam([self.model.input_embeddings.weight, self.model.item_embeddings.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                # opti_dense = optim.Adam([self.model.linear1.weight, self.model.linear2.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                opti_sparse = optim.SparseAdam(list(self.model.input_embeddings.parameters()) + list(self.model.item_embeddings.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
                opti_dense = optim.Adam(list(self.model.linear1.parameters()) + list(self.model.linear2.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
                optimizer = MultipleOptimizer(opti_sparse, opti_dense)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'asgd':
            optimizer = optim.ASGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        else:
            raise Exception('Unknown optimizer!')

        if self.lr_schedule:
            if self.optimizer == 'sparse_dense_adam':
                scheduler = MultipleScheduler(optimizer, self.iterations)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.iterations)
        else:
            scheduler = None
        running_loss = 0.0
        running_batch_time = 0.0

        iprint = self.iprint #len(self.dataloader) // 20

        self.file_handle_lists = get_files_in_dir(path_decoration(LINK_PAIRS_LOCATION, self.w2v_mimic))
        self.file_handle_lists = sorted(self.file_handle_lists)
        num_train_files = int(len(self.file_handle_lists) * (1 - self.testset_ratio))
        self.file_handle_lists_train = self.file_handle_lists[:num_train_files]
        self.file_handle_lists_test = self.file_handle_lists[num_train_files:]

        for iteration in range(self.iterations):

            print("\nIteration: " + str(iteration + 1))
            

            # Initialize file handle list

            if self.test:
                self.file_handle_lists_train = self.file_handle_lists_train[:2]

            # shuffle order of input for each epoch
            np.random.shuffle(self.file_handle_lists_train)
            if self.num_workers > 0:
                self.file_handle_lists_train_split = np.array_split(self.file_handle_lists_train, self.num_workers)
            else:
                self.timeout = 0
                self.file_handle_lists_train_split = self.file_handle_lists_train.copy()
                self.dataset.file_list = self.file_handle_lists_train.copy()
            dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers, 
                                     collate_fn=self.collate_fn, 
                                     worker_init_fn=partial(self.dataset.worker_init_fn, file_handle_lists=self.file_handle_lists_train_split),
                                     timeout = self.timeout,
                                     drop_last = True,
                                     pin_memory=True
                                    )
            # if self.amp:
            #     scaler = torch.cuda.amp.GradScaler()
            prev_time = time.time()
            prev_i = 0
            for i, sample_batched in enumerate(dataloader):
                if self.dataload_only:
                    continue
                if len(sample_batched[0]) == 0:
                    continue

                pos_u = sample_batched[0].to(self.device)
                pos_v = sample_batched[1].to(self.device)
                neg_v = sample_batched[2].to(self.device)
                
                optimizer.zero_grad()
                loss = self.model.forward(pos_u, pos_v, neg_v)
                # if self.amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                loss.backward()
                optimizer.step()

                running_loss = running_loss * (1 - 5/iprint) + loss.item() * (5/iprint)

                # running_batch_time = running_batch_time * (1 - 5/iprint) + (time_now - start_time) * (5/iprint)
                # start_time = time_now
                if i > 0 and i % iprint == 0:
                    time_now = time.time()
                    if self.optimizer == 'sparse_dense_adam':
                        lr = [param_group['lr'] for param_group in optimizer.optimizers[0].param_groups]
                    else:
                        lr = [param_group['lr'] for param_group in optimizer.param_groups]
                    print(f" Loss: {running_loss} lr: {lr}"
                        f" batch time = {(time_now - prev_time) / (i - prev_i)}" 
                    )
                    prev_time = time_now
                    prev_i = i

            print(i)
            if self.lr_schedule:
                scheduler.step()
            print(" Loss: " + str(running_loss))
            if self.save_embedding:
                path = path_decoration(f'wiki_data/wiki_embedding/embedding_iter_{iteration}_{self.entity_type}.npz', self.w2v_mimic)
                self.do_save_embedding(path, save_item_embedding=self.save_item_embedding)

        #self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
        if self.save_embedding:
            path = path_decoration(f'wiki_data/wiki_embedding/embedding_{self.entity_type}.npz', self.w2v_mimic)
            self.do_save_embedding(path, save_item_embedding=self.save_item_embedding)

    def do_save_embedding(self, path, save_item_embedding=True):
        print('Saving embeddings...', flush=True)


        output_dict = {
            'entity_type':self.entity_type,
            'model_init_kwargs': self.model_init_kwargs,
            'model_state_dict': self.model.state_dict(),
#            'user_embeddings':self.model.forward_to_user_embedding_layer(torch.LongTensor(range(self.model.corpus_size)).to(self.device), in_chunks = True).cpu().data.numpy(),
        }

        # if self.model.two_tower:
        #     output_dict['item_embeddings'] = self.model.forward_to_user_embedding_layer(torch.LongTensor(range(self.model.corpus_size)).to(self.device), in_chunks = True, user_tower = False).cpu().data.numpy(),
        # else:
        #     output_dict['item_embeddings'] = self.model.item_embeddings.weight.cpu().data.numpy()

        if self.entity_type == 'page':
            output_dict['emb2page_over_threshold'] = self.dataset.emb2page_over_threshold
        elif self.entity_type == 'word':
            output_dict['emb2word'] = self.dataset.emb2word
            output_dict['page_emb_to_word_emb_tensor'] = self.dataset.page_emb_to_word_emb_tensor
            output_dict['emb2page'] = self.dataset.emb2page
        torch.save(output_dict, path)

    def save_model(self, fname = 'trained_model'):
        path = path_decoration(f'wiki_data/saved_trained_model/{fname}_{self.entity_type}.npz', self.w2v_mimic)
        # set iterator to None to avoid pickle error on generator
        self.dataset.chunk_iterator = None
        torch.save(self, path)
    
    @classmethod
    def load_model(cls, fname, w2v_mimic, entity_type):
        path = path_decoration(f'wiki_data/saved_trained_model/{fname}_{entity_type}.npz', w2v_mimic)
        return torch.load(path, map_location = 'cpu')



class MultipleOptimizer(optim.Optimizer):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class MultipleScheduler(object):
    def __init__(self, optimizer, iter_num):
        self.optimizer = optimizer
        self.iter_num = iter_num
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(op, self.iter_num) for op in optimizer.optimizers]

    def step(self):
        for sc in self.schedulers:
            sc.step()
