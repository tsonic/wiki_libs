import gc
import pickle 
import torch
import pandas as pd
from wiki_libs.stats import PageWordStats
from wiki_libs.datasets import WikiDataset
from wiki_libs.models import OneTower
from wiki_libs.eval import compute_recall
from wiki_libs.knn import build_knn
import torch.optim as optim
from wiki_libs.preprocessing import (
    convert_to_w2v_mimic_path, get_files_in_dir, path_decoration, 
    LINK_PAIRS_LOCATION, read_files_in_chunks, read_page_data, is_colab,
    convert_to_colab_path
)
from torch.utils.data import DataLoader
from functools import partial
from IPython.core.display import display, HTML
import numpy as np
import time
import os
import json
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

BASE_CONFIG = {
    'hidden_dim1':128, 
    'item_embedding_dim':128, 
    #'output_file':"gdrive/My Drive/Projects with Wei/Wei_tmp_outputs/w2v_output/out.vec",
    #'min_count':5,
    'batch_size':4048,
    'num_negs':5,
    'iterations':1,
    'num_workers':7,
    'collate_fn':'custom',
    'iprint':10000,
    'n_chunk':20,
    'input_embedding_dim':128,
    'ns_exponent':0.75,
    'initial_lr':0.01,
#   optimizer_name='sparse_dense_adam',
    'optimizer_name':'sparse_adam',
    'single_layer':True,
    'sparse':True,
    'lr_schedule':False,
    'test':False,
    'save_embedding':True,
    'save_item_embedding':False,
    'w2v_mimic':False,
    'page_word_stats_path':"wiki_data/page_word_stats.json",
    'page_emb_to_word_emb_tensor_fname':"page_emb_to_word_emb_tensor.npz",
    'use_cuda':True,
    'page_min_count':50,
    'testset_ratio':0.05,
    'entity_type':'word',
    'amp':False,
    'title_category_trunc_len':30,
    'dataload_only':False,
    'title_only':False,
    'normalize':False,
    'temperature':1,
    'two_tower':False,
    'dataload_only': False,
    'model_name': 'baseline',
    'relu':True,
    'dense_lr_ratio':0.1,
    'repeat':0,
    'clamp':True,
    'softmax':False,
    'kaiming_init':False
}

def parse_config(base_config_update):
    actual_config_update = {}
    config = BASE_CONFIG.copy()
    for k,v in base_config_update.items():
        if k not in BASE_CONFIG or BASE_CONFIG[k] != v:
            actual_config_update[k] = v
    if actual_config_update:
        new_model_name = 'baseline'
        for k, v in actual_config_update.items():
            if isinstance(v, bool):
                if v:
                    new_model_name += f'_{k}'
                else:
                    new_model_name += f'_not_{k}'
            else:
                new_model_name += f"_{k}_{v}"
        actual_config_update['model_name'] = new_model_name
        config.update(actual_config_update)
    return config

def optimizer_to(optim, device):
    if isinstance(optim, MultipleOptimizer):
        optims = optim.optimizers
    else:
        optims = [optim]
    
    for op in optims:
        for param in op.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

class WikiTrainer:

    def __init__(self, hidden_dim1, item_embedding_dim, use_cuda, model_name, page_word_stats_path = None, input_embedding_dim=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, page_min_count=0, word_min_count=0, num_workers=0, collate_fn='custom', iprint=500, t=1e-3, ns_exponent=0.75, 
                 optimizer_name='adam', optimizer_kwargs=None, warm_start_model=None, lr_schedule=False, timeout=60, n_chunk=20,
                 sparse=False, single_layer=False, test=False, save_embedding=True, save_item_embedding = True, w2v_mimic=False, num_negs=5, 
                 testset_ratio = 0.1, entity_type = 'page', amp = False, page_emb_to_word_emb_tensor_fname = None, title_category_trunc_len = 30,
                 dataload_only = False, title_only = False, normalize = False, temperature = 1, two_tower = False, dense_lr_ratio = 0.1,
                 relu = True, repeat = 0, clamp = True, softmax = False, kaiming_init = False
                 ):


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
        self.save_embedding = save_embedding
        self.dense_lr_ratio = dense_lr_ratio
        self.model_name = model_name


        self.create_dir_structure()

        if self.test:
            self.num_workers = 0
            n_chunk = 1
    #        self.save_embedding = False

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
        
        self.save_item_embedding = save_item_embedding
        self.iprint = iprint
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.normalize = normalize
        self.temperature = temperature
        self.two_tower = two_tower
        self.single_layer = single_layer
        self.relu = relu
        self.clamp = clamp
        self.softmax = softmax
        self.kaiming_init = kaiming_init

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
            'relu':self.relu,
            'clamp':self.clamp,
            'softmax':self.softmax,
            'kaiming_init':self.kaiming_init
        }
        
        self.model = OneTower(**self.model_init_kwargs)
        
        print(f"total parameters is: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        if warm_start_model is not None:
            self.model.load_state_dict(torch.load(warm_start_model), strict=False)
        self.optimizer_name = optimizer_name
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_schedule = lr_schedule
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.model.cuda()

    def create_dir_structure(self):
        import shutil
        if self.test:
            self.prefix = f'wiki_data/test/{self.model_name}'
        else:
            self.prefix = f'wiki_data/experiments/{self.model_name}'
        if is_colab():
            self.prefix = convert_to_colab_path(self.prefix)
        self.saved_embeddings_dir = f'{self.prefix}/wiki_embedding'
        shutil.rmtree(self.prefix, ignore_errors=True)
        os.makedirs(self.prefix, exist_ok = False)
        os.makedirs(self.saved_embeddings_dir, exist_ok = False)

    def train(self):
        # clearn GPU memory cache
        gc.collect()
        torch.cuda.empty_cache()

        # self.final_lr = self.initial_lr

        # use sparse dense adam solver for non-single-layer network
        # writer = SummaryWriter(f'{self.prefix}/tensorboard/')
        if self.optimizer_name == 'sparse_adam' and not self.single_layer:
            self.optimizer_name = 'sparse_dense_adam'

        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer_name == 'sparse_adam':
            self.optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer_name == 'sparse_dense_adam':
            if self.model.two_tower:
                opti_sparse = optim.SparseAdam([self.model.input_embeddings.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                opti_dense = optim.Adam([self.model.linear1.weight, self.model.linear2.weight, self.model.linear1_item.weight, self.model.linear2_item.weight], lr=self.initial_lr*self.dense_lr_ratio, **self.optimizer_kwargs)
                self.optimizer = MultipleOptimizer(opti_sparse, opti_dense)                
            else:
                # opti_sparse = optim.SparseAdam([self.model.input_embeddings.weight, self.model.item_embeddings.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                # opti_dense = optim.Adam([self.model.linear1.weight, self.model.linear2.weight], lr=self.initial_lr, **self.optimizer_kwargs)
                opti_sparse = optim.SparseAdam(list(self.model.input_embeddings.parameters()) + list(self.model.item_embeddings.parameters()), lr=self.initial_lr, **self.optimizer_kwargs)
                opti_dense = optim.Adam(list(self.model.linear1.parameters()) + list(self.model.linear2.parameters()), lr=self.initial_lr*self.dense_lr_ratio, **self.optimizer_kwargs)
                self.optimizer = MultipleOptimizer(opti_sparse, opti_dense)
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer_name == 'asgd':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer_name == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)
        else:
            raise Exception('Unknown optimizer!')

        if self.lr_schedule:
            if self.optimizer == 'sparse_dense_adam':
                scheduler = MultipleScheduler(self.optimizer, self.iterations)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.iterations)
        else:
            scheduler = None
        running_loss = 0.0
        running_batch_time = 0.0
        total_training_instances = 0

        iprint = self.iprint #len(self.dataloader) // 20

        self.file_handle_lists = get_files_in_dir(path_decoration(LINK_PAIRS_LOCATION, self.w2v_mimic))
        self.file_handle_lists = sorted(self.file_handle_lists)
        num_train_files = int(len(self.file_handle_lists) * (1 - self.testset_ratio))
        self.file_handle_lists_train = self.file_handle_lists[:num_train_files]
        self.file_handle_lists_test = self.file_handle_lists[num_train_files:]

        self.df_eval_list = []

        if self.amp:
            scaler = torch.cuda.amp.GradScaler()

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
                
                self.optimizer.zero_grad()

                if self.amp:
                    with torch.cuda.amp.autocast():
                        loss = self.model.forward(pos_u, pos_v, neg_v, i)
                    scaler.scale(loss).backward()
                    if isinstance(self.optimizer, MultipleOptimizer):
                        for op in self.optimizer.optimizers:
                            scaler.step(op)
                    else:
                        scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss = self.model.forward(pos_u, pos_v, neg_v, i)
                    loss.backward()
                    self.optimizer.step()

                # roughly every iprint batches, running loss are flushed out 5 times
                running_loss = running_loss * (1 - 5/iprint) + loss.item() * (5/iprint)
                total_training_instances += sample_batched[0].shape[0]

                # running_batch_time = running_batch_time * (1 - 5/iprint) + (time_now - start_time) * (5/iprint)
                # start_time = time_now
                if i > 0 and i % iprint == 0:
                    time_now = time.time()
                    if self.optimizer_name == 'sparse_dense_adam':
                        lr = [param_group['lr'] for param_group in self.optimizer.optimizers[0].param_groups]
                    else:
                        lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                    print(f" Loss: {running_loss} lr: {lr}"
                        f" batch time = {(time_now - prev_time) / (i - prev_i)}" 
                    )
                    #self.write_tensorboard_stats(running_loss, total_training_instances)
                    prev_time = time_now
                    prev_i = i

            print(i)
            if self.lr_schedule:
                scheduler.step()
            print(f" Loss: {running_loss}")
            #self.write_tensorboard_stats(running_loss, total_training_instances)
            # saving embeddings per epoch if running locally
            if not is_colab():
                path = f'{self.saved_embeddings_dir}/embedding_iter_{iteration}_{self.entity_type}.npz'
                if self.w2v_mimic:
                    path = convert_to_w2v_mimic_path(path)
                    self.do_save_embedding(path)
            
            df_eval = (
                self.eval_model(iter_num = iteration)
                .pivot(index = 'iter_num', columns = 'k', values = 'recall')
                .assign(loss = running_loss)
                [['loss', 10, 50, 100, 300]]
            )
            display(df_eval)
            df_eval.to_csv(f'{self.prefix}/eval_result_iter_{iteration}.tsv', sep = '\t')
            self.df_eval_list.append(df_eval)

            gc.collect()
            torch.cuda.empty_cache()

        # # saving embeddings after all epochs
        if not is_colab():
            path = path_decoration(f'{self.saved_embeddings_dir}/embedding_{self.entity_type}.npz', self.w2v_mimic)
            self.do_save_embedding(path)

        # show recall evaluation
        df_result = pd.concat(self.df_eval_list)
        df_result.to_csv(f'{self.prefix}/eval_result.tsv', sep = '\t')
        #self.writer.add_graph(self.model, [pos_u, pos_v, neg_v])
        #self.writer.close()
        #self.writer = None
        display(df_result)
        self.save_train_config()
        # self.save_model()

    def save_train_config(self):
        trainer_config = self.__dict__
        with open(f'{self.prefix}/trainer_config.json', 'w') as f:
            f.write(json.dumps(trainer_config, default=lambda o: '<not serializable>', indent=4))

    def do_save_embedding(self, path):    
        output_dict = self.prep_embedding_output()
        if self.save_embedding:
            print('Saving embeddings...', flush=True)
            torch.save(output_dict, path)
    
    def prep_embedding_output(self):
        output_dict = {
            'entity_type':self.entity_type,
            'model_init_kwargs': self.model_init_kwargs,
            'model_state_dict': self.model.state_dict(),
        }

        if self.entity_type == 'page':
            output_dict['emb2page_over_threshold'] = self.dataset.emb2page_over_threshold
        elif self.entity_type == 'word':
            output_dict['emb2word'] = self.dataset.emb2word
            output_dict['page_emb_to_word_emb_tensor'] = self.dataset.page_emb_to_word_emb_tensor
            output_dict['emb2page'] = self.dataset.emb2page
        
        return output_dict

    def save_model(self, fname = 'trained_model'):
        path = path_decoration(f'{self.prefix}/{fname}_{self.entity_type}.npz', self.w2v_mimic)
        # set iterator to None to avoid pickle error on generator
        self.dataset.chunk_iterator = None
        torch.save(self, path)
    
    def eval_model(self, iter_num, quick = True):
        df_links_test = pd.concat(list(read_files_in_chunks(self.file_handle_lists_test, compression='gz', 
                                                       n_chunk = 10, progress_bar = True)))
        if quick:
            df_links_test = df_links_test.iloc[:100000]
        
        df_page = read_page_data(w2v_mimic = self.w2v_mimic)

        
        # torch.save(self.optimizer.state_dict(), '/tmp/optimizer_state_dict_cache.pkl')
        # optimizer_state_dict = torch.load('/tmp/optimizer_state_dict_cache.pkl', map_location='cpu')
        # self.optimizer.load_state_dict(optimizer_state_dict)

        print(f'self.model in cuda: {next(self.model.parameters()).is_cuda}')

        if is_colab():
            torch.save(self.optimizer.state_dict(), '/tmp/optimizer_state_dict_cache.pkl')
            self.optimizer.__setstate__({'state': defaultdict(dict)})
            gc.collect()
        else:
            self.model.cpu()
            optimizer_to(self.optimizer, 'cpu')

        embedding_output_dict = self.prep_embedding_output()
        embedding_output_dict['model'] = self.model

        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()

        df_embedding, nn = build_knn(
            emb_file=embedding_output_dict, 
            df_page=df_page, w2v_mimic=self.w2v_mimic, emb_name="item_embedding", 
            algorithm='faiss', device = 'gpu'
        )
        del embedding_output_dict
        gc.collect()
        torch.cuda.empty_cache()

        # self.model.to(self.device)
        # self.model = OneTower(**embedding_output_dict['model_init_kwargs'])
        # self.model.load_state_dict(embedding_output_dict['model_state_dict'])
        start_1 = time.time()
        print('compute recall...', flush = True)
        df_ret = (
            compute_recall(df_links_test, df_embedding, nn, [10, 50, 100, 300],use_user_emb = True)
            .assign(iter_num = iter_num)
        )
        del nn, df_embedding
        gc.collect()
        torch.cuda.empty_cache()
        if is_colab():
            optimizer_state_dict = torch.load('/tmp/optimizer_state_dict_cache.pkl', map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            if self.use_cuda:
                self.model.cuda()
            optimizer_to(self.optimizer, self.device)

        end = time.time()
        print(f"nn training time is {start_1 - start}, recall evaluation time is {end - start_1}")
        return df_ret
    
    @classmethod
    def load_model(cls, fname, w2v_mimic, entity_type):
        path = path_decoration(f'wiki_data/{fname}_{entity_type}.npz', w2v_mimic)
        return torch.load(path, map_location = 'cpu')

    def write_tensorboard_stats(self, running_loss, total_training_instances):
        self.writer.add_scalar('train loss', running_loss, total_training_instances)
        if not self.single_layer:
            self.writer.add_histogram('linear1', self.model.linear1.weight.detach().cpu().numpy(), total_training_instances)
            self.writer.add_histogram('linear1_diagonal', self.model.linear1.weight.detach().cpu().numpy().diagonal(), total_training_instances)
            self.writer.add_histogram('linear2', self.model.linear2.weight.detach().cpu().numpy(), total_training_instances)
            self.writer.add_histogram('linear2_diagonal', self.model.linear2.weight.detach().cpu().numpy().diagonal(), total_training_instances)
            if self.two_tower:
                self.writer.add_histogram('linear1_itme', self.model.linear1_item.weight.detach().numpy(), total_training_instances)
                self.writer.add_histogram('linear1_itme_diagonal', self.model.linear1_item.weight.detach().numpy().diagonal(), total_training_instances)

                self.writer.add_histogram('linear2_item', self.model.linear2_item.weight.detach().numpy(), total_training_instances)
                self.writer.add_histogram('linear2_item_diagonal', self.model.linear2_item.weight.detach().numpy().diagonal(), total_training_instances)

def train(config):
    wt = WikiTrainer(**config)
    wt.train()

class MultipleOptimizer(optim.Optimizer):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def to(self, device):
        for op in self.optimizers:
            optimizer_to(op, device)
    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dicts):
        for sd, op in zip(state_dicts, self.optimizers):
            op.load_state_dict(sd)

class MultipleScheduler(object):
    def __init__(self, optimizer, iter_num):
        self.optimizer = optimizer
        self.iter_num = iter_num
        self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(op, self.iter_num) for op in optimizer.optimizers]

    def step(self):
        for sc in self.schedulers:
            sc.step()
