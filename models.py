
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import gc

class OneTower(nn.Module):
    def __init__(self, corpus_size, input_embedding_dim, 
                item_embedding_dim, sparse, entity_type = 'page', normalize = False,
                temperature = 1, two_tower = False, relu = True, clamp = True, softmax = False, kaiming_init = False,
                last_layer_relu = False, layer_nodes = (512,128), in_batch_neg = False, neg_sample_prob_corrected = False,
                ):
        super(OneTower, self).__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.entity_type = entity_type
        self.two_tower = two_tower
        self.corpus_size = corpus_size
        self.relu = relu
        self.clamp = clamp
        self.softmax = softmax

        self.kaiming_init = kaiming_init
        self.last_layer_relu = last_layer_relu
        self.layer_nodes = layer_nodes
        self.in_batch_neg = in_batch_neg
        self.neg_sample_prob_corrected = neg_sample_prob_corrected

        if self.entity_type == 'page':
            self.input_embeddings = nn.Embedding(corpus_size, input_embedding_dim, sparse=sparse)
            if not self.two_tower:
                self.item_embeddings = nn.Embedding(corpus_size, item_embedding_dim, sparse=sparse)
        else:
            self.input_embeddings = nn.EmbeddingBag(corpus_size + 1, input_embedding_dim, sparse=sparse, padding_idx=-1, mode = 'sum')
            if not self.two_tower:
                self.item_embeddings = nn.EmbeddingBag(corpus_size + 1, item_embedding_dim, sparse=sparse, padding_idx=-1, mode = 'sum')

        print(f'layer_node is {self.layer_nodes}')

        self.linears = self.create_and_init_tower(layer_nodes, input_embedding_dim)
        if self.two_tower:
            self.linears_item = self.create_and_init_tower(layer_nodes, input_embedding_dim)
        else:
            self.linears_item = self.create_and_init_tower(tuple(), input_embedding_dim)

        input_initrange = 1.0 / input_embedding_dim
        init.uniform_(self.input_embeddings.weight, -input_initrange, input_initrange)
        if not self.two_tower:
            item_initrange = 1.0 / item_embedding_dim
            init.uniform_(self.item_embeddings.weight, -item_initrange, item_initrange)

        if self.entity_type == 'word':
            self.input_embeddings.weight.data[-1] = 0
            if not self.two_tower:
                self.item_embeddings.weight.data[-1] = 0

    def create_and_init_tower(self, layer_nodes, input_dims):
        linears = nn.ModuleList()
        n_prev = input_dims
        for n in layer_nodes:
            new_layer = nn.Linear(n_prev, n)
            if self.relu and self.kaiming_init:
                init.kaiming_normal_(new_layer.weight, nonlinearity='relu')
            init.zeros_(new_layer.bias)
            linears.append(new_layer)
            n_prev = n
        return linears

    def forward_one_tower(self, pos_input, linears):
        ret = pos_input
        for i, linear in enumerate(linears):
            ret = linear(ret)
            if self.relu and i != len(linears) - 1: # does not apply activation in last layer
                ret = F.relu(ret)
        return ret
    
    def forward_to_user_embedding_layer(self, pos_input, user_tower = True, force_cpu_output = False):

        gc_input_len = 100_000
        # input embedding
        
        chunks = torch.split(pos_input, gc_input_len)

        ret_list = []

        for i, chunk in enumerate(chunks):
            if next(self.parameters()).is_cuda and not chunk.is_cuda:
                chunk = chunk.to('cuda')
            if user_tower:
                ret = self.embedding_lookup(self.input_embeddings, chunk)
                ret = self.forward_one_tower(ret, self.linears)
            else:
                if self.two_tower:
                    ret = self.embedding_lookup(self.input_embeddings, chunk)
                    ret = self.forward_one_tower(ret, self.linears_item)
                else:
                    ret = self.embedding_lookup(self.item_embeddings, chunk)
            if self.last_layer_relu:
                ret = F.relu(ret)
            if self.normalize:
                ret = F.normalize(ret, p=2, dim=-1)
            if force_cpu_output and ret.is_cuda:
                ret = ret.cpu()
            ret_list.append(ret)
            if (i > 1):
                print(i, flush = True)
                gc.collect()
                torch.cuda.empty_cache()
        if len(ret_list) == 1:
            return ret_list[0]
        else:
            return torch.cat(ret_list)
                

    def forward(self, pos_input, pos_item, neg_item, i, neg_sample_prob = None, pos_item_page = None, neg_item_page = None):

        emb_user = self.forward_to_user_embedding_layer(pos_input, user_tower=True)

        # output embedding for positive instance
        emb_item = self.forward_to_user_embedding_layer(pos_item, user_tower=False)

        # output embedding for negative instance

        score = torch.sum(torch.mul(emb_user, emb_item), dim=1) / self.temperature
        

        if self.clamp:
            score = torch.clamp(score, max=10, min=-10)
        if self.neg_sample_prob_corrected:
            score = score - torch.log(neg_sample_prob[pos_item_page])
        score_copy = score
        if self.softmax:
            score = -score
        else:
            score = -F.logsigmoid(score)
        
        if not self.in_batch_neg:
            neg_item_shape = neg_item.shape
            neg_item = neg_item.reshape(neg_item_shape[0] * neg_item_shape[1], neg_item.shape[2])
            emb_neg_item = self.forward_to_user_embedding_layer(neg_item, user_tower=False)
            emb_neg_item = emb_neg_item.reshape(neg_item_shape[0], neg_item_shape[1], emb_neg_item.shape[-1])
            neg_score = torch.bmm(emb_neg_item, emb_user.unsqueeze(2)).squeeze() / self.temperature
            if self.neg_sample_prob_corrected:
                neg_score = neg_score - torch.log(neg_sample_prob[neg_item_page])
            if self.clamp:
                neg_score = torch.clamp(neg_score, max=10, min=-10)
            if self.softmax:
                neg_score = torch.logsumexp(torch.hstack([neg_score, score_copy.unsqueeze(-1)]), dim=1)
            else:
                neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        else:
            neg_score = torch.matmul(emb_user, emb_item.t()) / self.temperature
            if self.neg_sample_prob_corrected:
                neg_score = neg_score - torch.log(neg_sample_prob[pos_item_page]).unsqueeze(0)
            if self.clamp:
                neg_score = torch.clamp(neg_score, max=10, min=-10)
            if self.softmax:
                neg_score = torch.logsumexp(neg_score, dim = 1)
            else:
                neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1) + F.logsigmoid(-score_copy)

        return torch.mean(score + neg_score)   

    def embedding_lookup(self, embedding, embed_index):
        if self.entity_type == 'page':
            emb_input = embedding(embed_index)
        elif self.entity_type == 'word':
            # need to fix lookup -1 embedding index should return 0 embedding vector
            # mean pooling
            select = (embed_index != embedding.padding_idx)
            sentence_emb_input = embedding(embed_index)
            emb_input = sentence_emb_input / select.sum(axis = -1).unsqueeze(-1)
        
        return emb_input
        
    def embedding_lookup_n_chunk(self, embedding, embed_index):
        chunks = torch.split(embed_index, 100000)
        return torch.cat([self.embedding_lookup(embedding, chunk) for chunk in chunks])