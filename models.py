
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

class OneTower(nn.Module):
    def __init__(self, corpus_size, input_embedding_dim, hidden_dim1, item_embedding_dim, sparse, single_layer = False):
        super(OneTower, self).__init__()
        self.input_embeddings = nn.Embedding(corpus_size, input_embedding_dim, sparse=sparse)
        self.single_layer = single_layer

        # if single_layer is True, it essentially become w2v model with single hidden layer
        if not self.single_layer:
            self.linear1 = nn.Linear(input_embedding_dim, hidden_dim1)
            self.linear2 = nn.Linear(hidden_dim1, item_embedding_dim)
        self.item_embeddings = nn.Embedding(corpus_size, item_embedding_dim, sparse=sparse)

        input_initrange = 1.0 / input_embedding_dim
        item_initrange = 1.0 / item_embedding_dim
        init.uniform_(self.input_embeddings.weight.data, -input_initrange, input_initrange)
        init.uniform_(self.item_embeddings.weight.data, -item_initrange, item_initrange)
    
    def forward_to_user_embedding_layer(self, pos_input):
        # input embedding
        emb_input = self.input_embeddings(pos_input)

        if not self.single_layer:
            h1 = F.relu(self.linear1(emb_input))
            emb_user = F.relu(self.linear2(h1))
        else:
            emb_user = emb_input
        
        return emb_user

    def forward(self, pos_input, pos_item, neg_item):

        emb_user = self.forward_to_user_embedding_layer(pos_input)

        # output embedding for positive instance
        emb_item = self.item_embeddings(pos_item)
        # output embedding for negative instance
        emb_neg_item = self.item_embeddings(neg_item)

        score = torch.sum(torch.mul(emb_user, emb_item), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)


        neg_score = torch.bmm(emb_neg_item, emb_user.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)    



class two_tower(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim1, out_dim):
        super(two_tower, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, out_dim)

    def forward_left(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        h = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(h))
        # log_probs = F.log_softmax(out, dim=1)
        return out

    def forward_right(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        h = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(h))
        # log_probs = F.log_softmax(out, dim=1)
        return out

    def forward(self, inputs):
        inputs_left, inputs_right = inputs
        return self.forward_left(inputs_left), self.forward_right(inputs_right)