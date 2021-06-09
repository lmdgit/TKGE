import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
import ipdb
import numpy as np
def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


class TTransE_SED(Model):

    def __init__(self, ent_tot, rel_tot, temp_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TTransE_SED, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.temp_tot=temp_tot

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.temp_embeddings = nn.Embedding(self.temp_tot, self.dim)
        self.rels_embeddings1 = nn.Embedding(self.rel_tot, self.dim)
        self.rels_embeddings2 = nn.Embedding(self.rel_tot, self.dim)
        self.relo_embeddings1 = nn.Embedding(self.rel_tot, self.dim)
        self.relo_embeddings2 = nn.Embedding(self.rel_tot, self.dim)        
        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.temp_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rels_embeddings1.weight.data)
            nn.init.xavier_uniform_(self.rels_embeddings2.weight.data)
            nn.init.xavier_uniform_(self.relo_embeddings1.weight.data)
            nn.init.xavier_uniform_(self.relo_embeddings2.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, temp, rhs, rho, rts, rto, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
            rh = F.normalize(rhs*h, 2, -1)
            rho = F.normalize(rho, 2, -1)
            rt = F.normalize(rts*t, 2, -1)
            rto = F.normalize(rto, 2, -1)
            temp = F.normalize(temp, 2, -1)

            score = (h + r + temp - t)
            score_cs=(rh - rho) 
            score_co=(rt - rto)
            
        score = torch.norm(score, self.p_norm, -1).flatten()
        score_cs = torch.norm(score_cs, self.p_norm, -1).flatten()
        score_co = torch.norm(score_co, self.p_norm, -1).flatten()
        score = score + score_cs + score_co 
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        batch_temp = data['batch_temp']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        rhs = self.rels_embeddings1(batch_r)
        rho = self.rels_embeddings2(batch_r)
        rts = self.relo_embeddings1(batch_r)
        rto = self.relo_embeddings2(batch_r)
        temp = self.temp_embeddings(batch_temp)
        score = self._calc(h, t, r, temp, rhs, rho, rts, rto, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
