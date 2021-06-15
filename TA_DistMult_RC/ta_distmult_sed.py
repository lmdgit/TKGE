import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

from LSTMLinear import LSTMModel
#import ipdb

		
class TA_DistMult_SED(torch.nn.Module):
    
    def __init__(self, dataset, params):

        super(TA_DistMult_SED, self).__init__()
        self.dataset = dataset
        self.params = params
        self.tem_total = 32
        # Creating static embeddings.
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim).cuda()
        
        self.tem_embeddings = nn.Embedding(self.tem_total, params.s_emb_dim).cuda()
        self.lstm = LSTMModel(params.s_emb_dim, n_layer=1).cuda()
        
        self.rels1_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rels2_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.relo1_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.relo2_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
		

        

        

        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.tem_embeddings.weight)
        
        nn.init.xavier_uniform_(self.rels1_embs.weight)
        nn.init.xavier_uniform_(self.rels2_embs.weight)
        nn.init.xavier_uniform_(self.relo1_embs.weight)
        nn.init.xavier_uniform_(self.relo2_embs.weight)
        

        
            

    
    
    def get_rseq(self, r_e, tem):
        #r_e = self.rel_embeddings(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        bs = tem.shape[0]  # batch size
        tem_len = tem.shape[1]
        tem = tem.contiguous()
        tem = tem.view(bs * tem_len)
        token_e = self.tem_embeddings(tem)
        token_e = token_e.view(bs, tem_len, self.params.s_emb_dim)
        seq_e = torch.cat((r_e, token_e), 1)

        hidden_tem = self.lstm(seq_e)
        hidden_tem = hidden_tem[0, :, :]
        rseq_e = hidden_tem

        return rseq_e    
    
    def getEmbeddings(self, heads, rels, tails, temps, intervals = None):

        
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails) 
        
        r_seq = self.get_rseq(r, temps)
        
        rs1,rs2,ro1,ro2= self.rels1_embs(rels), self.rels2_embs(rels), self.relo1_embs(rels), self.relo2_embs(rels)
        
        h_c= rs1*h - rs2
        t_c= ro1*t - ro2


        return h,r_seq,t, h_c, t_c
        
    def forward(self, heads, rels, tails, temps):
        #ipdb.set_trace()
        h_embs, r_embs, t_embs, h_c, t_c= self.getEmbeddings(heads, rels, tails, temps)
        
        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
		
        h_c = F.dropout(h_c, p=self.params.dropout, training=self.training)
        h_c = -torch.norm(h_c, dim = 1)
        t_c = F.dropout(t_c, p=self.params.dropout, training=self.training)
        t_c = -torch.norm(t_c, dim = 1)
        
        
        return scores + h_c + t_c