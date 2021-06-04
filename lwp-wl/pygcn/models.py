import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.dc = InnerProductDecoder(dropout)

    def forward(self, x, adj, n1, n2, node_ids_to_index):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        n1_indices = torch.ones(n1.shape[0])
        for i, value in enumerate(n1):
            n1_indices[i] = node_ids_to_index[value.item()]
        n1_indices = n1_indices.cuda().long()

        n2_indices = torch.ones(n1.shape[0])
        for i, value in enumerate(n2):
            n2_indices[i] = node_ids_to_index[value.item()]
        n2_indices = n2_indices.cuda().long()

        n1_val = torch.index_select(x, 0, n1_indices)
        n2_val = torch.index_select(x, 0, n2_indices)
        
        x = self.dc(n1_val, n2_val)

        return x


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, n1, n2):
        adj = torch.sum(n1 * n2, dim=1)
        return adj