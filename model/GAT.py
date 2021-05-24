import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from model.GATLayer import MultiHeadLayer
from model.GATLayer import PositionwiseFeedForward, ESGATLayer, SEGATLayer

class ESEGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if self.layerType == "E2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=ESGATLayer)
        elif self.layerType == "S2E":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SEGATLayer)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, e, s):
        if self.layerType == "E2S":
            origin, neighbor = s, e
        elif self.layerType == "S2E":
            origin, neighbor = e, s
        elif self.layerType == "S2S":
            assert torch.equal(e, s)
            origin, neighbor = e, s
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, origin, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class MultiHeadedGAT(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedGAT, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if(self.use_final_linear):
            # self.final_linear = nn.Linear(model_dim, model_dim)
            self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, para, cluster, edge, para_center = True):
        """
        :param para: batch_size x n_paras x emb
        :param cluster: batch_size x n_clusters x emb
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        if not para_center:
            edge = edge.transpose(1, 2).contiguous()
        batch_size = para.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        n_paras = para.size(1)
        n_clusters = cluster.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_keys(cluster)       # batch_size x n_clusters x hidden
        value = self.linear_values(cluster)         # batch_size x n_clusters x hidden
        query = self.linear_query(para)         # batch_size x n_paras x hidden
        key = shape(key)            # batch_size x head_count x n_clusters x dim_per_head
        value = shape(value)        # batch_size x head_count x n_clusters x dim_per_head

        query = shape(query)        # batch_size x head_count x n_paras x dim_per_head

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))       # batch_size x head_count x n_paras x n_clusters

        mask = ~(edge > 0)         # batch_size x n_paras x n_clusters

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores).bool()
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)         # batch_size x head_count x n_paras x n_clusters

        edge = edge.unsqueeze(1).expand_as(attn)
        attn = attn * edge

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))           # batch_size x n_paras x hidden
            # output = self.final_linear(context)
            output = self.layer_norm(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context
