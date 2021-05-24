from model.transformer_encoder import NewTransformerEncoder, BertLSTMEncoder
from model.GAT import MultiHeadedGAT
from model.neural import PositionwiseFeedForward, sequence_mask
from model.roberta import RobertaEmbedding

import torch.nn as nn
import torch

class EMEncoder(nn.Module):
    def __init__(self, args, device, src_embeddings, padding_idx, bert_model):
        super(EMEncoder, self).__init__()
        self.args = args
        self.device = device
        self.padding_idx = padding_idx
        # self._TFembed = nn.Embedding(50, self.args.emb_size) # box=10 , embed_size = 256

        if args.use_bert:
            self.bert_model = bert_model
            self.para_encoder = BertLSTMEncoder(self.bert_model)
            self.cluster_encoder = BertLSTMEncoder(self.bert_model)
        else:
            self.para_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                   self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
            self.cluster_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                   self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)

        # Para attends to the clusters
        self.P2C = MultiHeadedGAT(self.args.heads, self.args.enc_hidden_size, self.args.enc_dropout)
        # CLuster attends to the paras
        self.C2P = MultiHeadedGAT(self.args.heads, self.args.enc_hidden_size, self.args.enc_dropout)

        self.layer_norm = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)


    def forward(self, src, cluster, edge):
        """
        :param src:  batch_size x n_paras x n_tokens
        :param cluster: batch_size x n_clusters x n_cluster_tokens
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        batch_size, n_paras, n_tokens = src.size()
        n_clusters, n_cluster_tokens = cluster.size(1), cluster.size(2)

        para_feature, para_context, _ = self.para_encoder(src)      #
        cluster_feature, _, __ = self.cluster_encoder(cluster)

        # the start state
        cluster_state = cluster_feature
        para_state = self.P2C(para_feature, cluster_feature, edge, para_center=True)        # batch_size x n_paras x hidden

        # TODO add gat_iter to args

        # the iterative updating process
        for i in range(self.args.gat_iter):
            cluster_state = self.C2P(cluster_state, para_state, edge, para_center=False)
            para_state = self.P2C(para_state, cluster_state, edge, para_center=True)

        _para_state = para_state.unsqueeze(2)
        para_context = self.feed_forward(para_context + _para_state)         # batch_size x n_paras x n_tokens x hidden
        mask_para = ~(src.data.eq(self.padding_idx).bool())
        mask_para = mask_para[:, :, :, None].float()
        para_context = para_context * mask_para
        # to be consistent with the predictor
        # para_context = para_context.view(batch_size, n_paras * n_tokens, -1).transpose(0, 1).contiguous()       # (n_paras*n_tokens) x batch_size x hidden

        return para_state, para_context, cluster_state





        # # paras_in_one_batch, n_tokens = src.size()
        # # n_paras = 20
        # # assert paras_in_one_batch % n_paras == 0
        # # batch_size = paras_in_one_batch // n_paras
        #
        # # the following steps are to concat the context vectorst to a long document, so as to feed to the decoder.
        # mask_local = ~(src.data.eq(self.padding_idx).view(-1, n_tokens).bool())
        # mask_hier = mask_local[:, :, None]  # n_paras * batch_size x n_tokens x 1
        # context = context * mask_hier  # n_paras * batch_size x n_tokens x embed_dim
        # context = context.view(batch_size, n_paras * n_tokens, -1)
        # context = context.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        #
        # mask_hier = mask_hier.view(batch_size, n_paras * n_tokens, -1).bool()
        # mask_hier = mask_hier.transpose(0, 1).contiguous()  # src_len, batch_size, 1
        #
        # unpadded = [torch.masked_select(context[:, i], mask_hier[:, i]).view([-1, context.size(-1)])
        #             for i in range(
        #         context.size(1))]  # [tensor(src_len1 x embed_dim), tensor(src_len2 x embed_dim), ...] without pad token
        # max_l = max([p.size(0) for p in unpadded])  # max_src_len
        # mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).bool().to(self.device)
        # mask_hier = ~mask_hier[:, None,
        #              :]  # real_batch_size x 1 x max_src_len, result after concat all the paras in one example
        # src_features = torch.stack(
        #     [torch.cat([p, torch.zeros(max_l - p.size(0), context.size(-1)).to(self.device)]) for p in unpadded],
        #     1)  # max_src_len x real_batch_size x embed_dim
        #
        # return src_features, mask_hier