from model.transformer_encoder import NewTransformerEncoder
from model.GAT import ESEGAT
from model.EMDecoder import TransformerDecoder
from model.neural import PositionwiseFeedForward, sequence_mask
from model.optimizer import Optimizer
from model.EMNewEncoder import EMEncoder
from model.roberta import RobertaEmbedding

import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
        args.optim, args.lr, args.max_grad_norm,
        beta1=args.beta1, beta2=args.beta2,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)


    if args.train_from != '':
        optim.optimizer.load_state_dict(checkpoint['optim'])
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    optim.set_parameters(list(model.named_parameters()))
    return optim


def get_generator(dec_hidden_size, vocab_size, emb_dim, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, emb_dim),
        nn.LeakyReLU(),
        nn.Linear(emb_dim, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator



class Summarizer(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None):
        super(Summarizer, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.device = device
        self.padding_idx = word_padding_idx

        if args.use_bert:
            self.bert_model = RobertaEmbedding()
            self.encoder = EMEncoder(self.args, self.device, None, self.padding_idx, self.bert_model)
            tgt_embeddings = self.bert_model._embedding
            tgt_embeddings.weight.requires_grad = False
            emb_dim = tgt_embeddings.weight.size(1)
        else:
            src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
            tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
            if self.args.share_embeddings:
                tgt_embeddings.weight = src_embeddings.weight
            self.encoder = EMEncoder(self.args, self.device, src_embeddings, self.padding_idx, None)
        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, emb_dim, device)
        if self.args.share_decoder_embeddings:
            self.generator[2].weight = tgt_embeddings.weight

        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for n, p in self.named_parameters():
                if 'RobertaModel' not in n:
                    if p.dim() > 1:
                        xavier_uniform_(p)


        self.to(device)

    def forward(self, src, cluster, tgt, edge):
        """
        :param src:  batch_size x n_paras x n_tokens
        :param cluster: batch_size x n_clusters x n_cluster_tokens
        :param tgt: n_tgt_tokens x batch_size
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        tgt = tgt[:-1]

        #

        # self.cnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        para_state, para_context, cluster_state = self.encoder(src, cluster, edge)
        # src = src.view(batch_size, n_paras, n_tokens)    # be consistent with the init decoder state
        dec_state = self.decoder.init_decoder_state(src, para_context)     # src: num_paras_in_one_batch x max_length
        decoder_outputs = self.decoder(tgt, para_context, dec_state, edge, cluster, cluster_state)
        # tgt, memory_bank, state, edge, cluster, cluster_feature
        return decoder_outputs

