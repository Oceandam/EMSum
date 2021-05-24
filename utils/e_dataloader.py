import gc
import glob
import random
import torch
import numpy as np

from utils.logging import logger


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_dataset_train(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)
        for i in range(6):
            pts.extend(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def load_dataset_val(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "val", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


class GraphDataLoader(object):
    def __init__(self, args, datasets, symbols, batch_size, device, shuffle, is_test):
        """
        :param args:
        :param datasets: yield _lazy_dataset_laoder
        :param symbols: special token id in spm
        :param batch_size:
        :param device: "cuda"
        :param shuffle:
        :param is_test:
        """
        self.args = args
        self.datasets = datasets
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for releasing memory
            if hasattr(self, "cur_iter"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)

        except StopIteration:
            return None

        return GraphIterator(args=self.args, dataset=self.cur_dataset, symbols=self.symbols,
                             batch_size=self.batch_size, device=self.device,
                             shuffle=self.shuffle, is_test=self.is_test)


class GraphIterator(object):
    def __init__(self, args, dataset, symbols, batch_size, device=None, shuffle=True, is_test=False):
        self.args = args
        self.dataset = dataset
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        self.iterations = 0
        self._iterations_this_epoch = 0

        self.secondary_sort_key = lambda x: sum([len(xi) for xi in x['src']])
        self.prime_sort_key = lambda x: len(x['tgt'])

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            c_len = 0
            for i in range(len(ex['clusters'])):
                c_len += len(ex['clusters'][i])
            if c_len == 0:
                continue
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def simple_batch_size_fn(self, new, count):
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
        if (self.args.hier):
            max_src_in_batch = max(max_src_in_batch, sum([len(p) for p in new['src']]))
        else:
            max_src_in_batch = max(max_src_in_batch, len(new['src']))
        src_elements = count * max_src_in_batch
        return src_elements

    def get_batch(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        data = self.data()
        for minibatch_buffer in self.batch_buffer(data, self.batch_size * 100):
            if self.args.mode != 'train':
                p_batch = self.get_batch(
                    sorted(sorted(minibatch_buffer, key=self.prime_sort_key), key=self.secondary_sort_key),
                    self.batch_size
                )
            else:
                p_batch = self.get_batch(
                    sorted(sorted(minibatch_buffer, key=self.secondary_sort_key), key=self.prime_sort_key),
                    self.batch_size
                )
            p_batch = list(p_batch)

            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = GraphBatch(minibatch, self.args.hier, self.symbols['PAD'], self.device, self.is_test,
                                   self.symbols['BOS'], self.symbols['EOS'])

                yield batch

            return


class GraphBatch(object):
    def __init__(self, data=None, hier=False, pad_id=None, device=None, is_test=False, bos_id=None, eos_id=None):
        if data is not None:
            self.batch_size = len(data)
            self.pad_id = pad_id
            self.bos_id = bos_id
            self.eos_id = eos_id
            src = [ex['src'] for ex in data]
            tgt = [ex['tgt'] for ex in data]
            clusters = [ex['clusters'] for ex in data]

            self.max_npara = max([len(ex) for ex in src])  # generally, it's 20
            self.max_ntoken = max([max([len(p) for p in e]) for e in src])  # generally, it's 100
            self.max_ncluster = max([len(cluster) for cluster in clusters])
            self.max_cluster_ntoken = 0
            c_len = []
            for cs in clusters:
                if cs is not None:
                    cs_len = [len(cluster) for cluster in cs]
                else:
                    cs_len = 0
                c_len.extend(cs_len)
            if len(c_len) != 0:
                self.max_cluster_ntoken = max(c_len)
            else:
                self.max_cluster_ntoken = 0
            # self.max_cluster_ntoken = max([max([len(c) for c in cs]) if cs is not None else 0 for cs in clusters])
            if self.max_cluster_ntoken > 50:
                self.max_cluster_ntoken = 50
            if self.max_ntoken > 100:
                self.max_ntoken = 100
            if self.max_npara > 20:
                self.max_npara =20
            _src = [self._pad_para(ex[:self.max_npara], self.max_npara, self.max_ntoken, self.pad_id, self.bos_id, self.eos_id) for ex in src]
            _cluster = [self._pad_para(ex[:self.max_ncluster], self.max_ncluster, self.max_cluster_ntoken, self.pad_id, self.bos_id, self.eos_id) for ex in clusters]
            src = torch.stack([torch.tensor(e[0]) for e in _src])  # batch_size * max_npara * max_ntoken
            cluster = torch.stack([torch.tensor(e[0]) for e in _cluster])

            batch_E = []
            for ex in data:
                Eij = []
                c2p = ex['tfidf']
                for i in range(self.max_npara):
                    Ei = []
                    if not c2p.__contains__(str(i)):
                        c2p[str(i)] = {}
                    for j in range(self.max_ncluster):
                        if not c2p[str(i)].__contains__(str(j)):
                            c2p[str(i)][str(j)] = 0.0
                        Ei.append(c2p[str(i)][str(j)])
                    Eij.append(Ei)
                batch_E.append(Eij)
            batch_E = torch.FloatTensor(batch_E).to(device)

            setattr(self, 'edge', batch_E)

            # graphs = []
            # for ex in data:
            #     G = self.createGraph(ex)
            #     graphs.append(G)
            #
            # batched_graph = dgl.batch(graphs)
            # setattr(self, 'graph', batched_graph.to(device))

            setattr(self, 'src', src.to(device))
            setattr(self, 'cluster', cluster.to(device))

            _tgt = self._pad_para(tgt, width=max([len(d) for d in tgt]), height=len(tgt), pad_id=pad_id, bos_id=self.bos_id, eos_id=self.eos_id)
            tgt = torch.tensor(_tgt[0]).transpose(0, 1)
            setattr(self, 'tgt', tgt.to(device))

            if (is_test):
                tgt_str = [ex['tgt_str'] for ex in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

    # def batchGraph(self, graphs):
    #     """
    #     :param graphs: a list of graphs
    #     :return: a batchedgraph
    #     """
    #     # graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # para node of graph
    #     # sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    #     # batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    #     batched_graph = dgl.batch(graphs)
    #     return batched_graph

    # def createGraph(self, ex):
    #     G = dgl.DGLGraph()
    #     # Todo ignore warning message here
    #
    #     # Create cluster nodes
    #     clusters_pad = self._pad_cluster(ex['clusters'], self.pad_id)
    #     c_nodes = len(clusters_pad)
    #     cluster_nid = [i for i in range(c_nodes)]
    #     G.add_nodes(c_nodes)
    #     G.set_n_initializer(dgl.init.zero_initializer)
    #     G.ndata["unit"] = torch.zeros(c_nodes)
    #     G.nodes[cluster_nid].data['id'] = torch.LongTensor(clusters_pad)
    #     G.ndata["dtype"] = torch.zeros(c_nodes)
    #
    #     # Create para nodes
    #     paras_pad, _ = self._pad_para(ex['src'], self.max_npara, self.max_ntoken, self.pad_id)
    #     p_nodes = len(paras_pad)
    #     G.add_nodes(p_nodes)
    #     G.ndata["unit"][c_nodes:] = torch.ones(p_nodes)
    #     G.ndata["dtype"][c_nodes:] = torch.ones(p_nodes)
    #     paraid2nid = [i + c_nodes for i in range(p_nodes)]
    #     G.nodes[paraid2nid].data["tokens"] = torch.LongTensor(paras_pad)
    #     G.nodes[paraid2nid].data["position"] = torch.arange(1, p_nodes + 1).view(-1, 1).long()
    #
    #     # Create edges
    #     c2p = ex['tfidf']
    #     G.set_e_initializer(dgl.init.zero_initializer)
    #     for i in range(p_nodes):
    #         if not c2p.__contains__(str(i)):
    #             c2p[str(i)] = {}
    #         para_nid = paraid2nid[i]
    #         para_tfidf = c2p[str(i)]
    #         for cluster_id, tfidf in para_tfidf.items():
    #             tfidf_box = np.round(tfidf * 9)
    #             G.add_edges(int(cluster_id), para_nid,
    #                         data={"ttfrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
    #             G.add_edges(para_nid, int(cluster_id),
    #                         data={"ttfrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
    #
    #     return G

    def _pad_cluster(self, clusters, pad_id, _max_cluster_token=50):
        """
        :param clusters:  [ [] ], src clusters in one example
        :return:
        """
        # there exits instance where clusters = []
        clusters_pad = []
        if len(clusters) == 0:
            clusters.append([])  # transform [] to [[]]

        if _max_cluster_token > self.max_cluster_ntoken:
            _max_cluster_token = self.max_cluster_ntoken

        trunc_cluster = [cluster[: _max_cluster_token] for cluster in clusters]
        # max_cluster_token_ex = max([len(cluster) for cluster in trunc_cluster])
        for i in range(len(clusters)):
            _cluster = clusters[i].copy()
            if len(_cluster) > _max_cluster_token:
                _cluster = _cluster[:_max_cluster_token]
            if len(_cluster) < _max_cluster_token:
                _cluster.extend([pad_id] * (_max_cluster_token - len(_cluster)))
            clusters_pad.append(_cluster)
        return clusters_pad

    def _pad_para(self, data, height, width, pad_id, bos_id, eos_id):
        """
        :param data:  [ [] ], src paras in one example
        :param height: num of paras in one example, generally, it's 20
        :param width:  num of max_tokens
        :param pad_id:
        :return:
        """
        # rtn_data = [para + [pad_id] * (width - len(para)) for para in data]
        rtn_data = []
        for para in data:
            if len(para) > width:
                para = para[:width]
            else:
                para += [pad_id] * (width - len(para))
            rtn_data.append(para)
        rtn_length = [len(para) for para in data]
        x = []
        x.append(bos_id)
        x.append(eos_id)
        x.extend([pad_id] * (width-2))
        rtn_data = rtn_data + [x] * (height - len(data))
        # rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        if len(rtn_data) == 0:
            rtn_data.append([])
        return rtn_data, rtn_length