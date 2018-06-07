import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nel.utils as utils
from nel.abstract_word_entity import AbstractWordEntity


class NTEE(AbstractWordEntity):
    """
    NTEE model, proposed in Yamada et al. "Learning Distributed Representations of Texts and Entities from Knowledge Base"
    """

    def __init__(self, config):
        config['word_embeddings_class'] = nn.EmbeddingBag
        config['entity_embeddings_class'] = nn.Embedding
        super(NTEE, self).__init__(config)
        self.linear = nn.Linear(self.emb_dims, self.emb_dims)

    def compute_sent_vecs(self, token_ids, token_offsets, use_sum=False):
        sum_vecs = self.word_embeddings(token_ids, token_offsets)
        if use_sum:
            return sum_vecs

        sum_vecs = F.normalize(sum_vecs)
        sent_vecs = self.linear(sum_vecs)
        return sent_vecs

    def forward(self, token_ids, token_offsets, entity_ids, use_sum=False):
        sent_vecs = self.compute_sent_vecs(token_ids, token_offsets, use_sum)
        entity_vecs = self.entity_embeddings(entity_ids)

        # compute scores
        batchsize, dims = sent_vecs.size()
        n_entities = entity_vecs.size(1)
        scores = torch.bmm(entity_vecs, sent_vecs.view(batchsize, dims, 1)).view(batchsize, n_entities)

        log_probs = F.log_softmax(scores, dim=1)
        return log_probs

    def predict(self, token_ids, token_offsets, entity_ids, gold_entity_ids=None):
        log_probs = self.forward(token_ids, token_offsets, entity_ids)
        _, pred_entity_ids = torch.max(log_probs, dim=1)

        acc = None
        if gold_entity_ids is not None:
            acc = torch.eq(gold_entity_ids, pred_entity_ids).sum()
        return pred_entity_ids, acc

    def loss(self, log_probs, true_pos):
        return F.nll_loss(log_probs, true_pos)


def create_ntee_from_components(dir_path):
    word_dict_path = dir_path + '/dict.word'
    word_embs_path = dir_path + '/word_embeddings.npy'
    entity_dict_path = dir_path + '/dict.entity'
    entity_embs_path = dir_path + '/entity_embeddings.npy'
    W_path = dir_path + '/W.npy'
    b_path = dir_path + '/b.npy'

    print('load voca and embeddings')
    word_voca, word_embs = utils.load_voca_embs(word_dict_path, word_embs_path)
    entity_voca, entity_embs = utils.load_voca_embs(entity_dict_path, entity_embs_path)
    config = {'word_embeddings': word_embs,
              'entity_embeddings': entity_embs,
              'word_voca':  word_voca,
              'entity_voca': entity_voca,
              'emb_dims': word_embs.shape[1]}
    print(word_embs.shape, entity_embs.shape)

    # create model
    print('create model')
    model = NTEE(config)

    W = np.load(W_path)
    b = np.load(b_path)
    model.linear.weight = nn.Parameter(torch.FloatTensor(W).t())
    model.linear.bias = nn.Parameter(torch.FloatTensor(b))

    return model

