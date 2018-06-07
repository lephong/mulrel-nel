import torch
import torch.nn as nn
import torch.nn.functional as F
import nel.utils as utils
from nel.abstract_word_entity import AbstractWordEntity


class LocalCtxAttRanker(AbstractWordEntity):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker, self).__init__(config)

        self.hid_dims = config['hid_dims']
        self.tok_top_n = config['tok_top_n']
        self.margin = config['margin']

        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.local_ctx_dr = nn.Dropout(p=0)

        self.score_combine_linear_1 = nn.Linear(2, self.hid_dims)
        self.score_combine_act_1 = nn.ReLU()
        self.score_combine_linear_2 = nn.Linear(self.hid_dims, 1)

    def print_weight_norm(self):
        print('att_mat_diag', self.att_mat_diag.data.norm())
        print('tok_score_mat_diag', self.tok_score_mat_diag.data.norm())
        print('f - l1.w, b', self.score_combine_linear_1.weight.data.norm(),  self.score_combine_linear_1.bias.data.norm())
        print('f - l2.w, b', self.score_combine_linear_2.weight.data.norm(),  self.score_combine_linear_2.bias.data.norm())

    def print_attention(self, gold_pos):
        token_ids = self._token_ids.data.cpu().numpy()
        entity_ids = self._entity_ids.data.cpu().numpy()
        att_probs = self._att_probs.data.cpu().numpy()
        top_tok_att_ids = self._top_tok_att_ids.data.cpu().numpy()
        gold_pos = gold_pos.data.cpu().numpy()
        scores = self._scores.data.cpu().numpy()

        print('===========================================')
        for tids, eids, ap, aids, gpos, ss in zip(token_ids, entity_ids, att_probs, top_tok_att_ids, gold_pos, scores):
            selected_tids = tids[aids]
            print('-------------------------------')
            print(utils.tokgreen(repr([(self.entity_voca.id2word[e], s) for e, s in zip(eids, ss)])),
                  utils.tokblue(repr(self.entity_voca.id2word[eids[gpos]] if gpos > -1 else 'UNKNOWN')))
            print([(self.word_voca.id2word[t], a[0]) for t, a in zip(selected_tids, ap)])

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m=None):
        batchsize, n_words = token_ids.size()
        n_entities = entity_ids.size(1)
        tok_mask = tok_mask.view(batchsize, 1, -1)

        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.entity_embeddings(entity_ids)

        # att
        ent_tok_att_scores = torch.bmm(entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(1e10))
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(batchsize, -1, 1)
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(batchsize, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(batchsize, n_entities)

        # combine with p(e|m) if p_e_m is not None
        if p_e_m is not None:
            inputs = torch.cat([ent_ctx_scores.view(batchsize * n_entities, -1),
                                torch.log(p_e_m + 1e-20).view(batchsize * n_entities, -1)], dim=1)
            hidden = self.score_combine_linear_1(inputs)
            hidden = self.score_combine_act_1(hidden)
            scores = self.score_combine_linear_2(hidden).view(batchsize, n_entities)
        else:
            scores = ent_ctx_scores

        scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

        # printing attention (debugging)
        self._token_ids = token_ids
        self._entity_ids = entity_ids
        self._att_probs = att_probs
        self._top_tok_att_ids = top_tok_att_ids
        self._scores = scores

        self._entity_vecs = entity_vecs
        self._local_ctx_vecs = ctx_vecs

        return scores

    def regularize(self, max_norm=1):
        l1_w_norm = self.score_combine_linear_1.weight.norm()
        l1_b_norm = self.score_combine_linear_1.bias.norm()
        l2_w_norm = self.score_combine_linear_2.weight.norm()
        l2_b_norm = self.score_combine_linear_2.bias.norm()

        if (l1_w_norm > max_norm).data.all():
            self.score_combine_linear_1.weight.data = self.score_combine_linear_1.weight.data * max_norm / l1_w_norm.data
        if (l1_b_norm > max_norm).data.all():
            self.score_combine_linear_1.bias.data = self.score_combine_linear_1.bias.data *  max_norm / l1_b_norm.data
        if (l2_w_norm > max_norm).data.all():
            self.score_combine_linear_2.weight.data = self.score_combine_linear_2.weight.data * max_norm / l2_w_norm.data
        if (l2_b_norm > max_norm).data.all():
            self.score_combine_linear_2.bias.data = self.score_combine_linear_2.bias.data *  max_norm / l2_b_norm.data

    def loss(self, scores, true_pos):
        loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        return loss
