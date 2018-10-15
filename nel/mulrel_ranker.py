import torch
import torch.nn.functional as F
from torch.autograd import Variable

from nel.local_ctx_att_ranker import LocalCtxAttRanker
import numpy as np


class STArgmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        max_values, _ = scores.max(dim=-1, keepdim=True)
        return (scores >= max_values).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MulRelRanker(LocalCtxAttRanker):
    """
    multi-relational global model with context token attention, using loopy belief propagation
    """

    def __init__(self, config):
        super(MulRelRanker, self).__init__(config)
        self.df = config['df']  # damping factor
        self.n_loops = config['n_loops']
        self.n_rels = config['n_rels']
        self.dr = config['dr']
        self.ew_hid_dims = self.emb_dims

        self.max_dist = 1000
        self.ent_top_n = 1000

        self.oracle = config.get('oracle', False)
        self.ent_ent_comp = config.get('ent_ent_comp', 'bilinear')  # bilinear, trans_e, fbilinear
        self.ctx_comp = config.get('ctx_comp', 'bow')  # bow or rnn

        self.mode = config.get('mulrel_type', 'ment-norm')  # ment-norm, rel-norm

        # options for ment-norm
        self.first_head_uniform = config.get('first_head_uniform', False)
        self.use_pad_ent = config.get('use_pad_ent', False)

        # options for rel-norm
        self.use_stargmax = config.get('use_stargmax', False)

        self.use_local = config.get('use_local', False)
        self.use_local_only = config.get('use_local_only', False)
        self.freeze_local = config.get('freeze_local', False)

        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False

        if self.use_local:
            self.ent_localctx_comp = torch.nn.Parameter(torch.ones(self.emb_dims))

        if self.use_pad_ent:
            self.pad_ent_emb = torch.nn.Parameter(torch.randn(1, self.emb_dims) * 0.1)
            self.pad_ctx_vec = torch.nn.Parameter(torch.randn(1, self.emb_dims) * 0.1)

        self.ctx_layer = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dims * 3, self.ew_hid_dims),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=self.dr))

        self.rel_embs = torch.randn(self.n_rels, self.emb_dims) * 0.01
        if self.ent_ent_comp == 'bilinear':
            self.rel_embs[0] = 1 + torch.randn(self.emb_dims) * 0.01
            if self.mode == 'ment-norm' and self.n_rels > 1 and self.first_head_uniform:
                self.rel_embs[1] = 1
            if self.mode == 'rel-norm':
                self.rel_embs.fill_(0).add_(torch.randn(self.n_rels, self.emb_dims) * 0.1)

        self.rel_embs = torch.nn.Parameter(self.rel_embs)

        self.ew_embs = torch.nn.Parameter(torch.randn(self.n_rels, self.ew_hid_dims) *
                                          (0.01 if self.mode == 'ment-norm' else 0.1))

        self._coh_ctx_vecs = None

        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(2, self.hid_dims),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dims, 1))

        print('---------------- model config -----------------')
        for k, v in self.__dict__.items():
            if not hasattr(v, '__dict__'):
                print(k, v)
        print('-----------------------------------------------')

    def print_weight_norm(self):
        LocalCtxAttRanker.print_weight_norm(self)
        print(self.ctx_layer[0].weight.data.norm(), self.ctx_layer[0].bias.data.norm())
        print('relations', self.rel_embs.data.norm(p=2, dim=1))
        X = F.normalize(self.rel_embs)
        diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        print(diff)

        print('ew_embs', self.ew_embs.data.norm(p=2, dim=1))
        X = F.normalize(self.ew_embs)
        diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).sqrt()
        print(diff)

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, gold=None):
        n_ments, n_cands = entity_ids.size()
        n_rels = self.n_rels

        if self.mode == 'ment-norm' and self.first_head_uniform:
            self.ew_embs.data[0] = 0

        if not self.oracle:
            gold = None

        if self.use_local:
            local_ent_scores = super(MulRelRanker, self).forward(token_ids, tok_mask,
                                                                 entity_ids, entity_mask,
                                                                 p_e_m=None)
            ent_vecs = self._entity_vecs
        else:
            ent_vecs = self.entity_embeddings(entity_ids)
            local_ent_scores = Variable(torch.zeros(n_ments, n_cands).cuda(), requires_grad=False)

        # compute context vectors
        ltok_vecs = self.snd_word_embeddings(self.s_ltoken_ids) * self.s_ltoken_mask.view(n_ments, -1, 1)
        local_lctx_vecs = torch.sum(ltok_vecs, dim=1) / torch.sum(self.s_ltoken_mask, dim=1, keepdim=True).add_(1e-5)
        rtok_vecs = self.snd_word_embeddings(self.s_rtoken_ids) * self.s_rtoken_mask.view(n_ments, -1, 1)
        local_rctx_vecs = torch.sum(rtok_vecs, dim=1) / torch.sum(self.s_rtoken_mask, dim=1, keepdim=True).add_(1e-5)
        mtok_vecs = self.snd_word_embeddings(self.s_mtoken_ids) * self.s_mtoken_mask.view(n_ments, -1, 1)
        ment_vecs = torch.sum(mtok_vecs, dim=1) / torch.sum(self.s_mtoken_mask, dim=1, keepdim=True).add_(1e-5)
        bow_ctx_vecs = torch.cat([local_lctx_vecs, ment_vecs, local_rctx_vecs], dim=1)

        if self.use_pad_ent:
            ent_vecs = torch.cat([ent_vecs, self.pad_ent_emb.view(1, 1, -1).repeat(1, n_cands, 1)], dim=0)
            tmp = torch.zeros(1, n_cands)
            tmp[0, 0] = 1
            tmp = Variable(tmp.cuda())
            entity_mask = torch.cat([entity_mask, tmp], dim=0)
            p_e_m = torch.cat([p_e_m, tmp], dim=0)
            local_ent_scores = torch.cat([local_ent_scores,
                                          Variable(torch.zeros(1, n_cands).cuda(), requires_grad=False)],
                                         dim=0)
            n_ments += 1

            if self.oracle:
                tmp = Variable(torch.zeros(1, 1).cuda().long())
                gold = torch.cat([gold, tmp], dim=0)

        if self.use_local_only:
            inputs = torch.cat([Variable(torch.zeros(n_ments * n_cands, 1).cuda()),
                                local_ent_scores.view(n_ments * n_cands, -1),
                                torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)
            scores = self.score_combine(inputs).view(n_ments, n_cands)
            return scores

        if n_ments == 1:
            ent_scores = local_ent_scores

        else:
            # distance - to consider only neighbor mentions
            ment_pos = torch.arange(0, n_ments).long().cuda()
            dist = (ment_pos.view(n_ments, 1) - ment_pos.view(1, n_ments)).abs()
            dist.masked_fill_(dist == 1, -1)
            dist.masked_fill_((dist > 1) & (dist <= self.max_dist), -1)
            dist.masked_fill_(dist > self.max_dist, 0)
            dist.mul_(-1)

            ctx_vecs = self.ctx_layer(bow_ctx_vecs)
            if self.use_pad_ent:
                ctx_vecs = torch.cat([ctx_vecs, self.pad_ctx_vec], dim=0)

            m1_ctx_vecs, m2_ctx_vecs = ctx_vecs, ctx_vecs
            rel_ctx_vecs = m1_ctx_vecs.view(1, n_ments, -1) * self.ew_embs.view(n_rels, 1, -1)
            rel_ctx_ctx_scores = torch.matmul(rel_ctx_vecs, m2_ctx_vecs.view(1, n_ments, -1).permute(0, 2, 1))  # n_rels x n_ments x n_ments

            rel_ctx_ctx_scores = rel_ctx_ctx_scores.add_((1 - Variable(dist.float().cuda())).mul_(-1e10))
            eye = Variable(torch.eye(n_ments).cuda()).view(1, n_ments, n_ments)
            rel_ctx_ctx_scores.add_(eye.mul_(-1e10))
            rel_ctx_ctx_scores.mul_(1 / np.sqrt(self.ew_hid_dims))  # scaling proposed by "attention is all you need"

            # get top_n neighbour
            if self.ent_top_n < n_ments:
                topk_values, _ = torch.topk(rel_ctx_ctx_scores, k=min(self.ent_top_n, n_ments), dim=2)
                threshold = topk_values[:, :, -1:]
                mask = 1 - (rel_ctx_ctx_scores >= threshold).float()
                rel_ctx_ctx_scores.add_(mask.mul_(-1e10))

            if self.mode == 'ment-norm':
                rel_ctx_ctx_probs = F.softmax(rel_ctx_ctx_scores, dim=2)
                rel_ctx_ctx_weights = rel_ctx_ctx_probs + rel_ctx_ctx_probs.permute(0, 2, 1)
                self._rel_ctx_ctx_weights = rel_ctx_ctx_probs
            elif self.mode == 'rel-norm':
                ctx_ctx_rel_scores = rel_ctx_ctx_scores.permute(1, 2, 0).contiguous()
                if not self.use_stargmax:
                    ctx_ctx_rel_probs = F.softmax(ctx_ctx_rel_scores.view(n_ments * n_ments, n_rels))\
                                        .view(n_ments, n_ments, n_rels)
                else:
                    ctx_ctx_rel_probs = STArgmax.apply(ctx_ctx_rel_scores)
                self._rel_ctx_ctx_weights = ctx_ctx_rel_probs.permute(2, 0, 1).contiguous()

            # compute phi(ei, ej)
            if self.mode == 'ment-norm':
                if self.ent_ent_comp == 'bilinear':
                    if self.ent_ent_comp == 'bilinear':
                        rel_ent_vecs = ent_vecs.view(1, n_ments, n_cands, -1) * self.rel_embs.view(n_rels, 1, 1, -1)
                    elif self.ent_ent_comp == 'trans_e':
                        rel_ent_vecs = ent_vecs.view(1, n_ments, n_cands, -1) - self.rel_embs.view(n_rels, 1, 1, -1)
                    else:
                        raise Exception("unknown ent_ent_comp")

                    rel_ent_ent_scores = torch.matmul(rel_ent_vecs.view(n_rels, n_ments, 1, n_cands, -1),
                                                      ent_vecs.view(1, 1, n_ments, n_cands, -1).permute(0, 1, 2, 4, 3))
                    # n_rels x n_ments x n_ments x n_cands x n_cands

                rel_ent_ent_scores = rel_ent_ent_scores.permute(0, 1, 3, 2, 4)  # n_rel x n_ments x n_cands x n_ments x n_cands
                rel_ent_ent_scores = (rel_ent_ent_scores * entity_mask).add_((entity_mask - 1).mul_(1e10))
                ent_ent_scores = torch.sum(rel_ent_ent_scores *
                                           rel_ctx_ctx_weights.view(n_rels, n_ments, 1, n_ments, 1), dim=0)\
                                 .mul(1. / n_rels)  # n_ments x n_cands x n_ments x n_cands

            elif self.mode == 'rel-norm':
                rel_vecs = torch.matmul(ctx_ctx_rel_probs.view(n_ments, n_ments, 1, n_rels),
                                        self.rel_embs.view(1, 1, n_rels, -1))\
                           .view(n_ments, n_ments, -1)
                ent_rel_vecs = ent_vecs.view(n_ments, 1, n_cands, -1) * rel_vecs.view(n_ments, n_ments, 1, -1)  # n_ments x n_ments x n_cands x dims
                ent_ent_scores = torch.matmul(ent_rel_vecs,
                                              ent_vecs.view(1, n_ments, n_cands, -1).permute(0, 1, 3, 2))\
                                 .permute(0, 2, 1, 3)

            if gold is None:
                # LBP
                prev_msgs = Variable(torch.zeros(n_ments, n_cands, n_ments).cuda())

                for _ in range(self.n_loops):
                    mask = 1 - Variable(torch.eye(n_ments).cuda())
                    ent_ent_votes = ent_ent_scores + local_ent_scores * 1 + \
                                    torch.sum(prev_msgs.view(1, n_ments, n_cands, n_ments) *
                                              mask.view(n_ments, 1, 1, n_ments), dim=3)\
                                    .view(n_ments, 1, n_ments, n_cands)
                    msgs, _ = torch.max(ent_ent_votes, dim=3)
                    msgs = (F.softmax(msgs, dim=1).mul(self.dr) +
                            prev_msgs.exp().mul(1 - self.dr)).log()
                    prev_msgs = msgs

                # compute marginal belief
                mask = 1 - Variable(torch.eye(n_ments).cuda())
                ent_scores = local_ent_scores * 1 + torch.sum(msgs * mask.view(n_ments, 1, n_ments), dim=2)
                ent_scores = F.softmax(ent_scores, dim=1)
            else:
                onehot_gold = Variable(torch.zeros(n_ments, n_cands).cuda()).scatter_(1, gold, 1)
                ent_scores = torch.sum(torch.sum(ent_ent_scores * onehot_gold, dim=3), dim=2)

        # combine with p_e_m
        inputs = torch.cat([ent_scores.view(n_ments * n_cands, -1),
                            torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1)], dim=1)
        scores = self.score_combine(inputs).view(n_ments, n_cands)

        if self.use_pad_ent:
            scores = scores[:-1]
        return scores

    def regularize(self, max_norm=1):
        super(MulRelRanker, self).regularize(max_norm)

    def loss(self, scores, true_pos, lamb=1e-7):
        loss = F.multi_margin_loss(scores, true_pos, margin=self.margin)
        if self.use_local_only:
            return loss

        # regularization
        X = F.normalize(self.rel_embs)
        diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        diff = diff * (diff < 1).float()
        loss -= torch.sum(diff).mul(lamb)

        X = F.normalize(self.ew_embs)
        diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
        diff = diff * (diff < 1).float()
        loss -= torch.sum(diff).mul(lamb)
        return loss
