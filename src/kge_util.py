#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import logging
import numpy as np
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

class KGEModels(object):

    def __init__(self, temp, n_corruptions, n_ent_clusters, n_rel_clusters):
        self.temp = temp
        self.n_corruptions = n_corruptions
        self.ent_temp = 1e-5/n_ent_clusters
        self.rel_temp = 1e-5/n_rel_clusters

    @staticmethod
    def cconv_score(h, t, r):
        bsize, n_dims = t.size()
        t = t.view(bsize, 1, n_dims)
        tprime = torch.cat([t.roll(-i, dims=2) for i in range(n_dims)], dim=1)  # dims: batch x n_dims x n_dims
        h = h.view(bsize, 1, n_dims)
        ht_product = (h * tprime).sum(dim=2)
        return torch.bmm(r.view(bsize, 1, n_dims), ht_product.unsqueeze(2)).squeeze()

    def cconv_score_batch(self, h, t, r):
        r = r.repeat((self.n_corruptions, 1, 1))
        if h.shape[0] == self.n_corruptions: t = t.repeat((self.n_corruptions, 1, 1))
        elif t.shape[0] == self.n_corruptions: h = h.repeat((self.n_corruptions, 1, 1))
        n_dims = t.shape[-1]
        t = t.unsqueeze(-2)
        h = h.unsqueeze(-2)
        tprime = torch.cat([t.roll(-i, dims=-1) for i in range(n_dims)], dim=-2)
        ht_product = (h * tprime).sum(dim=-1)
        return torch.sum(ht_product * r, dim=-1).T

    def get_negative_samples(self, prob_matrix, epsilon=1e-20):
        neg_samples = list()
        logits_value = torch.log(1.-prob_matrix+epsilon) - torch.log(prob_matrix+epsilon)  # To prevent NaNs.
        for _ in range(self.n_corruptions):
            neg_samples.append(F.gumbel_softmax(logits_value, tau=self.temp, hard=True))
        return torch.stack(neg_samples)  # dims: n_corruptions x batch_size x (n_ent_clusters OR n_rel_clusters)

    def get_embeddings(self, hrt_probs, entity_embeddings, rel_embeddings, status='pos'):
        result_embeddings = list()
        embeddings = [entity_embeddings, rel_embeddings]
        temperatures = [self.ent_temp, self.rel_temp, self.ent_temp]
        for idx, (probs, temp) in enumerate(zip(hrt_probs, temperatures)):
            if status == 'neg':
                sample = self.get_negative_samples(probs)# , temp)
            else:
                sample = F.softmax(probs/temp, dim=-1)
                sample = sample.unsqueeze(0)
            embs = embeddings[idx%2].unsqueeze(0)
            result_embeddings.append(torch.matmul(sample, embs))
        return result_embeddings  # A 3 element list, with each elem having dims: 1(or n_corruptions) x batch_size x emb_dims

    def hole_loss(self, hrt_probs, entity_embeddings, rel_embeddings):
        h_emb, r_emb, t_emb = self.get_embeddings(hrt_probs, entity_embeddings, rel_embeddings)
        pos_scores = self.cconv_score(h_emb[0], t_emb[0], r_emb[0])
        neg_h_emb, neg_r_emb, neg_t_emb = self.get_embeddings(hrt_probs, entity_embeddings, rel_embeddings, status='neg')
        my_neg_scores1 = self.cconv_score_batch(neg_h_emb, t_emb, r_emb)
        my_neg_scores2 = self.cconv_score_batch(h_emb, neg_t_emb, r_emb)
        my_neg_scores = [my_neg_scores1[:, i] if np.random.uniform(0, 1) < 0.5 else my_neg_scores2[:, i] for i in range(self.n_corruptions)]
        my_neg_scores = torch.stack(my_neg_scores, dim=-1)
        scores = torch.cat([pos_scores.unsqueeze(1), torch.logsumexp(my_neg_scores, dim=1, keepdim=True)], dim=1)
        loss = F.cross_entropy(scores, torch.zeros_like(pos_scores, dtype=torch.long), reduction='mean')
        return loss

    def transe_loss(self, hrt_probs, entity_embeddings, rel_embeddings):
        h_emb, r_emb, t_emb = self.get_embeddings(hrt_probs, entity_embeddings, rel_embeddings)
        pos_scores = torch.norm(h_emb[0]+r_emb[0]-t_emb[0], p=2, dim=-1)   # dims: batch_size
        pos_scores = torch.neg(pos_scores)   # TransE calculates L2 distance; larger the distance, lesser the score.
        neg_h_emb, neg_r_emb, neg_t_emb = self.get_embeddings(hrt_probs, entity_embeddings, rel_embeddings, status='neg')
        my_neg_scores1 = torch.norm(neg_h_emb+r_emb-t_emb, p=2, dim=-1).T    # dims: batch_size x n_corruptions
        my_neg_scores2 = torch.norm(h_emb+r_emb-neg_t_emb, p=2, dim=-1).T    # dims: batch_size x n_corruptions
        my_neg_scores = [my_neg_scores1[:, i] if np.random.uniform(0, 1) < 0.5 else my_neg_scores2[:, i] for i in range(self.n_corruptions)]
        my_neg_scores = torch.stack(my_neg_scores, dim=-1)
        my_neg_scores = torch.neg(my_neg_scores)    # TransE calculates L2 distance; larger the distance, lesser the score.
        scores = torch.cat([pos_scores.unsqueeze(1), torch.logsumexp(my_neg_scores, dim=1, keepdim=True)], dim=1)
        loss = F.cross_entropy(scores, torch.zeros_like(pos_scores, dtype=torch.long), reduction='mean')
        return loss
