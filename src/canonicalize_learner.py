#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import logging
import numpy as np
import torch.nn as nn

from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

from vae_gmm import VAE_GMM
from kge_util import KGEModels
from canonicalize_ds import BuildDataForCanonicalization

logging.basicConfig(level=logging.INFO)

class OpenKGCanonicalization(nn.Module):
    def __init__(self, data_dir, config_params_file, kge_algorithm, device):
        if config_params_file is None or not os.path.exists(config_params_file): raise FileNotFoundError
        super(OpenKGCanonicalization, self).__init__()
        self.kge_algorithm = kge_algorithm
        self.config_params = json.load(open(config_params_file, 'r'))
        if 'seed' in self.config_params:
            torch.manual_seed(self.config_params['seed'])
            np.random.seed(self.config_params['seed'])
        # ====== GET THE Datasets and Generators ============
        dataloader_params = self.get_train_dataloader_params()
        self.dataset = BuildDataForCanonicalization(data_dir)
        self.dataloader = DataLoader(self.dataset, **dataloader_params)
        logging.info('Sizes of the dataset:{0}'.format(len(self.dataset)))
        # ====== Instantiate the ENTITY and RELATION Vae ============
        self.train_stage = None
        models = self.load_models()
        self.reg_info = self.config_params['reg_info']
        self.n_ent_clusters = n_ent = self.config_params['clusters']['ent']
        self.n_rel_clusters = n_rel = self.config_params['clusters']['rel']
        self.initial_clusters = init_clusters = self.read_initial_clusters()
        self.ent_vae = VAE_GMM(self.config_params['dims'], n_ent, init_clusters['ent'], self.reg_info, models, self.dataset.ent2id, device)
        self.rel_vae = VAE_GMM(self.config_params['dims'], n_rel, init_clusters['rel'], self.reg_info, models, self.dataset.rel2id, device)
        # ================== Set Up the KBC Model.
        kge_util_params = self.config_params['kge_util_params']
        self.kge_model = KGEModels(temp=kge_util_params['temp'], n_corruptions=kge_util_params['n_corruptions'], n_ent_clusters=n_ent, n_rel_clusters=n_rel)

    def read_initial_clusters(self):
        init_clust_files = dict()
        work_dir = self.config_params['working_directory']
        for tag, init_file_tag in zip(['ent', 'rel'], ['init_ent_file', 'init_rel_file']):
            init_clusters_fname = os.path.join(work_dir, self.config_params[init_file_tag])
            if not os.path.exists(init_clusters_fname): raise FileNotFoundError('File not found: {0}'.format(init_clusters_fname))
            init_clust_files[tag] = init_clusters_fname
        return init_clust_files

    def get_train_dataloader_params(self):
         return dict(batch_size=self.config_params['batch_size']['train'],
                collate_fn=self.collate_for_train,
                shuffle=True, pin_memory=True)

    def load_models(self):
        model_dir = self.config_params['model_info']['dir']
        input_dims = self.config_params['dims']['input']
        latent_dims = self.config_params['dims']['latent']
        if model_dir is None or not os.path.exists(model_dir): raise FileNotFoundError
        model_source = self.config_params['model_info']['src']
        if model_source == 'GLOVE':
            input_fname = os.path.join(model_dir, 'glove.6B.{0}d_word2vec.txt'.format(input_dims))
            latent_fname = os.path.join(model_dir, 'glove.6B.{0}d_word2vec.txt'.format(latent_dims))
            if os.path.exists(input_fname): input_model = KeyedVectors.load_word2vec_format(input_fname, binary=False)
            else: input_model = None
            if os.path.exists(latent_fname): latent_model = KeyedVectors.load_word2vec_format(latent_fname, binary=False)
            else: latent_model = None
            return dict(source='GLOVE', input=input_model, latent=latent_model)
        else: raise NotImplementedError

    def edit_encoder_params(self, is_trainable):
        logging.info('{0}: Are ENCODER params trainable: {1}'.format(self.train_stage.lower(), is_trainable))
        self.ent_vae.enc_weights_mean.requires_grad_(requires_grad=is_trainable)
        self.ent_vae.enc_weights_log_sigma_sq.requires_grad_(requires_grad=is_trainable)
        for layer_ in self.ent_vae.enc_layers.nn_layers:
            for q in layer_.parameters():
                q.requires_grad = is_trainable
        self.rel_vae.enc_weights_mean.requires_grad_(requires_grad=is_trainable)
        self.rel_vae.enc_weights_log_sigma_sq.requires_grad_(requires_grad=is_trainable)
        for layer_ in self.rel_vae.enc_layers.nn_layers:
            for q in layer_.parameters():
                q.requires_grad = is_trainable

    def edit_decoder_params(self, is_trainable):
        logging.info('{0}: Are DECODER params trainable: {1}'.format(self.train_stage.lower(), is_trainable))
        self.ent_vae.dec_weights_mean.requires_grad_(requires_grad=is_trainable)
        self.ent_vae.dec_weights_log_sigma_sq.requires_grad_(requires_grad=is_trainable)
        for layer_ in self.ent_vae.dec_layers.nn_layers:
            for q in layer_.parameters():
                q.requires_grad = is_trainable
        self.rel_vae.dec_weights_mean.requires_grad_(requires_grad=is_trainable)
        self.rel_vae.dec_weights_log_sigma_sq.requires_grad_(requires_grad=is_trainable)
        for layer_ in self.rel_vae.dec_layers.nn_layers:
            for q in layer_.parameters():
                q.requires_grad = is_trainable

    def set_train_stage(self, stage_id):
        self.train_stage = stage_id

    @staticmethod
    def constraint_loss_mse(instance_pairs, word_embedding):
        e1, e2, scores = instance_pairs[:, 0], instance_pairs[:, 1], instance_pairs[:, 2]
        e1_emb = word_embedding(e1.long())
        e2_emb = word_embedding(e2.long()) # Dims: batch_size x n_input_dims
        dist_squared = torch.sum((e1_emb - e2_emb) ** 2, dim=-1)  # dims: (batch_size,)
        return torch.mean(scores * dist_squared)

    def forward(self, batch):
        hrt_triples, similar_ent_triples, similar_rel_triples = batch
        rel_idxs = hrt_triples[:, 1]
        ent_idxs = torch.cat((hrt_triples[:,0], hrt_triples[:,2]), dim=-1)
        e_latent_params, e_out_scores, e_clust_probs = self.ent_vae(ent_idxs, self.reg_info['mc_samples'])
        r_latent_params, r_out_scores, r_clust_probs = self.rel_vae(rel_idxs, self.reg_info['mc_samples'])
        if self.train_stage.lower() == 'stage1':
            e_vae_loss = self.ent_vae.stage_one_loss(ent_idxs, e_clust_probs)
            r_vae_loss = self.rel_vae.stage_one_loss(hrt_triples[:, 1], r_clust_probs)
            kbc_loss = 0.
        else:
            e_vae_loss = self.ent_vae.loss(ent_idxs, e_latent_params, e_out_scores, e_clust_probs, self.train_stage.lower())
            r_vae_loss = self.rel_vae.loss(rel_idxs, r_latent_params, r_out_scores, r_clust_probs, self.train_stage.lower())
            h_clust_probs = e_clust_probs[:e_clust_probs.shape[0]//2, :]
            t_clust_probs = e_clust_probs[e_clust_probs.shape[0]//2:, :]
            hrt_probs = [h_clust_probs, r_clust_probs, t_clust_probs]
            if self.kge_algorithm == 'HOLE':
                kbc_loss = self.kge_model.hole_loss(hrt_probs, self.ent_vae.cluster_means, self.rel_vae.cluster_means)
            elif self.kge_algorithm == 'TRANSE':
                kbc_loss = self.kge_model.transe_loss(hrt_probs, self.ent_vae.cluster_means, self.rel_vae.cluster_means)
            else: raise NotImplementedError
        vae_loss = e_vae_loss + r_vae_loss
        ent_constraint_loss = self.constraint_loss_mse(similar_ent_triples, self.ent_vae.word_embedding)
        rel_constraint_loss = self.constraint_loss_mse(similar_rel_triples, self.rel_vae.word_embedding)
        constraint_loss = ent_constraint_loss + rel_constraint_loss
        return vae_loss, kbc_loss, constraint_loss

    @staticmethod
    def collate_for_train(batch):
        trps = torch.LongTensor(list(map(lambda z: z[0], batch)))
        ent_si = torch.FloatTensor(list(map(lambda z: z[1], batch)))
        rel_si = torch.FloatTensor(list(map(lambda z: z[2], batch)))
        return trps, ent_si, rel_si
