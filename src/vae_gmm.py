#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import pickle
import logging
import numpy as np
import torch.nn as nn

from nltk.tokenize import word_tokenize
from torch.nn.init import xavier_uniform_

from typing import Dict

logging.basicConfig(level=logging.INFO)

class VAE_GMM(nn.Module):
    """ Implemention based on the following paper:
    Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering. IJCAI 2017.
    """
    def __init__(self, dims, n_clusters, init_clust_assignment_file, weights, pretrained_models, term2idx, device):
        super(VAE_GMM, self).__init__()
        self.device = device
        self.weights = weights
        self.n_clusters = n_clusters
        self.input_dims = dims['input']
        self.hidden_dims = dims['latent']
        # Initialize Cluster MEANS, Cluster LOG SIGMA Squared AND Cluster ASSIGNMENT PRIOR.
        logging.info('Using labels from a serialized HAC clustering to initialize GMMs')
        self.log_pi_c = None
        self.cluster_means = None
        self.cluster_log_sigma_sq = None
        self.init_clustering_labels = None
        with open(init_clust_assignment_file, 'rb') as f:
            labels = pickle.load(f)
        latent_embs = self.__initialize_embeddings(pretrained_models['latent'], pretrained_models['source'], term2idx)
        self.__initialize_gmm_parameters(latent_embs, labels)
        self.word_embedding = self.__initialize_input_embeddings(term2idx, pretrained_models)

        # Build the VAE Model.
        self.enc_layers = NeuralNetwork(self.input_dims, dims['interim'], self.hidden_dims)
        self.dec_layers = NeuralNetwork(self.hidden_dims, dims['interim'], self.input_dims)
        # Parameters that build \tilde{\mu} AND \log \tilde{\sigma}^2 for the generation/recognition networks.
        self.enc_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.hidden_dims, self.hidden_dims))), requires_grad=True)
        self.dec_weights_mean = nn.Parameter(xavier_uniform_(torch.empty(size=(self.input_dims, self.input_dims))), requires_grad=True)
        self.enc_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(self.hidden_dims, self.hidden_dims))), requires_grad=True)
        self.dec_weights_log_sigma_sq = nn.Parameter(xavier_uniform_(torch.empty(size=(self.input_dims, self.input_dims))), requires_grad=True)

    def __initialize_gmm_parameters(self, X, label):
        """ Initialize Gaussian Mixture model params using,
            1. Initial clustering results, and
            2. Pretrained embeddings in latent dimensions.
        X has dims: n_instances x self.hidden_dims
        """
        n_instances = X.shape[0]
        assignments = np.zeros((n_instances, self.n_clusters))
        assignments[np.arange(n_instances), label] = 1
        # Adds non zero EPS to clusters having 0 members.
        clust_size = assignments.sum(axis=0) + 10*np.finfo(assignments.dtype).eps
        # For each cluster, compute centroid of the cluster participants.
        means = np.dot(assignments.T, X) / clust_size[:, np.newaxis]
        # Use initial cluster assignment to build a prior.
        log_assign_prior = np.log(clust_size) - np.log(clust_size.sum())
        # Calculate the diagonal covariances.
        avg_means_sq = means ** 2
        avg_X_sq = np.dot(assignments.T, X * X) / clust_size[:, np.newaxis]
        avg_X_means = means * np.dot(assignments.T, X)/clust_size[:, np.newaxis]
        diag_covariances = avg_X_sq - 2*avg_X_means + avg_means_sq + 1e-6
        # Populate the instance variables.
        self.init_clustering_labels = torch.LongTensor(label).to(device=self.device)
        self.cluster_means = nn.Parameter(torch.from_numpy(means), requires_grad=True)
        self.cluster_log_sigma_sq = nn.Parameter(torch.from_numpy(diag_covariances), requires_grad=True)
        self.log_pi_c = nn.Parameter(torch.from_numpy(log_assign_prior), requires_grad=True)

    def __initialize_input_embeddings(self, term2idx, pretrained_models):
        """ Build an embedding lookup Table.
        Use pretrained_models if available, else initialize randomly. """
        input_embeddings = nn.Embedding(len(term2idx), self.input_dims)
        if pretrained_models['input']:
            input_term_vectors = self.__initialize_embeddings(pretrained_models['input'], pretrained_models['source'], term2idx)
            input_embeddings.load_state_dict({'weight': torch.FloatTensor(input_term_vectors)})
        else:
            logging.info('No Pretrained MODELS found for input layer ... Doing Random Initializations.')
        return input_embeddings

    def __initialize_embeddings(self, pretrained_model, source: str, term2idx: Dict[str, int]):
        if not pretrained_model: raise NotImplementedError
        input_dims = pretrained_model.vector_size
        weights_matrix = np.zeros((len(term2idx), input_dims))
        em, all_tokens = 0, 0
        for term, indx in term2idx.items():
            term = str(term)
            is_embedding_avail = False
            if source == 'GLOVE':
                tterm = term.replace(' ', '_')
                if tterm in pretrained_model:
                    em += 1
                    is_embedding_avail = True
                    weights_matrix[indx] = pretrained_model[tterm]
            else:
                raise NotImplementedError
            if not is_embedding_avail:
                tokens_list = word_tokenize(term)
                vector = np.zeros(input_dims, np.float32)
                if all(map(lambda z: z in pretrained_model, tokens_list)):
                    all_tokens += 1
                for tok in tokens_list:
                    if tok in pretrained_model: current_vec = pretrained_model[tok]
                    else: current_vec = np.random.randn(input_dims)
                    vector += (current_vec/np.linalg.norm(current_vec))
                vector /= len(tokens_list)
                weights_matrix[indx] = vector
        log_msg = '[Dims: {3}]: Pretrained embeddings found: EM: {0}, all tokens:{1} out of {2} terms.'
        logging.info(log_msg.format(em, all_tokens, len(term2idx), pretrained_model.vector_size))
        return weights_matrix

    def encode(self, batch):
        """ Run the batch through the encoder neural network. """
        z1 = self.enc_layers(batch)
        z_mean = torch.matmul(z1, self.enc_weights_mean)
        z_log_sigma_sq = torch.matmul(z1, self.enc_weights_log_sigma_sq.float())
        return z_mean, z_log_sigma_sq

    def decode(self, batch):
        """ Run through the decoder neural network. """
        out_mean_list = list()
        out_log_sigma_sq_list = list()
        for matrix in batch:  # dims for batch: mc_samples x batch_size x hidden_dims
            out = self.dec_layers(matrix)
            out_mean = torch.matmul(out, self.dec_weights_mean)
            out_log_sigma_sq = torch.matmul(out, self.dec_weights_log_sigma_sq.float())
            out_mean_list.append(out_mean)
            out_log_sigma_sq_list.append(out_log_sigma_sq)
        out_mean = torch.stack(out_mean_list)
        out_log_sigma_sq = torch.stack(out_log_sigma_sq_list)
        return out_mean, out_log_sigma_sq

    def prob_c_given_x(self, z_sampled):
        """ Implements q(c|x) = E_{q(z|x)} p(c|z), or in other words,
         computes cluster assignments that maximizes ELBO. """
        x = z_sampled.unsqueeze(2)                                 # mc_samples x batch_size x 1 x hidden_dims
        mean = self.cluster_means.unsqueeze(0).unsqueeze(0)        # 1 x 1 x n_clusters x hidden_dims
        cov_inv = 1./torch.exp(self.cluster_log_sigma_sq)
        cov_inv = cov_inv.unsqueeze(0).unsqueeze(0)                # 1 x 1 x n_clusters x hidden_dims
        log_exponent = -0.5 * torch.sum((x-mean) * cov_inv * (x-mean), dim=-1)      # mc_samples x batch_size x n_clusters
        log_det = -0.5 * torch.sum(self.cluster_log_sigma_sq, dim=-1, keepdim=True)
        log_det = torch.transpose(log_det, 0, 1).unsqueeze(0)
        log_p = log_exponent + log_det    # mc_samples x batch_size x n_clusters
        log_numerator = log_p + self.log_pi_c.unsqueeze(0).unsqueeze(0)  # mc_samples x batch_size x n_clusters
        log_normalizer = torch.logsumexp(log_numerator, dim=-1, keepdim=True)   # Numerically Stable option: mc_samples x batch_size x 1
        q_c_given_x = torch.exp(log_numerator - log_normalizer)    # mc_samples x batch_size x n_clusters
        return torch.mean(q_c_given_x, dim=0)

    def forward(self, batch, mc_samples):
        """ batch: A LongTensor of idxs of dimensions: (batch_size,) """
        self.log_pi_c.data -= torch.logsumexp(self.log_pi_c.data, dim=-1)
        batch_emb = self.word_embedding(batch)
        z_mean, z_log_sigma_sq = self.encode(batch_emb)
        # Use the Reparametrization trick. eps ~ N(0,1); dims of eps/z_sampled = mc_samples x batch_size x hidden_dims
        eps = torch.randn(mc_samples, batch.size(0), self.hidden_dims).to(device=self.device)
        z_sampled = torch.unsqueeze(z_mean, 0) + torch.unsqueeze(torch.sqrt(torch.exp(z_log_sigma_sq)), 0) * eps
        q_c_given_x = self.prob_c_given_x(z_sampled)
        out_mean, out_log_sigma_sq = self.decode(z_sampled)  # z_sampled: mc_samples x batch_size x hidden_dims
        latent_params = (z_mean, z_log_sigma_sq)
        out_params = (out_mean, out_log_sigma_sq)
        return latent_params, out_params, q_c_given_x

    def compute_regularizer_loss(self, p_val, train_stage):
        if train_stage == 'stage1':
            enc_mean_reg = torch.norm(self.enc_weights_mean, p=p_val)
            enc_log_sigma_sq_reg = torch.norm(self.enc_weights_log_sigma_sq, p=p_val)
            enc_regularizer = self.weights[train_stage+'_reg'] * (enc_mean_reg + enc_log_sigma_sq_reg)
            return enc_regularizer
        elif train_stage == 'stage2':
            dec_mean_reg = torch.norm(self.dec_weights_mean, p=p_val)
            dec_log_sigma_sq_reg = torch.norm(self.dec_weights_log_sigma_sq, p=p_val)
            dec_regularizer = self.weights[train_stage+'_reg'] * (dec_mean_reg + dec_log_sigma_sq_reg)
            return dec_regularizer
        else:
            raise NotImplementedError

    @staticmethod
    def reconstruction_loss(out_params, inp):
        """ Returns a vector of reconstruction loss for all instances within the batch. """
        reconstr_mean, reconstr_log_sigma_sq = out_params # dims: mc_samples x batch_size x input_dims
        diff = inp.unsqueeze(0) - reconstr_mean           # inp: batch_size x input_dims
        reconstr_var_inv = 1./torch.exp(reconstr_log_sigma_sq)
        exp_loss = torch.sum(diff * reconstr_var_inv * diff, dim=-1)                                   # mc_samples x batch_size
        # log_det_loss = torch.sum(reconstr_log_sigma_sq, dim=-1)                                      # mc_samples x batch_size
        # overall_loss = 0.5 * (exp_loss + log_det_loss + np.log(2 * np.pi) * self.input_dims)         # mc_samples x batch_size
        overall_loss = 0.5 * exp_loss
        return torch.mean(overall_loss, dim=0)  # dims: (batch_size,)

    def kld_loss(self, latent_params, q_c_given_x):
        """ Computes the KL Divergence between the variational posterior and the true posterior. """
        z_mean, z_log_sigma_sq = latent_params
        loss1 = -0.5 * torch.sum(1.+z_log_sigma_sq, dim=-1)     # -0.5*\sum_{j=1}^J (1+\log \tilde{\sigma_j}^2)
        q_c_given_x = torch.clamp(q_c_given_x, min=1e-20, max=1.0)
        interim = self.log_pi_c.unsqueeze(0) - torch.log(q_c_given_x)
        loss2 = -1. * torch.sum(q_c_given_x * interim, dim=-1)  # -\sum_{c=1}^K \gamma_c \log \frac {\pi_c}{\gamma_c}
        diff = (z_mean.unsqueeze(1) - self.cluster_means.unsqueeze(0))**2
        t2 = 1./torch.exp(self.cluster_log_sigma_sq).unsqueeze(0)
        s1 = diff * t2 # s1 = \frac{\(\tilde{\mu_j} - \mu^c_j\)^2}{(\sigma^c_j)^2} dims: batch_size x n_clusters x hidden_dims
        s2 = t2 * torch.exp(z_log_sigma_sq).unsqueeze(1)  # s2 = \frac{\tilde{\sigma_j}^2}{(\sigma^c_j)^2}; Same dims as s1.
        s3 = self.cluster_log_sigma_sq.unsqueeze(0)       # s3 = \log (\sigma^c_j)^2
        overall_sum = torch.sum(s1+s2+s3, dim=-1)         # overall_sum dims: batch_size x n_clusters
        loss3 = 0.5*torch.sum(q_c_given_x * overall_sum, dim=-1)  # s = 0.5*\sum_{c=1}^K \gamma_c \sum_{j=1}^J (s1+s2+s3)
        return loss1+loss2+loss3   # dims: (batch_size,)

    def loss(self, batch, latent_params, out_params, q_c_given_x, stage_id):
        """ Computes the Evidence Lower Bound loss (or ELBO loss).
        batch: A LongTensor of idxs of dimensions: (batch_size,) """
        batch_emb = self.word_embedding(batch)
        reconstruction_loss = self.reconstruction_loss(out_params, batch_emb)
        other_loss = self.kld_loss(latent_params, q_c_given_x)
        if torch.isnan(reconstruction_loss).any() or torch.isnan(other_loss).any():
            logging.error('Nans Found, Troublesome batch: {0}'.format(batch))
            raise ValueError
        data_loss = torch.mean(reconstruction_loss + other_loss)
        regularizer_loss = self.compute_regularizer_loss(p_val=1, train_stage=stage_id)
        return data_loss + regularizer_loss

    def stage_one_loss(self, batch, q_c_given_x):
        """ Loss function used to train the Encoder while keeping Decoder fixed.
        batch: A LongTensor of idxs of dimensions: (batch_size,)
        q_c_given_x: A FloatTensor of probabilities of dims: (batch_size, n_clusters) """
        loss = nn.NLLLoss(reduction='mean')
        true_labels = self.init_clustering_labels[batch]
        if torch.isnan(torch.log(q_c_given_x)).any():
            raise ValueError
        pretrain_loss = loss(torch.log(q_c_given_x), true_labels)
        reg_loss = self.compute_regularizer_loss(p_val=1, train_stage='stage1')
        return pretrain_loss + reg_loss

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, interim_size, output_size):
        super(NeuralNetwork, self).__init__()
        if type(interim_size) != list:
            raise ValueError('Interim_size should be a non-empty list.')
        modules = list()
        self.nn_layers = list()
        if not interim_size:
            self.nn_layers.append(nn.Linear(input_size, output_size))
        else:
            self.nn_layers.append(nn.Linear(input_size, interim_size[0]))
            for i in range(len(interim_size)-1):
                self.nn_layers.append(nn.Linear(interim_size[i], interim_size[i+1]))
            self.nn_layers.append(nn.Linear(interim_size[-1], output_size))
        for l in self.nn_layers:
            modules.extend([l, nn.Tanh()])
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def regularizer(self, p):
        return sum([torch.norm(x.weight, p=p) for x in self.nn_layers])
