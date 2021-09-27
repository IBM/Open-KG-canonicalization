#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import logging
import numpy as np
import torch.nn as nn

from typing import Dict, Any
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import Word2VecKeyedVectors

logging.basicConfig(level=logging.INFO)

class EmbeddingsLayer(nn.Module):
    def __init__(self, mention2idx: Dict[str, int], input_dims: int,
                 pretrained_models: Dict[str, Any]):
        super(EmbeddingsLayer, self).__init__()
        if not mention2idx: raise ValueError
        self.input_dims = input_dims
        self.mention2idx = mention2idx
        self.embeddings = self.build_lookup_table(pretrained_models)

    def forward(self, batch):
        return self.embeddings(batch)  # Dims for batch: (batch_size,)

    def build_lookup_table(self, pretrained_models):
        input_embeddings = nn.Embedding(len(self.mention2idx), self.input_dims)
        if pretrained_models['input']:
            input_term_vectors = self.initialize_embeddings(pretrained_models['source'], pretrained_models['input'])
            input_embeddings.load_state_dict({'weight': torch.FloatTensor(input_term_vectors)})
        else:
            logging.info('No Pretrained MODELS found for input layer ... Doing Random Initializations.')
        return input_embeddings

    def initialize_embeddings(self, source: str, pretrained_model: Word2VecKeyedVectors):
        if not pretrained_model: raise NotImplementedError
        input_dims = pretrained_model.vector_size
        weights_matrix = np.zeros((len(self.mention2idx), input_dims))
        em, all_tokens = 0, 0
        for term, indx in self.mention2idx.items():
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
        logging.info(log_msg.format(em, all_tokens, len(self.mention2idx), pretrained_model.vector_size))
        return weights_matrix