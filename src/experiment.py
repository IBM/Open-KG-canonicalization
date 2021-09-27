#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import time
import torch
import logging
import argparse
import numpy as np
import torch.optim as optim

from copy import deepcopy
from typing import Dict, Set
from collections import defaultdict as ddict

from helper import invertDict
from metrics_copied_from_cesi import get_metrics
from canonicalize_learner import OpenKGCanonicalization

logging.basicConfig(level=logging.INFO)

class Experiment(object):
    def __init__(self, data_id, data_dir, config_params_file, kge_algorithm, serialize_results):
        self.data_id = data_id
        if self.data_id == 'reverb': self.eval_fn = lambda z: z.rsplit('|', 1)[0]
        else: self.eval_fn = lambda z: z
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.openKG_canonicalize = OpenKGCanonicalization(data_dir, config_params_file, kge_algorithm, self.device).to(device=self.device)
        self.serialize_results = serialize_results
        self.save_dir = self.build_save_dir()
        self.num_entities = len(self.openKG_canonicalize.dataset.ent2id)
        logging.info('Number of entities: {0}'.format(self.num_entities))

    def evaluate_clustering(self, ent2truelinks, ent2predlinks, eval_category):
        if not ent2truelinks: return dict()
        pred_items = set([x[0] for x in ent2predlinks.items()])
        gold_items = set([self.eval_fn(x[0]) for x in ent2truelinks.items()])
        if self.data_id == 'nell165' and len(pred_items) > len(gold_items):
            ent2predlinks = dict(filter(lambda z: z[0] in gold_items, ent2predlinks.items()))
            pred_items = set([x[0] for x in ent2predlinks.items()])
        if len(pred_items) != len(gold_items):
            difference = gold_items.symmetric_difference(pred_items)
            raise AssertionError('A difference of {0} items found. They are: {1}'.
                                 format(len(difference), list(difference)))
        mention2predlinks = dict([(mention, ent2predlinks[self.eval_fn(mention)]) for mention in ent2truelinks])
        try:
            metrics = get_metrics(
                mention2predlinks,
                invertDict(mention2predlinks),
                ent2truelinks,
                invertDict(ent2truelinks)
            )
        except ValueError as e:
            logging.error('CESI metrics ARE NOT applicable ...\nOriginal error: {0}'.format(str(e)))
            raise
        log_msg = '[{0}]: GOLD/PRED items: {1}/{2}, GOLD/PRED Clusters: {3}/{4}, Macro F1: {5}, Micro F1: {6}, Pair F1: {7}'
        logging.info(
            log_msg.format(
                eval_category, len(gold_items), len(pred_items),
                len(invertDict(ent2truelinks)), len(invertDict(ent2predlinks)),
                metrics['macro_f1'], metrics['micro_f1'], metrics['pair_f1']
            )
        )
        return metrics

    def build_save_dir(self):
        out_idx = 1
        while True:
            out_folder = os.path.join(self.openKG_canonicalize.config_params['working_directory'], 'output_OpenKG_Canonicalize_{0}'.format(out_idx))
            if not os.path.exists(out_folder): break
            out_idx += 1
        if self.serialize_results:
            os.makedirs(out_folder)
            logging.info('Out RESULTS stored at: {0}\n'.format(out_folder))
        return out_folder

    def run(self):
        # For printing out predicted NP and RP Clusters.
        out_ent_fname = os.path.join(self.save_dir, 'vaegmm_joint_npclust.txt')
        out_rel_fname = os.path.join(self.save_dir, 'vaegmm_joint_rpclust.txt')
        for train_stage_id in self.openKG_canonicalize.config_params['train_stages']:
            self.openKG_canonicalize.set_train_stage(train_stage_id)
            optimizer = optim.Adam(self.openKG_canonicalize.parameters(), lr=self.openKG_canonicalize.config_params['learning_rate'][train_stage_id])
            if train_stage_id.lower() == 'stage1':
                self.openKG_canonicalize.edit_encoder_params(is_trainable=True)
                self.openKG_canonicalize.edit_decoder_params(is_trainable=False)
            elif train_stage_id.lower() == 'stage2':
                self.openKG_canonicalize.edit_encoder_params(is_trainable=False)
                self.openKG_canonicalize.edit_decoder_params(is_trainable=True)
            else: raise ValueError()
            metric, clustID2entList = self.train_stage(train_stage_id, optimizer)
            if self.serialize_results:
                # kge_util:hole_model/transe_model does NOT have any params to serialize.
                states = {
                    'ent_vae': self.openKG_canonicalize.ent_vae.state_dict(),
                    'rel_vae': self.openKG_canonicalize.rel_vae.state_dict()
                }
                torch.save(states, os.path.join(self.save_dir, 'best_state_{0}'.format(train_stage_id)))
                if train_stage_id == 'stage2':
                    self.write_cluster_results(out_ent_fname, clustID2entList)
        self.cluster_relations_and_serialize(out_rel_fname)
        if self.data_id == 'reverb':
            logging.info('Evaluating on all NP Mentions for ReVerb45K ... ')
            self.evaluate_for_clustering_entities('stage2', False)
        logging.info('Program terminated successfully.')

    def write_cluster_results(self, out_fname: str, clustID2itemList: Dict[str, Set[str]]):
        if os.path.exists(out_fname): raise FileExistsError
        with open(out_fname, 'w') as f:
            params_str = json.dumps(self.openKG_canonicalize.config_params)
            f.write(params_str+'\n\n')
            f.write('Number of Clusters: {0}\n'.format(len(clustID2itemList)))
            for clustID, item_set in clustID2itemList.items():
                curr_cluster_item_str = '\t'.join(map(str, item_set))
                count = len(item_set)
                f.write(str(clustID)+'\t'+str(count)+'\t'+curr_cluster_item_str+'\n')
        logging.info('Clusters written to {0}'.format(out_fname))

    def train_stage(self, stage_id, optimizer):
        max_epochs = self.openKG_canonicalize.config_params['n_epochs'][stage_id]
        loss_weights = self.openKG_canonicalize.config_params['loss_weights']
        eval_per = self.openKG_canonicalize.config_params['eval_per']
        for epoch in range(max_epochs):
            n_batches = 0
            epoch_loss = 0.
            start_time = time.time()
            total_vae, total_kbc, side_info = 0., 0., 0.
            for i, (instances, similar_ents, similar_rels) in enumerate(self.openKG_canonicalize.dataloader):
                n_batches += 1
                optimizer.zero_grad()
                cuda_batch = instances.to(device=self.device)
                cuda_similar_ents = similar_ents.to(device=self.device)
                cuda_similar_rels = similar_rels.to(device=self.device)
                vae_loss, kbc_loss, constraint_loss = self.openKG_canonicalize([cuda_batch, cuda_similar_ents, cuda_similar_rels])  # Run through one pass.
                logging.info('[Epoch: %d, BatchID: %5d] VAE Loss: %.5f, KBC Loss: %.5f, Constraint Loss: %.5f' % (epoch+1, i+1, vae_loss, kbc_loss, constraint_loss))
                loss = loss_weights['vae']*vae_loss + loss_weights['kbc']*kbc_loss + loss_weights['side_info']*constraint_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                total_vae += loss_weights['vae'] * vae_loss
                total_kbc += loss_weights['kbc'] * kbc_loss
                side_info += loss_weights['side_info'] * constraint_loss
            duration = time.time() - start_time
            logging_msg = '[%s, Epoch: %d, Time: %.3f sec] Vae LOSS: %.5f, KBC LOSS: %.5f, Constraint LOSS: %.5f, Overall MEAN loss: %.5f'
            logging.info(logging_msg % (stage_id, epoch+1, duration, total_vae/n_batches, total_kbc/n_batches, side_info/n_batches, epoch_loss/n_batches))
            if (1 + epoch) % eval_per[stage_id] == 0:
                with torch.no_grad():
                    metrics, clustID2entList = self.evaluate_for_clustering_entities(stage_id, self.data_id == 'reverb')
        return metrics, clustID2entList

    def evaluate_for_clustering_entities(self, stage_id: str, use_sub_only: bool) -> Dict[str, Set[str]]:
        """ Generates a cluster ID for all the entities. """
        ent2predlinks = ddict(set)
        curr_id2ent = self.openKG_canonicalize.dataset.id2ent
        num_entities = len(self.openKG_canonicalize.dataset.ent2id)
        results_identifier = '{0}-Joint VAE-GMM & KBC'.format(stage_id)
        batch_size = self.openKG_canonicalize.config_params['batch_size']['eval']
        n_batches = num_entities // batch_size
        if num_entities % batch_size != 0: n_batches += 1
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((1+i) * batch_size, num_entities)
            ent_idxes = np.arange(start_idx, end_idx)
            ent_idxs_batch = torch.LongTensor(ent_idxes).to(device=self.device)
            ent_clust_probs = self.openKG_canonicalize.ent_vae(ent_idxs_batch, self.openKG_canonicalize.reg_info['mc_samples'])[2]  # dims: eval_batch_size x n_ent_clusters
            ent_clust_probs = ent_clust_probs.cpu().data.numpy()
            clust_ids = np.argmax(ent_clust_probs, axis=-1)
            for ent_idx, cl_id in zip(ent_idxes, clust_ids):
                ent2predlinks[curr_id2ent[int(ent_idx)]].add(cl_id)
        allent2predlinks = deepcopy(ent2predlinks)
        if use_sub_only:
            all_head_mentions = set([h for h, r, t in self.openKG_canonicalize.dataset.triples])
            ent2predlinks = dict(filter(lambda z: z[0] in all_head_mentions, ent2predlinks.items()))
            results_identifier += '-Head ENT Mentions'
        else:
            results_identifier += '-All ENT Mentions'
        if self.data_id == 'reverb' and not use_sub_only:
            # Evaluate for all NP Mentions for ReVerb45K.
            gfile_all_entities = os.path.join(self.openKG_canonicalize.dataset.data_dir, 'gold_npclust_all.txt')
            ent2truelinks = self.openKG_canonicalize.dataset.read_gold_clust(gfile_all_entities)
        else:
            ent2truelinks = self.openKG_canonicalize.dataset.ent2truelinks
        metrics = self.evaluate_clustering(ent2truelinks, ent2predlinks, results_identifier)
        return metrics, invertDict(allent2predlinks)

    def cluster_relations_and_serialize(self, out_fname):
      """ Use our trained model to assign each Relation Phrase to a cluster.
      Serialize the clusters onto a output file"""
      rel2predlinks = ddict(set)
      curr_id2rel = self.openKG_canonicalize.dataset.id2rel
      num_relations = len(self.openKG_canonicalize.dataset.rel2id)
      batch_size = self.openKG_canonicalize.config_params['batch_size']['eval']
      n_batches = num_relations // batch_size
      if num_relations % batch_size != 0: n_batches += 1
      for i in range(n_batches):
          start = i * batch_size
          end = min(num_relations, (1+i) * batch_size)
          rel_idxes = np.arange(start, end)
          rel_idxs_batch = torch.LongTensor(rel_idxes).to(device=self.device)
          rel_clust_probs = self.openKG_canonicalize.rel_vae(rel_idxs_batch, self.openKG_canonicalize.reg_info['mc_samples'])[2]
          rel_clust_probs = rel_clust_probs.cpu().data.numpy()
          clust_ids = np.argmax(rel_clust_probs, axis=-1)
          for rel_idx, cl_id in zip(rel_idxes, clust_ids):
              rel2predlinks[curr_id2rel[int(rel_idx)]].add(cl_id)
      self.write_cluster_results(out_fname, invertDict(rel2predlinks))

def main():
    parser = argparse.ArgumentParser(description='A joint VAE-GMM+KBC Model for Canonicalization AND OpenKG Link Prediction.')
    parser.add_argument('--serialize_results', action='store_true', help='Serialize the model', default=False)
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Data directory containing the input')
    parser.add_argument('--config_file', dest='config_file', type=str, help='Configuration file containing hyper-parameters')
    parser.add_argument('--data_id', dest='data_id', choices=['nell165', 'reverb', 'lenovo'], help='Dataset to run VAE_GMM+KBC Canonicalization on.')
    parser.add_argument('--kge_algorithm', dest='kge_algorithm', type=str, choices=['TRANSE', 'HOLE'], default='HOLE',
                        help='Choice of Knowledge Graph Embedding Algorithm')
    args = parser.parse_args()
    logging.info('===== CONFIG PARAMS =====')
    logging.info(args)
    params = json.load(open(args.config_file, 'r'))
    logging.info(json.dumps(params, indent=4))
    logging.info('===============')
    experiment = Experiment(args.data_id, args.data_dir, args.config_file, args.kge_algorithm, args.serialize_results)
    experiment.run()

if __name__ == '__main__':
    main()