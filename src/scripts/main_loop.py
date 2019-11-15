# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import collections
import datetime
import logging
import os
import sys
import time
from typing import Dict

import mlflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient

from cupy_utils import *
from embeddings import load_embeddings, embeddings_normalization, length_normalize
from src.domain.matrix_operations import whitening_transformation, dropout
from src.initialization import get_seed_dictionary_indices, init_computing_engine
from src.utils import compute_matrix_size
from src.utils import topk_mean
from src.validations import whitening_arguments_validation

BATCH_SIZE = 500


def is_same_configuration(config: Dict, config_filter: Dict):
    for key, value in config_filter.items():
        if config.get(key) != value:
            logging.info("{} is different than {} for {}".format(config.get(key), value, key))
            return False
    return True


def run_experiment(_config):
    logging.info(_config)
    mlflow.log_params(_config)
    mlflow.log_metric('test', 0.9)

    whitening_arguments_validation(_config)

    if _config['test']:
        source_language = _config['source_language'] + '_slim'
        target_language = _config['target_language'] + '_slim'
    else:
        source_language = _config['source_language']
        target_language = _config['target_language']

    test_dictionary = './data/dictionaries/{}-{}.test.txt'.format(_config['source_language'],
                                                                  _config['target_language'])

    compute_engine = init_computing_engine(_config['cuda'], _config['seed'])
    dtype = _config['precision']

    src_words, x = load_embeddings(_config['embeddings_path'], source_language, _config['encoding'], dtype)
    trg_vocab, z = load_embeddings(_config['embeddings_path'], target_language, _config['encoding'], dtype)

    x = compute_engine.send_to_device(x)
    z = compute_engine.send_to_device(z)

    embeddings_normalization(x, z, normalization_method=_config['normalize'])

    # Allocate memory
    logging.info("Allocating memory")
    xw = compute_engine.engine.empty_like(x)
    zw = compute_engine.engine.empty_like(z)

    src_size, trg_size = compute_matrix_size(x, z, _config['vocabulary_cutoff'])
    simfwd = compute_engine.engine.empty((_config['batch_size'], trg_size), dtype=dtype)
    simbwd = compute_engine.engine.empty((_config['batch_size'], src_size), dtype=dtype)

    best_sim_forward = compute_engine.engine.full(src_size, -100, dtype=dtype)
    src_indices_forward = compute_engine.engine.arange(src_size)
    trg_indices_forward = compute_engine.engine.zeros(src_size, dtype=int)
    best_sim_backward = compute_engine.engine.full(trg_size, -100, dtype=dtype)
    src_indices_backward = compute_engine.engine.zeros(trg_size, dtype=int)
    trg_indices_backward = compute_engine.engine.arange(trg_size)
    knn_sim_fwd = compute_engine.engine.zeros(src_size, dtype=dtype)
    knn_sim_bwd = compute_engine.engine.zeros(trg_size, dtype=dtype)

    # Training loop
    logging.info("Beginning training loop")
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = _config['stochastic_initial']
    t = time.time()
    end = not _config['self_learning']
    src_indices, trg_indices = get_seed_dictionary_indices(_config['seed_dictionary_method'], compute_engine.engine,
                                                           src_words, trg_vocab, x, z, _config)
    while True:

        logging.info("Iteration number {}".format(it))
        # Increase the keep probability if we have not improve in _config['stochastic_interval iterations
        logging.info("Keep prob {}".format(keep_prob))
        if it - last_improvement > _config['stochastic_interval']:
            if keep_prob >= 1.0:
                logging.info("Training will end...")
                end = True
            keep_prob = min(1.0, _config['stochastic_multiplier'] * keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if _config['orthogonal'] or not end:  # orthogonal mapping
            u, s, vt = compute_engine.engine.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        elif _config['unconstrained']:  # unconstrained mapping
            x_pseudoinv = compute_engine.engine.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping

            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            if _config['whiten']:
                wx1 = whitening_transformation(xw[src_indices], compute_engine=compute_engine.engine)
                wz1 = whitening_transformation(zw[trg_indices], compute_engine=compute_engine.engine)
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = compute_engine.engine.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s**_config['reweight']
            zw *= s**_config['reweight']

            # STEP 4: De-whitening
            if _config['src_dewhiten'] == 'src':
                xw = xw.dot(wx2.T.dot(compute_engine.engine.linalg.inv(wx1)).dot(wx2))
            elif _config['src_dewhiten'] == 'trg':
                xw = xw.dot(wz2.T.dot(compute_engine.engine.linalg.inv(wz1)).dot(wz2))
            if _config['trg_dewhiten'] == 'src':
                zw = zw.dot(wx2.T.dot(compute_engine.engine.linalg.inv(wx1)).dot(wx2))
            elif _config['trg_dewhiten'] == 'trg':
                zw = zw.dot(wz2.T.dot(compute_engine.engine.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if _config['dim_reduction'] > 0:
                xw = xw[:, :_config['dim_reduction']]
                zw = zw[:, :_config['dim_reduction']]

        # Self-learning
        # Update the training dictionary
        if end:
            break
        else:
            if _config['direction'] in ('forward', 'union'):
                if _config['csls'] > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=_config['csls'], inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                    simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j - i], 1 - keep_prob,
                            compute_engine=compute_engine).argmax(axis=1, out=trg_indices_forward[i:j])
            if _config['direction'] in ('backward', 'union'):
                if _config['csls'] > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=_config['csls'], inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                    simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j - i], 1 - keep_prob,
                            compute_engine=compute_engine).argmax(axis=1, out=src_indices_backward[i:j])
            if _config['direction'] == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif _config['direction'] == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif _config['direction'] == 'union':
                src_indices = compute_engine.engine.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = compute_engine.engine.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if _config['direction'] == 'forward':
                objective = compute_engine.engine.mean(best_sim_forward).tolist()
            elif _config['direction'] == 'backward':
                objective = compute_engine.engine.mean(best_sim_backward).tolist()
            elif _config['direction'] == 'union':
                objective = (compute_engine.engine.mean(best_sim_forward) +
                             compute_engine.engine.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= _config['threshold']:
                last_improvement = it
                best_objective = objective

            # Logging
            duration = time.time() - t
            logging.info('ITERATION {0} ({1:.2f}s)'.format(it, duration))
            logging.info('\t- Objective:        {0:9.4f}%'.format(100 * objective))
            logging.info('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob))

        t = time.time()
        it += 1

    x = xw
    z = zw

    if _config['cuda'] is True:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        compute_engine.engine = get_cupy()
        x = compute_engine.engine.asarray(x)
        z = compute_engine.engine.asarray(z)
    else:
        compute_engine.engine = np
    compute_engine.engine.random.seed(_config['seed'])

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not _config['dot']:
        length_normalize(x)
        length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_vocab)}

    # Read dictionary and compute coverage
    f = open(test_dictionary, encoding=_config['encoding'], errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # Find translations
    translation = collections.defaultdict(int)
    if _config['retrieval'] == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]
    elif _config['retrieval'] == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k - i, l]
                    sim = sims[k - i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif _config['retrieval'] == 'invsoftmax':  # Inverted softmax
        sample = compute_engine.engine.arange(
            x.shape[0]) if _config['inv_sample'] is None else compute_engine.engine.random.randint(
                0, x.shape[0], _config['inv_sample'])
        partition = compute_engine.engine.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += compute_engine.engine.exp(_config['inv_temperature'] * z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = compute_engine.engine.exp(_config['inv_temperature'] * x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]
    elif _config['retrieval'] == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = compute_engine.engine.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=_config['csls'], inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2 * x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]

    # Compute accuracy
    accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
    mlflow.log_metric('coverage', coverage)
    mlflow.log_metric('accuracy', accuracy)
    logging.info('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))


def override_configs(source_language, target_language, i, configs):
    seed = configs['seed'] if configs['supercomputer'] else i
    configs.update({
        'iteration': i,
        'source_language': source_language,
        'target_language': target_language,
        'seed': seed
    })
    return configs


def create_filter(source_language, target_language, configs):
    filter = {}
    filter.update(configs)
    filter['source_language'] = source_language
    filter['target_language'] = target_language
    del filter['seed']
    del filter['cuda']
    del filter['normalize']
    del filter['iteration']
    del filter['num_runs']
    return filter


def retrieve_stats(runs):
    accuracies = list()
    coverages = list()
    times = list()
    for run in runs:
        if 'accuracy' in run.data.metrics:
            minutes = ((run.info.end_time - run.info.start_time) // 60 // 60) % 60
            times.append(minutes)
            accuracies.append(run.data.metrics['accuracy'])
            coverages.append(run.data.metrics['coverage'])
    return accuracies, coverages, times


class MlFlowHandler(logging.FileHandler):
    def emit(self, record):
        super(MlFlowHandler, self).emit(record)
        mlflow.log_artifact(self.baseFilename)


def get_mlflow_logging_handler(path_to_log_directory, log_level, formatter):
    log_filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.log'
    fh = MlFlowHandler(filename=os.path.join(path_to_log_directory, log_filename))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    return fh


def run_main(configs):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    path_to_log_directory = './output/logs'
    mlflow_logging_handler = get_mlflow_logging_handler(path_to_log_directory, logging.INFO, formatter)
    logger.addHandler(mlflow_logging_handler)

    mlflow.set_experiment(configs['experiment_name'])  # Create the experiment if it did not already existed
    mlflow_client = MlflowClient(tracking_uri=configs['mlflow_output_uri'])
    logging.info(mlflow_client.list_experiments())
    mlflow_experiment = mlflow_client.get_experiment_by_name(configs['experiment_name'])
    if not mlflow_experiment:
        mlflow_client.create_experiment(configs['experiment_name'])
        mlflow_experiment = mlflow_client.get_experiment_by_name(configs['experiment_name'])
    os.makedirs('{}/mapped_embeddings'.format(configs['embedding_output_uri']), exist_ok=True)

    with mlflow.start_run(experiment_id=mlflow_experiment.experiment_id):
        try:
            run_experiment(configs)
        except KeyboardInterrupt:
            logging.warning("Run exited.")

    logger.removeHandler(mlflow_logging_handler)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    options = argument_parser.parse_args()
    configs = vars(options)
    run_main(configs)
