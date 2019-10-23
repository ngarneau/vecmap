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

from typing import Dict
import os
import logging
import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import re
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient
import yaml

from src.factory.seed_dictionary import SeedDictionaryBuilderFactory
from src.utils import topk_mean


BATCH_SIZE = 500


class ComputeEngine:
    def __init__(self, name, engine, seed):
        self.name = name
        self.engine = engine
        self.engine.random.seed(seed)

    def send_to_device(self, data):
        if self.name == 'cupy':
            return self.engine.asarray(data)
        else:
            return data


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m * mask


def is_same_configuration(config: Dict, config_filter: Dict):
    for key, value in config_filter.items():
        if config.get(key) != value:
            logging.info("{} is different than {} for {}".format(config.get(key), value, key))
            return False
    return True


def whitening_arguments_validation(_config):
    if (_config['src_dewhiten'] is not None or _config['trg_dewhiten'] is not None) and not _config['whiten']:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)


def get_dtype(_config):
    if _config['precision'] == 'fp16':
        return 'float16'
    elif _config['precision'] == 'fp32':
        return 'float32'
    elif _config['precision'] == 'fp64':
        return 'float64'


def load_embeddings(embeddings_path, language, encoding, dtype):
    input_filename = embeddings_path.format(language)
    logging.info("Loading file {}".format(input_filename))
    input_file = open(input_filename, encoding=encoding, errors='surrogateescape')
    words, x = embeddings.read(input_file, dtype=dtype)
    logging.info("Loaded {} words of dimension {}".format(x.shape[0], x.shape[1]))
    return words, x


def get_compute_engine(use_cuda, seed):
    """
    This method will try to get cupy if the CUDA flag is activated.
    It will break if there are some problems with cupy
    """
    if use_cuda:
        if not supports_cupy():
            logging.error('Install CuPy for CUDA support')
            sys.exit(-1)
        return ComputeEngine('cupy', get_cupy(), seed)
    else:
        return ComputeEngine('numpy', np, seed)


def run_experiment(_config):
    logging.info(_config)
    mlflow.log_params(_config)
    mlflow.log_metric('test', 0.9)

    whitening_arguments_validation(_config)

    dtype = get_dtype(_config)

    src_words, x = load_embeddings(_config['embeddings_path'], _config['source_language'], _config['encoding'], dtype)
    trg_words, z = load_embeddings(_config['embeddings_path'], _config['target_language'], _config['encoding'], dtype)

    compute_engine = get_compute_engine(_config['cuda'], _config['seed'])
    xp = compute_engine.engine

    x = compute_engine.send_to_device(x)
    z = compute_engine.send_to_device(z)

    # Read input embeddings
    src_output = "./output/{}.{}.emb.{}.txt".format(_config['source_language'], _config['target_language'],
                                                    _config['iteration'])  # The output source embeddings
    trg_output = "./output/{}.{}.emb.{}.txt".format(_config['target_language'], _config['source_language'],
                                                    _config['iteration'])  # The output target embeddings
    init_dictionary = './data/dictionaries/{}-{}.train.txt'.format(
        _config['source_language'], _config['target_language'])  # the training dictionary file
    test_dictionary = './data/dictionaries/{}-{}.test.txt'.format(
        _config['source_language'], _config['target_language'])  # the test dictionary file

    # Build word to index map
    logging.info("Building word to index map")
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    logging.info("Normalize embeddings")
    embeddings.normalize(x, _config['normalize'])
    embeddings.normalize(z, _config['normalize'])

    # Build the seed dictionary
    seed_dictionary_builder = SeedDictionaryBuilderFactory.get_seed_dictionary_builder(
        _config['seed_dictionary_method'], xp, src_words, trg_words, x, z, _config)
    src_indices, trg_indices = seed_dictionary_builder.get_indices()

    # Allocate memory
    logging.info("Allocating memory")
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0] if _config['vocabulary_cutoff'] <= 0 else min(x.shape[0], _config['vocabulary_cutoff'])
    trg_size = z.shape[0] if _config['vocabulary_cutoff'] <= 0 else min(z.shape[0], _config['vocabulary_cutoff'])
    simfwd = xp.empty((_config['batch_size'], trg_size), dtype=dtype)
    simbwd = xp.empty((_config['batch_size'], src_size), dtype=dtype)
    if _config['validation']:
        simval = xp.empty((len(validation.keys()), z.shape[0]), dtype=dtype)

    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = _config['stochastic_initial']
    t = time.time()
    end = not _config['self_learning']
    logging.info("Beginning training loop")
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
            u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
            w = vt.T.dot(u.T)
            x.dot(w, out=xw)
            zw[:] = z
        elif _config['unconstrained']:  # unconstrained mapping
            x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping

            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1 / s)).dot(vt)

            if _config['whiten']:
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
            wz2 = wz2_t.T
            xw = xw.dot(wx2)
            zw = zw.dot(wz2)

            # STEP 3: Re-weighting
            xw *= s**_config['src_reweight']
            zw *= s**_config['trg_reweight']

            # STEP 4: De-whitening
            if _config['src_dewhiten'] == 'src':
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif _config['src_dewhiten'] == 'trg':
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if _config['trg_dewhiten'] == 'src':
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif _config['trg_dewhiten'] == 'trg':
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if _config['dim_reduction'] > 0:
                xw = xw[:, :_config['dim_reduction']]
                zw = zw[:, :_config['dim_reduction']]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
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
                    dropout(simfwd[:j - i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
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
                    dropout(simbwd[:j - i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if _config['direction'] == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif _config['direction'] == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif _config['direction'] == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if _config['direction'] == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif _config['direction'] == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif _config['direction'] == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= _config['threshold']:
                last_improvement = it
                best_objective = objective

            # Accuracy and similarity evaluation in validation
            if _config['validation']:
                src = list(validation.keys())
                xw[src].dot(zw.T, out=simval)
                nn = asnumpy(simval.argmax(axis=1))
                accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
                similarity = np.mean(
                    [max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

            # Logging
            duration = time.time() - t
            logging.info('ITERATION {0} ({1:.2f}s)'.format(it, duration))
            logging.info('\t- Objective:        {0:9.4f}%'.format(100 * objective))
            logging.info('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob))
            if _config['validation']:
                logging.info('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity))
                logging.info('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy))
                logging.info('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage))
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity, 100 * accuracy, 100 *
                    validation_coverage) if _config['validation'] is not None else ''
                logging.info('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)

        t = time.time()
        it += 1

    # Write mapped embeddings
    logging.info("Writing mapped embeddings to {}".format(src_output))
    srcfile = open(src_output, mode='w', encoding=_config['encoding'], errors='surrogateescape')
    embeddings.write(src_words, xw, srcfile)
    srcfile.close()
    logging.info("Done")

    logging.info("Writing mapped embeddings to {}".format(trg_output))
    trgfile = open(trg_output, mode='w', encoding=_config['encoding'], errors='surrogateescape')
    embeddings.write(trg_words, zw, trgfile)
    trgfile.close()
    logging.info("Done")

    srcfile = open(src_output, encoding=_config['encoding'], errors='surrogateescape')
    trgfile = open(trg_output, encoding=_config['encoding'], errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    if _config['cuda'] is True:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(_config['seed'])

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not _config['dot']:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

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
        sample = xp.arange(x.shape[0]) if _config['inv_sample'] is None else xp.random.randint(
            0, x.shape[0], _config['inv_sample'])
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(_config['inv_temperature'] * z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(_config['inv_temperature'] * x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j - i):
                translation[src[i + k]] = nn[k]
    elif _config['retrieval'] == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
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


def get_query_string(configs):
    return " and ".join(["params.{}='{}'".format(config, value) for config, value in configs.items()])


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
    times = list()
    for run in runs:
        if 'accuracy' in run.data.metrics:
            minutes = ((run.info.end_time - run.info.start_time)//60//60)%60
            times.append(minutes)
            accuracies.append(run.data.metrics['accuracy'])
    return accuracies, times


class MlFlowHandler(logging.FileHandler):
    def emit(self, record):
        super(MlFlowHandler, self).emit(record)
        mlflow.log_artifact(self.baseFilename)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    exp_name = os.getenv('EXP_NAME', default='vecmap')

    base_configs = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    argument_parser = argparse.ArgumentParser()
    for config, value in base_configs.items():
        argument_parser.add_argument('--{}'.format(config), type=type(value), default=value)
    options = argument_parser.parse_args()
    configs = vars(options)

    client = MlflowClient()
    mlflow.set_experiment(exp_name)
    experiment = client.get_experiment_by_name('vecmap')

    os.makedirs('./output/mapped_embeddings', exist_ok=True)

    os.makedirs('./output/logs', exist_ok=True)
    fh = MlFlowHandler('./output/logs')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if configs['test']:
        language_pairs = [
            ['en_slim', 'de_slim'],
        ]
        new_language_pairs = []
    else:
        language_pairs = [
            ['en', 'de'],
            ['en', 'es'],
            ['en', 'fi'],
            ['en', 'it'],
        ]
        new_language_pairs = [
            ['en', 'et'],
            ['en', 'fa'],
            ['en', 'lv'],
            ['en', 'tr'],
            ['en', 'vi'],
        ]

    if not configs['num_runs'] == 1 and configs['supercomputer']:
        configs['num_runs'] = 1
        print('Manually overriding num_runs attribute to 1 because supercomputer mode is enabled.')

    # These are languages from the original paper.
    for source_language, target_language in language_pairs:
        for i in range(configs['num_runs']):
            configs = override_configs(source_language, target_language, i, configs)
            try:
                with mlflow.start_run(experiment_id=experiment.experiment_id):
                    run_experiment(configs)
            except KeyboardInterrupt:
                logging.warning("Run exited.")

        filter = create_filter(source_language, target_language, configs)
        query_string = get_query_string(filter)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query_string)
        accuracies, times = retrieve_stats(runs)
        print(target_language, np.mean(accuracies), np.std(accuracies), np.mean(times))


    # We place new language in another loop since we will create a different table.
    # Refactoring may be needed here.
    for source_language, target_language in new_language_pairs:
        for i in range(configs['num_runs']):
            configs = override_configs(source_language, target_language, i, configs)
            try:
                with mlflow.start_run(experiment_id=experiment.experiment_id):
                    run_experiment(configs)
            except KeyboardInterrupt:
                logging.warning("Run exited.")

        filter = create_filter(source_language, target_language, configs)
        query_string = get_query_string(filter)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query_string)
        accuracies, times = retrieve_stats(runs)
        logging.info("Accuracies: {}".format(accuracies))
        logging.info("Language: {}, Mean Acc: {}, Std Acc: {}, Max Acc: {}, Min acc: {}, Mean Time: {}".format(
            target_language,
            np.mean(accuracies),
            np.std(accuracies),
            max(accuracies),
            min(accuracies),
            np.mean(times)
        ))



if __name__ == '__main__':
    main()

