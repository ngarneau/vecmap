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

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver


exp_name = os.getenv('EXP_NAME', default='vecmap')
db_url = os.getenv('DB_URL', default='localhost')
db_name = os.getenv('DB_NAME', default='vecmap')

experiment = Experiment(exp_name)
experiment.observers.append(
    MongoObserver.create(
        url=db_url,
        db_name=db_name
    )
)


BATCH_SIZE = 500

def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


@experiment.command
def run_experiment(_run, _config):
    logging.info(_config)
    # Check command line arguments
    if (_config['src_dewhiten'] is not None or _config['trg_dewhiten'] is not None) and not _config['whiten']:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if _config['precision'] == 'fp16':
        dtype = 'float16'
    elif _config['precision'] == 'fp32':
        dtype = 'float32'
    elif _config['precision'] == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    src_input = "./data/embeddings/{}.emb.txt".format(_config['source_language'])  # The input source embeddings
    trg_input = "./data/embeddings/{}.emb.txt".format(_config['target_language'])  # The input target embeddings
    src_output = "./data/mapped_embeddings/{}.{}.emb.{}.txt".format(_config['source_language'],_config['target_language'], _config['iteration'])  # The output source embeddings
    trg_output = "./data/mapped_embeddings/{}.{}.emb.{}.txt".format(_config['target_language'], _config['source_language'], _config['iteration'])  # The output target embeddings
    init_dictionary = './data/dictionaries/{}-{}.train.txt'.format(_config['source_language'], _config['target_language'])  # the training dictionary file
    test_dictionary = './data/dictionaries/{}-{}.test.txt'.format(_config['source_language'], _config['target_language'])  # the test dictionary file

    logging.info("Loading srcfile {}".format(src_input))
    srcfile = open(src_input, encoding=_config['encoding'], errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)

    logging.info("Loading target input {}".format(trg_input))
    trgfile = open(trg_input, encoding=_config['encoding'], errors='surrogateescape')
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    if _config['cuda']:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(_config['seed'])

    # Build word to index map
    logging.info("Building word to index map")
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    logging.info("Normalize embeddings")
    embeddings.normalize(x, _config['normalize'])
    embeddings.normalize(z, _config['normalize'])

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if _config['init_unsupervised']:
        sim_size = min(x.shape[0], z.shape[0]) if _config['unsupervised_vocab'] <= 0 else min(x.shape[0], z.shape[0], _config['unsupervised_vocab'])
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, _config['normalize'])
        embeddings.normalize(zsim, _config['normalize'])
        sim = xsim.dot(zsim.T)
        if _config['csls'] > 0:
            knn_sim_fwd = topk_mean(sim, k=_config['csls'])
            knn_sim_bwd = topk_mean(sim.T, k=_config['csls'])
            sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
        if _config['direction'] == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif _config['direction'] == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif _config['direction'] == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
    elif _config['init_numerals']:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif _config['init_identical']:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:
        logging.info("Init dictionary")
        f = open(init_dictionary, encoding=_config['encoding'], errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)
        logging.info("Done")

    # Read validation dictionary
    if _config['validation']:
        logging.info("Reading validation dict")
        f = open(_config['validation'], encoding=_config['encoding'], errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))
        logging.info("Done")

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
    logging.info("ENd: {}".format(end))
    logging.info("Beginning training loop")
    while True:

        logging.info("Iteration number {}".format(it))
        # Increase the keep probability if we have not improve in _config['stochastic_interval iterations
        logging.info("Keep prob {}".format(keep_prob))
        if it - last_improvement > _config['stochastic_interval']:
            if keep_prob >= 1.0:
                logging.info("Training will end...")
                end = True
            keep_prob = min(1.0, _config['stochastic_multiplier']*keep_prob)
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
                return vt.T.dot(xp.diag(1/s)).dot(vt)
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
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=_config['csls'], inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if _config['direction'] in ('backward', 'union'):
                if _config['csls'] > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=_config['csls'], inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
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
                similarity = np.mean([max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

            # Logging
            duration = time.time() - t
            logging.info('ITERATION {0} ({1:.2f}s)'.format(it, duration))
            logging.info('\t- Objective:        {0:9.4f}%'.format(100 * objective))
            logging.info('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob))
            if _config['validation']:
                logging.info('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity))
                logging.info('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy))
                logging.info('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage))
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity,
                    100 * accuracy,
                    100 * validation_coverage
                ) if _config['validation'] is not None else ''
                logging.info('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)

        t = time.time()
        it += 1

    # Write mapped embeddings
    logging.info("Writing mapped embeddings to {}".format(src_output))
    srcfile = open(src_output, mode='w', encoding=_config['encoding'])
    embeddings.write(src_words, xw, srcfile)
    srcfile.close()
    logging.info("Done")

    logging.info("Writing mapped embeddings to {}".format(trg_output))
    trgfile = open(trg_output, mode='w', encoding=_config['encoding'])
    embeddings.write(trg_words, zw, trgfile)
    trgfile.close()
    logging.info("Done")

    srcfile = open(src_output, encoding=_config['encoding'])
    trgfile = open(trg_output, encoding=_config['encoding'])
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    if _config['cuda']:
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
    f = open(test_dictionary, encoding=_config['encoding'])
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
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
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
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif _config['retrieval'] == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if _config['inv_sample'] is None else xp.random.randint(0, x.shape[0], _config['inv_sample'])
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(_config['inv_temperature']*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(_config['inv_temperature']*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif _config['retrieval'] == 'csls':  # Cross-domain similarity local scaling
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=_config['neighborhood'], inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]

    # Compute accuracy
    accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
    _run.log_scalar('coverage', coverage)
    _run.log_scalar('accuracy', accuracy)
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))


@experiment.config
def base_config():
    logging.info("Activating base config")
    seed = 42
    iteration = 0
    source_language = "en"
    target_language = "de"
    encoding = "utf-8"  # The character encoding for input/output 
    precision = "fp32"  # The floating-point precision 
    cuda = False
    supervised = False  # recommended if you have a large training dictionary
    semi_supervised = False  # recommended if you have a small seed dictionary
    identical = False  # recommended if you have no seed dictionary but can rely on identical words
    unsupervised = False  # recommended if you have no seed dictionary and do not want to rely on identical words
    acl2018 = False  # reproduce our ACL 2018 system
    aaai2018 = False  # reproduce our AAAI 2018 system
    acl2017 = False  # reproduce our ACL 2017 system with numeral initialization
    acl2017_seed = False  # reproduce our ACL 2017 system with a seed dictionary
    emnlp2016 = False  # reproduce our EMNLP 2016 system
    num_runs = 10
    csls = 0  # use CSLS for dictionary induction
    init_unsupervised = False  # use unsupervised initialization
    normalize = []  # the normalization actions to perform in order. choices=['unit', 'center', 'unitdim', 'centeremb', 'none']
    self_learning = False  # enable self-learning
    src_dewhiten = False  # de-whiten the source language embeddings
    trg_dewhiten = False  # de-whiten the target language embeddings
    src_reweight = 0  # re-weight the source language embeddings
    trg_reweight = 0  # re-weight the target language embeddings
    unsupervised_vocab = 0  # restrict the vocabulary to the top k entries for unsupervised initialization
    vocabulary_cutoff = 0  # restrict the vocabulary to the top k entries
    whiten = False  # whiten the embeddings
    batch_size = 10000 # Batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory
    init_identical = False  # use identical words as the seed dictionary
    init_numerals = False  # use latin numerals (i.e. words matching [0-9]+) as the seed dictionary
    dim_reduction = False  # apply dimensionality reduction
    orthogonal = False  # use orthogonal constrained mapping
    unconstrained = False  # use unconstrained mapping
    direction = "union"  # the direction for dictionary induction (defaults to union)
    csls = 10  # use CSLS for dictionary induction
    threshold = 0.000001  # the convergence threshold
    validation = False  # a dictionary file for validation at each iteration
    stochastic_initial = 0.1  # initial keep probability stochastic dictionary induction (defaults to 0.1)
    stochastic_multiplier = 2.0  # stochastic dictionary induction multiplier (defaults to 2.0)
    stochastic_interval = 50  # stochastic dictionary induction interval (defaults to 50)

    # Evaluation parameters
    retrieval = 'nn' # choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)'
    inv_temperature = 1  # the inverse temperature (only compatible with inverted softmax)'
    inv_sample = None  # 'use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)'
    neighborhood = 10  # the neighborhood size (only compatible with csls)')
    dot=False  # use the dot product in the similarity computations instead of the cosine')


@experiment.named_config
def supervised():
    logging.info("Activating supervised configuration")
    supervised=True
    normalize=['unit', 'center', 'unit']
    whiten=True
    src_reweight=0.5
    trg_reweight=0.5
    src_dewhiten='src'
    trg_dewhiten='trg'
    batch_size=1000


@experiment.named_config
def semi_supervised():
    normalize=['unit', 'center', 'unit']
    whiten=True
    src_reweight=0.5
    trg_reweight=0.5
    src_dewhiten='src'
    trg_dewhiten='trg'
    self_learning=True
    vocabulary_cutoff=20000
    csls=10


@experiment.named_config
def identical():
    init_identical=True
    normalize=['unit', 'center', 'unit']
    whiten=True
    src_reweight=0.5
    trg_reweight=0.5
    src_dewhiten='src'
    trg_dewhiten='trg'
    self_learning=True
    vocabulary_cutoff=20000
    csls=10


@experiment.named_config
def unsupervised():
    init_unsupervised=True
    unsupervised_vocab=4000
    normalize=['unit', 'center', 'unit']
    whiten=True
    src_reweight=0.5
    trg_reweight=0.5
    src_dewhiten='src'
    trg_dewhiten='trg'
    self_learning=True
    vocabulary_cutoff=20000
    csls=10


@experiment.named_config
def acl2018():
    init_unsupervised=True
    unsupervised_vocab=4000
    normalize=['unit', 'center', 'unit']
    whiten=True
    src_reweight=0.5
    trg_reweight=0.5
    src_dewhiten='src'
    trg_dewhiten='trg'
    self_learning=True
    vocabulary_cutoff=20000
    csls=10


@experiment.named_config
def aaai2018():
    normalize=['unit', 'center']
    whiten=True
    trg_reweight=1
    src_dewhiten='src'
    trg_dewhiten='trg'
    batch_size=1000


@experiment.named_config
def acl2017():
    init_numerals=True
    orthogonal=True
    normalize=['unit', 'center']
    self_learning=True
    direction='forward'
    stochastic_initial=1.0
    stochastic_interval=1
    batch_size=1000


@experiment.named_config
def acl2017_seed():
    orthogonal=True
    normalize=['unit', 'center']
    self_learning=True
    direction='forward'
    stochastic_initial=1.0
    stochastic_interval=1
    batch_size=1000


@experiment.named_config
def emnlp2016():
    orthogonal=True
    normalize=['unit', 'center']
    batch_size=1000


@experiment.main
def main(_config):
    logging.getLogger().setLevel(logging.INFO)

    os.makedirs('./data/mapped_embeddings', exist_ok=True)

    config_updates = {}
    if _config['supervised']:
        logging.info("Adding supervised configurations")
        config_updates = dict(supervised(), **config_updates)
    if _config['semi_supervised']:
        logging.info("Adding semi supervised configurations")
        config_updates = dict(semi_supervised(), **config_updates)
    if _config['identical']:
        logging.info("Adding identical configurations")
        config_updates = dict(identical(), **config_updates)
    if _config['unsupervised'] or _config['acl2018']:
        config_updates = dict(unsupervised(), **config_updates)
        logging.info("Adding unsupervised configurations")
    if _config['aaai2018']:
        logging.info("Adding aaai2018 configurations")
        config_updates = dict(aaai2018(), **config_updates)
    if _config['acl2017']:
        logging.info("Adding acl2017 configurations")
        config_updates = dict(acl2017(), **config_updates)
    if _config['acl2017_seed']:
        logging.info("Adding acl2017 seed configurations")
        config_updates = dict(acl2017_seed(), **config_updates)
    if _config['emnlp2016']:
        config_updates = dict(emnlp2016(), **config_updates)
        logging.info("Adding emnlp2016 configurations")

    language_pairs = [
        ['en', 'es'],
        ['en', 'fi'],
        ['en', 'it'],
        ['en', 'de'],
    ]

    for source_language, target_language in language_pairs:
        for i in range(_config['num_runs']):
            runtime_config_updates = {
                    'iteration': i,
                    'source_language': source_language,
                    'target_language': target_language,
                }
            config_updates = dict(config_updates, **runtime_config_updates)
            experiment.run(
                'run_experiment',
                # named_configs=named_configs,
                config_updates=config_updates
            )


if __name__ == '__main__':
    experiment.run_commandline()