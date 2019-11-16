import collections
import logging
import time

import mlflow
import numpy as np

from src.cupy_utils import get_cupy
from src.domain.compute_engine import CuPyEngine, NumPyEngine
from src.domain.embeddings import load_embeddings, embeddings_normalization_step, length_normalize, asnumpy, \
    supports_cupy
from src.domain.matrix_operations import whitening_transformation, dropout
from src.initialization import get_seed_dictionary_indices
from src.utils import resolve_language_source, compute_matrix_size, topk_mean
from src.validations import did_not_improve

BATCH_SIZE = 500


class VecMap:
    def __init__(self, config):
        self._config = config

        self.dtype = self._config['precision']

        self._init_computing_engine(self._config['cuda'], self._config['seed'])

        self.src_words = None
        self.x = None
        self.trg_vocab = None
        self.z = None

        self.xw = None
        self.zw = None
        self.src_size = None
        self.trg_size = None
        self.simfwd = None
        self.simbwd = None
        self.best_sim_forward = None
        self.src_indices_forward = None
        self.trg_indices_forward = None
        self.best_sim_backward = None
        self.src_indices_backward = None
        self.trg_indices_backward = None
        self.knn_sim_fwd = None
        self.knn_sim_bwd = None

        self.src_indices = None
        self.trg_indices = None

    def load_embeddings(self):
        source_language, target_language = resolve_language_source(self._config)
        self.src_words, x = load_embeddings(self._config['embeddings_path'], source_language,
                                            self._config['encoding'], self.dtype)
        self.trg_vocab, z = load_embeddings(self._config['embeddings_path'], target_language,
                                            self._config['encoding'], self.dtype)

        self.x = self.compute_engine.send_to_device(x)
        self.z = self.compute_engine.send_to_device(z)

    def embeddings_normalization_step(self):
        embeddings_normalization_step(self.x, self.z, normalization_method=self._config['normalize'])

    def allocate_memory(self):
        logging.info("Allocating memory")
        self.xw = self.compute_engine.engine.empty_like(self.x)
        self.zw = self.compute_engine.engine.empty_like(self.z)

        self.src_size, self.trg_size = compute_matrix_size(self.x, self.z, self._config['vocabulary_cutoff'])
        self.simfwd = self.compute_engine.engine.empty((self._config['batch_size'], self.trg_size), dtype=self.dtype)
        self.simbwd = self.compute_engine.engine.empty((self._config['batch_size'], self.src_size), dtype=self.dtype)

        self.best_sim_forward = self.compute_engine.engine.full(self.src_size, -100, dtype=self.dtype)
        self.src_indices_forward = self.compute_engine.engine.arange(self.src_size)
        self.trg_indices_forward = self.compute_engine.engine.zeros(self.src_size, dtype=int)
        self.best_sim_backward = self.compute_engine.engine.full(self.trg_size, -100, dtype=self.dtype)
        self.src_indices_backward = self.compute_engine.engine.zeros(self.trg_size, dtype=int)
        self.trg_indices_backward = self.compute_engine.engine.arange(self.trg_size)
        self.knn_sim_fwd = self.compute_engine.engine.zeros(self.src_size, dtype=self.dtype)
        self.knn_sim_bwd = self.compute_engine.engine.zeros(self.trg_size, dtype=self.dtype)

    def fully_unsupervised_initialization_step(self):
        self.src_indices, self.trg_indices = get_seed_dictionary_indices(self._config['seed_dictionary_method'],
                                                                         self.compute_engine.engine,
                                                                         self.src_words, self.trg_vocab, self.x,
                                                                         self.z, self._config)

    def robust_self_learning(self):
        logging.info("Beginning training loop")
        best_objective = objective = -100.
        it = 1
        last_improvement = 0
        keep_prob = self._config['stochastic_initial']
        time_stamp = time.time()
        end = not self._config['self_learning']

        while True:
            logging.info("Iteration number {}".format(it))
            logging.info("Keep prob {}".format(keep_prob))
            if did_not_improve(it - last_improvement, self._config['stochastic_interval']):
                if keep_prob >= 1.0:
                    logging.info("Training will end...")
                    end = True
                keep_prob = min(1.0, self._config['stochastic_multiplier'] * keep_prob)
                last_improvement = it

            # Update the embedding mapping
            if self._config['orthogonal'] or not end:  # orthogonal mapping
                u, s, vt = self.compute_engine.engine.linalg.svd(
                    self.z[self.trg_indices].T.dot(self.x[self.src_indices]))
                w = vt.T.dot(u.T)
                self.x.dot(w, out=self.xw)
                self.zw[:] = self.z
            elif self._config['unconstrained']:  # unconstrained mapping
                x_pseudoinv = self.compute_engine.engine.linalg.inv(
                    self.x[self.src_indices].T.dot(self.x[self.src_indices])).dot(
                    self.x[self.src_indices].T)
                w = x_pseudoinv.dot(self.z[self.trg_indices])
                self.x.dot(w, out=self.xw)
                self.zw[:] = self.z
            else:  # advanced mapping
                self.symmetric_re_weighting()

            # Self-learning
            # Update the training dictionary
            if end:
                break
            else:
                self._CSLS_retrieval(keep_prob)
                self._bidirectional_dictionary_induction()

                # Objective function evaluation
                if self._config['direction'] == 'forward':
                    objective = self.compute_engine.engine.mean(self.best_sim_forward).tolist()
                elif self._config['direction'] == 'backward':
                    objective = self.compute_engine.engine.mean(self.best_sim_backward).tolist()
                elif self._config['direction'] == 'union':
                    objective = (self.compute_engine.engine.mean(
                        self.best_sim_forward) + self.compute_engine.engine.mean(
                        self.best_sim_backward)).tolist() / 2
                if objective - best_objective >= self._config['threshold']:
                    last_improvement = it
                    best_objective = objective

                # Logging
                duration = time.time() - time_stamp
                logging.info('ITERATION {0} ({1:.2f}s)'.format(it, duration))
                logging.info('\t- Objective:        {0:9.4f}%'.format(100 * objective))
                logging.info('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob))

            time_stamp = time.time()
            it += 1

        self.x = self.xw
        self.z = self.zw

    def eval(self):
        self.compute_engine.engine.random.seed(
            self._config['seed'])  # todo verify is this step is legit (same as code base)

        # Length normalize embeddings so their dot product effectively computes the cosine similarity
        if not self._config['dot']:
            length_normalize(self.x)
            length_normalize(self.z)

        # Build word to index map
        src_word2ind = {word: i for i, word in enumerate(self.src_words)}
        trg_word2ind = {word: i for i, word in enumerate(self.trg_vocab)}

        # Read dictionary and compute coverage
        test_dictionary = './data/dictionaries/{}-{}.test.txt'.format(
            self._config['source_language'], self._config['target_language'])
        f = open(test_dictionary, encoding=self._config['encoding'], errors='surrogateescape')
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
        if self._config['retrieval'] == 'nn':  # Standard nearest neighbor
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                similarities = self.x[src[i:j]].dot(self.z.T)
                nn = similarities.argmax(axis=1).tolist()
                for k in range(j - i):
                    translation[src[i + k]] = nn[k]
        elif self._config['retrieval'] == 'invnn':  # Inverted nearest neighbor
            best_rank = np.full(len(src), self.x.shape[0], dtype=int)
            best_sim = np.full(len(src), -100, dtype=self.dtype)
            for i in range(0, self.z.shape[0], BATCH_SIZE):
                j = min(i + BATCH_SIZE, self.z.shape[0])
                similarities = self.z[i:j].dot(self.x.T)
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
        elif self._config['retrieval'] == 'invsoftmax':  # Inverted softmax
            sample = self.compute_engine.engine.arange(self.x.shape[0]) if self._config[
                                                                               'inv_sample'] is None else self.compute_engine.engine.random.randint(
                0, self.x.shape[0], self._config['inv_sample'])
            partition = self.compute_engine.engine.zeros(self.z.shape[0])
            for i in range(0, len(sample), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(sample))
                partition += self.compute_engine.engine.exp(
                    self._config['inv_temperature'] * self.z.dot(self.x[sample[i:j]].T)).sum(axis=1)
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                p = self.compute_engine.engine.exp(
                    self._config['inv_temperature'] * self.x[src[i:j]].dot(self.z.T)) / partition
                nn = p.argmax(axis=1).tolist()
                for k in range(j - i):
                    translation[src[i + k]] = nn[k]
        elif self._config['retrieval'] == 'csls':  # Cross-domain similarity local scaling
            knn_sim_bwd = self.compute_engine.engine.zeros(self.z.shape[0])
            for i in range(0, self.z.shape[0], BATCH_SIZE):
                j = min(i + BATCH_SIZE, self.z.shape[0])
                knn_sim_bwd[i:j] = topk_mean(self.z[i:j].dot(self.x.T), k=self._config['csls'],
                                             inplace=True)
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                similarities = 2 * self.x[src[i:j]].dot(
                    self.z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
                nn = similarities.argmax(axis=1).tolist()
                for k in range(j - i):
                    translation[src[i + k]] = nn[k]

        # Compute accuracy
        accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
        mlflow.log_metric('coverage', coverage)
        mlflow.log_metric('accuracy', accuracy)
        logging.info('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))

    def symmetric_re_weighting(self):
        self.xw[:] = self.x
        self.zw[:] = self.z

        wx1, wz1 = self._whiten()

        wx2, wz2, s = self._orthogonal_mapping()

        self._re_weighting(s)

        self._re_whitening(wx1, wz1, wx2, wz2)

        self._dimensionality_reduction()

    def _init_computing_engine(self, use_cuda, seed):
        """
        This method will try to set cupy as the compute engine if the CUDA flag is activated otherwise it will use NumPy.
        It will break if there are some problems with cupy
        """
        if use_cuda:
            if not supports_cupy():
                raise ImportError("Install CuPy for CUDA support.")
            self.compute_engine = CuPyEngine(get_cupy(), seed)
        else:
            self.compute_engine = NumPyEngine(np, seed)

    def _whiten(self):
        wx1 = whitening_transformation(self.xw[self.src_indices], compute_engine=self.compute_engine.engine)
        wz1 = whitening_transformation(self.zw[self.trg_indices], compute_engine=self.compute_engine.engine)
        self.xw = self.xw.dot(wx1)
        self.zw = self.zw.dot(wz1)
        return wx1, wz1

    def _orthogonal_mapping(self):
        wx2, s, wz2_t = self.compute_engine.engine.linalg.svd(
            self.xw[self.src_indices].T.dot(self.zw[self.trg_indices]))
        wz2 = wz2_t.T
        self.xw = self.xw.dot(wx2)
        self.zw = self.zw.dot(wz2)
        return wx2, wz2, s

    def _re_weighting(self, s):
        self.xw *= s ** self._config['reweight']
        self.zw *= s ** self._config['reweight']

    def _re_whitening(self, wx1, wz1, wx2, wz2):
        if self._config['src_dewhiten'] == 'src':
            self.xw = self.xw.dot(wx2.T.dot(self.compute_engine.engine.linalg.inv(wx1)).dot(wx2))
        elif self._config['src_dewhiten'] == 'trg':
            self.xw = self.xw.dot(wz2.T.dot(self.compute_engine.engine.linalg.inv(wz1)).dot(wz2))
        if self._config['trg_dewhiten'] == 'src':
            self.zw = self.zw.dot(wx2.T.dot(self.compute_engine.engine.linalg.inv(wx1)).dot(wx2))
        elif self._config['trg_dewhiten'] == 'trg':
            self.zw = self.zw.dot(wz2.T.dot(self.compute_engine.engine.linalg.inv(wz1)).dot(wz2))

    def _dimensionality_reduction(self):
        if self._config['dim_reduction'] > 0:
            self.xw = self.xw[:, :self._config['dim_reduction']]
            self.zw = self.zw[:, :self._config['dim_reduction']]

    def _CSLS_retrieval(self, keep_prob):
        if self._config['direction'] in ('forward', 'union'):
            if self._config['csls'] > 0:
                for i in range(0, self.trg_size, self.simbwd.shape[0]):
                    j = min(i + self.simbwd.shape[0], self.trg_size)
                    self.zw[i:j].dot(self.xw[:self.src_size].T, out=self.simbwd[:j - i])
                    self.knn_sim_bwd[i:j] = topk_mean(self.simbwd[:j - i], k=self._config['csls'], inplace=True)
            for i in range(0, self.src_size, self.simfwd.shape[0]):
                j = min(i + self.simfwd.shape[0], self.src_size)
                self.xw[i:j].dot(self.zw[:self.trg_size].T, out=self.simfwd[:j - i])
                self.simfwd[:j - i].max(axis=1, out=self.best_sim_forward[i:j])
                self.simfwd[:j - i] -= self.knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                dropout(self.simfwd[:j - i], 1 - keep_prob, compute_engine=self.compute_engine).argmax(axis=1,
                                                                                                       out=self.trg_indices_forward[
                                                                                                           i:j])
        if self._config['direction'] in ('backward', 'union'):
            if self._config['csls'] > 0:
                for i in range(0, self.src_size, self.simfwd.shape[0]):
                    j = min(i + self.simfwd.shape[0], self.src_size)
                    self.xw[i:j].dot(self.zw[:self.trg_size].T, out=self.simfwd[:j - i])
                    self.knn_sim_fwd[i:j] = topk_mean(self.simfwd[:j - i], k=self._config['csls'], inplace=True)
            for i in range(0, self.trg_size, self.simbwd.shape[0]):
                j = min(i + self.simbwd.shape[0], self.trg_size)
                self.zw[i:j].dot(self.xw[:self.src_size].T, out=self.simbwd[:j - i])
                self.simbwd[:j - i].max(axis=1, out=self.best_sim_backward[i:j])
                self.simbwd[:j - i] -= self.knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                dropout(self.simbwd[:j - i], 1 - keep_prob, compute_engine=self.compute_engine).argmax(axis=1,
                                                                                                       out=self.src_indices_backward[
                                                                                                           i:j])

    def _bidirectional_dictionary_induction(self):
        if self._config['direction'] == 'forward':
            self.src_indices = self.src_indices_forward
            self.trg_indices = self.trg_indices_forward
        elif self._config['direction'] == 'backward':
            self.src_indices = self.src_indices_backward
            self.trg_indices = self.trg_indices_backward
        elif self._config['direction'] == 'union':
            self.src_indices = self.compute_engine.engine.concatenate(
                (self.src_indices_forward, self.src_indices_backward))
            self.trg_indices = self.compute_engine.engine.concatenate(
                (self.trg_indices_forward, self.trg_indices_backward))
