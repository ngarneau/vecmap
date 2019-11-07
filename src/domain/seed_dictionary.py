import logging
import re
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np

import embeddings
from src.utils import topk_mean


class SeedDictionary(ABC):
    def __init__(self, src_words, trg_words):
        self.src_words = src_words
        self.trg_words = trg_words
        self.src_word2ind = {word: ind for ind, word in enumerate(self.src_words)}
        self.trg_word2ind = {word: ind for ind, word in enumerate(self.trg_words)}

    @abstractmethod
    def get_indices(self) -> Tuple[List[int], List[int]]:
        pass


class UnsupervisedSeedDictionary(SeedDictionary):
    def __init__(self, xp, src_words, trg_words, x, z, configurations):
        super().__init__(src_words, trg_words)
        self.xp = xp
        self.x = x
        self.z = z
        self.configurations = configurations

    def get_indices(self):
        sim_size = min(self.x.shape[0], self.z.shape[0]) if self.configurations['unsupervised_vocab'] <= 0 else min(
            self.x.shape[0], self.z.shape[0], self.configurations['unsupervised_vocab'])

        u, s, vt = self.xp.linalg.svd(self.x[:sim_size], full_matrices=False)
        xsim = (u * s).dot(u.T)  # This is equivalent to Mx in the original paper
        u, s, vt = self.xp.linalg.svd(self.z[:sim_size], full_matrices=False)
        zsim = (u * s).dot(u.T)  # This is equivalent to Mz in the original paper
        del u, s, vt

        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, self.configurations['normalize'])
        embeddings.normalize(zsim, self.configurations['normalize'])
        sim = xsim.dot(zsim.T)

        if self.configurations['csls'] > 0:
            knn_sim_fwd = topk_mean(sim, k=self.configurations['csls'])
            knn_sim_bwd = topk_mean(sim.T, k=self.configurations['csls'])
            sim -= knn_sim_fwd[:, self.xp.newaxis] / 2 + knn_sim_bwd / 2

        src_indices, trg_indices = self._dictionary_induction(sim, sim_size)

        del xsim, zsim, sim
        return src_indices, trg_indices

    def _dictionary_induction(self, sim, sim_size):
        induction_direction = self.configurations['direction']
        if induction_direction == 'forward':
            src_indices = self.xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif induction_direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = self.xp.arange(sim_size)
        elif induction_direction == 'union':
            src_indices = self.xp.concatenate((self.xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = self.xp.concatenate((sim.argmax(axis=1), self.xp.arange(sim_size)))
        else:
            raise ("Method {} not implemented.".format(induction_direction))

        del sim

        return src_indices, trg_indices


class NumeralsSeedDictionary(SeedDictionary):
    def __init__(self, src_words, trg_words):
        super().__init__(src_words, trg_words)
        self.numeral_regex = re.compile('^[0-9]+$')

    def get_indices(self):
        src_numerals = {word for word in self.src_words if self.numeral_regex.match(word) is not None}
        trg_numerals = {word for word in self.trg_words if self.numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)

        return self._dictionary_induction(numerals)

    def _dictionary_induction(self, numerals):
        src_indices = list()
        trg_indices = list()
        for word in numerals:
            src_indices.append(self.src_word2ind[word])
            trg_indices.append(self.trg_word2ind[word])
        return src_indices, trg_indices


class IdenticalSeedDictionary(SeedDictionary):
    def __init__(self, src_words, trg_words):
        super().__init__(src_words, trg_words)

    def get_indices(self):
        identical = set(self.src_words).intersection(set(self.trg_words))
        return self._dictionary_induction(identical)

    def _dictionary_induction(self, identical):
        src_indices = list()
        trg_indices = list()
        for word in identical:
            src_indices.append(self.src_word2ind[word])
            trg_indices.append(self.trg_word2ind[word])
        return src_indices, trg_indices


class DefaultSeedDictionary(SeedDictionary):
    def __init__(self, src_words, trg_words, configurations):
        super().__init__(src_words, trg_words)
        self.configurations = configurations

        self.training_dictionary_filename = './data/dictionaries/{}-{}.train.txt'.format(
            configurations['source_language'],
            configurations['target_language'])

    def get_indices(self):
        dict_data = open(self.training_dictionary_filename, encoding=self.configurations['encoding'],
                         errors='surrogateescape')
        return self._dictionary_induction(dict_data)

    def _dictionary_induction(self, dict_data):
        src_indices = list()
        trg_indices = list()
        for line in dict_data:
            src, trg = line.split()
            try:
                src_ind = self.src_word2ind[src]
                trg_ind = self.trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                logging.warning('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg))
        return src_indices, trg_indices


class RandomRawSeedDictionary(SeedDictionary):
    def __init__(self, src_words, trg_words, configurations):
        super().__init__(src_words, trg_words)
        self.configurations = configurations

    def get_indices(self):
        return self._generate_indices()

    def _generate_indices(self):
        src_indices = list()
        trg_indices = list()
        if len(self.src_words) > len(self.trg_words):
            random_mapping = np.random.randint(len(self.trg_words), size=len(self.src_words))

            for src_index, trg_index in enumerate(random_mapping):
                src_indices.append(src_index)
                trg_indices.append(trg_index)
        else:
            random_mapping = np.random.randint(len(self.src_words), size=len(self.trg_words))

            for trg_index, src_index in enumerate(random_mapping):
                src_indices.append(src_index)
                trg_indices.append(trg_index)

        src_indices = np.array(src_indices)
        trg_indices = np.array(trg_indices)

        return src_indices, trg_indices


class RandomCutoffSeedDictionary(RandomRawSeedDictionary):
    def __init__(self, src_words, trg_words, configurations):
        super().__init__(src_words, trg_words, configurations)

    def get_indices(self):
        self.src_words = self.src_words[:self.configurations['vocabulary_cutoff']]
        self.trg_words = self.trg_words[:self.configurations['vocabulary_cutoff']]

        return self._generate_indices()
