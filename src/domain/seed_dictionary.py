from typing import Tuple, List


import embeddings
from src.utils import topk_mean


class SeedDictionary:
    def __init__(self):
        pass

    def get_indices(self, x, z) -> Tuple[List[int], List[int]]:
        pass

class UnsupervisedSeedDictionary:
    def __init__(self, xp, configurations):
        self.xp = xp
        self.configurations = configurations

    def get_indices(self, x, z):
        sim_size = min(x.shape[0], z.shape[0]) if self.configurations['unsupervised_vocab'] <= 0 else min(x.shape[0], z.shape[0], self.configurations['unsupervised_vocab'])
        u, s, vt = self.xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = self.xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, self.configurations['normalize'])
        embeddings.normalize(zsim, self.configurations['normalize'])
        sim = xsim.dot(zsim.T)
        if self.configurations['csls'] > 0:
            knn_sim_fwd = topk_mean(sim, k=self.configurations['csls'])
            knn_sim_bwd = topk_mean(sim.T, k=self.configurations['csls'])
            sim -= knn_sim_fwd[:,self.xp.newaxis]/2 + knn_sim_bwd/2
        if self.configurations['direction'] == 'forward':
            src_indices = self.xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif self.configurations['direction'] == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = self.xp.arange(sim_size)
        elif self.configurations['direction'] == 'union':
            src_indices = self.xp.concatenate((self.xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = self.xp.concatenate((sim.argmax(axis=1), self.xp.arange(sim_size)))
        del xsim, zsim, sim
        return src_indices, trg_indices

