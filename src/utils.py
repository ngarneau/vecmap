import logging
import sys

import numpy as np

from cupy_utils import *
from src.domain.compute_engine import CuPyEngine, NumPyEngine


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


def set_compute_engine(use_cuda, seed):
    """
    This method will try to set cupy as the compute engine if the CUDA flag is activated otherwise it will use NumPy.
    It will break if there are some problems with cupy
    """
    if use_cuda:
        if not supports_cupy():
            logging.error('Install CuPy for CUDA support')
            sys.exit(-1)
        return CuPyEngine(get_cupy(), seed)
    else:
        return NumPyEngine(np, seed)

def output_embeddings_filename(_config):
    src_output = "./output/{}.{}.emb.{}.txt".format(_config['source_language'], _config['target_language'],
                                                    _config['iteration'])  # The output source embeddings
    trg_output = "./output/{}.{}.emb.{}.txt".format(_config['target_language'], _config['source_language'],
                                                    _config['iteration'])  # The output target embeddings

    return src_output, trg_output
