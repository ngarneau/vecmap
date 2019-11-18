import numpy as np

from cupy_utils import supports_cupy, get_cupy
from src.domain.compute_engine import CuPyEngine, NumPyEngine
from src.factory.seed_dictionary import SeedDictionaryFactory


def get_seed_dictionary_indices(seed_dictionary_method, compute_engine, src_vocab, trg_vocab, src_embedding_matrix,
                                trg_embedding_matrix, other_configs):
    return SeedDictionaryFactory.create_seed_dictionary_builder(seed_dictionary_method, compute_engine,
                                                                src_vocab, trg_vocab, src_embedding_matrix,
                                                                trg_embedding_matrix, other_configs).get_indices()


def init_computing_engine(use_cuda, seed):
    """
    This method will try to set cupy as the compute engine if the CUDA flag is activated otherwise it will use NumPy.
    It will break if there are some problems with cupy
    """
    if use_cuda:
        if not supports_cupy():
            raise ImportError("Install CuPy for CUDA support.")
        return CuPyEngine(get_cupy(), seed)
    else:
        return NumPyEngine(np, seed)
