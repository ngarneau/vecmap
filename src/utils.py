from src.cupy_utils import *


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


def compute_matrix_size(src_embedding_matrix, trg_embedding_matrix, vocabulary_cutoff):
    src_size = src_embedding_matrix.shape[0] if vocabulary_cutoff <= 0 else min(
        src_embedding_matrix.shape[0], vocabulary_cutoff)
    trg_size = trg_embedding_matrix.shape[0] if vocabulary_cutoff <= 0 else min(
        trg_embedding_matrix.shape[0], vocabulary_cutoff)
    return src_size, trg_size


def resolve_language_source(_config):
    if _config['test']:
        source_language = _config['source_language'] + '_slim'
        target_language = _config['target_language'] + '_slim'
    else:
        source_language = _config['source_language']
        target_language = _config['target_language']
    return source_language, target_language
