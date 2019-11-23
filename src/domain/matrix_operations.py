from src.cupy_utils import get_array_module


def whitening_transformation(m, compute_engine):
    u, s, vt = compute_engine.linalg.svd(m, full_matrices=False)
    return vt.T.dot(compute_engine.diag(1 / s)).dot(vt)


def dropout(matrix, dropout_prob, compute_engine):
    if dropout_prob <= 0.0:
        return matrix
    else:
        compute_engine.engine = get_array_module(matrix)
        mask = compute_engine.engine.random.rand(*matrix.shape) >= dropout_prob
        return matrix * mask
