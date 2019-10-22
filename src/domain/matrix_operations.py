def whitening_transformation(m, compute_engine):
    u, s, vt = compute_engine.linalg.svd(m, full_matrices=False)
    return vt.T.dot(compute_engine.diag(1 / s)).dot(vt)
