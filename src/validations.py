from src.domain.initialization_error import InitializationError


def whitening_arguments_validation(_config):
    if (_config['src_dewhiten'] is not None or _config['trg_dewhiten'] is not None) and not _config['whiten']:
        raise InitializationError('ERROR: De-whitening requires whitening first')


def did_not_improve(span_since_last_improvement, max_interval):
    return span_since_last_improvement > max_interval
