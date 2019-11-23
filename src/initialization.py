from src.factory.seed_dictionary import SeedDictionaryFactory


def get_seed_dictionary_indices(seed_dictionary_method, compute_engine, src_vocab, trg_vocab, src_embedding_matrix,
                                trg_embedding_matrix, other_configs):
    return SeedDictionaryFactory.create_seed_dictionary_builder(seed_dictionary_method, compute_engine,
                                                                src_vocab, trg_vocab, src_embedding_matrix,
                                                                trg_embedding_matrix, other_configs).get_indices()
