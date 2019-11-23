from src.domain.seed_dictionary import *


class SeedDictionaryFactory:
    UNSUPERVISED = 'unsupervised'
    NUMERALS = 'numerals'
    IDENTICAL = 'identical'
    RANDOM_RAW = 'random_raw'
    RANDOM_CUTOFF = 'random_cutoff'
    DEFAULT = 'default'

    @classmethod
    # todo revisit the naming.
    def create_seed_dictionary_builder(cls, method, xp, src_words, trg_words, x, z, configurations):
        if method == cls.UNSUPERVISED:
            return UnsupervisedSeedDictionary(xp, src_words, trg_words, x, z, configurations)
        elif method == cls.NUMERALS:
            return NumeralsSeedDictionary(src_words, trg_words)
        elif method == cls.IDENTICAL:
            return IdenticalSeedDictionary(src_words, trg_words)
        elif method == cls.RANDOM_RAW:
            return RandomRawSeedDictionary(src_words, trg_words, configurations)
        elif method == cls.RANDOM_CUTOFF:
            return RandomCutoffSeedDictionary(src_words, trg_words, configurations)
        elif method == cls.DEFAULT:
            return DefaultSeedDictionary(src_words, trg_words, configurations)
        else:
            raise ("Method {} not implemented.".format(method))
