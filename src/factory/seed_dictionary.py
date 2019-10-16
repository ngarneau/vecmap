from src.domain.seed_dictionary import *


class SeedDictionaryBuilderFactory:

    UNSUPERVISED = 'unsupervised'
    NUMERALS = 'numerals'
    IDENTICAL = 'identical'
    DEFAULT = 'default'

    @classmethod
    def get_seed_dictionary_builder(cls, method, xp, src_words, trg_words, x, z, configurations):
        if method == cls.UNSUPERVISED:
            return UnsupervisedSeedDictionary(xp, src_words, trg_words, x, z, configurations)
        elif method == cls.NUMERALS:
            return NumeralsSeedDictionary(xp, src_words, trg_words, x, z, configurations)
        elif method == cls.IDENTICAL:
            return IdenticalSeedDictionary(xp, src_words, trg_words, x, z, configurations)
        elif method == cls.DEFAULT:
            return DefaultSeedDictionary(xp, src_words, trg_words, x, z, configurations)
        else:
            raise("Method {} not implemented.".format(method))
