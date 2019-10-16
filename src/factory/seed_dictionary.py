from src.domain.seed_dictionary import UnsupervisedSeedDictionary


class SeedDictionaryBuilderFactory:

    UNSUPERVISED = 'unsupervised'
    NUMERALS = 'numerals'
    IDENTICAL = 'identical'
    DEFAULT = 'default'

    def __init__(self):
        pass

    @classmethod
    def get_seed_dictionary_builder(cls, method, xp, configurations):
        if method == cls.UNSUPERVISED:
            # return unsupervised
            return UnsupervisedSeedDictionary(xp, configurations)
        elif method == cls.NUMERALS:
            pass
        elif method == cls.IDENTICAL:
            pass
        elif method == cls.DEFAULT:
            pass
        else:
            raise("Method {} not implemented.".format(method))
