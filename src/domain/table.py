from typing import Dict, List, Iterable, Tuple

from src.domain.experiment import *

class Table:
    def __init__(self, experiments: Dict[str, Experiment]):
        self.experiments = experiments

    def get_experiments(self) -> Iterable[Tuple[str, Experiment]]:
        for experiment_name, experiment in self.experiments.items():
            yield experiment_name, experiment

    def write(self, output_path):
        raise NotImplementedError("Writing should be implemented by child class")


class Table1(Table):

    ORIGINAL_RESULTS = {}  # TODO: store results in the same way we retrieve them

    def write(self, output_path):
        # TODO
        pass


class Table2(Table1):
    def write(self, output_path):
        # TODO
        pass


def get_table1():
    return Table1({"Reproduced Results": OriginalExperiment})

def get_table2():
    return Table2({
        "Full System": OriginalExperiment,
        "Unsup. Init (Random)": RandomSeedDictionaryAblationExperiment,
        "Unsup. Init (Random Cutoff)": RandomCutoffSeedDictionaryAblationExperiment,
        "Stochastic ": StochasticAblationExperiment,
        "Cutoff (k=100k)": VocabularyCutOffAblationExperiment,
        "CSLS": CSLSAblationExperiment,
        "Bidrectional": DirectionAblationExperiment,
        "Re-weighting": ReweightAblationExperiment,
    })
