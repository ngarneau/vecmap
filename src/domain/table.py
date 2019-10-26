from collections import defaultdict
from typing import Dict, List, Iterable, Tuple

import mlflow

from src.domain.experiment import *

class Table:
    def __init__(self, experiments: Dict[str, Experiment]):
        self.experiments = experiments

    def get_experiments(self) -> Iterable[Tuple[str, Experiment]]:
        for experiment_name, experiment in self.experiments.items():
            yield experiment_name, experiment

    def aggregate_runs_by_language(self, runs):
        accuracies = defaultdict(list)
        times = defaultdict(list)
        for run in runs:
            minutes = ((run.info.end_time - run.info.start_time)//60//60)%60
            languages[run.data.params['target_language']].append(run.data.metrics['accuracy'])
            languages[run.data.params['target_language']].append(minutes)
        return {
            'accuracies': accuracies,
            'times': times
        }


    def write(self, output_path):
        raise NotImplementedError("Writing should be implemented by child class")


class Table1(Table):

    ORIGINAL_RESULTS = {}  # TODO: store results in the same way we retrieve them

    def write(self, output_path):
        experiment = self.experiments['Reproduced Results']
        runs = experiment.get_runs()
        metrics = self.aggregate_runs_by_language(runs)


class Table2(Table1):
    def write(self, output_path):
        # TODO
        pass


def get_table1(configs):
    return Table1({"Reproduced Results": OriginalExperiment(configs)})

def get_table2(configs):
    return Table2({
        "Full System": OriginalExperiment(configs),
        "Unsup. Init (Random)": RandomSeedDictionaryAblationExperiment(configs),
        "Unsup. Init (Random Cutoff)": RandomCutoffSeedDictionaryAblationExperiment(configs),
        "Stochastic ": StochasticAblationExperiment(configs),
        "Cutoff (k=100k)": VocabularyCutOffAblationExperiment(configs),
        "CSLS": CSLSAblationExperiment(configs),
        "Bidrectional": DirectionAblationExperiment(configs),
        "Re-weighting": ReweightAblationExperiment(configs),
    })
