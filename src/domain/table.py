from typing import Dict, List, Iterable, Tuple

import numpy as np
import mlflow
from python2latex import Document, Table as LatexTable, build, bold

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

    ORIGINAL_RESULTS = {
        'de': {
            'best': 48.47,
            'avg': 48.19,
            'time': 7.3
        },
        'es': {
            'best': 37.60,
            'avg': 37.33,
            'time': 9.1
        },
        'fi': {
            'best': 33.50,
            'avg': 32.63,
            'time': 12.9
        },
        'it': {
            'best': 48.53,
            'avg': 48.13,
            'time': 8.9
        },
    }
    CAPTION = 'The original results were taken from the original paper of \cite{artetxe-etal-2018-robust}. The reproduced results have been generated using the codebase of \cite{artetxe-etal-2018-robust} wrapped around the \texttt{reproduce\_original.sh} from our codebase.'

    def write(self, output_path):
        experiment = self.experiments['Reproduced Results']
        metrics = experiment.aggregate_runs()

        doc = Document(filename='table1', filepath=output_path, doc_type='article', options=('12pt',))
        sec = doc.new_section('Table 1')
        col, row = 13, 4
        table = sec.new(LatexTable(shape=(row, col), alignment=['l'] + ['c'] * 12, float_format='.2f', label='original_results'))
        table.caption = self.CAPTION

        # Main header
        table[0,1:4].multicell(bold('EN-DE'), h_align='c')
        table[0,4:7].multicell(bold('EN-ES'), h_align='c')
        table[0,7:10].multicell(bold('EN-FI'), h_align='c')
        table[0,10:13].multicell(bold('EN-IT'), h_align='c')
        table[0,1:4].add_rule(trim_left=True, trim_right='.3em')
        table[0,4:7].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,7:10].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,10:13].add_rule(trim_left='.3em', trim_right=True)

        # Sub header
        table[1,1:13] = (['best', 'avg', 'time'] * 4)
        table[1,0:13].add_rule(trim_left=True, trim_right=True)

        table[2, 0] = 'Original Results'
        table[2, 1] = self.ORIGINAL_RESULTS['de']['best']
        table[2, 2] = self.ORIGINAL_RESULTS['de']['avg']
        table[2, 3] = self.ORIGINAL_RESULTS['de']['time']
        table[2, 4] = self.ORIGINAL_RESULTS['es']['best']
        table[2, 5] = self.ORIGINAL_RESULTS['es']['avg']
        table[2, 6] = self.ORIGINAL_RESULTS['es']['time']
        table[2, 7] = self.ORIGINAL_RESULTS['fi']['best']
        table[2, 8] = self.ORIGINAL_RESULTS['fi']['avg']
        table[2, 9] = self.ORIGINAL_RESULTS['fi']['time']
        table[2, 10] = self.ORIGINAL_RESULTS['it']['best']
        table[2, 11] = self.ORIGINAL_RESULTS['it']['avg']
        table[2, 12] = self.ORIGINAL_RESULTS['it']['time']

        table[3, 0] = bold('Reproduced results')
        table[3, 1] = np.max(metrics['accuracies']['de'])
        table[3, 2] = np.average(metrics['accuracies']['de'])
        table[3, 3] = np.average(metrics['times']['de'])
        table[3, 4] = np.max(metrics['accuracies']['es'])
        table[3, 5] = np.average(metrics['accuracies']['es'])
        table[3, 6] = np.average(metrics['times']['es'])
        table[3, 7] = np.max(metrics['accuracies']['fi'])
        table[3, 8] = np.average(metrics['accuracies']['fi'])
        table[3, 9] = np.average(metrics['times']['fi'])
        table[3, 10] = np.max(metrics['accuracies']['it'])
        table[3, 11] = np.average(metrics['accuracies']['it'])
        table[3, 12] = np.average(metrics['times']['it'])

        tex = doc.build(save_to_disk=True, compile_to_pdf=False, show_pdf=False)


class Table2(Table1):
    def write(self, output_path):
        # TODO
        pass


def get_table1(configs) -> Table:
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
