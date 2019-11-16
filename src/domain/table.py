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
        'de': {'best': 48.47, 'avg': 48.19, 'time': 7.3, 'successful': 1.0},
        'es': {'best': 37.60, 'avg': 37.33, 'time': 9.1, 'successful': 1.0},
        'fi': {'best': 33.50, 'avg': 32.63, 'time': 12.9, 'successful': 1.0},
        'it': {'best': 48.53, 'avg': 48.13, 'time': 8.9, 'successful': 1.0},
    }
    CAPTION = 'The original results were taken from the original paper of \cite{artetxe-etal-2018-robust}. The reproduced results have been generated using the codebase of \cite{artetxe-etal-2018-robust} wrapped around the \texttt{reproduce\_original.sh} from our codebase.'

    def write(self, output_path):
        experiment = self.experiments['Reproduced Results']
        metrics = experiment.aggregate_runs()

        doc = Document(filename='table1', filepath=output_path, doc_type='article', options=('12pt',))
        sec = doc.new_section('Table 1')
        col, row = 17, 4
        table = sec.new(LatexTable(shape=(row, col), alignment=['l'] + ['c'] * 16, float_format='.1f', label='original_results'))
        table.caption = self.CAPTION
        table.label_pos = 'bottom'

        # Main header
        table[0,1:5].multicell(bold('EN-DE'), h_align='c')
        table[0,5:9].multicell(bold('EN-ES'), h_align='c')
        table[0,9:13].multicell(bold('EN-FI'), h_align='c')
        table[0,13:17].multicell(bold('EN-IT'), h_align='c')
        table[0,1:5].add_rule(trim_left=True, trim_right='.3em')
        table[0,5:9].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,9:13].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,13:17].add_rule(trim_left='.3em', trim_right=True)

        # Sub header
        table[1,1:17] = (['best', 'avg', 's', 't'] * 4)
        table[1,0:17].add_rule(trim_left=True, trim_right=True)

        table[2, 0] = 'Original'
        table[2, 1] = self.ORIGINAL_RESULTS['de']['best']
        table[2, 2] = self.ORIGINAL_RESULTS['de']['avg']
        table[2, 3] = self.ORIGINAL_RESULTS['de']['successful']
        table[2, 4] = self.ORIGINAL_RESULTS['de']['time']
        table[2, 5] = self.ORIGINAL_RESULTS['es']['best']
        table[2, 6] = self.ORIGINAL_RESULTS['es']['avg']
        table[2, 7] = self.ORIGINAL_RESULTS['es']['successful']
        table[2, 8] = self.ORIGINAL_RESULTS['es']['time']
        table[2, 9] = self.ORIGINAL_RESULTS['fi']['best']
        table[2, 10] = self.ORIGINAL_RESULTS['fi']['avg']
        table[2, 11] = self.ORIGINAL_RESULTS['fi']['successful']
        table[2, 12] = self.ORIGINAL_RESULTS['fi']['time']
        table[2, 13] = self.ORIGINAL_RESULTS['it']['best']
        table[2, 14] = self.ORIGINAL_RESULTS['it']['avg']
        table[2, 15] = self.ORIGINAL_RESULTS['it']['successful']
        table[2, 16] = self.ORIGINAL_RESULTS['it']['time']

        table[3, 0] = bold('Reproduced')
        table[3, 1] = np.max(metrics['accuracies']['de'])
        table[3, 2] = np.average(metrics['accuracies']['de'])
        table[3, 3] = np.sum(np.array(metrics['accuracies']['de']) > 1.0)/len(metrics['accuracies']['de'])
        table[3, 4] = np.average(metrics['times']['de'])
        table[3, 5] = np.max(metrics['accuracies']['es'])
        table[3, 6] = np.average(metrics['accuracies']['es'])
        table[3, 7] = np.sum(np.array(metrics['accuracies']['es']) > 1.0)/len(metrics['accuracies']['es'])
        table[3, 8] = np.average(metrics['times']['es'])
        table[3, 9] = np.max(metrics['accuracies']['fi'])
        table[3, 10] = np.average(metrics['accuracies']['fi'])
        table[3, 11] = np.sum(np.array(metrics['accuracies']['fi']) > 1.0)/len(metrics['accuracies']['fi'])
        table[3, 12] = np.average(metrics['times']['fi'])
        table[3, 13] = np.max(metrics['accuracies']['it'])
        table[3, 14] = np.average(metrics['accuracies']['it'])
        table[3, 15] = np.sum(np.array(metrics['accuracies']['it']) > 1.0)/len(metrics['accuracies']['it'])
        table[3, 16] = np.average(metrics['times']['it'])

        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[:, col].highlight_best('high', 'bold')

        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[:, col].highlight_best('low', 'bold')

        tex = doc.build(save_to_disk=True, compile_to_pdf=False, show_pdf=False)


class Table2(Table1):

    ORIGINAL_RESULTS = {
        'Full System': {
            'de': {'best': 48.47, 'avg': 48.19, 'time': 7.3, 'successful': 1.0},
            'es': {'best': 37.60, 'avg': 37.33, 'time': 9.1, 'successful': 1.0},
            'fi': {'best': 33.50, 'avg': 32.63, 'time': 12.9, 'successful': 1.0},
            'it': {'best': 48.53, 'avg': 48.13, 'time': 8.9, 'successful': 1.0},
        },
        'Unsup. Init': {
            'de': {'best': 0.00, 'avg': 0.00, 'time': 17.3, 'successful': 0.0},
            'es': {'best': 0.13, 'avg': 0.02, 'time': 15.9, 'successful': 0.0},
            'fi': {'best': 0.07, 'avg': 0.01, 'time': 13.8, 'successful': 0.0},
            'it': {'best': 0.07, 'avg': 0.02, 'time': 16.5, 'successful': 0.0},
        },
        'Stochastic': {
            'de': {'best': 48.13, 'avg': 48.13, 'time': 2.5, 'successful': 1.0},
            'es': {'best': 37.80, 'avg': 37.80, 'time': 2.6, 'successful': 1.0},
            'fi': {'best': 0.28, 'avg': 0.28, 'time': 4.3, 'successful': 0.0},
            'it': {'best': 48.20, 'avg': 48.20, 'time': 2.7, 'successful': 1.0},
        },
        'Cutoff (k=100k)': {
            'de': {'best': 48.27, 'avg': 48.12, 'time': 105.3, 'successful': 1.0},
            'es': {'best': 35.47, 'avg': 34.88, 'time': 185.2, 'successful': 1.0},
            'fi': {'best': 31.95, 'avg': 30.78, 'time': 162.5, 'successful': 1.0},
            'it': {'best': 46.87, 'avg': 46.46, 'time': 114.5, 'successful': 1.0},
        },
        'CSLS': {
            'de': {'best': 0.0, 'avg': 0.0, 'time': 13.8, 'successful': 0.0},
            'es': {'best': 0.0, 'avg': 0.0, 'time': 14.1, 'successful': 0.0},
            'fi': {'best': 0.0, 'avg': 0.0, 'time': 13.1, 'successful': 0.0},
            'it': {'best': 0.0, 'avg': 0.0, 'time': 15.0, 'successful': 0.0},
        },
        'Bidrectional': {
            'de': {'best': 48.27, 'avg': 48.02, 'time': 5.5, 'successful': 1.0},
            'es': {'best': 36.20, 'avg': 35.77, 'time': 7.3, 'successful': 1.0},
            'fi': {'best': 31.39, 'avg': 24.86, 'time': 7.8, 'successful': .8},
            'it': {'best': 46.00, 'avg': 45.37, 'time': 5.6, 'successful': 1.0},
        },
        'Re-weighting': {
            'de': {'best': 48.13, 'avg': 47.41, 'time': 7.0, 'successful': 1.0},
            'es': {'best': 36.00, 'avg': 35.45, 'time': 9.1, 'successful': 1.0},
            'fi': {'best': 32.94, 'avg': 31.77, 'time': 11.2, 'successful': 1.0},
            'it': {'best': 46.07, 'avg': 45.61, 'time': 8.4, 'successful': 1.0},
        },
    }
    CAPTION = 'The original results were taken from the original paper of \cite{artetxe-etal-2018-robust}. The reproduced results have been generated using the codebase of \cite{artetxe-etal-2018-robust} wrapped around the \texttt{reproduce\_original.sh} from our codebase.'

    def write_original_row(self, table, row, original_data):
        table[row, 1] = original_data['de']['best']
        table[row, 2] = original_data['de']['avg']
        table[row, 3] = original_data['de']['successful']
        table[row, 4] = original_data['de']['time']
        table[row, 5] = original_data['es']['best']
        table[row, 6] = original_data['es']['avg']
        table[row, 7] = original_data['es']['successful']
        table[row, 8] = original_data['es']['time']
        table[row, 9] = original_data['fi']['best']
        table[row, 10] = original_data['fi']['avg']
        table[row, 11] = original_data['fi']['successful']
        table[row, 12] = original_data['fi']['time']
        table[row, 13] = original_data['it']['best']
        table[row, 14] = original_data['it']['avg']
        table[row, 15] = original_data['it']['successful']
        table[row, 16] = original_data['it']['time']
        return table

    def write_new_row(self, table, row, metrics):
        table[row, 1] = np.max(metrics['accuracies']['de'])
        table[row, 2] = np.average(metrics['accuracies']['de'])
        table[row, 3] = np.sum(np.array(metrics['accuracies']['de']) > 1.0)/len(metrics['accuracies']['de'])
        table[row, 4] = np.average(metrics['times']['de'])
        table[row, 5] = np.max(metrics['accuracies']['es'])
        table[row, 6] = np.average(metrics['accuracies']['es'])
        table[row, 7] = np.sum(np.array(metrics['accuracies']['es']) > 1.0)/len(metrics['accuracies']['es'])
        table[row, 8] = np.average(metrics['times']['es'])
        table[row, 9] = np.max(metrics['accuracies']['fi'])
        table[row, 10] = np.average(metrics['accuracies']['fi'])
        table[row, 11] = np.sum(np.array(metrics['accuracies']['fi']) > 1.0)/len(metrics['accuracies']['fi'])
        table[row, 12] = np.average(metrics['times']['fi'])
        table[row, 13] = np.max(metrics['accuracies']['it'])
        table[row, 14] = np.average(metrics['accuracies']['it'])
        table[row, 15] = np.sum(np.array(metrics['accuracies']['it']) > 1.0)/len(metrics['accuracies']['it'])
        table[row, 16] = np.average(metrics['times']['it'])
        return table

    def write(self, output_path):
        doc = Document(filename='table2', filepath=output_path, doc_type='article', options=('12pt',))
        sec = doc.new_section('Table 2')
        col, row = 17, 17
        table = sec.new(LatexTable(shape=(row, col), alignment=['l'] + ['c'] * 16, float_format='.1f', label='ablation_study'))
        table.caption = self.CAPTION
        table.label_pos = 'bottom'

        # Main header
        table[0,1:5].multicell(bold('EN-DE'), h_align='c')
        table[0,5:9].multicell(bold('EN-ES'), h_align='c')
        table[0,9:13].multicell(bold('EN-FI'), h_align='c')
        table[0,13:17].multicell(bold('EN-IT'), h_align='c')
        table[0,1:5].add_rule(trim_left=True, trim_right='.3em')
        table[0,5:9].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,9:13].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,13:17].add_rule(trim_left='.3em', trim_right=True)

        # Sub header
        table[1,1:17] = (['best', 'avg', 's', 't'] * 4)
        table[1,0:17].add_rule(trim_left=True, trim_right=True)

        ### Full system metrics
        table[2, 0] = 'Full System'
        table = self.write_original_row(table, 2, self.ORIGINAL_RESULTS['Full System'])
        experiment = self.experiments['Full System']
        metrics = experiment.aggregate_runs()
        table[3, 0] = bold('Reproduced')
        table = self.write_new_row(table, 3, metrics)
        table[3,0:17].add_rule(trim_left=True, trim_right=True)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[2:4, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[2:4, col].highlight_best('low', 'bold')

        ### Unsup. Init
        table[4, 0] = 'Unsup. Init.'
        table = self.write_original_row(table, 4, self.ORIGINAL_RESULTS['Unsup. Init'])
        experiment = self.experiments['Unsup. Init (Random)']
        metrics = experiment.aggregate_runs()
        table[5, 0] = bold('Rand.')
        table = self.write_new_row(table, 5, metrics)
        experiment = self.experiments['Unsup. Init (Random Cutoff)']
        metrics = experiment.aggregate_runs()
        table[6, 0] = bold('Rand. Cut.')
        table = self.write_new_row(table, 6, metrics)
        table[6,0:17].add_rule(trim_left=True, trim_right=True)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[4:7, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[4:7, col].highlight_best('low', 'bold')

        ### Stochastic
        table[7, 0] = 'Stochastic'
        table = self.write_original_row(table, 7, self.ORIGINAL_RESULTS['Stochastic'])
        experiment = self.experiments['Stochastic']
        metrics = experiment.aggregate_runs()
        table[8, 0] = bold('Reproduced')
        table = self.write_new_row(table, 8, metrics)
        table[8,0:17].add_rule(trim_left=True, trim_right=True)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[7:9, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[7:9, col].highlight_best('low', 'bold')

        ### Cutoff
        table[9, 0] = 'Cutoff (k=100k)'
        # table = self.write_original_row(table, 9, self.ORIGINAL_RESULTS['Cutoff (k=100k)'])
        # experiment = self.experiments['Cutoff (k=100k)']
        # metrics = experiment.aggregate_runs()
        table[10, 0] = bold('Reproduced')
        # table = self.write_new_row(table, 10, metrics)
        table[10,0:17].add_rule(trim_left=True, trim_right=True)
        # # Columns that check for higher values
        # for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
        #     table[9:11, col].highlight_best('high', 'bold')
        # # Columns that check for lower values
        # for col in [4, 8, 12, 16]:
        #     table[9:11, col].highlight_best('low', 'bold')

        ### CSLS
        table[11, 0] = 'CSLS'
        table = self.write_original_row(table, 11, self.ORIGINAL_RESULTS['CSLS'])
        experiment = self.experiments['CSLS']
        metrics = experiment.aggregate_runs()
        table[12, 0] = bold('Reproduced')
        table = self.write_new_row(table, 12, metrics)
        table[12,0:17].add_rule(trim_left=True, trim_right=True)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[11:13, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[11:13, col].highlight_best('low', 'bold')

        ### Bidrectional
        table[13, 0] = 'Bidrectional'
        table = self.write_original_row(table, 13, self.ORIGINAL_RESULTS['Bidrectional'])
        experiment = self.experiments['Bidrectional']
        metrics = experiment.aggregate_runs()
        table[14, 0] = bold('Reproduced')
        table = self.write_new_row(table, 14, metrics)
        table[14,0:17].add_rule(trim_left=True, trim_right=True)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[13:15, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[13:15, col].highlight_best('low', 'bold')

        ### Re-weighting
        table[15, 0] = 'Re-weighting'
        table = self.write_original_row(table, 15, self.ORIGINAL_RESULTS['Re-weighting'])
        experiment = self.experiments['Re-weighting']
        metrics = experiment.aggregate_runs()
        table[16, 0] = bold('Reproduced')
        table = self.write_new_row(table, 16, metrics)
        # Columns that check for higher values
        for col in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            table[15:17, col].highlight_best('high', 'bold')
        # Columns that check for lower values
        for col in [4, 8, 12, 16]:
            table[15:17, col].highlight_best('low', 'bold')


        tex = doc.build(save_to_disk=True, compile_to_pdf=False, show_pdf=False)


class Table3(Table):

    CAPTION = 'The original results were taken from the original paper of \cite{artetxe-etal-2018-robust}. The reproduced results have been generated using the codebase of \cite{artetxe-etal-2018-robust} wrapped around the \texttt{reproduce\_original.sh} from our codebase.'

    def write(self, output_path):
        experiment = self.experiments['Other Languages']
        metrics = experiment.aggregate_runs()

        doc = Document(filename='table3', filepath=output_path, doc_type='article', options=('12pt',))
        sec = doc.new_section('Table 3')
        col, row = 17, 3
        table = sec.new(LatexTable(shape=(row, col), alignment=['l'] + ['c'] * 16, float_format='.1f', label='other_languages_results'))
        table.caption = self.CAPTION
        table.label_pos = 'bottom'

        # Main header
        table[0,1:5].multicell(bold('EN-ET'), h_align='c')
        table[0,5:9].multicell(bold('EN-FA'), h_align='c')
        table[0,9:13].multicell(bold('EN-LV'), h_align='c')
        table[0,13:17].multicell(bold('EN-VI'), h_align='c')
        table[0,1:5].add_rule(trim_left=True, trim_right='.3em')
        table[0,5:9].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,9:13].add_rule(trim_left='.3em', trim_right='.3em')
        table[0,13:17].add_rule(trim_left='.3em', trim_right=True)

        # Sub header
        table[1,1:17] = (['best', 'avg', 's', 't'] * 4)
        table[1,0:17].add_rule(trim_left=True, trim_right=True)

        table[2, 0] = bold('Vecmap')
        table[2, 1] = np.max(metrics['accuracies']['et'])
        table[2, 2] = np.average(metrics['accuracies']['et'])
        table[2, 3] = np.sum(np.array(metrics['accuracies']['et']) > 1.0)/len(metrics['accuracies']['et'])
        table[2, 4] = np.average(metrics['times']['et'])
        table[2, 5] = np.max(metrics['accuracies']['fa'])
        table[2, 6] = np.average(metrics['accuracies']['fa'])
        table[2, 7] = np.sum(np.array(metrics['accuracies']['fa']) > 1.0)/len(metrics['accuracies']['fa'])
        table[2, 8] = np.average(metrics['times']['fa'])
        table[2, 9] = np.max(metrics['accuracies']['lv'])
        table[2, 10] = np.average(metrics['accuracies']['lv'])
        table[2, 11] = np.sum(np.array(metrics['accuracies']['lv']) > 1.0)/len(metrics['accuracies']['lv'])
        table[2, 12] = np.average(metrics['times']['lv'])
        table[2, 13] = np.max(metrics['accuracies']['vi'])
        table[2, 14] = np.average(metrics['accuracies']['vi'])
        table[2, 15] = np.sum(np.array(metrics['accuracies']['vi']) > 1.0)/len(metrics['accuracies']['vi'])
        table[2, 16] = np.average(metrics['times']['vi'])

        tex = doc.build(save_to_disk=True, compile_to_pdf=False, show_pdf=False)


class Table4(Table):
    def write(self, output_path):
        pass

def get_table1(configs) -> Table:
    return Table1({"Reproduced Results": OriginalExperiment(configs)})

def get_table2(configs):
    return Table2({
        "Full System": OriginalExperiment(configs),
        "Unsup. Init (Random)": RandomSeedDictionaryAblationExperiment(configs),
        "Unsup. Init (Random Cutoff)": RandomCutoffSeedDictionaryAblationExperiment(configs),
        "Stochastic": StochasticAblationExperiment(configs),
        "Cutoff (k=100k)": VocabularyCutOffAblationExperiment(configs),
        "CSLS": CSLSAblationExperiment(configs),
        "Bidrectional": DirectionAblationExperiment(configs),
        "Re-weighting": ReweightAblationExperiment(configs),
    })

def get_table3(configs) -> Table:
    return Table3({
        "Other Languages": OtherLanguagesOriginalExperiment(configs),
        "Other Languages Stochastic": OtherLanguagesStochasticExperiment(configs),
    })

def get_table4(configs) -> Table:
    return Table4({
        "CSLS": CSLSGridSearchExperiment(configs),
        "Vocabulary cutoff": VocabularyCutoffGridSearchExperiment(configs),
        "Stochastic": StochasticGridSearchExperiment(configs)
    })