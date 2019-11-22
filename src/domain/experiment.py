from collections import defaultdict
import mlflow
from mlflow.tracking import MlflowClient

class Experiment:

    EXPERIMENT_NAME = 'None'
    FIXED_PARAMS = {}
    LANGUAGE_PARAMS = {}
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        self.base_config = base_config
        self.mlflow_client = MlflowClient(tracking_uri=base_config['mlflow_output_uri'])

    def get_runs(self):
        mlflow_experiment = self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME)
        return self.mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id])

    def get_parameters_combinations(self):
        run_params = self.base_config
        run_params.update(self.FIXED_PARAMS)
        for source_language in self.LANGUAGE_PARAMS['source_language']:
            for target_language in self.LANGUAGE_PARAMS['target_language']:
                if source_language != target_language:
                    run_params['source_language'] = source_language
                    run_params['target_language'] = target_language
                    if len(self.CHANGING_PARAMS) > 0:
                        for ablated_param, param_values in self.CHANGING_PARAMS.items():
                            for param_value in param_values:
                                # Apply the language params combination
                                run_params[ablated_param] = param_value
                                yield run_params
                    else:
                        yield run_params


class OriginalExperiment(Experiment):
    EXPERIMENT_NAME = 'original'
    FIXED_PARAMS = {
        'stochastic_initial': 0.1,
        'vocabulary_cutoff': 20000,
        'csls': 10,
        'direction': 'union',
        'reweight': 0.5,
        'seed_dictionary_method': 'unsupervised'
    }
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['de', 'es', 'fi', 'it']
    }
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        super().__init__(base_config)


    def __is_a_valid_run(self, run):
        return (
            'target_language' in run.data.params
            and 'source_language' in run.data.params
            and run.data.params['target_language'] in self.LANGUAGE_PARAMS['target_language']
            and run.data.params['source_language'] in self.LANGUAGE_PARAMS['source_language']
            and 'accuracy' in run.data.metrics
        )

    def aggregate_runs(self):
        runs = self.get_runs()
        accuracies = defaultdict(list)
        times = defaultdict(list)
        for run in runs:
            if self.__is_a_valid_run(run):
                minutes = ((run.info.end_time - run.info.start_time)//60//60)%60
                accuracies[run.data.params['target_language']].append(run.data.metrics['accuracy'] * 100)
                times[run.data.params['target_language']].append(minutes)
        return {
            'accuracies': accuracies,
            'times': times
        }




class StochasticAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'stochastic_ablation'
    CHANGING_PARAMS = {
        'stochastic_initial': [1.0],
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class VocabularyCutOffAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'vocabulary_cutoff_ablation'
    CHANGING_PARAMS = {
        'vocabulary_cutoff': [0],
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class CSLSAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'csls_ablation'
    CHANGING_PARAMS = {
        'csls': [0],
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class DirectionAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'direction_ablation'
    CHANGING_PARAMS = {
        'direction': ['forward'],
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class ReweightAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'reweight_ablation'
    CHANGING_PARAMS = {
        'reweight': [1.0],
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class RandomSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'random_seed_dictionary_ablation'
    CHANGING_PARAMS = {
        'seed_dictionary_method': ['random_raw']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class RandomCutoffSeedDictionaryAblationExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'random_cutoff_seed_dictionary_ablation'
    CHANGING_PARAMS = {
        'seed_dictionary_method': ['random_cutoff']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesOriginalExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'other_languages_original'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesStochasticExperiment(StochasticAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_stochastic'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesCSLSAblationExperiment(CSLSAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_csls_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesDirectionAblationExperiment(DirectionAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_direction_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesReweightAblationExperiment(ReweightAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_reweight_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesRandomSeedDictionaryAblationExperiment(RandomSeedDictionaryAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_random_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class OtherLanguagesRandomCutoffSeedDictionaryAblationExperiment(RandomCutoffSeedDictionaryAblationExperiment):
    EXPERIMENT_NAME = 'other_languages_random_cutoff_seed_dictionary_ablation'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': ['et', 'fa', 'lv', 'vi']
    }
    def __init__(self, base_config):
        super().__init__(base_config)


class HyperparametersExperiment(OriginalExperiment):
    EXPERIMENT_NAME = 'hyperparameters'
    LANGUAGE_PARAMS = {
        'source_language': ['en'],
        'target_language': [
            'de',
            'es',
            'et',
            'fa',
            'fi',
            'it'
            'lv',
            'vi'
        ]
    }

    # TODO Change a bunch of params here
    CHANGING_PARAMS = {}

    def __init__(self, base_config):
        super().__init__(base_config)

