
import copy


# Dictionary of preset hyperparameter configurations.
hyperparameter_config_dict = dict(
    # Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
    default={
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
        'custom': ['GBM'],
    },
    # Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
    light={
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'max_depth': 15, 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'max_depth': 15, 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'max_depth': 15, 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'max_depth': 15, 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'max_depth': 15, 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'max_depth': 15, 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'custom': ['GBM'],
    },
    # Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
    very_light={
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
    },
    # Results in extremely quick to train models. Only use this when prototyping, as the model accuracy will be severely reduced.
    toy={
        'NN': {'num_epochs': 10},
        'GBM': {'num_boost_round': 10},
        'CAT': {'iterations': 10},
        'XGB': {'n_estimators': 10},
    },
    # Default AutoGluon hyperparameters intended to maximize accuracy in multimodal tabular + text datasets. Requires GPU.
    multimodal={
        'NN': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        ],
        'CAT': {},
        'XGB': {},
        # 'FASTAI': {},  # FIXME: FASTAI is extremely slow if trained alongside TextNN due to forkserver.
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
        'TEXT_NN_V1': ['ELECTRA_BASE'],
        'custom': ['GBM'],
    },
)


def get_hyperparameter_config_options():
    return list(hyperparameter_config_dict.keys())


def get_hyperparameter_config(config_name):
    config_options = get_hyperparameter_config_options()
    if config_name not in config_options:
        raise ValueError(f'Valid hyperparameter config names are: {config_options}, but \'{config_name}\' was given instead.')
    return copy.deepcopy(hyperparameter_config_dict[config_name])
