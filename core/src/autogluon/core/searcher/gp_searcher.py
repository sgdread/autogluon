import ConfigSpace as CS
import multiprocessing as mp

from .bayesopt.autogluon.searcher_factory import gp_fifo_searcher_factory, \
    gp_multifidelity_searcher_factory, gp_fifo_searcher_defaults, \
    gp_multifidelity_searcher_defaults
from .searcher import BaseSearcher
from ..utils.default_arguments import check_and_merge_defaults

__all__ = ['GPFIFOSearcher',
           'GPMultiFidelitySearcher']


def _to_config_cs(config_space: CS.ConfigurationSpace, config: dict) \
        -> CS.Configuration:
    return CS.Configuration(config_space, values=config)


class GPFIFOSearcher(BaseSearcher):
    """Gaussian process Bayesian optimization for FIFO scheduler

    This searcher must be used with `FIFOScheduler`. It provides Bayesian
    optimization, based on a Gaussian process surrogate model. It is created
    along with the scheduler, using `searcher='bayesopt'`:

    Pending configurations (for which evaluation tasks are currently running)
    are dealt with by fantasizing (i.e., target values are drawn from the
    current posterior, and acquisition functions are averaged over this
    sample, see `num_fantasy_samples`).
    The GP surrogate model uses a Matern 5/2 covariance function with automatic
    relevance determination (ARD) of input attributes, and a constant mean
    function. The acquisition function is expected improvement (EI). All
    hyperparameters of the surrogate model are estimated by empirical Bayes
    (maximizing the marginal likelihood). In general, this hyperparameter
    fitting is the most expensive part of a `get_config` call.

    The following happens in `get_config`. For the first `num_init_random` calls,
    a config is drawn at random (the very first call results in the default
    config of the space). Afterwards, Bayesian optimization is used, unless
    there are no finished evaluations yet.
    First, model hyperparameter are refit. This step can be skipped (see
    `opt_skip*` parameters). Next, `num_init_candidates` configs are sampled at
    random, and ranked by a scoring function (`initial_scoring`). BFGS local
    optimization is then run starting from the top scoring config, where EI
    is minimized.

    Parameters
    ----------
    configspace : ConfigSpace.ConfigurationSpace
        Config space of `train_fn`, equal to `train_fn.cs`
    reward_attribute : str
        Name of reward attribute reported by `train_fn`, equal to `reward_attr`
        of `scheduler
    debug_log : bool (default: False)
        If True, both searcher and scheduler output an informative log, from
        which the configs chosen and decisions being made can be traced.
    first_is_default : bool (default: True)
        If True, the first config to be evaluated is the default one of the
        config space. Otherwise, this first config is drawn at random.
    random_seed : int
        Seed for pseudo-random number generator used.
    num_init_random : int
        Number of initial `get_config` calls for which randomly sampled configs
        are returned. Afterwards, Bayesian optimization is used
    num_init_candidates : int
        Number of initial candidates sampled at random in order to seed the
        search for `get_config`
    num_fantasy_samples : int
        Number of samples drawn for fantasizing (latent target values for
        pending candidates)
    initial_scoring : str
        Scoring function to rank initial candidates (local optimization of EI
        is started from top scorer). Values are 'thompson_indep' (independent
        Thompson sampling; randomized score, which can increase exploration),
        'acq_func' (score is the same (EI) acquisition function which is afterwards
        locally optimized).
    opt_nstarts : int
        Parameter for hyperparameter fitting. Number of random restarts
    opt_maxiter : int
        Parameter for hyperparameter fitting. Maximum number of iterations
        per restart
    opt_warmstart : bool
        Parameter for hyperparameter fitting. If True, each fitting is started
        from the previous optimum. Not recommended in general
    opt_verbose : bool
        Parameter for hyperparameter fitting. If True, lots of output
    opt_skip_init_length : int
        Parameter for hyperparameter fitting, skip predicate. Fitting is never
        skipped as long as number of observations below this threshold
    opt_skip_period : int
        Parameter for hyperparameter fitting, skip predicate. If >1, and number
        of observations above `opt_skip_init_length`, fitting is done only
        K-th call, and skipped otherwise
    map_reward : str or MapReward (default: '1_minus_x')
        AutoGluon is maximizing reward, while internally, Bayesian optimization
        is minimizing the criterion. States how reward is mapped to criterion.
        This must a strictly decreasing function. Values are '1_minus_x'
        (criterion = 1 - reward), 'minus_x' (criterion = -reward).
        From a technical standpoint, it does not matter what is chosen here,
        because criterion is only used internally. Also note that criterion
        data is always normalized to mean 0, variance 1 before fitted with a
        GP.

    Examples
    --------
    >>> import autogluon.core as ag
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True))
    >>> def train_fn(args, reporter):
    ...     reporter(accuracy = args.lr ** 2)
    >>> searcher_options = {
    ...     'map_reward': 'minus_x',
    ...     'opt_skip_period': 2}
    >>> scheduler = ag.scheduler.FIFOScheduler(
    ...     train_fn, searcher='bayesopt', searcher_options=searcher_options,
    ...     num_trials=10, reward_attr='accuracy')
    """
    def __init__(self, configspace, **kwargs):
        _gp_searcher = kwargs.get('_gp_searcher')
        if _gp_searcher is None:
            kwargs['configspace'] = configspace
            _kwargs = check_and_merge_defaults(
                kwargs, *gp_fifo_searcher_defaults(),
                dict_name='search_options')
            _gp_searcher = gp_fifo_searcher_factory(**_kwargs)
        super().__init__(
            _gp_searcher.hp_ranges.config_space,
            reward_attribute=kwargs.get('reward_attribute'))
        self.gp_searcher = _gp_searcher
        # This lock protects gp_searcher. We are not using self.LOCK, this
        # can lead to deadlocks when superclass methods are called
        self._gp_lock = mp.RLock()

    def configure_scheduler(self, scheduler):
        from ..scheduler import FIFOScheduler

        assert isinstance(scheduler, FIFOScheduler), \
            "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)

    def get_config(self, **kwargs):
        print('GPFIFOSearcher.get_config waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.get_config got lock')
            config_cs = self.gp_searcher.get_config()
        print('GPFIFOSearcher.get_config released lock')
        return config_cs.get_dictionary()

    def update(self, config, **kwargs):
        super().update(config, **kwargs)
        print('GPFIFOSearcher.update waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.update got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.update(
                config_cs, reward=kwargs[self._reward_attribute])
        print('GPFIFOSearcher.update released lock')

    def register_pending(self, config, milestone=None):
        print('GPFIFOSearcher.register_pending waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.register_pending got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.register_pending(config_cs)
        print('GPFIFOSearcher.register_pending released lock')

    def evaluation_failed(self, config, **kwargs):
        print('GPFIFOSearcher.evaluation_failed waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.evaluation_failed got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.evaluation_failed(config_cs)
        print('GPFIFOSearcher.evaluation_failed released lock')

    def dataset_size(self):
        print('GPFIFOSearcher.dataset_size waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.dataset_size got lock')
            result = self.gp_searcher.dataset_size()
        print('GPFIFOSearcher.dataset_size released lock')
        return result

    def cumulative_profile_record(self):
        print('GPFIFOSearcher.cumulative_profile_record waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.cumulative_profile_record got lock')
            result = self.gp_searcher.cumulative_profile_record()
        print('GPFIFOSearcher.cumulative_profile_record released lock')
        return result

    def model_parameters(self):
        print('GPFIFOSearcher.model_parameters waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.model_parameters got lock')
            result = self.gp_searcher.get_params()
        print('GPFIFOSearcher.model_parameters released lock')
        return result

    def get_state(self):
        print('GPFIFOSearcher.get_state waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.get_state got lock')
            result = self.gp_searcher.get_state()
        print('GPFIFOSearcher.get_state released lock')
        return result

    def clone_from_state(self, state):
        print('GPFIFOSearcher.clone_from_state waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.clone_from_state got lock')
            _gp_searcher = self.gp_searcher.clone_from_state(state)
        print('GPFIFOSearcher.clone_from_state released lock')
        # Use copy constructor
        return GPFIFOSearcher(
            reward_attribute=self._reward_attribute,
            _gp_searcher=_gp_searcher)

    @property
    def debug_log(self):
        print('GPFIFOSearcher.debug_log waiting lock')
        with self._gp_lock:
            print('GPFIFOSearcher.debug_log got lock')
            result = self.gp_searcher.debug_log
        print('GPFIFOSearcher.debug_log released lock')
        return result

    def _to_config_cs(self, config):
        return _to_config_cs(self.gp_searcher.hp_ranges.config_space, config)


class GPMultiFidelitySearcher(BaseSearcher):
    """Gaussian process Bayesian optimization for Hyperband scheduler

    This searcher must be used with `HyperbandScheduler`. It provides a novel
    combination of Bayesian optimization, based on a Gaussian process surrogate
    model, with Hyperband scheduling. In particular, observations across
    resource levels are modelled jointly. It is created along with the
    scheduler, using `searcher='bayesopt'`:

    Most of `GPFIFOSearcher` comments apply here as well.
    In multi-fidelity HPO, we optimize a function f(x, r), x the configuration,
    r the resource (or time) attribute. The latter must be a positive integer.
    In most applications, `time_attr` == 'epoch', and the resource is the number
    of epochs already trained.

    We model the function f(x, r) jointly over all resource levels r at which
    it is observed (but see `searcher_data` in `HyperbandScheduler`). The kernel
    and mean function of our surrogate model are over (x, r). The surrogate
    model is selected by `gp_resource_kernel`. More details about the supported
    kernels is in:

        Tiao, Klein, Lienart, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter and Neural Architecture Search
        https://arxiv.org/abs/2003.10865

    The acquisition function (EI) which is optimized in `get_config`, is obtained
    by fixing the resource level r to a value which is determined depending on
    the current state. If `resource_acq` == 'bohb', r is the largest value
    <= max_t, where we have seen >= dimension(x) metric values. If
    `resource_acq` == 'first', r is the first milestone which config x would
    reach when started.

    Parameters
    ----------
    configspace : ConfigSpace.ConfigurationSpace
        Config space of `train_fn`, equal to `train_fn.cs`
    reward_attribute : str
        Name of reward attribute reported by `train_fn`, equal to `reward_attr`
        of scheduler
    resource_attribute : str
        Name of resource (or time) attribute reported by `train_fn`, equal to
        `time_attr` of scheduler
    debug_log : bool (default: False)
        If True, both searcher and scheduler output an informative log, from
        which the configs chosen and decisions being made can be traced.
    first_is_default : bool (default: True)
        If True, the first config to be evaluated is the default one of the
        config space. Otherwise, this first config is drawn at random.
    random_seed : int
        Seed for pseudo-random number generator used.
    num_init_random : int
        See `GPFIFOSearcher`
    num_init_candidates : int
        See `GPFIFOSearcher`
    num_fantasy_samples : int
        See `GPFIFOSearcher`
    initial_scoring : str
        See `GPFIFOSearcher`
    opt_nstarts : int
        See `GPFIFOSearcher`
    opt_maxiter : int
        See `GPFIFOSearcher`
    opt_warmstart : bool
        See `GPFIFOSearcher`
    opt_verbose : bool
        See `GPFIFOSearcher`
    opt_skip_init_length : int
        See `GPFIFOSearcher`
    opt_skip_period : int
        See `GPFIFOSearcher`
    map_reward : str or MapReward (default: '1_minus_x')
        See `GPFIFOSearcher`
    gp_resource_kernel : str
        Surrogate model over criterion function f(x, r), x the config, r the
        resource. Note that x is encoded to be a vector with entries in [0, 1],
        and r is linearly mapped to [0, 1], while the criterion data is
        normalized to mean 0, variance 1. The reference above provides details
        on the models supported here. For the exponential decay kernel, the
        base kernel over x is Matern 5/2 ARD.
        Values are 'matern52' (Matern 5/2 ARD kernel over [x, r]),
        'matern52-res-warp' (Matern 5/2 ARD kernel over [x, r], with additional
        warping on r),
        'exp-decay-sum' (exponential decay kernel, with delta=0. This is the
        additive kernel from Freeze-Thaw Bayesian Optimization),
        'exp-decay-delta1' (exponential decay kernel, with delta=1),
        'exp-decay-combined' (exponential decay kernel, with delta in [0, 1]
        a hyperparameter).
    resource_acq : str
        Determines how the EI acquisition function is used (see above).
        Values: 'bohb', 'first'
    opt_skip_num_max_resource : bool
        Parameter for hyperparameter fitting, skip predicate. If True, and
        number of observations above `opt_skip_init_length`, fitting is done
        only when there is a new datapoint at r = max_t, and skipped otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import autogluon.core as ag
    >>>
    >>> @ag.args(
    ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
    ...     wd=ag.space.Real(1e-3, 1e-2))
    >>> def train_fn(args, reporter):
    ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
    ...     for e in range(9):
    ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
    ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr,
    ...         wd=args.wd)
    >>> searcher_options = {
    ...     'gp_resource_kernel': 'matern52-res-warp',
    ...     'opt_skip_num_max_resource': True}
    >>> scheduler = ag.scheduler.HyperbandScheduler(
    ...     train_fn, searcher='bayesopt', searcher_options=searcher_options,
    ...     num_trials=10, reward_attr='accuracy', time_attr='epoch',
    ...     max_t=10, grace_period=1, reduction_factor=3)

    See Also
    --------
    GPFIFOSearcher
    """
    def __init__(self, configspace, **kwargs):
        _gp_searcher = kwargs.get('_gp_searcher')
        if _gp_searcher is None:
            kwargs['configspace'] = configspace
            _kwargs = check_and_merge_defaults(
                kwargs, *gp_multifidelity_searcher_defaults(),
                dict_name='search_options')
            _gp_searcher = gp_multifidelity_searcher_factory(**_kwargs)
        super().__init__(
            _gp_searcher.hp_ranges.config_space,
            reward_attribute=kwargs.get('reward_attribute'))
        self.gp_searcher = _gp_searcher
        self._resource_attribute = kwargs.get('resource_attribute')
        # This lock protects gp_searcher. We are not using self.LOCK, this
        # can lead to deadlocks when superclass methods are called
        self._gp_lock = mp.Lock()

    def configure_scheduler(self, scheduler):
        from ..scheduler import HyperbandScheduler

        assert isinstance(scheduler, HyperbandScheduler), \
            "This searcher requires HyperbandScheduler scheduler"
        super().configure_scheduler(scheduler)
        print('GPMultiFidelitySearcher.configure_scheduler waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.configure_scheduler got lock')
            self.gp_searcher.set_map_resource_to_index(
                scheduler.map_resource_to_index())
        print('GPMultiFidelitySearcher.configure_scheduler released lock')
        self._resource_attribute = scheduler._time_attr

    def get_config(self, **kwargs):
        print('GPMultiFidelitySearcher.get_config waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.get_config got lock')
            config_cs = self.gp_searcher.get_config(**kwargs)
        print('GPMultiFidelitySearcher.get_config released lock')
        return config_cs.get_dictionary()

    def update(self, config, **kwargs):
        super().update(config, **kwargs)
        print('GPMultiFidelitySearcher.update waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.update released lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.update(
                config_cs, reward=kwargs[self._reward_attribute],
                resource=int(kwargs[self._resource_attribute]))
            # If evaluation task has terminated, cleanup pending evaluations
            # which may have been overlooked
            if kwargs.get('terminated', False):
                self.gp_searcher.cleanup_pending(config_cs)
        print('GPMultiFidelitySearcher.update got lock')

    def register_pending(self, config, milestone=None):
        assert milestone is not None, \
            "This searcher works with a multi-fidelity scheduler only"
        print('GPMultiFidelitySearcher.register_pending waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.register_pending got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.register_pending(config_cs, milestone)
        print('GPMultiFidelitySearcher.register_pending released lock')

    def remove_case(self, config, **kwargs):
        print('GPMultiFidelitySearcher.remove_case waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.remove_case got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.remove_case(
                config_cs, resource=int(kwargs[self._resource_attribute]))
        print('GPMultiFidelitySearcher.remove_case released lock')

    def evaluation_failed(self, config, **kwargs):
        print('GPMultiFidelitySearcher.evaluation_failed waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.evaluation_failed got lock')
            config_cs = self._to_config_cs(config)
            self.gp_searcher.evaluation_failed(config_cs)
        print('GPMultiFidelitySearcher.evaluation_failed released lock')

    def dataset_size(self):
        print('GPMultiFidelitySearcher.dataset_size waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.dataset_size got lock')
            result = self.gp_searcher.dataset_size()
        print('GPMultiFidelitySearcher.dataset_size released lock')
        return result

    def cumulative_profile_record(self):
        print('GPMultiFidelitySearcher.cumulative_profile_record waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.cumulative_profile_record got lock')
            result = self.gp_searcher.cumulative_profile_record()
        print('GPMultiFidelitySearcher.cumulative_profile_record released lock')
        return result

    def model_parameters(self):
        print('GPMultiFidelitySearcher.model_parameters waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.model_parameters got lock')
            result = self.gp_searcher.get_params()
        print('GPMultiFidelitySearcher.model_parameters released lock')
        return result

    def get_state(self):
        print('GPMultiFidelitySearcher.get_state waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.get_state got lock')
            result = self.gp_searcher.get_state()
        print('GPMultiFidelitySearcher.get_state released lock')
        return result

    def clone_from_state(self, state):
        print('GPMultiFidelitySearcher.clone_from_state waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.clone_from_state got lock')
            _gp_searcher = self.gp_searcher.clone_from_state(state)
        print('GPMultiFidelitySearcher.clone_from_state released lock')
        # Use copy constructor
        return GPMultiFidelitySearcher(
            reward_attribute=self._reward_attribute,
            resource_attribute=self._resource_attribute,
            _gp_searcher=_gp_searcher)

    @property
    def debug_log(self):
        print('GPMultiFidelitySearcher.debug_log waiting lock')
        with self._gp_lock:
            print('GPMultiFidelitySearcher.debug_log got lock')
            result = self.gp_searcher.debug_log
        print('GPMultiFidelitySearcher.debug_log released lock')
        return result

    def _to_config_cs(self, config):
        return _to_config_cs(self.gp_searcher.hp_ranges.config_space, config)
