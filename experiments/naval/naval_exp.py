"""
Naval
"""
import sys
sys.path.insert(0, '~/bnn-analysis/')
sys.path.insert(0, '~/bnn-analysis/models')
sys.path.insert(0, '~/bnn-analysis/experiments')
import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict

from naval_env import NavalEnv
from experiment import Experiment
from hmc_samplers import HMCSampler, SGHMCSampler
from ld_samplers import LDSampler, SGLDSampler, pSGLDSampler


class NavalExp(Experiment):
    def __init__(self):
        super().__init__()

    def _setup_sampler_defaults(self, sampler_params):
        sampler_params['noise_precision'] = 5.
        sampler_params['weights_precision'] = 1.

    def run_baseline_hmc(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'hmc'
        env.test_case_name = 'baseline'

        env.chains_num = 1
        env.n_samples = 100
        env.thinning = 4

        sampler_params = dict()
        sampler_params['step_sizes'] = .0005
        sampler_params['hmc_steps'] = 10
        sampler_params['mh_correction'] = True

        sampler_params['batch_size'] = None
        sampler_params['seek_step_sizes'] = False
        sampler_params['fade_in_velocities'] = True
        env.setup_data_dir()
        self.configure_env_mcmc(env, HMCSampler, sampler_params)
        env.run()

    def run_sgld(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'sgld'

        env.n_samples = 100
        env.thinning = 29

        sampler_params = dict()
        sampler_params['step_sizes'] = .001

        sampler_params['fade_in_velocities'] = True

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGLDSampler, sampler_params)
        env.run()

    def run_sghmc(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'sghmc'

        env.chains_num = 1
        env.n_samples = 100
        env.thinning = 1

        sampler_params = dict()
        sampler_params['step_sizes'] = .0005
        sampler_params['hmc_steps'] = 10
        sampler_params['friction'] = 1.

        sampler_params['fade_in_velocities'] = True

        env.setup_data_dir()
        self.configure_env_mcmc(env, SGHMCSampler, sampler_params)
        env.run()

    def run_psgld(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'psgld'

        env.n_samples = 100
        env.thinning = 19

        sampler_params = dict()
        sampler_params['step_sizes'] = .001
        sampler_params['preconditioned_alpha'] = .999
        sampler_params['preconditioned_lambda'] = .01

        sampler_params['fade_in_velocities'] = True

        env.setup_data_dir()
        self.configure_env_mcmc(env, pSGLDSampler, sampler_params)
        env.run()

    def run_dropout(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'dropout'

        env.n_samples = 100

        sampler_params = dict()
        sampler_params['n_epochs'] = 20

        dropout = 0.05
        tau = 0.159707652696

        env.setup_data_dir()
        self.configure_env_dropout(env, sampler_params=sampler_params, dropout=dropout, tau=tau)
        env.run()

    def run_bbb(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'bbb'
        n_epochs = 25

        env.setup_data_dir()
        self.configure_env_bbb(env, n_epochs=n_epochs)
        env.run()

    def run_pbp(self):
        env = NavalEnv()
        self.setup_env_defaults(env)

        env.model_name = 'pbp'
        env.n_samples = 100
        env.n_chunks = 20

        env.setup_data_dir()
        self.configure_env_pbp(env, n_epochs=5)
        env.run()


def main():
    experiment = NavalExp()

    queue = OrderedDict()
    queue['HMC'] = experiment.run_baseline_hmc
    queue['SGHMC'] = experiment.run_sghmc

    experiment.run_queue(queue, skip_completed=False, cpu=False)
    experiment.report_metrics_table(queue)

    del queue['HMC']


if __name__ == '__main__':
    main()
