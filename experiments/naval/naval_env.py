"""
Naval
"""

from env import Env


class NavalEnv(Env):
    """ Test environment for experiments with Naval data set. """

    def __init__(self):
        super().__init__()

        # setup defaults
        self.env_name = 'nv'
        self.layers_description = [[17, 0.0], [50, 0.0], [1, 0.0]]
        self.n_splits = 4
        self.batch_size = 128
