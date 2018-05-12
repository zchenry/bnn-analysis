"""
Protein
"""

from env import Env


class ProEnv(Env):
    """ Test environment for experiments with Protein data set. """

    def __init__(self):
        super().__init__()

        # setup defaults
        self.env_name = 'pro'
        self.layers_description = [[9, 0.0], [50, 0.0], [1, 0.0]]
        self.n_splits = 4
        self.batch_size = 512
