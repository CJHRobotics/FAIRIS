import sys
import os

# Path for FAIRIS
sys.path.append("/home/brendon/PhDResearch/REU/FAIRIS/")
os.chdir("/home/brendon/PhDResearch/REU/FAIRIS/")

import torch
import torch.nn as nn
import numpy as np
from collections.abc import Callable
from numbers import Number
from torch.utils.tensorboard import SummaryWriter
import datetime

from tianshou.algorithm import DiscreteSAC
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSACPolicy#, DiscreteSAC
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, ReplayBuffer #VectorReplayBuffer
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils import BaseLogger, LazyLogger
from tianshou.utils import TensorboardLogger

from FAIRIS_Env_Gym import FAIRISEnv

# Hyperparameters
hidden_size = 256
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
tau= 0.005
gamma= 0.99
n_step= 3
frames_stack= 1
buffer_size= 100000
epoch= 100
epoch_num_steps= 10000
collection_step_num_env_steps = 100
batch_size=256
update_per_step=1.0
hidden_sizes = [256, 256]

# Env parameters
maze_file = f'/home/brendon/PhDResearch/REU/FAIRIS/simulation/worlds/mazes/Experiment1/WM10.xml'
horizon = 100

VALID_LOG_VALS_TYPE = int | Number | np.number | np.ndarray | float

# Simple Logger to print to screen or file
class EpisodeLengthLogger(BaseLogger):
    def __init__(self):
        super().__init__(training_interval=500)
        self.data = []
        self.printFile = False
        if self.printFile:
            self.file_name = r"/home/brendon/PhDResearch/REU/FAIRIS/reinforcement_lib/torchrl/log.txt"
            with open(self.file_name, 'w') as file:
                pass

    def prepare_dict_for_logging(
        self,
        data: dict[str, VALID_LOG_VALS_TYPE],
    ) -> dict[str, VALID_LOG_VALS_TYPE]:
        return data

    def write(self, step_type: str, step: int, data: dict[str, VALID_LOG_VALS_TYPE]) -> None:
        """The LazyLogger writes nothing."""
        if ('lens' in data.keys()) and ('returns' in data.keys()) and (len(data['lens']) > 0):
            if self.printFile:
                with open(self.file_name, 'a') as file:
                    file.write(f"Step: {step}, lens: {data['lens']}, returns: {data['returns']}\n")
            else:
                print(f"Step: {step}, lens: {data['lens']}, returns: {data['returns']}, collected_eps: {data['n_collected_episodes']}")

    def finalize(self) -> None:
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        update_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        pass

    def restore_data(self) -> tuple[int, int, int]:
        return 0, 0, 0

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        return {}



# Build Env
# Create Function to make the env to be passed to the DummyVectorEnv
def create_env():
    return FAIRISEnv(maze_filet=maze_file, horizont=horizon)
vector_env = DummyVectorEnv([create_env])

# Action shape and Obs shape
action_shape = vector_env.action_space[0].n
obs_shape = vector_env.observation_space[0].shape

# Define networks used for actor and both critics
a_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes)
c1_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes)
c2_model = Net(state_shape=obs_shape[0], hidden_sizes=hidden_sizes)

# Create models for actor and both critics in addition to the optimizers for each model
actor = DiscreteActor(preprocess_net=a_model, action_shape=action_shape, softmax_output=False)
actor_optim = AdamOptimizerFactory(lr=actor_lr)
critic1 = DiscreteCritic(preprocess_net=c1_model, last_size=action_shape)
critic1_optim = AdamOptimizerFactory(lr=critic_lr)
critic2 = DiscreteCritic(preprocess_net=c2_model, last_size=action_shape)
critic2_optim = AdamOptimizerFactory(lr=critic_lr)

# Define automatic alpha using the target entropy and the AutoAlpha class
target_entropy = 0.5 * np.log(action_shape)
log_alpha = 0.0
alpha_optim = AdamOptimizerFactory(lr=alpha_lr)
alpha_param = AutoAlpha(target_entropy, log_alpha, alpha_optim)

# Define policy and algorithms
policy = DiscreteSACPolicy(
    actor=actor,
    action_space=vector_env.action_space[0],
)

algorithm = DiscreteSAC(
    policy=policy,
    policy_optim=actor_optim,
    critic=critic1,
    critic_optim=critic1_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=tau,
    gamma=gamma,
    alpha=alpha_param,
    n_step_return_horizon=n_step,
)


# Define Replay Buffer
buffer = ReplayBuffer(
    buffer_size,
)

# Define collector
collector = Collector[CollectStats](
    algorithm, vector_env, buffer, exploration_noise=False,    
)

# Create the path to the log into
now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
log_path_base = r"/home/brendon/PhDResearch/REU/FAIRIS/reinforcement_lib/torchrl/logs/"
log_path = os.path.join(log_path_base, now)

# Create our tensorboard logger
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer, training_interval=50, update_interval=1)

# Warmup our buffer with 4*batch_size
collector.reset()
collector.collect(n_step=4*batch_size)
# Train
result = algorithm.run_training(
    OffPolicyTrainerParams(
        training_collector=collector,
        max_epochs=epoch,
        epoch_num_steps=epoch_num_steps,
        collection_step_num_env_steps=collection_step_num_env_steps,
        batch_size=batch_size,
        update_step_num_gradient_steps_per_sample=update_per_step,
        logger=logger,
    ),
)
print(result)
