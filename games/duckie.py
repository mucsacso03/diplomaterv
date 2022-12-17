import datetime
import pathlib
import time
import os
import logging
import numpy as np
from gym_duckietown.simulator import NotInLane
from matplotlib import pyplot as plt    
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

logger = logging.getLogger(__name__)
from gym import spaces
import gym
import numpy
import torch
from .abstract_game import AbstractGame
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

from datetime import datetime as dt

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 120, 160)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = 2  # Number of dimensions in the action space
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 5  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 9  # Number of simultaneous threads self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
        self.num_simulations = 5  # Number of future moves self-simulated
        self.discount = 0.99  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)
        self.node_prior = 'uniform'  # 'uniform' or 'density'

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Progressive widening parameter
        self.pw_alpha = 0.49

        ### Network
        # self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2 # Number of channels in the ResNet
        self.reduced_channels_reward = 1  # Number of channels in reward head
        self.reduced_channels_value = 1  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = [128]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [128]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 15
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_mu_policy_layers = [64]  # Define the hidden layers in the policy network
        self.fc_log_std_policy_layers = [64]  # Define the hidden layers in the policy network

    

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000 # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.entropy_loss_weight = 0  # Scale the entropy loss
        self.log_std_clamp = (-20, 2)  # Clamp the standard deviation
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01
        self.lr_decay_rate = 0.99 
        self.lr_decay_steps = 5000



        ### Replay Buffer
        self.replay_buffer_size = 256  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1/5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on
        self.record_env: True

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.25 * self.training_steps:
            return 0.75
        elif trained_steps < 0.5 * self.training_steps:
            return 0.25
        elif trained_steps < 0.75 * self.training_steps:
            return 0.1
        else:
            return 0.05

from gym.wrappers.monitoring.video_recorder import VideoRecorder

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = launch_env()
        print("Initialized environment")

        # Wrappers
        try:
            self.env = NormalizeWrapper(self.env)
        except Exception as e:
            print(e)
        self.env = DtRewardPosAngle(self.env)
        self.env = DtRewardWrapperDistanceTravelled(self.env)

        self.env = ActionWrapper(self.env)     
        self.env = SetSpeedWrapper(self.env, 1)   
        self.env = ActionSmoothingWrapper(self.env)

        self.rec = VideoRecorder(self.env, path=f'/root/workdir/muzero-general/vid/video-{dt.now()}.mp4')
        self.rec.capture_frame()

        print("Initialized Wrappers")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        observation = cv2.resize(observation, (160, 120), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        self.rec.capture_frame()
        return observation, reward, done

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation = self.env.reset()
        observation = cv2.resize(observation, (160, 120), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="float32") / 255.0
        observation = numpy.moveaxis(observation, -1, 0)
        return observation

    def close(self):
        """
        Properly close the game.
        """
        self.rec.close()

        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

class DtRewardPosAngle(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardPosAngle, self).__init__(env)

        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 1./2.
        self.orientation_reward = 0.

    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle
        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
        except NotInLane:
            return -200.
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        self.orientation_reward = self.scale * (angle_narrow_reward + angle_wide_reward * 2)
        early_termination_penalty = 0.

        return self.orientation_reward + early_termination_penalty

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide


class SetSpeedWrapper(gym.ActionWrapper):
    def __init__(self, env, speed=1):
        super(SetSpeedWrapper, self).__init__(env)
        self.speed = speed

    def action(self, action):
        action = [e / max(action) for e in action]
        return action * self.speed

class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, ):
        super(ActionSmoothingWrapper, self).__init__(env)
        self.last_action = [0, 0]
        self.new_action_ratio = 0.75

    def action(self, action):
        diff = abs(action[0] - action[1])

        if action[0] < action[1]:
            action[0] += diff/2
        else:
            action[1] += diff/2
        return action


    def reset(self, **kwargs):
        self.last_action = [0, 0]
        return self.env.reset(**kwargs)

        
class DtRewardWrapperDistanceTravelled(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapperDistanceTravelled, self).__init__(env)
        self.prev_pos = None

    def reward(self, reward):
        # Baseline reward is a for each step
        my_reward = 0

        # Get current position and store it for the next step
        pos = self.unwrapped.cur_pos
        prev_pos = self.prev_pos
        self.prev_pos = pos
        if prev_pos is None:
            return 0

        # Get the closest point on the curve at the current and previous position
        angle = self.unwrapped.cur_angle
        curve_point, tangent = self.unwrapped.closest_curve_point(pos, angle)
        prev_curve_point, prev_tangent = self.unwrapped.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            logger.error("self.unwrapped.closest_curve_point(pos, angle) returned None!!!")
            return my_reward
        # Calculate the distance between these points (chord of the curve), curve length would be more accurate
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        try:
            lp = self.unwrapped.get_lane_pos2(pos, self.unwrapped.cur_angle)
        except NotInLane:
            return my_reward

        # Dist is negative on the left side of the rignt lane center and is -0.1 on the lane center.
        # The robot is 0.13 (m) wide, to keep the whole vehicle in the right lane, dist should be > -0.1+0.13/2)=0.035
        # 0.05 is a little less conservative
        if lp.dist < -0.05:
            return my_reward
        # Check if the agent moved in the correct direction
        if np.dot(tangent, diff) < 0:
            return my_reward

        # Reward is proportional to the distance travelled at each step
        my_reward = 50 * dist
        if np.isnan(my_reward):
            my_reward = 0.
            logger.error("Reward is nan!!!")
        return my_reward