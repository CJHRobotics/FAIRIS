# Import Required Libs
import torch
import numpy as np
import gymnasium as gym
import math

# FAIRIS libs
from fairis_lib.robot_lib import hambot

class FAIRISEnv(gym.Env):
    def __init__(self, maze_filet, horizont, device="cpu"):

        # Env Variables
        self.robot = hambot.HamBot(use_camera=False)
        self.maze_file = maze_filet
        self.first_run = True # Var to see if we need to load maze
        self.horizon = horizont
        self.length = 0
        self.max_lidar = 20
#        self.observation_space_size = 2
#        self.action_space_size = 8

        self.observation_space = gym.spaces.Box(low=-20, high=20, shape=(364,))
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, seed=None, options=None):
        if self.first_run:
            self.robot.load_environment(self.maze_file)
            self.first_run = False

        self.robot.move_to_random_experiment_start()
        self.robot.experiment_supervisor.simulationResetPhysics()

        # Get first state
        state = self.getState()

        self.length = 0

        info = {}

        return state, info

    def step(self, action):
        done = False
        reward = -0.5

        # Do action
        value = self.robot.perform_action_with_PID(int(action))

        # Get new state
        state = self.getState()

        # Calculate reward
        if self.robot.check_at_goal():
            reward = 10.0
            done = True
        elif self.length >= self.horizon:
            done = True
            reward = -1.0
        elif value == -1:
            reward = -1.0
        else:
#            reward = 0
            goal_x, goal_y = self.robot.maze.get_goal_location()
            dist = math.sqrt((goal_x - state[0]) ** 2 + (goal_y - state[1]) ** 2) / 7
            reward = -0.1 - dist
            #reward = -0.5

        self.length += 1

        info = {}

        return state, reward, done, False, info

    def getState(self):
        robot_x, robot_y, robot_theta = self.robot.get_robot_pose()

        goal_x, goal_y = self.robot.maze.get_goal_location()

        lidar_image = self.robot.get_lidar_range_image()
        for idx in range(len(lidar_image)):
            if lidar_image[idx] > 20:
                lidar_image[idx] = 20

#        print(lidar_image.shape)

        return np.array([robot_x, robot_y] + lidar_image + [goal_x, goal_y])
