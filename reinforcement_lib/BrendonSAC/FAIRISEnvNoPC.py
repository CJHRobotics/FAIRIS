from fairis_lib.robot_lib import hambot#rosbot #TODO Fix
#from rosbot import RosBot
#from rosbot import RosBot

#from fairis_tools.experiment_tools.place_cell.PlaceCellLibrary import PlaceCellNetwork
#from fairis_tools.biology_tools import PlaceCellNetwork
from fairis_tools.experiment_tools.paths.path_planning.MazePaths import MazePaths
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

class FAIRISEnvTF:
    def __init__(self, maze_filet, horizont, noiset=False, noise_stdt=0.1, noise_meant=0.0):
        self.robot = hambot.HamBot(use_camera=False)#rosbot.RosBot()
        self.maze_file = maze_filet

        self.first_run = True
        self.horizon = horizont
        self.length = 0
        self.store_path = False
        self.cur_op = 0
        self.path = []
        self.ops = []
        self.last_reward = False
        self.cur_reward = 0
        self.noise = noiset
        self.noise_std = noise_stdt
        self.noise_mean = noise_meant
        self.changeStartXY = None
#        self.x = []
#        self.y = []

        # Get optimal path len
        #maze_path = MazePaths(maze_filet)
        #maze_path.calculate_shortest_paths()
        #self.optimal_len = len(maze_path.paths[0].path)
        #print(f"Optimal Path Length is: {self.optimal_len}")

    def set_maze(self, new_maze):
        self.maze_file = new_maze
        self.first_run = True

    def changeStart(self, xy):
        self.changeStartXY = xy

    def add_noise(self, x, y, mean=0, std_dev=0.1, clip_amount=5):
        noise_x = np.random.normal(mean, std_dev)
        noise_y = np.random.normal(mean, std_dev)
        x_noisy = np.clip(x + noise_x, -1*clip_amount, clip_amount)
        y_noisy = np.clip(y + noise_y, -1*clip_amount, clip_amount)
        return x_noisy, y_noisy

#        return abs(path_len - self.optimal_len) / self.optimal_len

#    def get_place_cell(self, x, y):
#        pcs = self.pc_net.get_all_pc_activations_normalized(x, y)
#        return pcs

#    def get_pc_xy(self):
#        xy_list = []
#        for pc in self.pc_net.pc_list:
#            xy_list.append([pc.center_x, pc.center_y])
#        return xy_list

    def getState(self):
        # Update pcs
        robot_x, robot_y, robot_theta = self.robot.get_robot_pose()
        if self.store_path:
            self.path.append((robot_x, robot_y))
            self.ops.append(self.cur_op)

        if self.noise:
            robot_x, robot_y = self.add_noise(robot_x, robot_y, self.noise_mean, self.noise_std)

        # Now get pcs
        #pcs = self.pc_net.get_all_pc_activations_normalized(robot_x, robot_y)

        return robot_x, robot_y

    def reset(self):
        if self.first_run:
            self.robot.load_environment(self.maze_file)
            self.first_run = False
#            print(f"Env subgoals: {self.robot.maze.subgoals}, goals: {self.robot.maze.goal_locations}")

        self.robot.move_to_random_experiment_start()
        if self.changeStartXY:
            self.robot.teleport_robot(self.changeStartXY[0], self.changeStartXY[1])
        self.robot.experiment_supervisor.simulationResetPhysics()

        # Get state
        state = self.getState()

        self.length = 0

        return state

    def step(self, action):
        done = False
        reward = -0.2

        # Do action
        value = self.robot.perform_action_with_PID(int(action))

        # Get state
        state = self.getState()

        # Calculate reward
#        if self.robot.check_at_goal():
        if self.robot.check_at_goal():
            reward = 10
            done = True
#                print("robot at goal")
        if self.length >= self.horizon:
            done = True
            reward = -1
        elif value == -1:
            reward = -1
#                if self.cur_reward == 0:
#                    reward = -1.5
#                else:
#                    reward = (-1.0 / self.cur_reward)

        self.length += 1

        return state, reward, done

    def start_path(self):
        self.store_path = True
        self.path = []
        self.ops = []

    def get_path(self):
        self.store_path = False
        return self.path

    def closeSim(self, status):
        self.robot.experiment_supervisor.simulationQuit(status)
