import os
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

# Path for FAIRIS
sys.path.append("/home/b/brendon45/FAIRIS/")
os.chdir("/home/b/brendon45/FAIRIS/")

import numpy as np
import torch
#from reinforcement_lib.PPO.PPONetworks import Agent
from reinforcement_lib.SAC.Agent import Agent
from fairis_tools.experiment_tools.loggers.experiment_logger import ExperimentLogger, EpisodeLogger
from fairis_lib.simulation_lib.webots_torch_environment_no_pc import WebotsEnv
from fairis_tools.experiment_tools.paths.path_planning.MazePaths import MazePaths

#********** Gets the shortest path for a maze*************
def get_shortest_path(filename):
    maze_path = MazePaths(filename)
    maze_path.calculate_shortest_paths()
    return len(maze_path.paths[0].path)

# Used to visualize the algorithm during a run by draw its path during an episode
def draw_path(path, filename, walls, borders):
    plt.clf()
    fig, ax = plt.subplots()
    rect = patches.Rectangle(borders[0], borders[1][0], borders[1][1], linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.relim()
    ax.autoscale_view()

#    for wall in walls:
#        ax.plot(wall[0], wall[1], 'black')

    # Plot path
    data = np.asarray(path)
    ax.plot(data[:, 0], data[:, 1], 'g')

    # Save file
    plt.savefig(filename)
    plt.close()

    
maze_name = "fourrooms"
#maze_name = "simple_ten_obs"
print(f"Maze name is {maze_name}")
maze_file = f'/home/b/brendon45/FAIRIS/simulation/worlds/mazes/Samples/WM00.xml'#f'/home/b/brendon45/oc_tests/mazes/{maze_name}.xml'
#pc_network_name = '/home/b/brendon45/ppo_tests/uniform_32'

#pc_network_name = '/home/b/brendon45/ppo_tests/pc_w6_h6'

env = WebotsEnv(maze_file=maze_file,
#                pc_network_name=pc_network_name,
                max_steps_per_episode=250)
#                action_length=0.3)
load_checkpoint = False
ver_name = '_'+maze_file+'_rewardV3'
batch_size = 32 #10
alpha = 0.0003
gamma = 0.9
max_steps = 250000

p_data = {}

        
agent = Agent(n_actions=8,
              batch_size=batch_size,
              gamma=gamma,
              alpha=alpha,
#              n_epochs=n_epochs,
              input_dims=(2,),
              ver_name=ver_name,
              n_layers=3,
              layer_size=128)
    
#        num_params = sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters())
    
#        print(f"Number of parameters: {num_params}")
    
        #if load_checkpoint:
        #    agent.load_models()
        #    env.set_mode(mode='train',
        #                 noise=False,
        #                 noise_intensity=0.01,
        #                 PC_decay=False,
        #                 PC_decay_percent=0.1)
        
best_score = 0.0
avg_score = 0
learn_iters = 0
n_steps = 0
#cur_eps = 1
eps = 0.9
lengths = []
steps = []
shortest_path = []
min_len = 5000
        
#for i in range(n_episodes):
while n_steps < max_steps:
    obs = [[], []]
    observation = env.reset()
    path_length = 0
    done = False
    start_x, start_y, start_theta = env.get_robot_pose()
    episode_logger = EpisodeLogger(start_x, start_y)
    prev_action = None
    path = [(start_x, start_y)]
    losses = []
    cum_reward = 0
    rewards = []

    while not done:
        #action, prob, val = agent.choose_action(observation)
        #print(type(observation))
        obs[0].append(observation[0])
        obs[1].append(observation[1])
        action = agent.choose_action(observation)
        prev_action = action
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        path.append((info[0], info[1]))
        path_length += 1
        cum_reward += reward
        rewards.append(reward)
        #agent.remember(observation, action, prob, val, reward, done)
        agent.remember(observation, action, reward, observation_, done)
        loss = agent.learn()
        losses.append(0)
        #losses.append(loss.item())
        observation = observation_


#    score_history.append(score)
#    avg_score = np.mean(score_history[-100:])
    lengths.append(path_length)
    steps.append(n_steps)

            
    if (max(lengths[-20:]) < 10) and (len(lengths) > 10):
        finished = True
        print(f"Converge time: {n_steps}")
        break
        
    

    # Plot path
#     walls = [[[-4, 3], [0, 0]]]
#     border = [(-4, -2), (8, 4)]
#     draw_path(path, f"/home/b/brendon45/ppo_tests/data/ppo_path_{maze_name}.png", walls, border)

    if path_length < min_len:
        shortest_path = path
        min_len = path_length

    print(f"{datetime.datetime.now()}: step: {n_steps}, path len: {path_length}, loss: {np.mean(losses):.3f}, cum_r: {cum_reward}, re_a: {np.mean(rewards)}, re_max: {np.amax(rewards)}, re_min: {np.amin(rewards)}, obs_x_a: {np.mean(obs[0]):3f}, obs_x_std: {np.mean(obs[0]):.3f}, obs_y_a: {np.mean(obs[1]):.3f}, obs_y_std: {np.mean(obs[1]):.3f}")


print(f"Finished conv time: {n_steps}")
print(min(lengths))

#p_data = {"": shortest_path}
#p_data[f"{n_layers_vals[idx]}-{layer_sizes[idy]}"] = {"lens": lengths, "steps": steps}#, "param": num_params}


#path_lengths = {"lengths": lengths, "steps": steps}
#with open(f"/home/b/brendon45/ppo_tests/data/path_{maze_name}_lengths.pkl", 'wb') as fp:
#    pickle.dump(path_lengths, fp)

# with open('data/'+maze_file+'_test_10_decay_V3','wb') as file:
#     pickle.dump(experiment_logger,file)

env.reset_environment()


with open(f"/home/b/brendon45/ppo_tests/data/{maze_name}_param_data_{idy}.pkl", 'wb') as fp:
    pickle.dump(p_data, fp)
