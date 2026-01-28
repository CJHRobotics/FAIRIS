import os
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

sys.path.append("/home/b/brendon45/FAIRIS/")
os.chdir("/home/b/brendon45/FAIRIS/")
#cwd = os.getcwd()
#files = os.listdir(cwd)
#print(files)

import numpy as np
import torch
#from reinforcement_lib.PPO.PPONetworks import Agent
from reinforcement_lib.SAC.Agent import Agent
from fairis_tools.experiment_tools.loggers.experiment_logger import ExperimentLogger, EpisodeLogger
from fairis_lib.simulation_lib.webots_torch_environment import WebotsEnv
from fairis_tools.experiment_tools.paths.path_planning.MazePaths import MazePaths

def get_shortest_path(filename):
    maze_path = MazePaths(filename)
    maze_path.calculate_shortest_paths()
    return len(maze_path.paths[0].path)
    #print('Starting')
    #print(type(maze_path.paths))
    #print(len(maze_path.paths[0].path))
    #print('Finished')
#    walls = [[[-4, 3], [0, 0]]]
    #walls = []
    #borders = [(-4, -4), (4, 4)]
    #draw_path(maze_path.paths[0].path, "/home/b/brendon45/ppo_tests/short.png", walls, borders)

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
maze_file = f'/home/b/brendon45/oc_tests/mazes/{maze_name}.xml'
pc_network_name = '/home/b/brendon45/ppo_tests/uniform_32'

#pc_network_name = '/home/b/brendon45/ppo_tests/pc_w6_h6'

#optimal_path = get_shortest_path(maze_file)
#print(f"Optimal path: {optimal_path}")

env = WebotsEnv(maze_file=maze_file,
                pc_network_name=pc_network_name,
                max_steps_per_episode=2500)
#                action_length=0.3)
load_checkpoint = False
ver_name = pc_network_name +'_'+maze_file+'_rewardV3'
N = 5#20
batch_size = 32 #10
n_epochs = 4
alpha = 0.0003
gamma = 0.9
experiment_logger = ExperimentLogger(maze_file,'uniform_32')
max_steps = 350000

n_layers_vals = [2, 3, 4]
layer_sizes = [32, 64, 128, 256]
p_data = {}
idy = 3
print(idy)

#for idy in range(5):
for idx in range(3):
        print(f"Starting run: {n_layers_vals[idx]}, {layer_sizes[idy]}\n\n")
        
        agent = Agent(n_actions=env.action_space.n,
                              batch_size=batch_size,
                #              gamma=gamma,
                              alpha=alpha,
#                              n_epochs=n_epochs,
                              input_dims=env.observation_spec().shape,
                              ver_name=ver_name,
                              n_layers=n_layers_vals[idx],
                              layer_size=layer_sizes[idy])
    
    #    print("Layer size")
    #    for name, module in agent.critic.named_modules():
    #        print(name, module)
    
#        num_params = sum(p.numel() for p in agent.actor.parameters()) + sum(p.numel() for p in agent.critic.parameters())
    
#        print(f"Number of parameters: {num_params}")
    
        #if load_checkpoint:
        #    agent.load_models()
        #    env.set_mode(mode='train',
        #                 noise=False,
        #                 noise_intensity=0.01,
        #                 PC_decay=False,
        #                 PC_decay_percent=0.1)
        
        n_episodes = 5#2500
        
        best_score = 0.0
        score_history = []
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
            observation = env.reset()
            path_length = 0
            done = False
            score = 0
            start_x, start_y, start_theta = env.get_robot_pose()
            episode_logger = EpisodeLogger(start_x, start_y)
            prev_action = None
            path = [(start_x, start_y)]
            losses = []
        
            while not done:
#                action, prob, val = agent.choose_action(observation)
                action = agent.choose_action(observation)
                if np.random.rand() < eps:
                    # Random sample
                    dist = torch.distributions.Categorical(probs=(torch.ones(env.action_space.n) / env.action_space.n))
                    action = dist.sample()
                    prob = torch.squeeze(dist.log_prob(action)).item()
                    action = torch.squeeze(action).item()
                #    cur_eps = cur_eps * eps
                prev_action = action
                observation_, reward, done, info = env.step(action)
                n_steps += 1
                episode_logger.log_step(robot_x=info[0],robot_y=info[1])
                path.append((info[0], info[1]))
                path_length += 1
                score += reward
                #agent.remember(observation, action, prob, val, reward, done)
                agent.remember(observation, action, reward, observation_, done)
                if n_steps % N == 0:
                    loss = agent.learn()
                    losses.append(loss)
                    learn_iters += 1
                observation = observation_
        
        
        #    if (path_length < 600) and (cur_eps > 0.75):
        #        cur_eps = 0.55
            
        #    if (path_length < 100) and not(eps == 0.999):
        #        eps = 0.999
        
            experiment_logger.log_episode(episode_logger,score,path_length)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            lengths.append(path_length)
            steps.append(n_steps)
        
            if len(lengths) > 5 and max(lengths[-4:]) < 1500 and eps == 0.9:
                print("First change")
                eps = 0.8
        
            if max(lengths[-5:]) < 750 and eps == 0.8:
                print("Second change")
                eps = 0.5
        
            if max(lengths[-5:]) < 150 and eps == 0.5:
                print("Third change")
                eps = 0.2
        
            if max(lengths[-20:]) < 50:
                finished = True
                print(f"Converge time: {n_steps}")
                break
                
            
        #    if max(lengths[-5:]) < 100 and not(eps == 0.999):
        #        print("Change Eps")
        #        eps = 0.999
        
            # Plot path
    #        walls = [[[-4, 3], [0, 0]]]
    #        border = [(-4, -2), (8, 4)]
    #        draw_path(path, f"/home/b/brendon45/ppo_tests/data/ppo_path_{maze_name}.png", walls, border)
    
            if path_length < min_len:
                shortest_path = path
                min_len = path_length
        
        #    optimality_ratio = abs(path_length - optimal_path) / optimal_path
        
        #    print('time', datetime.datetime.now(), 'episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
        #            'Path Length: ', path_length, 'learn iters', learn_iters, 'cur_eps %.3f', cur_eps,
        #            'loss_mean %.3f', np.mean(losses))#, 'optimality_ratio', optimality_ratio)
            print(f"{datetime.datetime.now()}: step: {n_steps}, path len: {path_length}, learn inters: {learn_iters}, cur_eps: {eps:.3f}")
        
        #    if optimality_ratio == 1.0:
        #        break
        
        print(f"Finished conv time: {n_steps}")
        print(min(lengths))
    
    #    p_data = {"": shortest_path}
        p_data[f"{n_layers_vals[idx]}-{layer_sizes[idy]}"] = {"lens": lengths, "steps": steps}#, "param": num_params}
    
        
        #path_lengths = {"lengths": lengths, "steps": steps}
        #with open(f"/home/b/brendon45/ppo_tests/data/path_{maze_name}_lengths.pkl", 'wb') as fp:
        #    pickle.dump(path_lengths, fp)
        
        # with open('data/'+maze_file+'_test_10_decay_V3','wb') as file:
        #     pickle.dump(experiment_logger,file)
        
        env.reset_environment()


with open(f"/home/b/brendon45/ppo_tests/data/{maze_name}_param_data_{idy}.pkl", 'wb') as fp:
    pickle.dump(p_data, fp)
