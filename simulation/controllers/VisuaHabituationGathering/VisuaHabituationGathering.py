import os
os.chdir("../../..")
print(os.getcwd())
from fairis_lib.robot_lib.hambot import HamBot
from fairis_tools.experiment_tools.loggers.visual_data_set import PovDataset
from tqdm import tqdm


maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'


maze_files = ['LM4','LM6','LM8', 'LM8D', 'LMO8', 'LMO8D', 'LM8_addition', 'LMO8_test', "BR"]
maze_index = 2
wall_test = 'True'
# create the robot/supervisor instance.
robot = HamBot(enable_cnn_features=False,cnn_extractor_model='mobilenetv3')
robot.load_environment(maze_file_dir + maze_files[maze_index] + '.xml')
training_dataset = PovDataset()
testing_dataset = PovDataset()

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]
total_steps = len(robot.maze.experiment_starting_location)
with tqdm(total=total_steps, desc="Collecting training data") as pbar:
    training_or_testing = 1
    for start_position in robot.maze.experiment_starting_location:
        robot.teleport_robot(x=start_position.x, y=start_position.y, theta=start_position.theta)
        robot_x, robot_y, robot_theta = robot.get_robot_pose()
        multimodal_features, cnn_features = robot.get_full_robot_pov_features(thetas)
        print(len(multimodal_features))
        if not wall_test:
            training_or_testing *= -1
            if training_or_testing < 0:
                training_dataset.add_observations(multimodal_features,
                                                  cnn_features,
                                                  robot_x,
                                                  robot_y,
                                                  robot_theta)
            else:
                testing_dataset.add_observations(multimodal_features,
                                                 cnn_features,
                                                 robot_x,
                                                 robot_y,
                                                 robot_theta)
        if wall_test:
            testing_dataset.add_observations(multimodal_features,
                                             cnn_features,
                                             robot_x,
                                             robot_y,
                                             robot_theta)
        pbar.update(1)
if not wall_test:
    training_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"_Training")
    testing_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"_Testing")
else:
    testing_dataset.save_dataset("data/VisualPlaceCellData/" + maze_files[maze_index] + "Wall_Testing")
robot.experiment_supervisor.simulationReset()

