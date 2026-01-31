#!/bin/bash -l
#SBATCH -p general
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=./output_logs/%x-%j.out
srun singularity run --nv /home/b/brendon45/ppo_tests/webots_v2.sif bash sac_script.sh $1
##srun singularity exec --nv -H /home/b/brendon45:/root ./webots_tf_env.sif ls ..
