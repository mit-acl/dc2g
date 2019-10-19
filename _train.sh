#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard
# rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add PYTHONPATH
export PYTHONPATH=$DIR/gym-minigrid:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
python3.6 main.py \
--env-name "MiniGrid-EmptySLAM-32x32-v0" \
--algorithm "DQN" \
--ep-max-timesteps 50 \
