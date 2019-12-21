#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add PYTHONPATH
export PYTHONPATH=$DIR/../gym-minigrid:$PYTHONPATH
export PYTHONPATH=$DIR/..:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
python3.6 main.py \
--env-name "MiniGrid-EmptySLAM-32x32-v0" \
--algorithm "DQN" \
--discount 0.95 \
