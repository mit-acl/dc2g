#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

cd $DIR
source venv/bin/activate
export PYTHONPATH=$DIR/venv

# Add PYTHONPATH
export PYTHONPATH=$DIR/gym-minigrid:$PYTHONPATH
export PYTHONPATH=$DIR/dc2g:$PYTHONPATH

# Train tf 
print_header "Running Example"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
# python3 dc2g/driveway_env.py
python3 dc2g/run_episode.py