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

print_header "Running One Episode"
cd $DIR

python -m dc2g.run_episode
