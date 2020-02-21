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
export PYTHONPATH=$DIR/venv
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# Add PYTHONPATH
export PYTHONPATH=$DIR/..:$PYTHONPATH
export PYTHONPATH=$DIR/../gym-minigrid:$PYTHONPATH

# Allow jupyter notebook to use venv as kernel
python3 -m pip install ipykernel
python3 -m ipykernel install --name=venv

print_header "Finished installing DC2G"
