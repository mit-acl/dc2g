Get Started

```sh
cd <code directory>
git clone --recursive <this repo>
cd dc2g
python -m pip install -e gym-minigrid
python -m pip install -e .
```

Train & export a network:
```sh
export dataset_name=driveways_bing_iros19
python pix2pix-tensorflow/tools/dockrun.py python pix2pix-tensorflow/pix2pix.py --mode train --output_dir data/trained_networks/${dataset_name}_masked --max_epochs 10 --input_dir data/datasets/${dataset_name}/tf_records --which_direction AtoB  --dataset ${dataset_name}
python pix2pix-tensorflow/pix2pix.py --mode export --output_dir data/trained_networks/${dataset_name}_masked2 --checkpoint data/trained_networks/${dataset_name}_masked --dataset ${dataset_name}
```

Test a network in the gridworld:
```sh
# Use semantic coloring in the map:
python -m dc2g.run_episode

# Don't semantic coloring in the map:
python -m dc2g.run_episode -u False
```