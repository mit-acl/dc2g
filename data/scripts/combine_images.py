"""Converts DC2G data to TFRecords file format with Example protos."""

import os
import sys

import matplotlib.pyplot as plt
import csv
from dc2g.util import get_training_testing_houses, get_object_goal_names

from PIL import Image
import numpy as np

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))
# dataset = "house3d"
# dataset = "driveways_iros19"
dataset = "driveways_bing_iros19"

def convert(mode, houses, goal_names):
    print("Converting {}".format(mode))
    # goal_names = ["television", "toilet"]
    # goal_names = ["television"]
    # houses = ["0004d52d1aeeb8ae6de39d6bd993e992", "000514ade3bcc292a613a4c2755a5050", "00a42e8f3cb11489501cfeba86d6a297"]

    if mode == "train":
        if dataset == "house3d":
            num_masks = 105
        elif "driveways" in dataset:
            num_masks = 256
        masks = range(num_masks)
    if mode == "val":
        if dataset == "house3d":
            num_masks = 105
        elif "driveways" in dataset:
            num_masks = 80
        masks = range(num_masks)
    elif mode == "test":
        # masks = [52, 29, 21, 5, 89]
        # masks = [1,2,3,4]
        masks = range(15)

    count = 0
    for world_id in houses:
        print("Combining {}.".format(world_id))
        for mask_id in masks:
            print("mask {}".format(mask_id))
            for goal_name in goal_names:
                mask_id_str = str(mask_id).zfill(3)
                semantic_map_filename = "{dir_path}/training_data/{dataset}/masked_semantic/{mode}/world{world_id}_{mask_id_str}.png".format(mask_id_str=mask_id_str, world_id=world_id, goal_name=goal_name, dir_path=dir_path, mode=mode, dataset=dataset)
                c2g_map_filename = "{dir_path}/training_data/{dataset}/masked_c2g/{mode}/world{world_id}_{mask_id_str}-{goal_name}.png".format(mask_id_str=mask_id_str, world_id=world_id, goal_name=goal_name, dir_path=dir_path, mode=mode, dataset=dataset)
                combined_filename = "{dir_path}/training_data/{dataset}/masked_combined/{mode}/world{world_id}_{mask_id_str}.jpg".format(mask_id_str=mask_id_str, world_id=world_id, goal_name=goal_name, dir_path=dir_path, mode=mode, dataset=dataset)
                if os.path.isfile(semantic_map_filename) and os.path.isfile(c2g_map_filename):
                    count += 1

                    with open(semantic_map_filename, 'rb') as f:
                        semantic_map = plt.imread(f)
                    with open(c2g_map_filename, 'rb') as f:
                        c2g_map = plt.imread(f)
                    combined_array = 255*np.hstack([semantic_map, c2g_map])[:,:,:3]
                    plt.imsave(combined_filename, combined_array)

    print("Combined {count} images.".format(count=count))

def main():
    training_houses, validation_houses, testing_houses = get_training_testing_houses(dataset)
    object_goal_names = get_object_goal_names(dataset)

    # Convert to Examples and write the result to TFRecords.
    convert('train', training_houses, object_goal_names)
    # convert('val', validation_houses, object_goal_names)
    # convert('test', testing_houses, object_goal_names)

if __name__ == '__main__':
    main()