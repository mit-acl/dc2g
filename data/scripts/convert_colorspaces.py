import matplotlib.pyplot as plt
import numpy as np
import dijkstra
import os
import glob
import csv
import multiprocessing
from dc2g.util import get_goal_colors, get_traversable_colors, find_traversable_inds, find_goal_inds, inflate, get_object_goal_names, get_training_testing_houses, get_colormap_dict

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

resolution = 1 # not sure if still used

from_colormap = "driveways_icra19"
to_colormap = "driveways_iros19"
# dataset = "house3d"
# dataset = "driveways_icra19"
dataset = "driveways_iros19"

training_houses, testing_houses = get_training_testing_houses(dataset)
from_colormap_dict = get_colormap_dict(from_colormap)
to_colormap_dict = get_colormap_dict(to_colormap)

def load_semantic_arrays(semantic_filename, room_type_filename):
    semantic_array = plt.imread(semantic_filename)
    # NOTE: room_type_array has been disabled temporarily...
    # room_type_array = (plt.imread(room_type_filename)*255).astype(np.uint16)
    # return semantic_array, room_type_array
    return semantic_array, None

def convert_colorspace_training(world_id):
    convert_colorspace(world_id, "train")
def convert_colorspace_testing(world_id):
    convert_colorspace(world_id, "test")

def convert_colorspace(world_id, mode):
    map_name_and_world_id = "world{world_id}".format(world_id=world_id)
    print(map_name_and_world_id)
    semantic_filename = "{dir_path}/training_data/{dataset}/full_semantic/{mode}/{map_name_and_world_id}.png".format(dir_path=dir_path, mode=mode, map_name_and_world_id=map_name_and_world_id, dataset=dataset)
    if not os.path.isfile(semantic_filename): # that map doesn't exist
        return
    semantic_array, _ = load_semantic_arrays(semantic_filename, None)
    new_semantic_array = np.zeros_like(semantic_array)

    for obj in from_colormap_dict:
        from_color = from_colormap_dict[obj]
        try:
            to_color = to_colormap_dict[obj]
        except:
            print("That object doesn't have a color in the new colormap, leaving black.")
            continue
        inds = np.where(np.logical_and.reduce((semantic_array[:,:,0] == from_color[0], semantic_array[:,:,1] == from_color[1], semantic_array[:,:,2] == from_color[2])))
        new_semantic_array[inds] = to_color
    plt.imsave(semantic_filename, new_semantic_array)

if __name__ == "__main__":

    # pool = multiprocessing.Pool(2*multiprocessing.cpu_count()-2 or 1)
    pool = multiprocessing.Pool(1)
    print("\n\n -------------------- \n Making Training gridmaps \n -------------------- \n\n")
    pool.map(convert_colorspace_training, training_houses)
    print("\n\n -------------------- \n Making Testing gridmaps \n -------------------- \n\n")
    pool.map(convert_colorspace_testing, testing_houses)

    print("--- All done ---")
