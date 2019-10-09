import matplotlib.pyplot as plt
import numpy as np
import dijkstra
import os
import glob
import csv
import multiprocessing
from dc2g.util import get_goal_colors, get_traversable_colors, find_traversable_inds, find_goal_inds, inflate, get_object_goal_names, get_training_testing_houses

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

resolution = 1 # not sure if still used

# dataset = "house3d"
# dataset = "driveways_icra19"
# dataset = "driveways_iros19"
dataset = "driveways_bing_iros19"

object_goal_names = get_object_goal_names(dataset)
room_goal_names = []

traversable_colors = get_traversable_colors(dataset)
room_goal_color_dict = get_goal_colors(dataset, room_goal_names, room_or_object_goal="room")
object_goal_color_dict = get_goal_colors(dataset, object_goal_names, room_or_object_goal="object")
training_houses, validation_houses, testing_houses = get_training_testing_houses(dataset)

def load_semantic_arrays(semantic_filename, room_type_filename):
    semantic_array = plt.imread(semantic_filename)
    # NOTE: room_type_array has been disabled temporarily...
    # room_type_array = (plt.imread(room_type_filename)*255).astype(np.uint16)
    # return semantic_array, room_type_array
    return semantic_array, None

def find_non_free_space(semantic_array, room_type_array, dataset, goal_color, traversable_colors, room_or_object_goal):
    if room_or_object_goal == "object":
        semantic_array = semantic_array
    elif room_or_object_goal == "room":
        semantic_array = room_type_array
    goal_array, goal_inds, _ = find_goal_inds(semantic_array, goal_color, room_or_object_goal)
    if len(goal_inds[0]) == 0:
        # print("[Error] Couldn't find anything with color {} in the semantic gridmap ==> skipping that goal for this world.".format(goal_color))
        # Couldn't find anything with goal_name's color in the semantic gridmap
        # ==> The goal object probably doesn't exist in this world! Move on to the next goal object
        return None, None, None

    # Find all the inds that have semantic color equivalent to one of the traversable_colors, then set those to be traversable
    traversable_array, _, _ = find_traversable_inds(semantic_array, traversable_colors)

    # Inflate the size of the goal object until it intersects with the traversable terrain 
    inflated_goal_array, inflated_traversable_array = inflate(goal_array, traversable_array, semantic_array)
    non_free_space_array = 1 - inflated_traversable_array

    non_traversable_inflated_goal_array = np.logical_and(inflated_goal_array, 1 - traversable_array)
    return non_free_space_array, goal_inds, non_traversable_inflated_goal_array

def compute_c2g(non_free_space_array, goal_inds, inflated_goal_array, c2g_filename, rgb=False, resolution=1.0):
    # Compute c2g over traversable regions
    c2g_array = dijkstra.og_to_c2g(non_free_space_array, goal_inds, inflated_goal_array, rgb=rgb, resolution=resolution)
    plt.imsave(c2g_filename, c2g_array)
    return c2g_array

def full_semantic_gridworld_to_full_c2g(semantic_filename, room_type_filename, c2g_filename, dataset, resolution, goal_color, traversable_colors, room_or_object_goal):
    semantic_array, room_type_array = load_semantic_arrays(semantic_filename, room_type_filename)
    non_free_space_array, goal_inds, non_traversable_inflated_goal_array = find_non_free_space(semantic_array, room_type_array, dataset, goal_color, traversable_colors, room_or_object_goal)
    if non_free_space_array is None:
        return None # couldn't find goal_name in the semantic gridmap of this world
    c2g_array = compute_c2g(non_free_space_array, goal_inds, non_traversable_inflated_goal_array, c2g_filename, rgb=True, resolution=resolution)
    return c2g_array

def generate_c2g_training(world_id):
    generate_c2g(world_id, "train")
def generate_c2g_validation(world_id):
    generate_c2g(world_id, "val")
def generate_c2g_testing(world_id):
    generate_c2g(world_id, "test")

def generate_c2g(world_id, mode):
    map_name_and_world_id = "world{world_id}".format(world_id=world_id)
    print(map_name_and_world_id)
    semantic_filename = "{dir_path}/training_data/{dataset}/full_semantic/{mode}/{map_name_and_world_id}.png".format(dir_path=dir_path, mode=mode, map_name_and_world_id=map_name_and_world_id, dataset=dataset)
    if not os.path.isfile(semantic_filename): # that map doesn't exist
        return
    room_type_filename = "{dir_path}/training_data/{dataset}/full_room_type/{mode}/{map_name_and_world_id}.png".format(dir_path=dir_path, mode=mode, map_name_and_world_id=map_name_and_world_id, dataset=dataset)
    for goal_name in room_goal_names:
        c2g_filename = "{dir_path}/training_data/{dataset}/full_c2g/{mode}/{map_name_and_world_id}-{goal_name}.png".format(dir_path=dir_path, mode=mode, map_name_and_world_id=map_name_and_world_id, dataset=dataset, goal_name=goal_name)
        try:
            goal_color = room_goal_color_dict[goal_name]
            # print("Searching for {goal_name}, which is bit # {goal_color}".format(goal_name=goal_name, goal_color=goal_color))
        except:
            # print("Couldn't find {goal_name}'s color in the colormap ==> skipping that goal for this world.".format(goal_name=goal_name))
            continue
            c2g_array = full_semantic_gridworld_to_full_c2g(semantic_filename, room_type_filename, c2g_filename, dataset, resolution, goal_color, traversable_colors, room_or_object_goal="room")
    for goal_name in object_goal_names:
        c2g_filename = "{dir_path}/training_data/{dataset}/full_c2g/{mode}/{map_name_and_world_id}-{goal_name}.png".format(dir_path=dir_path, mode=mode, map_name_and_world_id=map_name_and_world_id, dataset=dataset, goal_name=goal_name)
        try:
            goal_color = object_goal_color_dict[goal_name]
        except:
            # print("Couldn't find {goal_name}'s color in the colormap ==> skipping that goal for this world.".format(goal_name=goal_name))
            continue
        if os.path.isfile(c2g_filename):
            continue
        c2g_array = full_semantic_gridworld_to_full_c2g(semantic_filename, room_type_filename, c2g_filename, dataset, resolution, goal_color, traversable_colors, room_or_object_goal="object")

if __name__ == "__main__":

    pool = multiprocessing.Pool(2*multiprocessing.cpu_count()-2 or 1)
    print("\n\n -------------------- \n Making Training gridmaps \n -------------------- \n\n")
    pool.map(generate_c2g_training, training_houses)
    print("\n\n -------------------- \n Making validation gridmaps \n -------------------- \n\n")
    pool.map(generate_c2g_validation, validation_houses)
    print("\n\n -------------------- \n Making Testing gridmaps \n -------------------- \n\n")
    pool.map(generate_c2g_testing, testing_houses)

    print("--- All done ---")
