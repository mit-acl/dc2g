import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import scipy.ndimage.morphology

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

def get_traversable_colors(dataset):
    if dataset == "house3d":
        traversable_colors = np.array([[0, 255, 255]]) / 255. # cyan movable
    elif "driveways" in dataset:
        traversable_classes = ["driveway", "path", "front_door", "road", "sidewalk", "traversable"]
        traversable_colors = []
        colormap_filename = "{dir_path}/data/datasets/{dataset}/labels_colors.csv".format(dir_path=dir_path, dataset=dataset)
        with open(colormap_filename) as colorcsvfile:
            color_reader = csv.DictReader(colorcsvfile)
            for color_row in color_reader:
                if color_row["class"] in traversable_classes:
                    traversable_colors.append([int(color_row["r"]), int(color_row["g"]), int(color_row["b"])])
        traversable_colors = np.array(traversable_colors) / 255.
    else:
        raise NotImplementedError
    # elif dataset == "driveways_icra19":
    #     traversable_colors = np.array([[255, 0, 0], [0, 0, 255]]) / 255.
    # elif dataset == "driveways_iros19" or dataset == "driveways_bing_iros19":
    #     traversable_colors = np.array([[128, 128, 0], [0, 0, 128], [0, 128, 128], [64, 0, 0]]) / 255.
    return traversable_colors

def get_wall_colors(dataset):
    if dataset == "house3d":
        wall_colors = np.array([[200, 200, 200]]) / 255.
    elif "driveways" in dataset:
        wall_colors = np.array([])
    else:
        raise NotImplementedError
    return wall_colors

def get_training_testing_houses(dataset):
    if dataset == "house3d":
        with open('{dir_path}/House3D/House3D/metadata/training_houses.csv'.format(dir_path=dir_path)) as csvfile:
            reader = csv.DictReader(csvfile)
            training_houses = [rows["house_id"] for rows in reader]

        with open('{dir_path}/House3D/House3D/metadata/testing_houses.csv'.format(dir_path=dir_path)) as csvfile:
            reader = csv.DictReader(csvfile)
            testing_houses = [rows["house_id"] for rows in reader]
    elif dataset == "driveways_icra19":
        print('--- not implemented yet ---')
        assert(0)
    elif dataset == "driveways_iros19" or dataset == "driveways_bing_iros19":
        with open('{dir_path}/data/datasets/{dataset}/training_houses.csv'.format(dir_path=dir_path, dataset=dataset)) as csvfile:
            reader = csv.DictReader(csvfile)
            training_houses = [rows["house_id"] for rows in reader]
        with open('{dir_path}/data/datasets/{dataset}/validation_houses.csv'.format(dir_path=dir_path, dataset=dataset)) as csvfile:
            reader = csv.DictReader(csvfile)
            validation_houses = [rows["house_id"] for rows in reader]
        with open('{dir_path}/data/datasets/{dataset}/testing_houses.csv'.format(dir_path=dir_path, dataset=dataset)) as csvfile:
            reader = csv.DictReader(csvfile)
            testing_houses = [rows["house_id"] for rows in reader]
    return training_houses, validation_houses, testing_houses

def get_object_goal_names(dataset):
    if dataset == "house3d":
        target_objects_filename = "{dir_path}/House3D/House3D/metadata/target_objects.csv".format(dir_path=dir_path)
        with open(target_objects_filename) as targetobjectscsvfile:
            object_reader = csv.DictReader(targetobjectscsvfile)
            object_goal_names = [obj_row["object_name"] for obj_row in object_reader]
    elif "driveways" in dataset:
        object_goal_names = ["front_door"]
    else:
        raise NotImplementedError
    return object_goal_names

def get_goal_colors(dataset, goal_names, room_or_object_goal="object"):
    goal_color_dict = {}
    if dataset == "house3d":
        if room_or_object_goal == "object":
            colormap_filename = "{dir_path}/House3D/House3D/metadata/colormap_coarse.csv".format(dir_path=dir_path)
            with open(colormap_filename) as colorcsvfile:
                color_reader = csv.DictReader(colorcsvfile)
                for color_row in color_reader:
                    if color_row["name"] in goal_names:
                        goal_color_dict[color_row["name"]] = (int(color_row["r"])/255., int(color_row["g"])/255., int(color_row["b"])/255.)
                        # print("goal_name: {}, color: {}".format(color_row["name"], goal_color_dict[color_row["name"]]))
        elif room_or_object_goal == "room":
            room_type_dict_filename = "{dir_path}/House3D/House3D/metadata/all_room_types.csv".format(dir_path=dir_path)
            with open(room_type_dict_filename) as csvfile:
                reader = csv.DictReader(csvfile)
                goal_color_dict = dict((rows["room_name"],int(rows["id"])) for rows in reader)
    elif "driveways" in dataset:
        if room_or_object_goal == "object":
            colormap_filename = "{dir_path}/data/datasets/{dataset}/labels_colors.csv".format(dir_path=dir_path, dataset=dataset)
            with open(colormap_filename) as colorcsvfile:
                color_reader = csv.DictReader(colorcsvfile)
                for color_row in color_reader:
                    if color_row["class"] in goal_names:
                        goal_color_dict[color_row["class"]] = (int(color_row["r"])/255., int(color_row["g"])/255., int(color_row["b"])/255.)
        # else:
        #     raise NotImplementedError
    else:
        raise NotImplementedError
    #     if room_or_object_goal == "object":
    #         goal_color_dict = {goal_names[0]: np.array([255, 255, 0]) / 255.}
    # elif dataset == "driveways_iros19" or dataset == "driveways_bing_iros19":
    #     if room_or_object_goal == "object":
    #         goal_color_dict = {goal_names[0]: np.array([128, 0, 0]) / 255.}
    return goal_color_dict

def get_colormap_dict(dataset):
    colormap_dict = {}
    if dataset == "house3d":
        colormap_filename = "{dir_path}/House3D/House3D/metadata/colormap_coarse.csv".format(dir_path=dir_path)
    elif "driveways" in dataset:
        colormap_filename = "{dir_path}/data/datasets/{dataset}/labels_colors.csv".format(dir_path=dir_path, dataset=dataset)
    else:
        print("That dataset doesn't seem to have a colormap yet.")
        raise NotImplementedError
    with open(colormap_filename) as colorcsvfile:
        color_reader = csv.DictReader(colorcsvfile)
        for color_row in color_reader:
            colormap_dict[color_row["class"]] = [int(color_row["r"])/255., int(color_row["g"])/255., int(color_row["b"])/255.]
    return colormap_dict

def find_traversable_inds(semantic_array, traversable_colors):
    '''
    Description: Look for any pixels in the semantic_array image that correspond to traversable terrain (road, driveway, goal)
    inputs:
        - semantic_array: 32x32x3 np array of robot's current partial knowledge of gridworld
    outputs:
        - observed_traversable_inds: tuple of inds within semantic_array that are traversable
                (np.array([x1, x2, ...]), np.array([y1, y2, ...]))
        - observed_traversable_inds_arr: nx2 np array of inds within semantic_array that are traversable
                np.array([[x1, y1], [x2, y2], ...])
    '''

    # Extract red / blue / yellow pixels (corresponding to road / driveway / goal)
    traversable_inds = (np.array([], dtype=int), np.array([], dtype=int))
    for traversable_color in traversable_colors:
        if np.max(semantic_array) > 1:
            color = 255*traversable_color
        else:
            color = traversable_color
        inds = np.where(np.linalg.norm(semantic_array - color, axis=2) < 0.1)
        # inds = np.where(np.logical_and.reduce((semantic_array[:,:,0] == color[0], semantic_array[:,:,1] == color[1], semantic_array[:,:,2] == color[2])))
        traversable_inds = (np.hstack([traversable_inds[0], inds[0]]), np.hstack([traversable_inds[1], inds[1]]))

    # Re-organize inds into pairs of indices
    traversable_inds_arr = np.dstack([traversable_inds[1], traversable_inds[0]])[0]

    traversable_array = np.zeros_like(semantic_array[:,:,0])
    traversable_array[traversable_inds] = 1

    return traversable_array, traversable_inds, traversable_inds_arr

def find_goal_inds(semantic_array, goal_color, room_or_object_goal):
    # Find all the inds that have semantic color equivalent to goal_color, then set those to be traversable
    goal_array = np.zeros_like(semantic_array[:,:,0])
    if room_or_object_goal == "object":
        # goal_inds = np.where(np.logical_and.reduce((semantic_array[:,:,0] == goal_color[0], semantic_array[:,:,1] == goal_color[1], semantic_array[:,:,2] == goal_color[2])))
        goal_inds = np.where(np.linalg.norm(semantic_array - goal_color, axis=2) < 0.1)
        # eps = 0.01
        # goal_inds = np.where(np.logical_and.reduce((semantic_array[:,:,0] > (1-eps)*goal_color[0], semantic_array[:,:,0] < (1+eps)*goal_color[0],
        #                                             semantic_array[:,:,1] > (1-eps)*goal_color[1], semantic_array[:,:,1] < (1+eps)*goal_color[1],
        #                                             semantic_array[:,:,2] > (1-eps)*goal_color[2], semantic_array[:,:,2] < (1+eps)*goal_color[2])))
    elif room_or_object_goal == "room":
        # goal_color is an int (which bit is set for that room type)
        goal_inds = np.where(np.bitwise_and(semantic_array[:,:,0], 1 << goal_color) > 0)
    goal_array[goal_inds] = 1
    if len(goal_inds[0]) > 0:
        goal_inds_arr = np.dstack([goal_inds[1], goal_inds[0]])[0]
    else:
        goal_inds_arr = None
    return goal_array, goal_inds, goal_inds_arr

def inflate(goal_array, traversable_array, semantic_array):
    wall_colors = get_wall_colors("house3d")
    wall_array = np.zeros_like(goal_array)
    for wall_color in wall_colors:
        wall_color_array = np.logical_and.reduce((semantic_array[:,:,0] == wall_color[0], semantic_array[:,:,1] == wall_color[1], semantic_array[:,:,2] == wall_color[2]))
        wall_array = np.logical_or(wall_array, wall_color_array)
    non_wall_array = np.invert(wall_array)

    # Inflate the size of the goal object until it intersects with the traversable terrain 
    num_inflations = 0
    inflated_goal_array = goal_array.copy()
    struct2 = scipy.ndimage.generate_binary_structure(2, 2)
    while np.sum(np.logical_and(inflated_goal_array, traversable_array)) == 0:
        # print('inflating obstacle...{} times'.format(num_inflations))
        inflated_goal_array = scipy.ndimage.morphology.binary_dilation(inflated_goal_array, struct2).astype(inflated_goal_array.dtype)
        inflated_goal_array = np.logical_and(inflated_goal_array, non_wall_array) # make sure inflation doesn't seep through a wall
        num_inflations += 1
    if num_inflations > 0: # do one more for good measure lol
        inflated_goal_array = scipy.ndimage.morphology.binary_dilation(inflated_goal_array).astype(inflated_goal_array.dtype)
    inflated_traversable_array = np.logical_or(traversable_array, inflated_goal_array)
    return inflated_goal_array, inflated_traversable_array

def inflate_old(goal_array, traversable_array):
    # Inflate the size of the goal object until it intersects with the traversable terrain 
    num_inflations = 0
    inflated_goal_array = goal_array.copy()
    struct2 = scipy.ndimage.generate_binary_structure(2, 2)
    while np.sum(np.logical_and(inflated_goal_array, traversable_array)) == 0:
        # print('inflating obstacle...{} times'.format(num_inflations))
        inflated_goal_array = scipy.ndimage.morphology.binary_dilation(inflated_goal_array, struct2).astype(inflated_goal_array.dtype)
        num_inflations += 1
    if num_inflations > 0: # do one more for good measure lol
        inflated_goal_array = scipy.ndimage.morphology.binary_dilation(inflated_goal_array).astype(inflated_goal_array.dtype)
    inflated_traversable_array = np.logical_or(traversable_array, inflated_goal_array)
    return inflated_goal_array, inflated_traversable_array

# keep angle between [-pi, pi]
def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def round_base(x, base):
    if type(x) == np.ndarray:
        return (base * np.around(x.astype(float)/base)).astype(int)
    else:
        return int(base * round(float(x)/base))

def round_base_down(x, base):
    if type(x) == np.ndarray:
        return (base * np.floor(x.astype(float)/base)).astype(int)
    else:
        return int(base * np.floor(float(x)/base))