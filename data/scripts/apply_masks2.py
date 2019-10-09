import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import numpy as np
import os
import glob
import multiprocessing
from skimage.transform import resize
from dc2g.util import get_object_goal_names, get_training_testing_houses

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

# dataset = "house3d"
# dataset = "driveways_icra19"
# dataset = "driveways_iros19"
dataset = "driveways_bing_iros19"

object_goal_names = get_object_goal_names(dataset)
room_goal_names = []
training_houses, validation_houses, testing_houses = get_training_testing_houses(dataset)

def apply_mask(world_id, mode):
    map_name_and_id = "world" + world_id
    mask_filenames = "{dir_path}/training_data/{dataset}/masks/{mode}/transformed/*.png".format(dataset=dataset, dir_path=dir_path, mode=mode)
    mask_ids = [filename.split('.')[0].split('/')[-1] for filename in glob.glob(mask_filenames)]
    semantic_filename = "{dir_path}/training_data/{dataset}/full_semantic/{mode}/{map_name_and_id}.png".format(dataset=dataset, map_name_and_id=map_name_and_id, dir_path=dir_path, mode=mode)
    if not os.path.isfile(semantic_filename):
        return
    semantic_array = plt.imread(semantic_filename)
    for object_goal_name in object_goal_names:
        c2g_filename = "{dir_path}/training_data/{dataset}/full_c2g/{mode}/{map_name_and_id}-{goal_name}.png".format(dataset=dataset, map_name_and_id=map_name_and_id, dir_path=dir_path, mode=mode, goal_name=object_goal_name)
        if not os.path.isfile(c2g_filename):
            continue
        c2g_array = plt.imread(c2g_filename)
        for mask_id in mask_ids:
            mask_filename = "{dir_path}/training_data/{dataset}/masks/{mode}/transformed/{mask_id}.png".format(dataset=dataset, mode=mode, mask_id=mask_id, dir_path=dir_path)
            # mask_filename = "{dir_path}/training_data/{dataset}/masks/{mode}/transformed/{map_name_and_id}_{mask_id}.png".format(dataset=dataset, map_name_and_id=map_name_and_id, mode=mode, mask_id=mask_id, dir_path=dir_path)
            mask = plt.imread(mask_filename)
            if mask.shape[:-1] != semantic_array.shape[:-1]:
                mask = resize(mask, semantic_array.shape[:-1], order=0)
            actually_apply_mask(c2g_array.copy(), semantic_array.copy(), mask, mask_id, dataset, object_goal_name, mode, map_name_and_id)
    print(map_name_and_id)

def actually_apply_mask(c2g_array, semantic_array, mask, mask_id, dataset, object_goal_name, mode, map_name_and_id):
    masked_semantic_filename = "{dir_path}/training_data/{dataset}/masked_semantic/{mode}/{map_name_and_id}_{mask_id}.png".format(dataset=dataset, mode=mode, map_name_and_id=map_name_and_id, mask_id=mask_id, dir_path=dir_path)
    masked_c2g_filename = "{dir_path}/training_data/{dataset}/masked_c2g/{mode}/{map_name_and_id}_{mask_id}-{goal_name}.png".format(goal_name=object_goal_name, dataset=dataset, mode=mode, map_name_and_id=map_name_and_id, mask_id=mask_id, dir_path=dir_path)
    if not os.path.isfile(masked_semantic_filename):
        semantic_array[mask[:,:,0] == 0] = 0
        semantic_array = semantic_array[:,:,:3]
        plt.imsave(masked_semantic_filename, semantic_array)
    if not os.path.isfile(masked_c2g_filename):
        c2g_array[mask[:,:,0] == 0] = 0 # mask out the unobserved regions
        c2g_array = c2g_array[:,:,:3]

        # re-scale the gray intensity of the un-masked region
        hsv = plt_colors.rgb_to_hsv(c2g_array)
        grayscale_inds = np.where(hsv[:, :, 1] < 0.3)
        max_intensity = np.max(c2g_array[grayscale_inds])
        if max_intensity > 0:
            c2g_array[grayscale_inds] /= max_intensity

        plt.imsave(masked_c2g_filename, c2g_array)

def apply_mask_training(world_id):
    apply_mask(world_id, "train")
def apply_mask_validation(world_id):
    apply_mask(world_id, "val")
def apply_mask_testing(world_id):
    apply_mask(world_id, "test")

if __name__ == "__main__":

    pool = multiprocessing.Pool(2*multiprocessing.cpu_count()-2 or 1)
    # pool = multiprocessing.Pool(1)
    print("\n\n -------------------- \n Applying Training masks \n -------------------- \n\n")
    pool.map(apply_mask_training, training_houses)
    print("\n\n -------------------- \n Applying Validation masks \n -------------------- \n\n")
    pool.map(apply_mask_validation, validation_houses)
    print("\n\n -------------------- \n Applying Testing masks \n -------------------- \n\n")
    pool.map(apply_mask_testing, testing_houses)

    print("--- All done ---")