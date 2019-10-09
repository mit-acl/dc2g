import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

def resize_mask(mask):
    return cv2.resize(mask, dsize=(600, 450), interpolation=cv2.INTER_NEAREST)

def transform_mask(mask, transformation):
    if transformation == 'none':
        return mask
    elif transformation == 'rotate_90_right':
        return np.rot90(mask, 1)
    elif transformation == 'rotate_90_left':
        return np.rot90(mask, -1)
    elif transformation == 'rotate_180':
        return np.rot90(mask, 2)
    elif transformation == 'invert_rotate_90_right':
        return np.rot90(1 - mask, 1)
    elif transformation == 'invert_rotate_90_left':
        return np.rot90(1 - mask, -1)
    elif transformation == 'invert_rotate_180':
        return np.rot90(1 - mask, 2)
    elif transformation == 'invert':
        return 1 - mask
    else:
        assert False, "Mask Transformation undefined."

if __name__ == "__main__":

    # dataset = "house3d"
    dataset = "driveways_bing_iros19"

    # mode = "train"
    # mode = "test"
    mode = "val"

    mask_path = "{dir_path}/training_data/{dataset}/masks/{mode}".format(dir_path=dir_path, mode=mode, dataset=dataset)
    mask_transformations = ['none', 'rotate_90_right', 'rotate_90_left', 'rotate_180', 'invert', 'invert_rotate_90_right', 'invert_rotate_90_left', 'invert_rotate_180']

    mask_num = 0
    num_masks = len(glob.glob("{mask_path}/raw/*.png".format(mask_path=mask_path)))
    for mask_index in range(num_masks):
        mask_id = str(mask_index).zfill(3)
        mask = plt.imread("{mask_path}/raw/{mask_id}.png".format(mask_path=mask_path, mask_id=mask_id))
        for i in range(len(mask_transformations)):
            mask_id = str(mask_num).zfill(3)
            mask_copy = mask.copy()
            transformed_mask = transform_mask(mask, mask_transformations[i])
            if dataset != "driveways_icra19": transformed_mask = resize_mask(transformed_mask)
            plt.imsave("{mask_path}/transformed/{mask_id}.png".format(mask_path=mask_path, mask_id=mask_id), transformed_mask, cmap=plt.cm.binary)
            mask_num += 1
            if np.all(mask == 0) or np.all(mask == 1):
                break
    print("--- All done ---")
