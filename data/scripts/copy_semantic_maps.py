import shutil
import os
import glob

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    dataset = "house3d"
    # dataset = "driveways_icra19"

    mode = "train"
    # mode = "test"
    
    num_masks = len(glob.glob("{dir_path}/training_data/{dataset}/masks/{mode}/transformed/*.png".format(dir_path=dir_path, mode=mode, dataset=dataset)))

    map_name = "world"
    world_filenames = glob.glob("{dir_path}/training_data/{dataset}/full_semantic/{mode}/*.png".format(dir_path=dir_path, mode=mode, dataset=dataset))
    world_ids = [w.split('/')[-1][5:-4] for w in world_filenames]
    for world_id in world_ids:
        print("Masking world: {world_id}".format(world_id=world_id))
        map_name_and_id = map_name + world_id
        for i in range(num_masks):
            full_name = map_name_and_id + "_" + str(i).zfill(3)
            orig_name = "{dir_path}/training_data/{dataset}/full_semantic/{mode}/{map_name_and_id}.png".format(mode=mode, map_name_and_id=map_name_and_id, dir_path=dir_path, dataset=dataset)
            new_name = "{dir_path}/training_data/{dataset}/copied_semantic/{mode}/{full_name}.png".format(mode=mode, full_name=full_name, dir_path=dir_path, dataset=dataset)
            shutil.copy(orig_name, new_name)
