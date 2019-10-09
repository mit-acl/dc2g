import tensorflow as tf
import numpy as np
import argparse
import json
import base64

import scipy.signal, scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.gridspec as gridspec
import glob
import pickle

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 17


# image_type = "full_semantic"
houses = [["n001m003h001","000"],["n001m004h001","000"],["n002m000h000","000"],["n002m002h015","000"],["n004m000h000","000"]]
# image_type = "masked_semantic"
# houses = [["n001m004h002","067"],["n001m003h002","047"],["n002m001h000","028"],["n002m001h006","034"],["n004m000h003","013"]]

sess = tf.Session()
def load_model(model_dir):
    saver = tf.train.import_meta_graph(model_dir + "/export.meta")
    saver.restore(sess, model_dir + "/export")
    input_vars = json.loads(tf.get_collection("inputs")[0])
    output_vars = json.loads(tf.get_collection("outputs")[0])
    input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
    goal_rgb = tf.get_default_graph().get_tensor_by_name(input_vars["goal_rgb"])
    output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])
    return input, output, goal_rgb

dataset = "driveways_bing_iros19"
mode = "test"
image_filename = "/home/mfe/code/dc2g/training_data/{dataset}/{image_type}/{mode}/{world_id}{mask}{goal}.{image_ending}"

model_dir = "/home/mfe/code/dc2g/pix2pix-tensorflow/driveways_bing_iros19_l1_full_test"
input, output, goal_rgb = load_model(model_dir)

# fig, ax = plt.subplots(figsize=(10, 10))
# fig, ax = plt.subplots(figsize=(len(houses), 4))
# fig, ax = plt.subplots(len(houses), 4)
# gs1 = gridspec.GridSpec(len(houses), 4)
# gs1.update(wspace=0.00025, hspace=0.05) # set the spacing between axes. 
# # fig.tight_layout()
# plt.subplot(len(houses), 4, 1)

# fig = plt.figure(figsize=(2*4,2*len(houses))) # Notice the equal aspect ratio

# for house_i, house in enumerate(houses):
#     world_id, mask_id = house
#     world = "world"+world_id
#     mask = "_"+mask_id

#     raw_data = plt.imread(image_filename.format(dataset=dataset, image_type="raw", goal="", world_id=world, mode=mode, image_ending="jpg", mask=""))
#     input_data = plt.imread(image_filename.format(dataset=dataset, image_type="masked_semantic", goal="", world_id=world, mode=mode, image_ending="png", mask=mask))
#     target_data = plt.imread(image_filename.format(dataset=dataset, image_type="masked_c2g", goal="-front_door", world_id=world, mode=mode, image_ending="png", mask=mask))

#     if input_data.shape[2] == 3:
#         input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
#     raw_data = scipy.misc.imresize(raw_data, (256,256), interp='nearest')
#     input_data = scipy.misc.imresize(input_data, (256,256), interp='nearest')
#     target_data = scipy.misc.imresize(target_data, (256,256), interp='nearest')
#     if np.max(input_data) > 1:
#         input_data = input_data / 255.
#     if input_data.shape[2] == 4:
#         input_data = input_data[:,:,:3]
#     goal_rgb_val = np.array([128., 0., 0.])/255.
#     output_data = sess.run(output, feed_dict={input: input_data, goal_rgb: goal_rgb_val})
    
#     # this_ax = plt.subplot(gs1[house_i*4+0])
#     this_ax = plt.subplot(len(houses), 4, house_i*4+1)
#     this_ax.imshow(raw_data)
#     this_ax.axis('off')
#     this_ax.set_aspect('equal')
#     if house_i == 0:
#         this_ax.set_title("Satellite")
#     # this_ax = plt.subplot(gs1[house_i*4+1])
#     this_ax = plt.subplot(len(houses), 4, house_i*4+2)
#     this_ax.imshow(input_data)
#     this_ax.axis('off')
#     this_ax.set_aspect('equal')
#     if house_i == 0:
#         this_ax.set_title("Semantic")
#     # this_ax = plt.subplot(gs1[house_i*4+2])
#     this_ax = plt.subplot(len(houses), 4, house_i*4+3)
#     this_ax.imshow(output_data)
#     this_ax.axis('off')
#     this_ax.set_aspect('equal')
#     if house_i == 0:
#         this_ax.set_title("Predicted C2G")
#     # this_ax = plt.subplot(gs1[house_i*4+3])
#     this_ax = plt.subplot(len(houses), 4, house_i*4+4)
#     this_ax.imshow(target_data)
#     this_ax.axis('off')
#     this_ax.set_aspect('equal')
#     if house_i == 0:
#         this_ax.set_title("True C2G")
# fig.subplots_adjust(wspace=0.05, hspace=0.05)
# this_ax = plt.subplot(5,4,1)
# ll=this_ax.plot((256,256), (0,256), '--r') #Let's plot it in red to show it better
# ll[0].set_clip_on(False)

# # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.1)
# plt.show()

fig = plt.figure(figsize=(2*4,2*len(houses))) # Notice the equal aspect ratio
gs1 = gridspec.GridSpec(len(houses),1)
gs1.update(left = 0.15, right = .3375 , wspace=0.02)
gs2 = gridspec.GridSpec(len(houses),3)
gs2.update(left = 0.3875, right = .575, wspace=.25)
sp1 = [plt.subplot(gs1[i,0]) for i in range(len(houses))]
sp2 = [plt.subplot(gs2[i,0]) for i in range(len(houses))]

for house_i, house in enumerate(houses):
    world_id, mask_id = house
    world = "world"+world_id
    mask = "_"+mask_id

    raw_data = plt.imread(image_filename.format(dataset=dataset, image_type="raw", goal="", world_id=world, mode=mode, image_ending="jpg", mask=""))
    input_data = plt.imread(image_filename.format(dataset=dataset, image_type="masked_semantic", goal="", world_id=world, mode=mode, image_ending="png", mask=mask))
    target_data = plt.imread(image_filename.format(dataset=dataset, image_type="masked_c2g", goal="-front_door", world_id=world, mode=mode, image_ending="png", mask=mask))

    if input_data.shape[2] == 3:
        input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
    raw_data = scipy.misc.imresize(raw_data, (256,256), interp='nearest')
    input_data = scipy.misc.imresize(input_data, (256,256), interp='nearest')
    target_data = scipy.misc.imresize(target_data, (256,256), interp='nearest')
    if np.max(input_data) > 1:
        input_data = input_data / 255.
    if input_data.shape[2] == 4:
        input_data = input_data[:,:,:3]
    goal_rgb_val = np.array([128., 0., 0.])/255.
    output_data = sess.run(output, feed_dict={input: input_data, goal_rgb: goal_rgb_val})
    
    # this_ax = plt.subplot(gs1[house_i*4+0])
    this_ax = plt.subplot(len(houses), 4, house_i*4+1)
    this_ax.imshow(raw_data)
    this_ax.axis('off')
    this_ax.set_aspect('equal')
    if house_i == 0:
        this_ax.set_title("Satellite")
    # this_ax = plt.subplot(gs1[house_i*4+1])
    this_ax = plt.subplot(len(houses), 4, house_i*4+2)
    this_ax.imshow(input_data)
    this_ax.axis('off')
    this_ax.set_aspect('equal')
    if house_i == 0:
        this_ax.set_title("Semantic")
    # this_ax = plt.subplot(gs1[house_i*4+2])
    this_ax = plt.subplot(len(houses), 4, house_i*4+3)
    this_ax.imshow(output_data)
    this_ax.axis('off')
    this_ax.set_aspect('equal')
    if house_i == 0:
        this_ax.set_title("Predicted C2G")
    # this_ax = plt.subplot(gs1[house_i*4+3])
    this_ax = plt.subplot(len(houses), 4, house_i*4+4)
    this_ax.imshow(target_data)
    this_ax.axis('off')
    this_ax.set_aspect('equal')
    if house_i == 0:
        this_ax.set_title("True C2G")
fig.subplots_adjust(wspace=0.05, hspace=0.05)
this_ax = plt.subplot(5,4,1)

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.1)
plt.show()