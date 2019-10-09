#!/usr/bin/env python3

from __future__ import division, print_function
from __future__ import absolute_import

import sys
import gym
import time
from optparse import OptionParser

import gym_minigrid

import tensorflow as tf
import numpy as np
import argparse
import json
import base64
import scipy.signal, scipy.misc

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import itertools

import util

np.set_printoptions(suppress=True, precision=8)

make_individual_figures = False
save_individual_figures = False
save_panel_figures = True

def mps(actions):
    '''
    Create a static motion primitive library given a list of possible actions
    inputs:
        actions: Actions class (defined in gym-minigrid)
    outputs:
        action_mps:
            np array of all permutations of actions of a certain length
            e.g. [[0,0,0,0], [0,0,0,1], [0,0,0,2], ...]
        coordinate_mps:
            np array of the change in (x,y,theta) coodinates undergone as a
            result of taking each action in each MP
            e.g. [[[0,0,pi/2], [0,0,pi/2], [0,0,pi/2], [0,0,pi/2]],
                  [[0,0,pi/2], [0,0,pi/2], [0,0,pi/2], [0,0,-pi/2]],
                    ...
                    ]
        accrued_coordinate_mps: 
            cumulative sum of change in coordinates along each MP
            e.g. [[[0,0,pi/2], [0,0,pi], [0,0,3pi/2], [0,0,0]],
                  [[0,0,pi/2], [0,0,pi], [0,0,3pi/2], [0,0,pi]],
                    ...
                    ]
    '''
    fwd = actions.forward
    r90 = actions.right
    l90 = actions.left
    action_mps = np.array([p for p in itertools.product([fwd, r90, l90], repeat=4)])
    coordinate_mps = np.zeros((action_mps.shape[0], action_mps.shape[1], 3)) # 3 elements == [x, y, theta]
    coordinate_mps[action_mps == l90] = np.array([0,0, np.pi / 2.0])
    coordinate_mps[action_mps == r90] = np.array([0,0, -np.pi / 2.0])
    coordinate_mps = np.cumsum(coordinate_mps, axis=1)
    forward_inds = np.where(action_mps == fwd)
    for i in range(len(forward_inds[0])):
        x_ind = forward_inds[0][i]
        y_ind = forward_inds[1][i]
        theta = coordinate_mps[x_ind, y_ind, 2]
        coordinate_mps[forward_inds[0][i], forward_inds[1][i], :] = np.array([np.cos(theta), np.sin(theta), 0])

    coordinate_mps = np.hstack([np.zeros((coordinate_mps.shape[0],1,3)), coordinate_mps]) # prepend (0,0,0) pose to each MP
    accrued_coordinate_mps = np.cumsum(coordinate_mps, axis=1)
    return action_mps, coordinate_mps, accrued_coordinate_mps

def rotate_and_shift_mps(mps, start_pos, start_theta):
    # start_pos = [x, y] (2-element np.array)
    # start_theta = angle (float)
    rotation_matrix = np.array([[np.cos(start_theta), np.sin(start_theta)],
                                [-np.sin(start_theta), np.cos(start_theta)]])
    rotated_mps = np.empty_like(mps)
    rotated_mps[:, :, :2] = np.dot(mps[:, :, :2], rotation_matrix)
    rotated_mps[:, :, 2] = util.wrap(mps[:, :, 2] + start_theta)
    rotated_and_shifted_mps = np.empty_like(mps)
    rotated_and_shifted_mps[:, :, :2] = rotated_mps[:, :, :2] + start_pos
    rotated_and_shifted_mps[:, :, 2] = rotated_mps[:, :, 2]
    return rotated_and_shifted_mps

def find_next_action(action_mps, mps, start_pos, start_theta, semantic_array, c2g_array, sess):
    # action_mps: sequences of actions (ints) that agent should take to follow that mp
    #   [[2,2,2], [2,1,2], [2,0,2]]
    # mps: sequences of coordinates (x,y,theta) that will be agent's state if that mp is followed, assuming initial pose is (0,0,0)
    #   [[1,0,0], [2,0,0], [3,0,0],
    #    [1,0,0], [1,0,pi/2], [1,1,pi/2],
    #    ...]
    # rotated_shifted_mps: take mps and adjust them to the actual poses the agent would occupy given its inital pose
    # c2g_array: 32x32 np array with the terminal cost of being at that grid cell
    # sess: tensorflow session

    print('--- find_next_action ---')
    print("action_mps:", action_mps)
    print("start_pos:", start_pos)
    print("start_theta:", start_theta)
    print("mps:", mps)
    rotated_shifted_mps = rotate_and_shift_mps(mps, start_pos, start_theta)
    print("rotated_shifted_mps:", rotated_shifted_mps)
    x_inds = np.clip(np.rint(rotated_shifted_mps[:, :, 0]).astype(int), 0, c2g_array.shape[0]-1)
    y_inds = np.clip(np.rint(rotated_shifted_mps[:, :, 1]).astype(int), 0, c2g_array.shape[1]-1)
    terminal_cost_along_each_mp = c2g_array[y_inds, x_inds]
    print("terminal_cost_along_each_mp:", terminal_cost_along_each_mp)

    step_cost_along_each_mp = np.cumsum(np.ones_like(terminal_cost_along_each_mp), axis=1) - 1
    print("step_cost_along_each_mp:", step_cost_along_each_mp)

    grass_inds = np.where(np.all(semantic_array == [0, 1, 0], axis=-1))
    house_inds = np.where(np.all(semantic_array == [1, 0, 1], axis=-1))
    terrain_cost_array = np.zeros_like(c2g_array)
    terrain_cost_array[grass_inds] = 1
    terrain_cost_array[house_inds] = 1
    terrain_cost_at_each_mp_pt = terrain_cost_array[y_inds, x_inds]
    terrain_cost_along_each_mp = np.cumsum(terrain_cost_at_each_mp_pt, axis=1)
    print("terrain_cost_along_each_mp:", terrain_cost_along_each_mp)

    w_terminal = 1
    w_terrain = -1e5
    w_step = -1
    total_cost_along_each_mp = w_terminal * terminal_cost_along_each_mp + w_terrain * terrain_cost_along_each_mp + w_step * step_cost_along_each_mp

    plt.figure("DC2G")
    plt.subplot(236)
    mp_costs = np.zeros_like(c2g_array)
    mp_costs[y_inds, x_inds] = total_cost_along_each_mp
    mp_costs[mp_costs < 0] = -1
    plt.cla()
    plt.imshow(mp_costs, cmap=plt.cm.Greens)
    semantic_array_alpha = np.zeros((semantic_array.shape[0], semantic_array.shape[1], 4))
    semantic_array_alpha[:,:,:3] = semantic_array
    semantic_array_alpha[:,:,3] = 0.5
    plt.imshow(semantic_array_alpha)
    # c2g_array_alpha = np.zeros((c2g_array.shape[0], c2g_array.shape[1], 4))
    # c2g_array_alpha[:,:,0] = c2g_array / 255.0
    # c2g_array_alpha[:,:,3] = 0.5
    # plt.imshow(c2g_array_alpha)

    # plt.figure("Motion Primitives")
    # plt.imshow(mp_costs, cmap=plt.cm.Greens)


    # Plot Motion Primitive Terrain Costs
    # accrued_cost_at_each_pt = terrain_cost_array[x_inds, y_inds]
    # plt.imshow(terrain_cost_array, cmap=plt.cm.Blues)
    
    # mp_c2gs_rgba = np.zeros((c2g_array.shape[0], c2g_array.shape[1], 4)) # rgba
    # # mp_c2gs_rgba[y_inds, x_inds, 0] = 255.0
    # mp_c2gs_rgba[y_inds, x_inds, 0] = c2g_array[y_inds, x_inds] / 255.0
    # mp_c2gs_rgba[y_inds, x_inds, 3] = 1.0
    # print(mp_c2gs_rgba)

    # plt.cla()
    # # world_array = plt.imread('/home/mfe/code/dc2g/training_data/full_semantic/world1.png')
    # plt.imshow(mp_c2gs_rgba)
    # # plt.imshow(world_array, alpha=0.1)
    # plt.pause(0.1)



    print('total_cost_along_each_mp:', total_cost_along_each_mp)
    if np.amax(total_cost_along_each_mp) < 0:
        print('------- WARNING: Stuck in infeasible terrain! --------')
        return 3 # null action
    # index_of_min_total_cost_per_mp = (total_cost_along_each_mp.shape[1]-1)*np.ones(total_cost_along_each_mp.shape[0], dtype=int)
    index_of_min_total_cost_per_mp = np.argmax(total_cost_along_each_mp, axis=1)
    print('index_of_min_total_cost_per_mp:', index_of_min_total_cost_per_mp)
    print('total_cost_along_each_mp[index_of_min_total_cost_per_mp]:', total_cost_along_each_mp[np.arange(index_of_min_total_cost_per_mp.shape[0]), index_of_min_total_cost_per_mp])
    index_of_best_mp = np.argmax(total_cost_along_each_mp[np.arange(index_of_min_total_cost_per_mp.shape[0]), index_of_min_total_cost_per_mp])
    index_of_best_cell_in_best_mp = index_of_min_total_cost_per_mp[index_of_best_mp]
    print('index_of_best_mp:', index_of_best_mp)
    print('index_of_best_cell_in_best_mp:', index_of_best_cell_in_best_mp)

    if index_of_best_cell_in_best_mp == 0:
        print('------- WARNING: Stuck at a local minimum! --------')
        return 3 # null action
    best_mp = action_mps[index_of_best_mp, :]
    print('best_mp:', best_mp)
    action = best_mp[0]
    print('action:', action)
    return action

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-EmptySLAM-32x32-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        first_obs = env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
        return first_obs

    obs = resetEnv()

    sess = tf.Session()
    model_dir = "/home/mfe/code/pix2pix-tensorflow/c2g_test/world1_25_256"
    saver = tf.train.import_meta_graph(model_dir + "/export.meta")
    saver.restore(sess, model_dir + "/export")
    input_vars = json.loads(tf.get_collection("inputs")[0])
    output_vars = json.loads(tf.get_collection("outputs")[0])
    input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
    output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

    action_mps, coordinate_mps, accrued_coordinate_mps = mps(env.actions)


    # Create a window to render into
    renderer = env.render('human')


    if make_individual_figures:
        fig = plt.figure('C2G', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Observation', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Environment', figsize=(12, 12))
        plt.axis('off')

    fig = plt.figure('DC2G', figsize=(12, 12))
    ax = fig.add_subplot(231)
    ax.set_axis_off()
    ax.set_title('World Semantic Map')
    ax = fig.add_subplot(232)
    ax.set_axis_off()
    ax.set_title('True Cost-to-Go')
    ax = fig.add_subplot(233)
    ax.set_axis_off()
    ax.set_title('Agent in Semantic Map')
    ax = fig.add_subplot(234)
    ax.set_axis_off()
    ax.set_title('Agent Observation')
    ax = fig.add_subplot(235)
    ax.set_axis_off()
    ax.set_title('Agent Estimated Cost-to-Go')
    ax = fig.add_subplot(236)
    ax.set_axis_off()
    ax.set_title('Agent Estimated Cost-to-Go for Current MPs')
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    total_world_array = plt.imread(env.world_image_filename)
    c2g_image_filename = '/home/mfe/code/dc2g/training_data/full_c2g/test/world' + str(env.world_id).zfill(3) + '.png'
    total_c2g_array = plt.imread(c2g_image_filename)
    plt.figure("DC2G")
    plt.subplot(231)
    plt.imshow(total_world_array)
    plt.figure("DC2G")
    plt.subplot(232)
    plt.imshow(total_c2g_array)

    # time.sleep(10)

    max_num_steps = 100
    while env.step_count < max_num_steps:
        plt.figure("DC2G")
        plt.suptitle('Step:' + str(env.step_count), fontsize=20)
        plt.subplot(233)
        render = env.render(mode='rgb_array', show_trajectory=True)
        plt.imshow(render)
        if make_individual_figures:
            plt.figure("Environment")
            plt.imshow(render)

        input_value = obs['image'].repeat(8, axis=0).repeat(8, axis=1)
        # plt.figure("DC2G")
        # plt.subplot(234)
        # plt.imshow(input_value)
        if make_individual_figures:
            plt.figure("Observation")
            plt.imshow(input_value)

        output_value = sess.run(output, feed_dict={input: input_value})
        
        # red_inds = np.where(np.all((output_value[:,:,0] > 0.9)))
        hsv = plt_colors.rgb_to_hsv(output_value)
        c2g_array = hsv[:, :, 2]
        c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map
        # plt.figure("DC2G")
        # plt.subplot(234)
        # plt.imshow(c2g_array, cmap=plt.cm.gray)
        # c2g_array = scipy.signal.medfilt(c2g_array, kernel_size=5)
        plt.figure("DC2G")
        plt.subplot(234)
        plt.imshow(c2g_array, cmap=plt.cm.gray)

        # c2g_array = c2g_array[::8, ::8] # downsample to 32x32 grid (from 256x256)
        # c2g_array = scipy.misc.imresize(c2g_array, (32,32), interp='bilinear')
        c2g_array = scipy.misc.imresize(c2g_array, (32,32), interp='nearest')
        plt.figure("DC2G")
        plt.subplot(235)
        plt.imshow(c2g_array, cmap=plt.cm.gray)

        # c2g_array[np.argmax(c2g_array)] = 100
        # print(c2g_array)
        # print(np.unravel_index(c2g_array.argmax(), c2g_array.shape))
        # c2g_array[np.unravel_index(c2g_array.argmax(), c2g_array.shape)] = 1000
        # c2g_array[(hsv[:,:,2] > 0.8)] = 1
        # c2g_array[(output_value[:,:,0] > 0.8) & (output_value[:,:,1] < 0.2) & (output_value[:,:,2] < 0.2)] = 1
        # print(red_inds)
        # output_value[red_inds] = np.array([0,0,0])
        
        if make_individual_figures:
            plt.figure("C2G")
            plt.imshow(output_value)

        action = find_next_action(action_mps, accrued_coordinate_mps, obs['pos'], obs['theta'], obs['image'], c2g_array, sess)
        if save_panel_figures:
            plt.figure("DC2G")
            plt.savefig('/home/mfe/code/gyms/gym-minigrid/results/8_29_18/panel/step_'+str(env.step_count).zfill(3)+'.png')
        if save_individual_figures:
            plt.figure("Observation")
            plt.savefig('/home/mfe/code/gyms/gym-minigrid/results/8_29_18/observation/step_'+str(env.step_count).zfill(3)+'.png')
            plt.figure("C2G")
            plt.savefig('/home/mfe/code/gyms/gym-minigrid/results/8_29_18/c2g/step_'+str(env.step_count).zfill(3)+'.png')
            plt.figure("Environment")
            plt.savefig('/home/mfe/code/gyms/gym-minigrid/results/8_29_18/environment/step_'+str(env.step_count).zfill(3)+'.png')
        plt.pause(0.1)
        obs, reward, done, info = env.step(action)
        env.render('human')
        print('step=%s, reward=%.2f' % (env.step_count, reward))
        if done:
            print('done!')
            break

if __name__ == "__main__":
    main()