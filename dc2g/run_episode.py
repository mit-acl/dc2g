from __future__ import division, print_function
from __future__ import absolute_import

import sys
import gym
import time
from optparse import OptionParser

import tensorflow as tf
import numpy as np
import argparse
import json
import base64
import scipy.signal, scipy.misc
from scipy.ndimage.interpolation import shift
from skimage.transform import resize
from queue import *
import collections

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import itertools

import cv2
import scipy.ndimage.morphology
import pickle

from dc2g.util import get_traversable_colors, get_goal_colors, find_traversable_inds, find_goal_inds, inflate, wrap, round_base_down, round_base

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(suppress=True, precision=4)
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

dir_path = os.path.dirname(os.path.realpath(__file__))


make_individual_figures = False
save_individual_figures = True
save_panel_figures = False
plot_panels = True

import sys, signal
def signal_handler(signal, frame):
    try:
        print("Shutting down environment gracefully...")
        ENVIRONMENT.on_shutdown()
    except:
        print("Environment doesn't support graceful shutdown.")
    print("\nprogram exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

global ENVIRONMENT

def check_if_goal_reachable(goal_array, reachable_array):
    # This hasn't really been tested. supposed to tell you if any of the goal inds are within reachable inds ==> your goal is reachable
    if goal_array.ndim > 2:
        goal_array = np.any(goal_array, axis=2)
    reachable_goal_inds = np.where(np.logical_and(goal_array, reachable_array))
    goal_is_reachable = len(reachable_goal_inds[0]) > 0
    return goal_is_reachable, reachable_goal_inds

def find_reachable_frontier_indices2(semantic_array, reachable_array):
    # Find the array of observed & traversable inds that have at least 1 unobserved neighbor
    semantic_array_shifted_left = shift(semantic_array, (0,-1,0), cval=np.nan)
    semantic_array_shifted_right = shift(semantic_array, (0,1,0), cval=np.nan)
    semantic_array_shifted_up = shift(semantic_array, (-1,0,0), cval=np.nan)
    semantic_array_shifted_down = shift(semantic_array, (1,0,0), cval=np.nan)

    frontier_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1], 4)) # 4 is num actions

    frontier_up_inds = np.where(np.all([np.sum(semantic_array[:,:,:3], axis=2) > 0.1, semantic_array_shifted_up[:,:,0] < 0.1, semantic_array_shifted_up[:,:,1] < 0.1, semantic_array_shifted_up[:,:,2] < 0.1], axis=0))
    frontier_down_inds = np.where(np.all([np.sum(semantic_array[:,:,:3], axis=2) > 0.1, semantic_array_shifted_down[:,:,0] < 0.1, semantic_array_shifted_down[:,:,1] < 0.1, semantic_array_shifted_down[:,:,2] < 0.1], axis=0))
    frontier_left_inds = np.where(np.all([np.sum(semantic_array[:,:,:3], axis=2) > 0.1, semantic_array_shifted_left[:,:,0] < 0.1, semantic_array_shifted_left[:,:,1] < 0.1, semantic_array_shifted_left[:,:,2] < 0.1], axis=0))
    frontier_right_inds = np.where(np.all([np.sum(semantic_array[:,:,:3], axis=2) > 0.1, semantic_array_shifted_right[:,:,0] < 0.1, semantic_array_shifted_right[:,:,1] < 0.1, semantic_array_shifted_right[:,:,2] < 0.1], axis=0))

    frontier_array[(frontier_up_inds[0], frontier_up_inds[1], 1*np.ones_like(frontier_up_inds[0]))] = 1
    frontier_array[(frontier_down_inds[0], frontier_down_inds[1], 3*np.ones_like(frontier_down_inds[0]))] = 1
    frontier_array[(frontier_left_inds[0], frontier_left_inds[1], 0*np.ones_like(frontier_left_inds[0]))] = 1
    frontier_array[(frontier_right_inds[0], frontier_right_inds[1], 2*np.ones_like(frontier_right_inds[0]))] = 1

    reachable_frontier_array = np.zeros_like(frontier_array)
    for i in range(frontier_array.shape[2]):
        reachable_frontier_array[:,:,i] = np.logical_and(frontier_array[:,:,i], reachable_array)

    fov_aware_frontier_array = get_fov_aware_goal_array2(frontier_array)
    fov_aware_reachable_frontier_array = np.zeros_like(fov_aware_frontier_array)
    for i in range(fov_aware_frontier_array.shape[2]):
        fov_aware_reachable_frontier_array[:,:,i] = np.logical_and(fov_aware_frontier_array[:,:,i], reachable_array)

    return frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array

def find_reachable_frontier_indices(semantic_array, reachable_array):
    # Find the array of observed & traversable inds that have at least 1 unobserved neighbor
    semantic_array_shifted_left = shift(semantic_array, (0,-1,0), cval=np.nan)
    semantic_array_shifted_right = shift(semantic_array, (0,1,0), cval=np.nan)
    semantic_array_shifted_up = shift(semantic_array, (-1,0,0), cval=np.nan)
    semantic_array_shifted_down = shift(semantic_array, (1,0,0), cval=np.nan)

    # # Shift semantic map up & find unobserved regions ==> should be pointed downward to see those
    # unobserved_up_inds = np.where(np.all(abs(semantic_array_shifted_up) < 1e-5, axis=-1))
    # unobserved_up_inds_arr = np.dstack([unobserved_up_inds[1], unobserved_up_inds[0], 1*np.ones_like(unobserved_up_inds[0], dtype=int)])[0]
    # unobserved_down_inds = np.where(np.all(abs(semantic_array_shifted_down) < 1e-5, axis=-1))
    # unobserved_down_inds_arr = np.dstack([unobserved_down_inds[1], unobserved_down_inds[0], 3*np.ones_like(unobserved_down_inds[0], dtype=int)])[0]
    # unobserved_left_inds = np.where(np.all(abs(semantic_array_shifted_left) < 1e-5, axis=-1))
    # unobserved_left_inds_arr = np.dstack([unobserved_left_inds[1], unobserved_left_inds[0], 0*np.ones_like(unobserved_left_inds[0], dtype=int)])[0]
    # unobserved_right_inds = np.where(np.all(abs(semantic_array_shifted_right) < 1e-5, axis=-1))
    # unobserved_right_inds_arr = np.dstack([unobserved_right_inds[1], unobserved_right_inds[0], 2*np.ones_like(unobserved_right_inds[0], dtype=int)])[0]
    # # print("unobserved_up_inds_arr: {}".format(unobserved_up_inds_arr))

    # observed_traversable_inds_with_theta_arr = np.tile(np.hstack([observed_traversable_inds_arr, np.zeros((observed_traversable_inds_arr.shape[0], 1), dtype=int)]), (4, 1))
    # # print("observed_traversable_inds_with_theta_arr: {}".format(observed_traversable_inds_with_theta_arr))
    # num_observable_inds = observed_traversable_inds_arr.shape[0]
    # # print("num_observable_inds: {}".format(num_observable_inds))
    # for i in range(1, 4):
    #     observed_traversable_inds_with_theta_arr[num_observable_inds*i:num_observable_inds*(i+1), 2] = i
    # # print("observed_traversable_inds_with_theta_arr: {}".format(observed_traversable_inds_with_theta_arr))
    # frontier_inds_arr = np.array([x for x in set(tuple(x) for x in observed_traversable_inds_with_theta_arr) & (set(tuple(x) for x in unobserved_up_inds_arr) | set(tuple(x) for x in unobserved_down_inds_arr) | set(tuple(x) for x in unobserved_left_inds_arr) | set(tuple(x) for x in unobserved_right_inds_arr))])
    # # print("frontier_inds_arr: {}".format(frontier_inds_arr))
    # frontier_inds = (frontier_inds_arr[:, 1], frontier_inds_arr[:, 0])
    # frontier_headings = frontier_inds_arr[:,2]

    ############# New
    frontier_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1], 4)) # 4 is num actions

    frontier_up_inds = np.where(np.all([reachable_array == 1, semantic_array_shifted_up[:,:,0] < 0.1, semantic_array_shifted_up[:,:,1] < 0.1, semantic_array_shifted_up[:,:,2] < 0.1], axis=0))
    frontier_down_inds = np.where(np.all([reachable_array == 1, semantic_array_shifted_down[:,:,0] < 0.1, semantic_array_shifted_down[:,:,1] < 0.1, semantic_array_shifted_down[:,:,2] < 0.1], axis=0))
    frontier_left_inds = np.where(np.all([reachable_array == 1, semantic_array_shifted_left[:,:,0] < 0.1, semantic_array_shifted_left[:,:,1] < 0.1, semantic_array_shifted_left[:,:,2] < 0.1], axis=0))
    frontier_right_inds = np.where(np.all([reachable_array == 1, semantic_array_shifted_right[:,:,0] < 0.1, semantic_array_shifted_right[:,:,1] < 0.1, semantic_array_shifted_right[:,:,2] < 0.1], axis=0))

    frontier_array[(frontier_up_inds[0], frontier_up_inds[1], 1*np.ones_like(frontier_up_inds[0]))] = 1
    frontier_array[(frontier_down_inds[0], frontier_down_inds[1], 3*np.ones_like(frontier_down_inds[0]))] = 1
    frontier_array[(frontier_left_inds[0], frontier_left_inds[1], 0*np.ones_like(frontier_left_inds[0]))] = 1
    frontier_array[(frontier_right_inds[0], frontier_right_inds[1], 2*np.ones_like(frontier_right_inds[0]))] = 1

    # print("{} reachable pts.".format(np.sum(reachable_array)))
    # print("{} right frontier pts.".format(np.sum(frontier_array[:,:,2])))
    # plt.figure("tmp")
    # # plt.subplot(241)
    # # plt.imshow(frontier_array[:,:,0])
    # # plt.subplot(242)
    # # plt.imshow(frontier_array[:,:,1])
    # plt.subplot(243)
    # plt.imshow(frontier_array[:,:,2])
    # # plt.subplot(244)
    # # plt.imshow(frontier_array[:,:,3])
    # plt.subplot(245)
    # plt.imshow(reachable_array)
    # plt.pause(10)

    return frontier_array

def check_if_at_goal(goal_array, pos, theta_ind, verbose=False):
    at_goal = goal_array[pos[1], pos[0]]
    if type(at_goal) == np.ndarray: # goal array is only based on positions, not positions and orientations
        # if verbose: print(at_goal)
        started_at_goal = bool(at_goal[theta_ind])
        theta_in_goal = True
    else:
        started_at_goal = bool(at_goal)
        theta_in_goal = False
    return started_at_goal, theta_in_goal

# def get_fov_aware_goal_array(raw_goal_array):
#     if np.sum(raw_goal_array) == 0:
#         return raw_goal_array
#     global ENVIRONMENT
#     # If the raw goal_array contains an axis that defines what theta_ind will be able to see that goal, that info can be ignored with this function.
#     if raw_goal_array.ndim > 2:
#         goal_array = np.any(raw_goal_array, axis=2)
#     else:
#         goal_array = raw_goal_array.copy()

#     plt.figure("goal_array")
#     plt.imshow(goal_array)
#     plt.pause(1)
#     goal_inds = np.where(goal_array == 1)
#     camera_fov = np.pi/12 # full FOV in radians
#     camera_range_in_meters = 2
#     num_theta_inds = 4
#     fov_aware_goal_array = np.repeat(goal_array[:, :, np.newaxis], num_theta_inds, axis=2)
#     # fov_aware_goal_array = goal_array.copy()
#     tx1, ty1, tx2, ty2 = ENVIRONMENT.rescale(0, 0, camera_range_in_meters, camera_range_in_meters)
#     camera_range_x = tx2 - tx1; camera_range_y = ty2 - ty1
#     grid_inds = np.indices(goal_array.shape)
#     grid_array = np.dstack([grid_inds[1], grid_inds[0]])
#     for i in range(len(goal_inds[0])):
#         print("{}/{}".format(i, len(goal_inds[0])))
#         goal_pos = np.array([goal_inds[1][i], goal_inds[0][i]])
#         rel_pos = goal_pos - grid_array
#         ellipse_r = rel_pos**2 / np.array([camera_range_x, camera_range_y])**2
#         r_arr = np.sum(ellipse_r, axis=2) <= 1

#         rel_angle = np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) # angle from a particular grid cell to the current cam pos
#         for theta_ind in range(num_theta_inds):
#             cam_angle = wrap(np.radians((theta_ind+2) * 90))
#             angle_offset = wrap(rel_angle + cam_angle)
#             angle_arr = abs(angle_offset) < (camera_fov/2)
#             observable_arr = np.bitwise_and(r_arr, angle_arr)
#             fov_aware_goal_array[:,:,theta_ind][observable_arr == 1] = 1
#     plt.figure("fov_aware_goal_array")
#     plt.imshow(fov_aware_goal_array[:,:,2])
#     plt.imshow(np.any(fov_aware_goal_array, axis=2))
#     plt.pause(1)

#     # TODO!!!!
#     return fov_aware_goal_array

def get_fov_aware_goal_array2(raw_goal_array):
    num_theta_inds = 4
    if np.sum(raw_goal_array) == 0:
        # none of the points are goals, so no point in using the FOV to see which points see the non-existent goal
        return raw_goal_array
        # if raw_goal_array.ndim > 2:
        #     return raw_goal_array
        # else:
        #     return np.repeat(raw_goal_array[:, :, np.newaxis], num_theta_inds, axis=2)
    global ENVIRONMENT
    # If the raw goal_array contains an axis that defines what theta_ind will be able to see that goal, that info can be ignored with this function.
    if raw_goal_array.ndim > 2:
        goal_array = np.any(raw_goal_array, axis=2)
    else:
        goal_array = raw_goal_array.copy()

    goal_inds = np.where(goal_array == 1)

    camera_fov = ENVIRONMENT.camera_fov # full FOV in radians
    # camera_range_x = ENVIRONMENT.camera_range_x-1; camera_range_y = ENVIRONMENT.camera_range_y-1;
    camera_range_x = ENVIRONMENT.camera_range_x; camera_range_y = ENVIRONMENT.camera_range_y;
    # camera_range_x = int(ENVIRONMENT.camera_range_x / ENVIRONMENT.grid_resolution); camera_range_y = int(ENVIRONMENT.camera_range_y / ENVIRONMENT.grid_resolution);
    # camera_fov = np.pi/2 # full FOV in radians
    # camera_range_x = 10; camera_range_y = 10;

    padded_goal_array = np.pad(goal_array,((camera_range_y,camera_range_y),(camera_range_x,camera_range_x)), 'constant',constant_values=0)
    fov_aware_goal_array = np.repeat(padded_goal_array[:, :, np.newaxis], num_theta_inds, axis=2)

    window = np.empty((2*camera_range_y+1, 2*camera_range_x+1))
    grid_inds = np.indices(window.shape)
    grid_array = np.dstack([grid_inds[1], grid_inds[0]])
    goal_pos = np.array([camera_range_x, camera_range_y])
    rel_pos = goal_pos.astype(np.float32) - grid_array
    ellipse_r = rel_pos**2 / np.array([camera_range_x, camera_range_y])**2
    r_arr = np.sum(ellipse_r, axis=2) < 1

    rel_angle = np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) # angle from a particular grid cell to the current cam pos
    observable_arr = np.repeat(r_arr[:, :, np.newaxis], num_theta_inds, axis=2)
    for theta_ind in range(num_theta_inds):
        cam_angle = wrap(np.radians((theta_ind+2) * 90))
        angle_offset = wrap(rel_angle + cam_angle)
        angle_arr = abs(angle_offset) < (camera_fov/2)
        observable_arr[:,:,theta_ind] = np.bitwise_and(r_arr, angle_arr)
        struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        # if theta_ind == 3:
        #     plt.figure('b4')
        #     plt.imshow(observable_arr[:,:,3])
        observable_arr[:,:,theta_ind] = scipy.ndimage.morphology.binary_erosion(observable_arr[:,:,theta_ind], struct2).astype(observable_arr.dtype)
        observable_arr[camera_range_y,camera_range_x,theta_ind] = 1
        observable_arr[camera_range_y+int(np.sin(cam_angle)),camera_range_x+int(np.cos(cam_angle)),theta_ind] = 1
        observable_arr[camera_range_y+2*int(np.sin(cam_angle)),camera_range_x+2*int(np.cos(cam_angle)),theta_ind] = 1
        # if theta_ind == 3:
        #     plt.figure('after')
        #     plt.imshow(observable_arr[:,:,3])

    for i in range(len(goal_inds[0])):
        gy = goal_inds[0][i]; gx = goal_inds[1][i]
        fov_aware_goal_array[gy:gy+2*camera_range_y+1,gx:gx+2*camera_range_x+1, :] += observable_arr

    fov_aware_goal_array = fov_aware_goal_array > 0
    # fov_aware_goal_array = (fov_aware_goal_array > 0).astype(int)

    # for i in range(len(goal_inds[0])):
    #     gy = goal_inds[0][i]; gx = goal_inds[1][i]
    #     fov_aware_goal_array[gy+camera_range_y:gy+camera_range_y+1,gx+camera_range_x:gx+camera_range_x+1, :] = 2

    # plt.figure("fov_goal_array")
    # plt.subplot(2,2,1)
    # plt.imshow(fov_aware_goal_array[:,:,0])
    # plt.subplot(2,2,2)
    # plt.imshow(fov_aware_goal_array[:,:,1])
    # plt.subplot(2,2,3)
    # plt.imshow(fov_aware_goal_array[:,:,2])
    # plt.subplot(2,2,4)
    # plt.imshow(fov_aware_goal_array[:,:,3])
    # plt.pause(0.01)

    unpadded_fov_aware_goal_array = fov_aware_goal_array[camera_range_y:-camera_range_y, camera_range_x:-camera_range_x]
    return unpadded_fov_aware_goal_array

def breadth_first_search2(traversable_array, goal_array, start_pos, start_theta_ind, exhaustive=False):
    '''
    Description: Starting from start_pos, start_theta_ind, execute a BFS among traversable nodes
        in the graph. If a goal position is found, stop the search -- unless the exhaustive flag is set,
        in which case keep searching until all graph nodes have been explored fully to determine
        all reachable nodes from start.
    inputs:
        - traversable_array: nxn binary np array of which positions are traversable, or which are reachable from current state
        - goal_array: either
            nxn binary np array of goal states, when final orientation doesn't matter (reaching a position in the map)
            nxnx4 binary np array of goal, when final orientation matters (reaching a frontier)
        - start_pos: current position of robot in gridworld (e.g. np.array([px, py]))
        - start_theta_ind: current heading index of robot in gridworld (e.g. 2) - some int btwn 0-3 inclusive
        - exhaustive: whether to search til something in goal_array is found, or to search til queue is empty
    outputs:
        - if no goal was provided or goal == position ==> returns None
        - if not exhaustive:
            - if goal not found ==> returns None
            - if goal found ==> returns action_list
        - if exhaustive ==> returns dict of child coord -> (parent coord, action)
    '''
    # print("There are {} traversable pts, {} goal points.".format(np.sum(traversable_array), np.sum(goal_array)))
    # print("start_pos: {}, start_theta_ind: {}".format(start_pos, start_theta_ind))
    if traversable_array[start_pos[1], start_pos[0]] == 0:
        print('[breadth_first_search] the starting position is not traversable. that seems pretty wrong.')
        return
    started_at_goal, theta_in_goal = check_if_at_goal(goal_array, start_pos, start_theta_ind, verbose=True)
    if started_at_goal:
        # If currently at goal position, remove it from consideration
        # print('[breadth_first_search] we are currently at a goal position. removing it from goal list.')
        if theta_in_goal:
            goal_array[start_pos[1], start_pos[0], start_theta_ind] = 0
        else:
            goal_array[start_pos[1], start_pos[0]] = 0
    if np.sum(goal_array) == 0 and not exhaustive:
        # If there aren't any goals, then quit, unless you're doing exhaustive search which has no goal by definition
        # print('[breadth_first_search] something got messed up: len(goal_positions_list) == 0.')
        return
    meta = dict()
    root = (start_pos[0], start_pos[1], start_theta_ind)
    visited_array = np.zeros((traversable_array.shape[0], traversable_array.shape[1], 4)) # 4 == number of theta_ind values
    queue = collections.deque([root])
    meta[root] = (None, None)
    num_vertices_popped = 0
    while queue:
        num_vertices_popped += 1
        vertex = queue.popleft()
        # print("[breadth_first_search] vertex: {}".format(vertex))
        if not exhaustive:
            vertex_at_goal, _ = check_if_at_goal(goal_array, vertex[:2], vertex[2])
            if vertex_at_goal:
                # print("BFS found one of the goals. A path exists to {}".format([vertex[0], vertex[1], vertex[2]]))
                return construct_path(vertex, meta)
        px, py, theta_ind = vertex
        children, actions = get_children(px, py, theta_ind, traversable_array.shape) # TODO: This probably should be environment-specific (action set)
        for i in range(len(children)):
            # print("[breadth_first_search] children[i]: {}".format(children[i]))
            try:
                skip = traversable_array[children[i][1], children[i][0]] == 0
            except IndexError:
                skip = True
            if skip:
                # print("child is not traversable")
                continue
            if visited_array[children[i][1], children[i][0], children[i][2]] == 0:
                visited_array[children[i][1], children[i][0], children[i][2]] = 1
                queue.append(children[i])
                if children[i] not in meta:
                    meta[children[i]] = (vertex, actions[i])
        # if num_vertices_popped % 100 == 0:
        # # if num_vertices_popped % 1 == 0:
        #     print("[breadth_first_search] visualizing visited_array...")
        #     plt.figure('bfs')
        #     plt.imshow(visited_array[:,:,0])
        #     plt.pause(0.01)
    if not exhaustive:
        print("[breadth_first_search] warning: queue is empty. while loop ended.")
        return
    return meta, visited_array[:,:,0]

def get_children(gridmap_x, gridmap_y, theta_ind, gridmap_upper_bnds):

    real_x, real_y = ENVIRONMENT.to_coor(gridmap_x, gridmap_y)
    next_states, actions = ENVIRONMENT.next_coords(real_x, real_y, theta_ind)
    next_gridmap_x, next_gridmap_y = ENVIRONMENT.to_grid(next_states[:,0], next_states[:,1])
    # print("started at gridmap_x, gridmap_y, theta_ind: ({},{},{})".format(gridmap_x, gridmap_y, theta_ind))
    # print("real_x, real_y, theta_ind: ({},{},{})".format(real_x, real_y, theta_ind))
    # print("next_states: {}".format(next_states))
    # print("next_gridmap_x, next_gridmap_y: ({},{})".format(next_gridmap_x, next_gridmap_y))

    gridmap_discretization = int(1./ENVIRONMENT.grid_resolution) # TODO: this could become a property of the environment... or at least based on the minimum action
    # gridmap_discretization = 9 # TODO: this could become a property of the environment... or at least based on the minimum action

    num_jumps_x = np.around((next_gridmap_x - gridmap_x) / gridmap_discretization).astype(int)
    next_gridmap_x = gridmap_x + gridmap_discretization*num_jumps_x
    # print("num_jumps_x, next_gridmap_x: ({},{})".format(num_jumps_x, next_gridmap_x))
    num_jumps_y = np.around((next_gridmap_y - gridmap_y) / gridmap_discretization).astype(int)
    next_gridmap_y = gridmap_y + gridmap_discretization*num_jumps_y
    # print("num_jumps_y, next_gridmap_y: ({},{})".format(num_jumps_y, next_gridmap_y))

    # next_gridmap = np.zeros_like(next_states, dtype=int)
    # gridmap_offset_x = gridmap_x % gridmap_discretization
    # gridmap_offset_y = gridmap_y % gridmap_discretization
    # print("gridmap_offset_x, gridmap_offset_y: ({},{})".format(gridmap_offset_x, gridmap_offset_y))
    # next_gridmap_x_tmp = round_base(next_gridmap_x, gridmap_discretization)
    # next_gridmap_y_tmp = round_base(next_gridmap_y, gridmap_discretization)
    # print("tmp next_gridmap_x, next_gridmap_y: ({},{})".format(next_gridmap_x_tmp, next_gridmap_y_tmp))
    # next_gridmap_x = round_base_down(next_gridmap_x, gridmap_discretization) + gridmap_offset_x
    # next_gridmap_y = round_base_down(next_gridmap_y, gridmap_discretization) + gridmap_offset_y
    # print("discretized next_gridmap_x, next_gridmap_y: ({},{})".format(next_gridmap_x, next_gridmap_y))

    next_gridmap_list = []
    actions_in_bounds = []
    for i in range(next_states.shape[0]):
        if next_gridmap_x[i] >= 0 and next_gridmap_x[i] < gridmap_upper_bnds[1] and next_gridmap_y[i] >= 0 and next_gridmap_y[i] < gridmap_upper_bnds[0]:
            next_gridmap_list.append((next_gridmap_x[i], next_gridmap_y[i], int(next_states[i,2])))
            actions_in_bounds.append(actions[i])

    # print("next_gridmap_list: {}".format(next_gridmap_list))

    return next_gridmap_list, actions_in_bounds

    # straight_gridmap_x = np.clip(straight_gridmap_x, 0, gridmap_upper_bnds[0]-1)
    # straight_gridmap_y = np.clip(straight_gridmap_y, 0, gridmap_upper_bnds[1]-1)
    # # print("straight_gridmap_x: {}, straight_gridmap_y: {}".format(straight_gridmap_x, straight_gridmap_y))
    # action_dict = {0: (straight_gridmap_x, straight_gridmap_y, theta_ind),
    #                1: (gridmap_x, gridmap_y, (theta_ind + 1) % 4),
    #                2: (gridmap_x, gridmap_y, (theta_ind - 1) % 4)}
    # # print("action_dict: {}".format(action_dict))
    # return list(action_dict.values()), list(action_dict.keys()) 

# Produce a backtrace of the actions taken to find the goal node, using the
# recorded meta dictionary
def construct_path(state, meta):
    if len(state) == 2:
        # If we don't specify a final theta (only give position), try all possibilities and return shortest path
        shortest_action_list = None
        for theta_ind in range(4):
            full_state = (state[0], state[1], theta_ind)
            # print("full_state:", full_state)
            if full_state in meta.keys():
                # print("full_state is in meta.")
                action_list, final_state, path = construct_path_full_state(full_state, meta)
                # print("action_list:", action_list)
                # print("path:", path)
                if shortest_action_list is None or len(shortest_action_list) > len(action_list):
                    # print("shorter path found!")
                    shortest_action_list = action_list
                    quickest_final_state = final_state
                    shortest_path = path
        return shortest_action_list, quickest_final_state, shortest_path
    else:
        # state is already fully defined, so just compute optimal path to that one state
        return construct_path_full_state(state, meta)

def construct_path_full_state(state, meta):
    final_state = state
    action_list = list()
    path = list()
    # Continue until you reach root meta data (i.e. (None, None))
    while meta[state][0] is not None:
        last_state = state
        state, action = meta[state]
        action_list.append(action)
        path.append(state)

    action_list.reverse()
    return action_list, final_state, path

# def resize_semantic_array(semantic_array, reachable_inds_arr):
#     observed_inds = np.where(np.any(abs(semantic_array) > 1e-5, axis=-1))
#     min_x_ind = np.min(observed_inds[0])
#     min_y_ind = np.min(observed_inds[1])
#     max_x_ind = np.max(observed_inds[0])
#     max_y_ind = np.max(observed_inds[1])
#     diff = max(max_x_ind - min_x_ind, max_y_ind - min_y_ind) + 1
#     cropped_semantic_array = semantic_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff]
#     cropped_padded_semantic_array = np.pad(cropped_semantic_array, ((2,2),(2,2),(0,0)), 'constant')
#     resized_semantic_array = resize(cropped_padded_semantic_array, (256,256,3), order=0)
#     return resized_semantic_array

def size_agnostic_c2g_query(semantic_array, tf_sess, tf_tensors):
    padding = 2

    # Find largest square that contains observed portion of map
    observed_inds = np.where(np.any(abs(semantic_array) > 1e-5, axis=-1))
    min_x_ind = np.min(observed_inds[0])
    min_y_ind = np.min(observed_inds[1])
    max_x_ind = np.max(observed_inds[0])
    max_y_ind = np.max(observed_inds[1])
    diff = max(max_x_ind - min_x_ind, max_y_ind - min_y_ind) + 1
    # Crop the map down to that square
    cropped_semantic_array = semantic_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff]
    # Add some padding to reduce edge effects of image translator - TODO: verify if needed
    cropped_padded_semantic_array = np.pad(cropped_semantic_array, ((padding,padding),(padding,padding),(0,0)), 'constant')
    # Scale image up to translator's desired size (256x256)
    # resized_semantic_array = scipy.misc.imresize(cropped_padded_semantic_array, tf_tensors['input'].shape, interp='nearest')
    # resized_semantic_array = resize(cropped_padded_semantic_array, tf_tensors['input'].shape, order=0)

    input_data = cropped_padded_semantic_array
    if input_data.shape[2] == 3:
        input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
    input_data = scipy.misc.imresize(input_data, tf_tensors['input'].shape[:2], interp='nearest')
    if np.max(input_data) > 1:
        input_data = input_data / 255.
    if input_data.shape[2] == 4:
        input_data = input_data[:,:,:3]

    # Query the encoder-decoder network for image translation
    feed_dict = {tf_tensors['input']: input_data}
    if 'goal_rgb' in tf_tensors:
        goal_rgb = goal_rgb_val = np.array([128., 0., 0.])/255.
        feed_dict[tf_tensors['goal_rgb']] = goal_rgb
    output_value = tf_sess.run(tf_tensors['output'], feed_dict=feed_dict)

    # Remove red from image, convert to grayscale via HSV saturation
    hsv = plt_colors.rgb_to_hsv(output_value)
    c2g_array = hsv[:, :, 2]
    c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map

    # Scale down to padded, cropped size
    c2g_array = scipy.misc.imresize(c2g_array, (cropped_padded_semantic_array.shape[0], cropped_padded_semantic_array.shape[1]), interp='nearest')
    # Remove padding
    c2g_array = c2g_array[padding:-padding, padding:-padding]

    # Stick remaining square back into right slot within full semantic map
    final_c2g_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
    final_c2g_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff] = c2g_array

    return final_c2g_array, semantic_array, output_value

def c2g_query(semantic_array, tf_sess, tf_tensors):

    # input_value_ = scipy.misc.imresize(semantic_array, tf_tensors['input'].shape, interp='nearest')[:,:,:3]
    input_data = semantic_array
    if input_data.shape[2] == 3:
        input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
    input_data = scipy.misc.imresize(input_data, tf_tensors['input'].shape[:2], interp='nearest')
    if np.max(input_data) > 1:
        input_data = input_data / 255.
    if input_data.shape[2] == 4:
        input_data = input_data[:,:,:3]
    # print("after:", input_data)
    # input_value = semantic_array.repeat(8, axis=0).repeat(8, axis=1)
    # plt.figure('tmps')
    # plt.imshow(np.hstack([input_value, input_data]))
    # plt.show()
    feed_dict = {tf_tensors['input']: input_data}
    if 'goal_rgb' in tf_tensors:
        goal_rgb = goal_rgb_val = np.array([128., 0., 0.])/255.
        feed_dict[tf_tensors['goal_rgb']] = goal_rgb
    output_value = tf_sess.run(tf_tensors['output'], feed_dict=feed_dict)
    output_value_resized = scipy.misc.imresize(output_value, semantic_array.shape[:2], interp='nearest')
    # plt.figure('output')
    # plt.imshow(output_value)
    # plt.show()
    hsv = plt_colors.rgb_to_hsv(output_value)
    c2g_array = hsv[:, :, 2]
    c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map
    c2g_array = scipy.misc.imresize(c2g_array, semantic_array.shape[:2], interp='nearest')
    return c2g_array, semantic_array, output_value_resized

def dc2g_planner(position, theta_ind, semantic_array, reachable_array, tf_sess, tf_tensors, bfs_parent_dict, step_number, rescale_semantic_map=False):
    '''
    Description: TODO
    inputs:
        - position: current position of robot in gridworld (e.g. np.array([px, py]))
        - theta_ind: current heading index of robot in gridworld (e.g. 2) - some int btwn 0-3 inclusive
        - semantic_array: 32x32x3 np array of robot's current partial knowledge of gridworld
        - reachable_inds_arr: nx2 np array of grid coordinates the agent can definitely reach given its current partial semantic map knowledge
        - tf_sess: tensorflow session
        - tf_input: tensorflow shortcut to refer to 256x256x3 image input
        - tf_output: tensorflow shortcut to refer to 256x256x3 image output
        - bfs_parent_dict: dictionary keyed by each reachable (px, py, theta_ind) coordinate, s.t. child coord -> (parent coord, action)
                            created by running exhaustive BFS on grid from current coordinate
    outputs:
        - action: int of action to take
    '''
    if rescale_semantic_map:
        c2g_array, input_value, raw_c2g = size_agnostic_c2g_query(semantic_array, tf_sess, tf_tensors)
    else:
        c2g_array, input_value, raw_c2g = c2g_query(semantic_array, tf_sess, tf_tensors)
    if plot_panels:
        plt.figure("DC2G")
        plt.subplot(132)
        plt.imshow(raw_c2g, cmap=plt.cm.gray)
        # plt.imshow(c2g_array, cmap=plt.cm.gray)
        plt.figure("DC2G")
        plt.subplot(131)
        plt.imshow(input_value)
    if save_individual_figures:
        plt.imsave("{dir_path}/results/c2g/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), raw_c2g)

    # print("Looking for frontier pts in semantic map...")
    frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array = find_reachable_frontier_indices2(semantic_array, reachable_array)

    # print("frontier_array.shape: {}".format(frontier_array.shape))
    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_frontiers_fov_aware.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), np.any(fov_aware_reachable_frontier_array, axis=2))
    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_frontiers.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), np.any(frontier_array, axis=2))
    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_c2g_array.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), c2g_array)

    frontier_c2gs = np.zeros_like(c2g_array)
    frontier_c2gs[np.any(fov_aware_reachable_frontier_array, axis=2) == 1] = c2g_array[np.any(fov_aware_reachable_frontier_array, axis=2) == 1]

    # print("c2g_array:", c2g_array)
    # print("reachable:", np.any(fov_aware_reachable_frontier_array, axis=2))
    if np.max(frontier_c2gs) == 0:
        print("none of the frontier pts have a non-zero c2g. Very bad news.")
        lowest_cost_frontier_ind = np.unravel_index(np.any(fov_aware_reachable_frontier_array, axis=2).argmax(), frontier_c2gs.shape)
    else:
        lowest_cost_frontier_ind = np.unravel_index(frontier_c2gs.argmax(), frontier_c2gs.shape)

    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_frontier_c2gs.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), frontier_c2gs)
    # frontier_c2gs = c2g_array.copy()
    # frontier_c2gs[np.all(frontier_array, axis=2) == 0] = 0

    # print("c2g_array.shape: {}".format(c2g_array.shape))
    # print("frontier_c2gs.shape: {}".format(frontier_c2gs.shape))
    # print("frontier_c2gs:", frontier_c2gs)

    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_frontier_c2gs.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), frontier_c2gs)
    # frontier_c2gs[lowest_cost_frontier_ind] = 2
    # plt.imsave("{dir_path}/results/c2g/step_{step_num}_frontier_c2gs.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), frontier_c2gs)
    # lowest_cost_frontier_ind_arr = frontier_array[lowest_cost_frontier_ind, :]

    # print("lowest_cost_frontier_ind: {}".format(lowest_cost_frontier_ind))
    # print("bfs_parent_dict: {}".format(bfs_parent_dict))
    # print("fov_aware_reachable_frontier_array[lowest_cost_frontier_ind]:", fov_aware_reachable_frontier_array[lowest_cost_frontier_ind])
    lowest_cost_frontier_state = (lowest_cost_frontier_ind[1], lowest_cost_frontier_ind[0])
    actions_to_frontier, _, path = construct_path(lowest_cost_frontier_state, bfs_parent_dict)
    # print("actions_to_frontier: {}, \npath: {}".format(actions_to_frontier, path))

    if position[0] == lowest_cost_frontier_state[0] and position[1] == lowest_cost_frontier_state[1]:
        print("[dc2g_planner] warning: currently at the frontier position ==> spin toward nearest theta that is a frontier theta.")
        # print("theta_ind:", theta_ind)
        frontier_thetas = np.where(fov_aware_reachable_frontier_array[lowest_cost_frontier_ind])[0]
        # print("frontier_thetas:", frontier_thetas)
        closest_theta_arg = np.argmin((theta_ind - frontier_thetas) % fov_aware_reachable_frontier_array.shape[2])
        # print("closest_theta_arg:", closest_theta_arg)
        closest_theta = frontier_thetas[closest_theta_arg]
        # print("closest_theta:", closest_theta)
        lowest_cost_frontier_state = (lowest_cost_frontier_state[0], lowest_cost_frontier_state[1], closest_theta)
        # print("lowest_cost_frontier_state:", lowest_cost_frontier_state)
        actions_to_frontier, _, path = construct_path(lowest_cost_frontier_state, bfs_parent_dict)
        # print("actions_to_frontier:", actions_to_frontier)
        # print("path:", path)


    if len(actions_to_frontier) == 0:
        print("[dc2g_planner] warning: len(actions_to_frontier)==0 ==> somehow we are at a frontier pt, which means it shouldn't be called a frontier pt. disagreement btwn camera in env and frontier pt finder. spin?")
        # with open("semantic_array.p","wb") as f:
        #     pickle.dump(semantic_array, f)
        # with open("reachable_array.p","wb") as f:
        #     pickle.dump(reachable_array, f)
        actions_to_frontier = [1] # super arbitrary which dir to spin
        path = [lowest_cost_frontier_state]

    action = actions_to_frontier[0]

    ############## auxiliary just for viz
    # path_color = np.linspace(0.5, 0.5, len(path))
    if plot_panels:
        path_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
        path_array[path_inds] = 1

        # fov_aware_reachable_frontier_array = np.any(fov_aware_reachable_frontier_array, axis=2)
        num_inflations = 1
        struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        for i in range(num_inflations):
            for i in range(4):
                fov_aware_reachable_frontier_array[:,:,i] = scipy.ndimage.morphology.binary_dilation(fov_aware_reachable_frontier_array[:,:,i], struct2).astype(fov_aware_reachable_frontier_array.dtype)
            reachable_array = scipy.ndimage.morphology.binary_dilation(reachable_array, struct2).astype(reachable_array.dtype)
            path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

        planner_array = np.ones((semantic_array.shape[0], semantic_array.shape[1], 4))
        colors = {'reachable': {'color': [0.7,0.7,0.7], 'condition': reachable_array == 1},
                  'frontier':  {'color': [0,0,1], 'condition': np.any(frontier_array, axis=2) == 1},
                  'reachable_frontier':  {'color': [0,1,0], 'condition': np.any(fov_aware_reachable_frontier_array, axis=2) == 1},
                  'path':      {'color': [1,0,0], 'condition': path_array == 1}}
                  # 'fov_aware_reachable_frontier':  {'color': [1,0,0], 'condition': fov_aware_reachable_frontier_array == 1}}

        for key in ['frontier', 'reachable', 'reachable_frontier', 'path']:
        # for key in ['reachable_frontier']:
            param = colors[key]
            for i in range(len(param['color'])):
                planner_array[:,:,i][param['condition']] = param['color'][i]

        width = 1+num_inflations
        for i in range(position[0]-width, position[0]+width):
            for j in range(position[1]-width, position[1]+width):
                planner_array[j,i,:] = [0,1,1,1]

        plt.figure("DC2G")
        plt.subplot(133)
        # plt.imsave("{dir_path}/results/frontiers/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), planner_array)
        plt.imshow(planner_array)
    ###################################

    # if plot_panels:
    #     frontier_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
    #     for ind in bfs_parent_dict.keys():
    #         frontier_array[ind[1], ind[0]] = 0.2
    #     frontier_array[frontier_inds] = 0.1*frontier_headings
    #     frontier_array[lowest_cost_frontier_ind_arr[1], lowest_cost_frontier_ind_arr[0]] = 1.0
    #     # frontier_array[path_inds] = path_color
    #     plt.figure("DC2G")
    #     plt.subplot(133)
    #     plt.imshow(frontier_array, cmap=plt.cm.binary)
    return action

def bfs_planner(position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, step_number):
    '''
    Description: TODO
    inputs:
        - position: current position of robot in gridworld (e.g. np.array([px, py]))
        - theta_ind: current heading index of robot in gridworld (e.g. 2) - some int btwn 0-3 inclusive
        - semantic_array: 32x32x3 np array of robot's current partial knowledge of gridworld
        - reachable_inds_arr: nx2 np array of grid coordinates the agent can definitely reach given its current partial semantic map knowledge
        - tf_sess: tensorflow session
        - tf_input: tensorflow shortcut to refer to 256x256x3 image input
        - tf_output: tensorflow shortcut to refer to 256x256x3 image output
        - bfs_parent_dict: dictionary keyed by each reachable (px, py, theta_ind) coordinate, s.t. child coord -> (parent coord, action)
                            created by running exhaustive BFS on grid from current coordinate
    outputs:
        - action: int of action to take
    '''

    # print("Looking for frontier pts in semantic map...")
    frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array = find_reachable_frontier_indices2(semantic_array, reachable_array)
    actions_to_frontier, point_on_frontier, path = breadth_first_search2(reachable_array, fov_aware_reachable_frontier_array, position, theta_ind)
    # print("path: {}, point_on_frontier: {}".format(path, point_on_frontier))
    # print("actions_to_frontier: {}".format(actions_to_frontier))
    action = actions_to_frontier[0]

    ############## auxiliary just for viz
    # path_color = np.linspace(0.5, 0.5, len(path))
    if plot_panels:
        path_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
        path_array[path_inds] = 1

        # fov_aware_reachable_frontier_array = np.any(fov_aware_reachable_frontier_array, axis=2)
        num_inflations = 0
        struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        for i in range(num_inflations):
            for i in range(4):
                fov_aware_reachable_frontier_array[:,:,i] = scipy.ndimage.morphology.binary_dilation(fov_aware_reachable_frontier_array[:,:,i], struct2).astype(fov_aware_reachable_frontier_array.dtype)
            reachable_array = scipy.ndimage.morphology.binary_dilation(reachable_array, struct2).astype(reachable_array.dtype)
            path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

        planner_array = np.ones((semantic_array.shape[0], semantic_array.shape[1], 4))
        colors = {'reachable': {'color': [0.7,0.7,0.7], 'condition': reachable_array == 1},
                  'frontier':  {'color': [0,0,1], 'condition': np.any(frontier_array, axis=2) == 1},
                  'reachable_frontier':  {'color': [0,1,0], 'condition': np.any(fov_aware_reachable_frontier_array, axis=2) == 1},
                  'path':      {'color': [1,0,0], 'condition': path_array == 1}}
                  # 'fov_aware_reachable_frontier':  {'color': [1,0,0], 'condition': fov_aware_reachable_frontier_array == 1}}

        for key in ['frontier', 'reachable', 'reachable_frontier', 'path']:
        # for key in ['reachable_frontier']:
            param = colors[key]
            for i in range(len(param['color'])):
                planner_array[:,:,i][param['condition']] = param['color'][i]

        plt.figure("DC2G")
        plt.subplot(133)
        # plt.imsave("{dir_path}/results/frontiers/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), planner_array)
        plt.imshow(planner_array)
    ###################################

    return action

def bfs_backtracking_planner(bfs_parent_dict, goal_state):
    actions_to_goal, _, path = construct_path(goal_state, bfs_parent_dict)
    action = actions_to_goal[0]
    path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
    path_color = np.linspace(1, 0.2, len(path))

    # planner_array = np.zeros((32, 32))
    # planner_array[path_inds] = path_color

    # if plot_panels:
    #     plt.figure("DC2G")
    #     plt.subplot(133)
    #     plt.imshow(planner_array, cmap=plt.cm.binary)
    return action

def dc2g(obs, sess, tf_tensors, step_number, traversable_colors, goal_color, room_or_object_goal, rescale_semantic_map=False):
    traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], traversable_colors)
    # struct2 = scipy.ndimage.generate_binary_structure(2, 2)
    # traversable_array = scipy.ndimage.morphology.binary_dilation(traversable_array, struct2).astype(traversable_array.dtype)
    # for i in range(int(2./ENVIRONMENT.grid_resolution)):
    #     struct2 = scipy.ndimage.generate_binary_structure(2, 2)
    #     traversable_array = scipy.ndimage.morphology.binary_erosion(traversable_array, struct2).astype(traversable_array.dtype)
    goal_array, _, _ = find_goal_inds(obs['semantic_gridmap'], goal_color, room_or_object_goal)
    # fov_aware_goal_array = get_fov_aware_goal_array2(goal_array)

    # plt.figure('obs')
    # plt.imshow(obs['semantic_gridmap'])
    # plt.figure('trav')
    # plt.imshow(traversable_array)
    # plt.show()

    bfs_parent_dict, reachable_array = breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], exhaustive=True)

    # bfs_parent_dict = breadth_first_search(obs['pos'], obs['theta_ind'], np.array([]), traversable_inds_arr, exhaustive=True)
    # reachable_inds_arr_with_duplicates = np.array([[x[0], x[1]] for x in bfs_parent_dict.keys()])
    # reachable_inds_arr = reachable_inds_arr_with_duplicates.copy()
    # TODO: Remove duplicates from reachable_inds_arr (if they exist)
    # _, goal_inds, goal_inds_arr = find_goal_inds(obs['semantic_gridmap'], goal_color, room_or_object_goal)
    # print("[main] goal_inds: {}".format(goal_inds))
    if np.sum(goal_array) == 0:
        # Haven't seen the goal yet ==> use dc2g
        # print("Haven't seen the goal yet ==> Using DC2G.")
        action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, sess, tf_tensors, bfs_parent_dict, step_number, rescale_semantic_map)
    else:
        goal_is_reachable, reachable_goal_inds = check_if_goal_reachable(goal_array, reachable_array)
        if goal_is_reachable:
            # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
            # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
            goal_states = np.where(goal_array == 1)
            goal_state = (reachable_goal_inds[1][0], reachable_goal_inds[0][0]) # just pick the first goal_state - if there are multiple, may wanna do something smarter
            action = bfs_backtracking_planner(bfs_parent_dict, goal_state)
        else:
            # Have seen goal, but no path to it exists yet ==> use dc2g
            # print("Have seen the goal, but no path exists to it ==> Using DC2G.")
            action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, sess, tf_tensors, bfs_parent_dict, step_number, rescale_semantic_map)
    return action

def frontier_nav(obs, step_number, traversable_colors, goal_color, room_or_object_goal):
    # plt.imsave("{dir_path}/results/raw_observation/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), obs['semantic_gridmap'])
    traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], traversable_colors)
    goal_array, _, _ = find_goal_inds(obs['semantic_gridmap'], goal_color, room_or_object_goal)
    # fov_aware_goal_array = goal_array.copy()
    # fov_aware_goal_array = get_fov_aware_goal_array2(goal_array)
    bfs_parent_dict, reachable_array = breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], exhaustive=True)    

    if np.sum(goal_array) == 0:
        # Haven't seen the goal yet ==> use frontier_nav
        # print("Haven't seen the goal yet ==> Using frontier_nav.")
        action = bfs_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, step_number)
    else:
        goal_is_reachable, reachable_goal_inds = check_if_goal_reachable(goal_array, reachable_array)
        if goal_is_reachable:
            # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
            # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
            goal_states = np.where(goal_array == 1)
            goal_state = (reachable_goal_inds[1][0], reachable_goal_inds[0][0]) # just pick the first goal_state - if there are multiple, may wanna do something smarter
            action = bfs_backtracking_planner(bfs_parent_dict, goal_state)
        else:
            # Have seen goal, but no path to it exists yet ==> use frontier_nav
            # print("Have seen the goal, but no path exists to it ==> Using frontier_nav.")
            action = bfs_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, step_number)
    if make_individual_figures:
        plt.figure("Observation")
        plt.imshow(obs['semantic_gridmap'])
    return action

def oracle(obs, entire_semantic_array, traversable_colors, goal_color, room_or_object_goal):
    traversable_array, _, _ = find_traversable_inds(entire_semantic_array, traversable_colors)
    goal_array, _, _ = find_goal_inds(entire_semantic_array, goal_color, room_or_object_goal)
    # fov_aware_goal_array = get_fov_aware_goal_array(goal_array)
    # fov_aware_goal_array = goal_array.copy()
    # inflated_goal_array, inflated_traversable_array = inflate(goal_array, traversable_array)
    # plt.figure('oracle')
    # plt.imshow(entire_semantic_array)
    # plt.show()
    actions_to_goal, _, path = breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'])
    # actions_to_goal, _, path = breadth_first_search2(inflated_traversable_array, inflated_goal_array, obs['pos'], obs['theta_ind'])

    ############## auxiliary just for viz
    if plot_panels:
        path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
        path_color = np.linspace(0.5, 0.5, len(path))
        path_array = np.zeros((entire_semantic_array.shape[0], entire_semantic_array.shape[1]))
        path_array[path_inds] = path_color

        # num_inflations = 5
        # struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        # for i in range(num_inflations):
        #     path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

        planner_array = np.zeros_like(entire_semantic_array)
        planner_array[:,:,0] = path_array
        planner_array[:,:,1] = path_array
        # planner_array[:,:,3] = 1
        planner_array[path_array == 0] = entire_semantic_array[path_array == 0]
        # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 0] = 1
        # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 1] = 0
        # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 2] = 0
        plt.figure("DC2G")
        plt.subplot(133)
        plt.title("Oracle Path to Goal")
        plt.imshow(planner_array)
    ###################################

    return actions_to_goal

def plot(render_mode, step_count, full_semantic_array):
    if plot_panels:
        global ENVIRONMENT
        try:
            render = ENVIRONMENT.render(mode=render_mode, show_trajectory=True) # TODO: add support for show_trajectory to House3D
        except:
            render = np.zeros((100,100))
        plt.figure("DC2G")
        plt.suptitle('Step:' + str(step_count), fontsize=20)
        # plt.subplot(233)
        # plt.imshow(render)
        if make_individual_figures:
            plt.figure("Environment")
            plt.imshow(render)

        plt.figure("DC2G")
        plt.subplot(131)
        plt.imshow(full_semantic_array)

        if save_panel_figures:
            plt.figure("DC2G")
            plt.savefig("{dir_path}/results/panels/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_count).zfill(3)))
        if save_individual_figures:
            plt.imsave("{dir_path}/results/environment/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_count).zfill(3)), render)
            plt.imsave("{dir_path}/results/observation/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_count).zfill(3)), full_semantic_array)
        plt.pause(0.01)

def setup_plots(world_image_filename, dataset, target):
    if make_individual_figures:
        fig = plt.figure('C2G', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Observation', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Environment', figsize=(12, 12))
        plt.axis('off')
    if plot_panels:
        fig = plt.figure('DC2G', figsize=(12, 5))
        # ax = fig.add_subplot(231)
        # ax.set_axis_off()
        # ax.set_title('World Semantic Map')
        # ax = fig.add_subplot(232)
        # ax.set_axis_off()
        # ax.set_title('True Cost-to-Go')
        # ax = fig.add_subplot(233)
        # ax.set_axis_off()
        # ax.set_title('Agent in Semantic Map')
        ax = fig.add_subplot(131)
        ax.set_axis_off()
        ax.set_title('2D Partial Semantic Map')
        ax = fig.add_subplot(132)
        ax.set_axis_off()
        ax.set_title('Predicted Cost-to-Go')
        ax = fig.add_subplot(133)
        ax.set_axis_off()
        ax.set_title('Planned Path')
        ax.set_xlabel("Red: Path\nBlue: Frontiers\nGray: Reachable Pts\nGreen: Reachable Cells to Push Frontier")
        plt.xticks(visible=False)
        plt.yticks(visible=False)


        # total_world_array = plt.imread(world_image_filename)
        # world_id = world_image_filename.split('/')[-1].split('.')[0] # e.g. world00asdasdf
        # world_filetype = world_image_filename.split('/')[-1].split('.')[-1] # e.g. png
        # c2g_image_filename = '/home/mfe/code/dc2g/training_data/{dataset}/full_c2g/test/{world}{target}.{filetype}'.format(filetype=world_filetype, target=target, dataset=dataset, world=world_id)
        # total_c2g_array = plt.imread(c2g_image_filename)
        # plt.figure("DC2G")
        # plt.subplot(231)
        # plt.imshow(total_world_array)
        # plt.figure("DC2G")
        # plt.subplot(232)
        # plt.imshow(total_c2g_array)


def run_episode(planner, seed, env, env_type, difficulty_level='easy'):
    # Load the gym environment
    env.seed(seed=int(seed))

    if planner in ['dc2g', 'dc2g_rescale']:
        # Set up the deep cost-to-go network (load network weights)
        sess = tf.Session()
        model_dir = "/home/mfe/code/dc2g_new/data/trained_networks/driveways_bing_iros19_full_test_works"
        # model_dir = "/home/mfe/code/dc2g/pix2pix-tensorflow/c2g_test/world1_25_256_icra2019"
        # model_dir = "/home/mfe/code/pix2pix-tensorflow/c2g_test/l1_20000"
        saver = tf.train.import_meta_graph(model_dir + "/export.meta")
        saver.restore(sess, model_dir + "/export")
        input_vars = json.loads(tf.get_collection("inputs")[0])
        output_vars = json.loads(tf.get_collection("outputs")[0])
        input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])
        tf_tensors = {'input': input, 'output': output}
        try:
            goal_rgb = tf.get_default_graph().get_tensor_by_name(input_vars["goal_rgb"])
            tf_tensors['goal_rgb'] = goal_rgb
        except:
            pass

    env.set_difficulty_level(difficulty_level)
    obs = reset_env(env)

    if env_type == "MiniGrid" or "AirSim":
        dataset = "driveways_bing_iros19"
        render_mode = "rgb_array"
        target = "front_door"
        target_str = ""
        object_goal_names = ["front_door"]
        room_goal_names = []
        room_or_object_goal = "object"
    elif env_type == "House3D":
        dataset = "house3d"
        render_mode = "rgb"
        target = env.info['target_room']
        target_str = "-{target}".format(target=target)
        object_goal_names = ["desk", "television", "table", "household_appliance", "sofa"]
        room_goal_names = ["bedroom", "dining_room", "kitchen", "office"]
        if target in room_goal_names:
            room_or_object_goal = "room"
        elif target in object_goal_names:
            room_or_object_goal = "object"
        else:
            print("--- Error: goal type ({}) is invalid!! ---".format(target))

    traversable_colors = get_traversable_colors(dataset)
    goal_color = get_goal_colors(dataset, [target], room_or_object_goal=room_or_object_goal)[target]

    # Create a window to render into
    # renderer = env.render(mode=render_mode)

    setup_plots(env.world_image_filename, dataset, target_str)

    # if env.world_id != 'worldn000m001h001':
    #     return True, 10, env.world_id

    while env.step_count < env.max_steps:

        # print("Current position: {}, angle: {}.".format(obs['pos'], obs['theta_ind']))
        ######################################################
        # Select Planner
        #################
        if obs['semantic_gridmap'] is None:
            action = 0
        else:
            if planner == 'dc2g':
                action = dc2g(obs, sess, tf_tensors, env.step_count, traversable_colors, goal_color, room_or_object_goal)
            elif planner == 'dc2g_rescale':
                action = dc2g(obs, sess, tf_tensors, env.step_count, traversable_colors, goal_color, room_or_object_goal, rescale_semantic_map=True)
            elif planner == 'frontier':
                action = frontier_nav(obs, env.step_count, traversable_colors, goal_color, room_or_object_goal)
            elif planner == 'oracle':
                if env.step_count == 0: # get the full shortest path on step 1, then just follow it
                    full_semantic_gridmap = plt.imread(env.world_image_filename) # only used for oracle planner
                    full_size_semantic_gridmap = env.world_array
                    # full_size_semantic_gridmap = cv2.resize(full_semantic_gridmap, (obs['semantic_gridmap'].shape[0], obs['semantic_gridmap'].shape[1]), interpolation=cv2.INTER_NEAREST)
                    actions = oracle(obs, full_size_semantic_gridmap, traversable_colors, goal_color, room_or_object_goal)
                #print("actions: {}".format(actions))
                action = actions.pop(0)
        ######################################################
        plot(render_mode, env.step_count, obs['semantic_gridmap'])


        # print("Sending action to the environment...")
        obs, reward, done, info = env.step(action)
        # print("Environment completed the desired action.")
        # env.render('human')
        if done:
            print('Done! Took {} steps.'.format(env.step_count))
            break
    return done, env.step_count, env.world_id

def start_experiment(env_name, env_type):
    global ENVIRONMENT
    if env_type == "MiniGrid":
        import gym_minigrid
        env = gym.make(env_name)
        ENVIRONMENT = env
    elif env_type == "House3D":
        from House3D.common import load_config

        from House3D.house import House
        from House3D.core import Environment, MultiHouseEnv
        from House3D.roomnav import objrender, RoomNavTask
        from House3D.objrender import RenderMode

        api = objrender.RenderAPI(
                w=400, h=300, device=0)
        cfg = load_config('/home/mfe/code/dc2g/House3D/House3D/config.json')

        house = '00a42e8f3cb11489501cfeba86d6a297'
        # houses = ['00065ecbdd7300d35ef4328ffe871505',
        # 'cf57359cd8603c3d9149445fb4040d90', '31966fdc9f9c87862989fae8ae906295', 'ff32675f2527275171555259b4a1b3c3',
        # '7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86', '775941abe94306edc1b5820e3a992d75',
        # '32e53679b33adfcc5a5660b8c758cc96', '4383029c98c14177640267bd34ad2f3c', '0884337c703e7c25949d3a237101f060',
        # '492c5839f8a534a673c92912aedc7b63', 'a7e248efcdb6040c92ac0cdc3b2351a6', '2364b7dcc432c6d6dcc59dba617b5f4b',
        # 'e3ae3f7b32cf99b29d3c8681ec3be321', 'f10ce4008da194626f38f937fb9c1a03', 'e6f24af5f87558d31db17b86fe269cf2',
        # '1dba3a1039c6ec1a3c141a1cb0ad0757', 'b814705bc93d428507a516b866efda28', '26e33980e4b4345587d6278460746ec4',
        # '5f3f959c7b3e6f091898caa8e828f110', 'b5bd72478fce2a2dbd1beb1baca48abd', '9be4c7bee6c0ba81936ab0e757ab3d61']
        #env = MultiHouseEnv(api, houses[:3], cfg)  # use 3 houses
        house_env = Environment(api, house, cfg)
        env = RoomNavTask(house_env, hardness=0.6, discrete_action=True)
        ENVIRONMENT = env.house
    elif env_type == "AirSim":
        import gym_airsim
        env = gym.make(env_name)
        ENVIRONMENT = env

    return env

def reset_env(env):
    first_obs = env.reset()
    return first_obs


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        # default='AirSim-v0'
        # default='House3D-RoomNav'
        default='MiniGrid-EmptySLAM-32x32-v0'
    )
    parser.add_option(
        "-p",
        "--planner",
        dest="planner",
        help="name of planner to use (e.g. dc2g, frontier)",
        default='dc2g'
        # default='dc2g_rescale'
        # default='frontier'
        # default='oracle'
    )
    parser.add_option(
        "-s",
        "--seed",
        dest="seed",
        help="seed for deterministically defining random environment behavior",
        default='1337'
    )
    (options, args) = parser.parse_args()

    if "MiniGrid" in options.env_name:
        env_type = "MiniGrid"
    elif "House3D" in options.env_name:
        env_type = "House3D"
    elif "AirSim" in options.env_name:
        env_type = "AirSim"

    env = start_experiment(env_name=options.env_name, env_type=env_type)
    success, num_steps, world_id = run_episode(planner=options.planner, seed=options.seed, env=env, env_type=env_type, difficulty_level='test_scenario')


if __name__ == "__main__":
    main()


# To duplicate ICRA 19 search experiment (dc2g vs frontier on a single setup), use this command:
# python3 /home/mfe/code/dc2g/run_episode.py --planner frontier --seed 1324

### using 2018-09-13_15_51_57.pkl ###############################################################
# # steps improvement over oracle (on each episode)
# root@9a7ec89d3925:/home/mfe/code/baselines# python3 /home/mfe/code/dc2g/run_experiment.py 
# easy dc2g 7.833333333333333 7.7977917101930565
# easy dc2g_rescale 17.133333333333333 15.683820396262584
# easy frontier 75.9 47.22029930725415
# medium dc2g 44.86666666666667 28.126776487104873
# medium dc2g_rescale 50.93333333333333 34.22955188462482
# medium frontier 75.2 51.99384578967014
# hard dc2g 205.33333333333334 154.34557186895762
# hard dc2g_rescale 181.53333333333333 177.00955404221045
# hard frontier 88.46666666666667 74.58003903339163

# # pct increase over oracle (on each episode)
# easy dc2g 0.270599881202737 0.28855391250361495
# easy dc2g_rescale 0.4816548830645126 0.4077706428167549
# easy frontier 2.8237796603954193 2.8788060235515505
# medium dc2g 1.4896587519378828 1.410429295315634
# medium dc2g_rescale 1.390809750658729 1.0523059787623652
# medium frontier 2.4351604780402436 2.981008615136291
# hard dc2g 10.717210471352766 12.422022585185918
# hard dc2g_rescale 7.577344522715777 7.680369559640723
# hard frontier 4.17461877934621 3.9921799250063414

# # total num steps (on each episode)
# easy oracle 32.166666666666664 10.456523747827903
# easy dc2g 40.0 13.30663994653296
# easy dc2g_rescale 49.3 23.279676400958266
# easy frontier 108.06666666666666 46.99924349272964
# medium oracle 37.7 15.106621064950295
# medium dc2g 82.56666666666666 31.538001768589517
# medium dc2g_rescale 88.63333333333334 44.16369801343885
# medium frontier 112.9 54.6847632648559
# hard oracle 23.566666666666666 12.093202865889399
# hard dc2g 228.9 159.91859387409167
# hard dc2g_rescale 205.1 184.19253513647072
# hard frontier 112.03333333333333 78.17564980364551



