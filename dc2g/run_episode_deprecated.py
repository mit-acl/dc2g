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
from scipy.ndimage.interpolation import shift
from skimage.transform import resize
from queue import *
import collections

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import itertools

import util

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.set_printoptions(suppress=True, precision=8)
np.warnings.filterwarnings('ignore')
# np.set_printoptions(threshold=np.inf)

dir_path = os.path.dirname(os.path.realpath(__file__))


make_individual_figures = False
save_individual_figures = True
save_panel_figures = False
plot_panels = True


def find_frontier_indices(semantic_array):
    # Find the array of observed inds that have at least 1 unobserved neighbor
    # t_start = time.time()
    observed_inds = np.where(np.any(abs(semantic_array) > 1e-5, axis=-1))
    observed_inds_arr = np.dstack([observed_inds[0], observed_inds[1]])[0]

    semantic_array_shifted_left = shift(semantic_array, (1,0,0), cval=np.nan)
    semantic_array_shifted_right = shift(semantic_array, (-1,0,0), cval=np.nan)
    semantic_array_shifted_up = shift(semantic_array, (0,1,0), cval=np.nan)
    semantic_array_shifted_down = shift(semantic_array, (0,-1,0), cval=np.nan)

    unobserved_up_inds = np.where(np.all(abs(semantic_array_shifted_up) < 1e-5, axis=-1))
    unobserved_up_inds_arr = np.dstack([unobserved_up_inds[0], unobserved_up_inds[1]])[0]
    unobserved_down_inds = np.where(np.all(abs(semantic_array_shifted_down) < 1e-5, axis=-1))
    unobserved_down_inds_arr = np.dstack([unobserved_down_inds[0], unobserved_down_inds[1]])[0]
    unobserved_left_inds = np.where(np.all(abs(semantic_array_shifted_left) < 1e-5, axis=-1))
    unobserved_left_inds_arr = np.dstack([unobserved_left_inds[0], unobserved_left_inds[1]])[0]
    unobserved_right_inds = np.where(np.all(abs(semantic_array_shifted_right) < 1e-5, axis=-1))
    unobserved_right_inds_arr = np.dstack([unobserved_right_inds[0], unobserved_right_inds[1]])[0]

    frontier_inds_arr = np.array([x for x in set(tuple(x) for x in observed_inds_arr) & (set(tuple(x) for x in unobserved_up_inds_arr) | set(tuple(x) for x in unobserved_down_inds_arr) | set(tuple(x) for x in unobserved_left_inds_arr) | set(tuple(x) for x in unobserved_right_inds_arr))])
    frontier_inds = (frontier_inds_arr[:,0], frontier_inds_arr[:,1])
    
    # print(time.time() - t_start)
    return frontier_inds, frontier_inds_arr, frontier_headings

def find_reachable_frontier_indices(semantic_array, observed_traversable_inds_arr):
    # Find the array of observed & traversable inds that have at least 1 unobserved neighbor

    semantic_array_shifted_left = shift(semantic_array, (0,-1,0), cval=np.nan)
    semantic_array_shifted_right = shift(semantic_array, (0,1,0), cval=np.nan)
    semantic_array_shifted_up = shift(semantic_array, (-1,0,0), cval=np.nan)
    semantic_array_shifted_down = shift(semantic_array, (1,0,0), cval=np.nan)

    # Shift semantic map up & find unobserved regions ==> should be pointed downward to see those
    unobserved_up_inds = np.where(np.all(abs(semantic_array_shifted_up) < 1e-5, axis=-1))
    unobserved_up_inds_arr = np.dstack([unobserved_up_inds[1], unobserved_up_inds[0], 1*np.ones_like(unobserved_up_inds[0], dtype=int)])[0]
    unobserved_down_inds = np.where(np.all(abs(semantic_array_shifted_down) < 1e-5, axis=-1))
    unobserved_down_inds_arr = np.dstack([unobserved_down_inds[1], unobserved_down_inds[0], 3*np.ones_like(unobserved_down_inds[0], dtype=int)])[0]
    unobserved_left_inds = np.where(np.all(abs(semantic_array_shifted_left) < 1e-5, axis=-1))
    unobserved_left_inds_arr = np.dstack([unobserved_left_inds[1], unobserved_left_inds[0], 0*np.ones_like(unobserved_left_inds[0], dtype=int)])[0]
    unobserved_right_inds = np.where(np.all(abs(semantic_array_shifted_right) < 1e-5, axis=-1))
    unobserved_right_inds_arr = np.dstack([unobserved_right_inds[1], unobserved_right_inds[0], 2*np.ones_like(unobserved_right_inds[0], dtype=int)])[0]

    observed_traversable_inds_with_theta_arr = np.tile(np.hstack([observed_traversable_inds_arr, np.zeros((observed_traversable_inds_arr.shape[0], 1), dtype=int)]), (4, 1))
    num_observable_inds = observed_traversable_inds_arr.shape[0]
    for i in range(1, 4):
        observed_traversable_inds_with_theta_arr[num_observable_inds*i:num_observable_inds*(i+1), 2] = i
    frontier_inds_arr = np.array([x for x in set(tuple(x) for x in observed_traversable_inds_with_theta_arr) & (set(tuple(x) for x in unobserved_up_inds_arr) | set(tuple(x) for x in unobserved_down_inds_arr) | set(tuple(x) for x in unobserved_left_inds_arr) | set(tuple(x) for x in unobserved_right_inds_arr))])
    frontier_inds = (frontier_inds_arr[:, 1], frontier_inds_arr[:, 0])
    frontier_headings = frontier_inds_arr[:,2]

    return frontier_inds, frontier_inds_arr, frontier_headings

def find_traversable_inds(semantic_array):
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
    road_inds = np.where(np.all(semantic_array == [1, 0, 0], axis=-1))
    driveway_inds = np.where(np.all(semantic_array == [0, 0, 1], axis=-1))
    goal_inds = np.where(np.all(semantic_array == [1, 1, 0], axis=-1))
    # Combine all traversable inds into a single object
    observed_traversable_inds = (np.hstack([road_inds[0], driveway_inds[0], goal_inds[0]]), np.hstack([road_inds[1], driveway_inds[1], goal_inds[1]]))

    # Re-organize inds into pairs of indices
    observed_traversable_inds_arr = np.dstack([observed_traversable_inds[1], observed_traversable_inds[0]])[0]

    return observed_traversable_inds, observed_traversable_inds_arr

def breadth_first_search(start_pos, start_theta_ind, goal_positions, traversable_inds_arr, exhaustive=False):
    '''
    Description: Starting from start_pos, start_theta_ind, execute a BFS among traversable nodes
        in the graph. If a goal position is found, stop the search -- unless the exhaustive flag is set,
        in which case keep searching until all graph nodes have been explored fully to determine
        all reachable nodes from start.
    inputs:
        - start_pos: current position of robot in gridworld (e.g. np.array([px, py]))
        - start_theta_ind: current heading index of robot in gridworld (e.g. 2) - some int btwn 0-3 inclusive
        - goal_positions: nx3 np array of goal states that, if found, should terminate any non-exhaustive search
        - traversable_inds_arr: nx2 np array of grid coordinates the agent can definitely reach given its current partial semantic map knowledge
        - exhaustive: whether to search til something in goal_positions is found, or to search til queue is empty
    outputs:
        - if no goal was provided or goal == position ==> returns None
        - if not exhaustive:
            - if goal not found ==> returns None
            - if goal found ==> returns action_list
        - if exhaustive ==> returns dict of child coord -> (parent coord, action)
    '''
    traversable_inds_list = traversable_inds_arr.tolist()
    goal_positions_list = goal_positions.tolist()
    if [start_pos[0], start_pos[1], start_theta_ind] in goal_positions_list:
        # If currently at goal position, remove it from consideration
        # print('[breadth_first_search] we are currently at a goal position. removing it from goal list.')
        goal_positions_list.remove([start_pos[0], start_pos[1], start_theta_ind])
    if len(goal_positions_list) == 0 and not exhaustive:
        # If there aren't any goals, then quit
        print('[breadth_first_search] something got messed up: len(goal_positions_list) == 0.')
        return
    # print("[breadth_first_search] goal_positions_list: {}".format(goal_positions_list))
    # print("traversable_inds_list: {}".format(traversable_inds_list))
    meta = dict()
    root = (start_pos[0], start_pos[1], start_theta_ind)
    visited, queue = set(), collections.deque([root])
    meta[root] = (None, None)
    while queue:
        vertex = queue.popleft()
        # print("[breadth_first_search] vertex: {}".format(vertex))
        if list(vertex) in goal_positions_list and not exhaustive:
            # print("BFS found one of the goals. A path exists to {}".format([vertex[0], vertex[1], vertex[2]]))
            return construct_path(vertex, meta)
        px, py, theta_ind = vertex
        actions = [2, 0, 1]
        children = [(int(round(px + 1*np.cos(np.pi * theta_ind / 2.0))), int(round(py + 1*np.sin(np.pi * theta_ind / 2.0))), theta_ind), (px, py, (theta_ind + 1) % 4), (px, py, (theta_ind - 1) % 4)]
        for i in range(len(children)):
            # print("[breadth_first_search] children[i]: {}".format(children[i]))
            if [children[i][0], children[i][1]] not in traversable_inds_list:
                continue
            if children[i] not in visited:
                visited.add(children[i])
                queue.append(children[i])
                if children[i] not in meta:
                    meta[children[i]] = (vertex, actions[i])
    if not exhaustive:
        print("[breadth_first_search] warning: queue is empty. while loop ended.")
        return
    return meta

# Produce a backtrace of the actions taken to find the goal node, using the
# recorded meta dictionary
def construct_path(state, meta):
    if len(state) == 2:
        # TODO: If you don't know theta, this might give a slightly crappier path than optimal, because the exhaustive search finds ways to get to this node from many directions

        # If we don't specify a final theta (only give position), find one.
        # print("[construct_path] State was only provided as len-2: {}. Looking through meta to see if it exists.".format(state))
        for theta_ind in range(4):
            final_state = (state[0], state[1], theta_ind)
            if final_state in meta.keys():
                state = final_state
                break
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

def look_for_goal(semantic_array):
    goal_inds = np.where(np.all(semantic_array == [1, 1, 0], axis=-1))
    if len(goal_inds[0]) > 0:
        goal_inds_arr = [goal_inds[1][0], goal_inds[0][0]]
        return goal_inds, goal_inds_arr
    else:
        return goal_inds, None

def resize_semantic_array(semantic_array, reachable_inds_arr):
    observed_inds = np.where(np.any(abs(semantic_array) > 1e-5, axis=-1))
    min_x_ind = np.min(observed_inds[0])
    min_y_ind = np.min(observed_inds[1])
    max_x_ind = np.max(observed_inds[0])
    max_y_ind = np.max(observed_inds[1])
    diff = max(max_x_ind - min_x_ind, max_y_ind - min_y_ind) + 1
    print(min_x_ind, min_x_ind+diff, min_y_ind, min_y_ind+diff)
    cropped_semantic_array = semantic_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff]
    cropped_padded_semantic_array = np.pad(cropped_semantic_array, ((2,2),(2,2),(0,0)), 'constant')
    resized_semantic_array = resize(cropped_padded_semantic_array, (256,256,3), order=0)
    return resized_semantic_array

def size_agnostic_c2g_query(semantic_array, tf_sess, tf_input, tf_output):
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
    resized_semantic_array = resize(cropped_padded_semantic_array, (256,256,3), order=0)

    # Query the encoder-decoder network for image translation
    output_value = tf_sess.run(tf_output, feed_dict={tf_input: resized_semantic_array})

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

    return final_c2g_array, resized_semantic_array

def c2g_query(semantic_array, tf_sess, tf_input, tf_output):
    input_value = semantic_array.repeat(8, axis=0).repeat(8, axis=1)
    output_value = tf_sess.run(tf_output, feed_dict={tf_input: input_value})
    hsv = plt_colors.rgb_to_hsv(output_value)
    c2g_array = hsv[:, :, 2]
    c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map
    c2g_array = scipy.misc.imresize(c2g_array, (32,32), interp='nearest')
    return c2g_array, semantic_array, output_value

def dc2g_planner(position, theta_ind, semantic_array, reachable_inds_arr, tf_sess, tf_input, tf_output, bfs_parent_dict, step_number, rescale_semantic_map=False):
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
        c2g_array, input_value = size_agnostic_c2g_query(semantic_array, tf_sess, tf_input, tf_output)
    else:
        c2g_array, input_value, raw_c2g = c2g_query(semantic_array, tf_sess, tf_input, tf_output)
    if plot_panels:
        plt.figure("DC2G")
        plt.subplot(235)
        plt.imshow(c2g_array, cmap=plt.cm.gray)
        plt.figure("DC2G")
        plt.subplot(234)
        plt.imshow(input_value)
    if save_individual_figures:
        plt.imsave("{dir_path}/results/c2g/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), raw_c2g)


    frontier_inds, frontier_inds_arr, frontier_headings = find_reachable_frontier_indices(semantic_array, reachable_inds_arr)
    frontier_c2gs = c2g_array[frontier_inds]
    lowest_cost_frontier_inds_ind = np.argmax(frontier_c2gs)
    lowest_cost_frontier_ind_arr = frontier_inds_arr[lowest_cost_frontier_inds_ind, :]

    # print("lowest_cost_frontier_ind_arr: {}".format(lowest_cost_frontier_ind_arr))
    # print("bfs_parent_dict: {}".format(bfs_parent_dict))
    actions_to_frontier, _, path = construct_path(tuple(lowest_cost_frontier_ind_arr), bfs_parent_dict)

    if len(actions_to_frontier) == 0:
        print("[dc2g_planner] warning: len(actions_to_frontier)==0 ==> we are at the desired pt. spin?")
        action = 0
    else:
        action = actions_to_frontier[0]
        path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
        path_color = np.linspace(0.8, 0.4, len(path))


    if plot_panels:
        frontier_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        for ind in bfs_parent_dict.keys():
            frontier_array[ind[1], ind[0]] = 0.2
        frontier_array[frontier_inds] = 0.1*frontier_headings
        frontier_array[lowest_cost_frontier_ind_arr[1], lowest_cost_frontier_ind_arr[0]] = 1.0
        # frontier_array[path_inds] = path_color
        plt.figure("DC2G")
        plt.subplot(236)
        plt.imshow(frontier_array, cmap=plt.cm.binary)
    return action

def bfs_planner(position, theta_ind, semantic_array, reachable_inds_arr, bfs_parent_dict, step_number):
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

    frontier_inds, frontier_inds_arr, frontier_headings = find_reachable_frontier_indices(semantic_array, reachable_inds_arr)
    actions_to_frontier, point_on_frontier, path = breadth_first_search(position, theta_ind, frontier_inds_arr, reachable_inds_arr)
    action = actions_to_frontier[0]

    path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
    path_color = np.linspace(0.8, 0.4, len(path))

    if plot_panels:
        frontier_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        for ind in bfs_parent_dict.keys():
            frontier_array[ind[1], ind[0]] = 0.2  # Reachable places
        frontier_array[frontier_inds] = 0.5  # Pixels along the frontier
        # frontier_array[frontier_inds] = 0.2*(frontier_headings+2)
        # frontier_array[path_inds] = path_color
        plt.figure("DC2G")
        plt.subplot(236)
        plt.imshow(frontier_array, cmap=plt.cm.binary)
    return action

def bfs_backtracking_planner(bfs_parent_dict, goal_inds_arr):
    actions_to_goal, _, path = construct_path(goal_inds_arr, bfs_parent_dict)
    action = actions_to_goal[0]
    path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
    path_color = np.linspace(1, 0.2, len(path))

    planner_array = np.zeros((32, 32))
    planner_array[path_inds] = path_color

    if plot_panels:
        plt.figure("DC2G")
        plt.subplot(236)
        plt.imshow(planner_array, cmap=plt.cm.binary)
    return action

def dc2g(obs, sess, input, output, step_number, rescale_semantic_map=False):
    traversable_inds, traversable_inds_arr = find_traversable_inds(obs['image'])
    bfs_parent_dict = breadth_first_search(obs['pos'], obs['theta_ind'], np.array([]), traversable_inds_arr, exhaustive=True)
    reachable_inds_arr_with_duplicates = np.array([[x[0], x[1]] for x in bfs_parent_dict.keys()])
    reachable_inds_arr = reachable_inds_arr_with_duplicates.copy()
    # TODO: Remove duplicates from reachable_inds_arr (if they exist)
    goal_inds, goal_inds_arr = look_for_goal(obs['image'])
    # print("[main] goal_inds: {}".format(goal_inds))
    if len(goal_inds[0]) == 0:
        # Haven't seen the goal yet ==> use dc2g
        # print("Haven't seen the goal yet ==> Using DC2G.")
        action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['image'], reachable_inds_arr, sess, input, output, bfs_parent_dict, step_number, rescale_semantic_map)
    elif goal_inds_arr in reachable_inds_arr.tolist():
        # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
        # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
        action = bfs_backtracking_planner(bfs_parent_dict, goal_inds_arr)
    else:
        # Have seen goal, but no path to it exists yet ==> use dc2g
        # print("Have seen the goal, but no path exists to it ==> Using DC2G.")
        action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['image'], reachable_inds_arr, sess, input, output, bfs_parent_dict, step_number, rescale_semantic_map)
    return action


def frontier_nav(obs, step_number):
    traversable_inds, traversable_inds_arr = find_traversable_inds(obs['image'])
    bfs_parent_dict = breadth_first_search(obs['pos'], obs['theta_ind'], np.array([]), traversable_inds_arr, exhaustive=True)
    reachable_inds_arr_with_duplicates = np.array([[x[0], x[1]] for x in bfs_parent_dict.keys()])
    reachable_inds_arr = reachable_inds_arr_with_duplicates.copy()
    # TODO: Remove duplicates from reachable_inds_arr (if they exist)
    goal_inds, goal_inds_arr = look_for_goal(obs['image'])
    # print("[main] goal_inds: {}".format(goal_inds))
    if len(goal_inds[0]) == 0:
        # Haven't seen the goal yet ==> use dc2g
        # print("Haven't seen the goal yet ==> Using DC2G.")
        action = bfs_planner(obs['pos'], obs['theta_ind'], obs['image'], reachable_inds_arr, bfs_parent_dict, step_number)
    elif goal_inds_arr in reachable_inds_arr.tolist():
        # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
        # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
        action = bfs_backtracking_planner(bfs_parent_dict, goal_inds_arr)
    else:
        # Have seen goal, but no path to it exists yet ==> use dc2g
        # print("Have seen the goal, but no path exists to it ==> Using DC2G.")
        action = bfs_planner(obs['pos'], obs['theta_ind'], obs['image'], reachable_inds_arr, bfs_parent_dict, step_number)
    if make_individual_figures:
        plt.figure("Observation")
        plt.imshow(obs['image'])
    return action

def oracle(obs, entire_semantic_array):
    traversable_inds, traversable_inds_arr = find_traversable_inds(entire_semantic_array)
    goal_inds, goal_inds_arr = look_for_goal(entire_semantic_array)
    actions_to_goal, _, path = breadth_first_search(obs['pos'], obs['theta_ind'], np.array([(goal_inds[1][0], goal_inds[0][0], 0)]), traversable_inds_arr)
    action = actions_to_goal[0]
    return action


def plot(render, step_count, semantic_array):
    if plot_panels:
        plt.figure("DC2G")
        plt.suptitle('Step:' + str(step_count), fontsize=20)
        plt.subplot(233)
        plt.imshow(render)
        if make_individual_figures:
            plt.figure("Environment")
            plt.imshow(render)

        plt.figure("DC2G")
        plt.subplot(234)
        plt.imshow(semantic_array)

        if save_panel_figures:
            plt.figure("DC2G")
            plt.savefig('/home/mfe/code/gyms/gym-minigrid/results/8_29_18/panel/step_'+str(step_count).zfill(3)+'.png')
        if save_individual_figures:
            plt.imsave("{dir_path}/results/environment/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_count).zfill(3)), render)
            plt.imsave("{dir_path}/results/observation/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_count).zfill(3)), semantic_array)
        plt.pause(0.1)

def setup_plots(world_image_filename):
    if make_individual_figures:
        fig = plt.figure('C2G', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Observation', figsize=(12, 12))
        plt.axis('off')
        fig = plt.figure('Environment', figsize=(12, 12))
        plt.axis('off')
    if plot_panels:
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
        ax.set_title('Agent Path to Nearest Frontier')
        plt.xticks(visible=False)
        plt.yticks(visible=False)

        total_world_array = plt.imread(world_image_filename)
        world_id = world_image_filename.split('/')[-1]
        c2g_image_filename = '/home/mfe/code/dc2g/training_data/full_c2g/test/{world}'.format(world=world_id)
        total_c2g_array = plt.imread(c2g_image_filename)
        plt.figure("DC2G")
        plt.subplot(231)
        plt.imshow(total_world_array)
        plt.figure("DC2G")
        plt.subplot(232)
        plt.imshow(total_c2g_array)


def run_episode(planner, seed, env, difficulty_level='easy'):
    # Load the gym environment
    env.seed(seed=int(seed))

    if planner in ['dc2g', 'dc2g_rescale']:
        # Set up the deep cost-to-go network (load network weights)
        sess = tf.Session()
        model_dir = "/home/mfe/code/pix2pix-tensorflow/c2g_test/world1_25_256_icra2019"
        # model_dir = "/home/mfe/code/pix2pix-tensorflow/c2g_test/l1_20000"
        saver = tf.train.import_meta_graph(model_dir + "/export.meta")
        saver.restore(sess, model_dir + "/export")
        input_vars = json.loads(tf.get_collection("inputs")[0])
        output_vars = json.loads(tf.get_collection("outputs")[0])
        input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])


    env.set_difficulty_level(difficulty_level)
    obs = reset_env(env)

    # Create a window to render into
    renderer = env.render('human')

    setup_plots(env.world_image_filename)

    while env.step_count < env.max_steps:
        render = env.render(mode='rgb_array', show_trajectory=True)

        ######################################################
        # Select Planner
        #################
        if planner == 'dc2g':
            action = dc2g(obs, sess, input, output, env.step_count)
        elif planner == 'dc2g_rescale':
            action = dc2g(obs, sess, input, output, env.step_count, rescale_semantic_map=True)
        elif planner == 'frontier':
            action = frontier_nav(obs, env.step_count)
        elif planner == 'oracle':
            action = oracle(obs, env.grid.encode_for_oracle())
        ######################################################

        plot(render, env.step_count, obs['image'])
        obs, reward, done, info = env.step(action)
        # env.render('human')
        # print('step=%s, reward=%.2f' % (env.step_count, reward))
        if done:
            print('Done! Took {} steps.'.format(env.step_count))
            break
    return done, env.step_count, env.world_id

def start_experiment(env_name):
    env = gym.make(env_name)
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
        default='MiniGrid-EmptySLAM-32x32-v0'
    )
    parser.add_option(
        "-p",
        "--planner",
        dest="planner",
        help="name of planner to use (e.g. dc2g, frontier)",
        default=''
    )
    parser.add_option(
        "-s",
        "--seed",
        dest="seed",
        help="seed for deterministically defining random environment behavior",
        default=''
    )
    (options, args) = parser.parse_args()

    env = start_experiment(env_name=options.env_name)
    success, num_steps = run_episode(planner=options.planner, seed=options.seed, env=env)


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



