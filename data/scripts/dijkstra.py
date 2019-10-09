#!/usr/bin/python

'''
BSD 2-Clause License

Copyright (c) 2017, Andrew Dahdouh
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import pprint

def og_to_c2g(occupancy_map, start_inds, inflated_start_array, rgb=False, resolution=1.0):
    """
    Implements Dijkstra's shortest path algorithm
    Input:
    occupancy_map - an N by M numpy array of boolean values (represented
        as integers 0 and 1) that represents the locations of the obstacles
        in the world
    start_inds - a tuple w/ 2 elements: indices corresponding to set of starting positions 
    Output: 
    path: list of the indices of the nodes on the shortest path found
        starting with "start" and ending with "end" (each node is in
        metric coordinates)
    """

    # allowed_movements = "straight"
    allowed_movements = "straight_and_diagonal"

    # Setup Map Visualizations:
    viz_map = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1], 3))
    viz_map[:,:,0] = occupancy_map.copy()

    my_dpi = 76
    fig = plt.figure(figsize=(occupancy_map.shape[0]/my_dpi, occupancy_map.shape[1]/my_dpi), dpi=my_dpi)
    # fig = plt.figure(figsize=(32.0/my_dpi, 32.0/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_aspect('equal')

    # Check that the entire traversable space isn't considered the goal... this means there's no planning task.
    start_array = np.ones_like(occupancy_map)
    start_array[start_inds] = 0
    if np.array_equal(start_array, occupancy_map):
        print("Entire traversable space is goal ==> just returning white viz_map")
        # Invert & Scale the colors on reachable nodes (==> white is goal, black is far from goal)
        viz_map[occupancy_map == 0] = 1 - (viz_map[occupancy_map == 0] / np.max(viz_map[:,:,0]))
        plt.imshow(viz_map, origin='upper', interpolation='none')
        return viz_map

    if allowed_movements == "straight":
        delta = [[-1, 0],  # go up
                 [0, -1],  # go left
                 [1, 0],  # go down
                 [0, 1]]  # go right
    elif allowed_movements == "straight_and_diagonal":
        delta = [[-1, 0],  # go up
                 [0, -1],  # go left
                 [1, 0],  # go down
                 [0, 1],  # go right
                 [1, -1],  # go up/left
                 [1, 1],  # go up/right
                 [-1, -1],  # go down/left
                 [-1, 1]]  # go down/right
        # cost = resolution*np.array([1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    cost = resolution*np.linalg.norm(delta, axis=1)

    num_motions = len(delta)
    assert(num_motions == len(cost))  # make sure each motion has a cost value.

    # Convert numpy array of map to list of map, makes it easier to search.
    occ_map = occupancy_map.tolist()

    start_inds_arr = np.array([start_inds[1], start_inds[0]])

    # Make a map to keep track of all the nodes and their cost distance values.
    possible_nodes = np.zeros_like(occupancy_map)

    # The g_value will count the number of steps each node is from the start.
    # Since we are at the start node, the total cost is 0.
    g_value = 0
    frontier_nodes = [] # dist, x, y
    for ind in range(start_inds_arr.shape[1]):
        frontier_nodes.append((g_value, start_inds_arr[0, ind], start_inds_arr[1, ind]))
    searched_nodes = []
    parent_node = {}  # Dictionary that Maps {child node : parent node}
    loopcount = 0

    while len(frontier_nodes) != 0:
        # print(len(frontier_nodes))
        frontier_nodes.sort(reverse=True) #sort from shortest distance to farthest
        current_node = frontier_nodes.pop()

        g_value, col, row = current_node

        # Check surrounding neighbors.
        for i in range(num_motions):
            motion = delta[i]
            possible_expansion_x = col + motion[0]
            possible_expansion_y = row + motion[1]
            valid_expansion = 0 <= possible_expansion_y < occupancy_map.shape[0] and 0 <= possible_expansion_x < occupancy_map.shape[1]

            if valid_expansion:
                try:
                    unsearched_node = possible_nodes[possible_expansion_y, possible_expansion_x] == 0
                    open_node = occ_map[possible_expansion_y][possible_expansion_x] == 0
                except:
                    unsearched_node = False
                    open_node = False
                if unsearched_node and open_node:
                    travel_cost = cost[i]
                    # Using  instead of 1 to make it easier to read This node has been searched.
                    possible_nodes[possible_expansion_y, possible_expansion_x] = 3
                    possible_node = (g_value + travel_cost, possible_expansion_x, possible_expansion_y)
                    frontier_nodes.append(possible_node)
                    # if not (possible_expansion_x == start[0] and possible_expansion_y == start[1]):
                    viz_map[possible_expansion_y, possible_expansion_x, :] = g_value + travel_cost

                    # This now builds parent/child relationship
                    parent_node[possible_node] = current_node
        loopcount = loopcount+1
        # if loopcount % 5000 == 0:
        # # if loopcount % 50000 == 0:
        #     print "Loopcount"
        #     plt.imshow(viz_map, interpolation='none', cmap=plt.cm.binary)
        #     max_c2g = np.max(viz_map)
        #     plt.clim(0, max_c2g)
        #     plt.pause(0.2)


    # Invert & Scale the colors on reachable nodes (==> white is goal, black is far from goal)
    viz_map[occupancy_map == 0] = 1 - (viz_map[occupancy_map == 0] / np.max(viz_map[:,:,0]))

    # Color unreachable, yet traversable, terrain black.
    unsearched_inds = np.where(np.logical_and(possible_nodes == 0, occupancy_map == 0))
    viz_map[unsearched_inds] = 0

    # Remove the inflated goal object from the c2g map (since we don't actually care about its cost)
    viz_map[inflated_start_array == 1] = [1,0,0]

    # plt.imshow(viz_map, origin='upper', interpolation='none')
    # plt.pause(5)
    plt.close()
    return viz_map

