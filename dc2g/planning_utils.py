import numpy as np
import collections
from scipy.ndimage.interpolation import shift
from dc2g.util import get_traversable_colors, get_goal_colors, find_traversable_inds, find_goal_inds, inflate, wrap, round_base_down, round_base
import scipy.ndimage.morphology
import matplotlib.pyplot as plt

def bfs_backtracking_planner(bfs_parent_dict, goal_state):
    actions_to_goal, _, path = construct_path(goal_state, bfs_parent_dict)
    action = actions_to_goal[0]
    path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
    path_color = np.linspace(1, 0.2, len(path))
    return action

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

def breadth_first_search2(traversable_array, goal_array, start_pos, start_theta_ind, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, exhaustive=False):
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
        return None, None
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
        children, actions = get_children(px, py, theta_ind, traversable_array.shape, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution) # TODO: This probably should be env-specific (action set)
        for i in range(len(children)):
            # print("[breadth_first_search] children[i]: {}".format(children[i]))
            try:
                skip = traversable_array[children[i][1], children[i][0]] == 0
                # print("child in traversable array")
            except IndexError:
                # print("child *not* in traversable array")
                skip = True
            if skip:
                # print("child is not traversable")
                continue
            if visited_array[children[i][1], children[i][0], children[i][2]] == 0:
                visited_array[children[i][1], children[i][0], children[i][2]] = 1
                queue.append(children[i])
                if children[i] not in meta:
                    # print("adding child to meta.")
                    meta[children[i]] = (vertex, actions[i])
                # else:
                #     print("child already in meta.")
            # else:
            #     print("child already visited.")
        # # if num_vertices_popped % 100 == 0:
        # if num_vertices_popped % 1 == 0:
        #     print("[breadth_first_search] visualizing visited_array...")
        #     plt.figure('bfs')
        #     plt.imshow(visited_array[:,:,0])
        #     plt.pause(0.01)
    if not exhaustive:
        print("[breadth_first_search] warning: queue is empty. while loop ended.")
        return
    return meta, visited_array[:,:,0]

def get_children(gridmap_x, gridmap_y, theta_ind, gridmap_upper_bnds, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution):

    real_x, real_y = env_to_coor(gridmap_x, gridmap_y)
    next_states, actions, gridmap_discretization = env_next_coords(real_x, real_y, theta_ind)
    next_gridmap_x, next_gridmap_y = env_to_grid(next_states[:,0], next_states[:,1])
    
    # print("started at gridmap_x, gridmap_y, theta_ind: ({},{},{})".format(gridmap_x, gridmap_y, theta_ind))
    # print("real_x, real_y, theta_ind: ({},{},{})".format(real_x, real_y, theta_ind))
    # print("next_states: {}".format(next_states))
    # print("next_gridmap_x, next_gridmap_y: ({},{})".format(next_gridmap_x, next_gridmap_y))

    # gridmap_discretization = int(1./env_grid_resolution)
    # gridmap_discretization = 2

    num_jumps_x = np.around((next_gridmap_x - gridmap_x) / gridmap_discretization).astype(int)
    next_gridmap_x = gridmap_x + gridmap_discretization*num_jumps_x
    # print("num_jumps_x, next_gridmap_x: ({},{})".format(num_jumps_x, next_gridmap_x))
    num_jumps_y = np.around((next_gridmap_y - gridmap_y) / gridmap_discretization).astype(int)
    next_gridmap_y = gridmap_y + gridmap_discretization*num_jumps_y
    # print("num_jumps_y, next_gridmap_y: ({},{})".format(num_jumps_y, next_gridmap_y))


    #####
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
    #####

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

def get_fov_aware_goal_array2(raw_goal_array, camera_fov, camera_range_x, camera_range_y):
    num_theta_inds = 4
    if np.sum(raw_goal_array) == 0:
        # none of the points are goals, so no point in using the FOV to see which points see the non-existent goal
        return raw_goal_array
        # if raw_goal_array.ndim > 2:
        #     return raw_goal_array
        # else:
        #     return np.repeat(raw_goal_array[:, :, np.newaxis], num_theta_inds, axis=2)
    # If the raw goal_array contains an axis that defines what theta_ind will be able to see that goal, that info can be ignored with this function.
    if raw_goal_array.ndim > 2:
        goal_array = np.any(raw_goal_array, axis=2)
    else:
        goal_array = raw_goal_array.copy()

    goal_inds = np.where(goal_array == 1)

    camera_fov = camera_fov # full FOV in radians
    camera_range_x = camera_range_x; camera_range_y = camera_range_y;

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

def check_if_goal_reachable(goal_array, reachable_array):
    # This hasn't really been tested. supposed to tell you if any of the goal inds are within reachable inds ==> your goal is reachable
    if goal_array.ndim > 2:
        goal_array = np.any(goal_array, axis=2)
    reachable_goal_inds = np.where(np.logical_and(goal_array, reachable_array))
    goal_is_reachable = len(reachable_goal_inds[0]) > 0
    return goal_is_reachable, reachable_goal_inds

def find_reachable_frontier_indices2(semantic_array, reachable_array, camera_fov, camera_range_x, camera_range_y):
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

    fov_aware_frontier_array = get_fov_aware_goal_array2(frontier_array, camera_fov, camera_range_x, camera_range_y)
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
