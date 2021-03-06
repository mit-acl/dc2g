from dc2g.planners.Planner import Planner
from dc2g.util import find_traversable_inds, find_goal_inds
import dc2g.planning_utils as planning_utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

class FrontierPlanner(Planner):
    def __init__(self, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render, output_name="output_masked", name="Frontier"):
        super(FrontierPlanner, self).__init__(name, traversable_colors, goal_color, room_or_object_goal, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render)

        # These parameters determine what a frontier pt is
        self.camera_fov = camera_fov
        self.camera_range_x = camera_range_x
        self.camera_range_y = camera_range_y

        self.env_to_coor = env_to_coor
        self.env_next_coords = env_next_coords
        self.env_to_grid = env_to_grid
        self.env_grid_resolution = env_grid_resolution

        # What planner to call when the goal is not in sight/not yet reachable
        self.search_planner = self.bfs_planner

        ### Traversability parameters
        # With perfect semantic map
        self.num_inflations_traversability = 0
        # # With imperfect semantic map, allow a little fudging on traversability
        # self.num_inflations_traversability = 3

        ### Visualization parameters
        # For small env, no need to inflate path viz
        self.num_inflations_visualize_path = 0
        # # For large env, inflate path viz
        # self.num_inflations_visualize_path = 1

    def plan(self, obs):
        # self.path = [[obs['pos'][0]+i, obs['pos'][1]] for i in range(10)]
        # return 0

        self.step_number += 1
        self.setup_plots_()
        traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], self.traversable_colors)

        struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        for i in range(self.num_inflations_traversability):
            traversable_array = scipy.ndimage.morphology.binary_dilation(traversable_array, struct2).astype(traversable_array.dtype)

        goal_array, _, _ = find_goal_inds(obs['semantic_gridmap'], self.goal_color, self.room_or_object_goal)
        bfs_parent_dict, reachable_array = planning_utils.breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], self.env_to_coor, self.env_next_coords, self.env_to_grid, self.env_grid_resolution, exhaustive=True)

        if bfs_parent_dict is None and reachable_array is None:
            action = 3
            path = []
        elif np.sum(goal_array) == 0:
            # Haven't seen the goal yet ==> use searcher
            # print("Haven't seen the goal yet ==> Using searcher.")
            action, path = self.search_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, traversable_array)
        else:
            goal_is_reachable, reachable_goal_inds = planning_utils.check_if_goal_reachable(goal_array, reachable_array)
            if goal_is_reachable:
                # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
                # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
                goal_states = np.where(goal_array == 1)
                goal_state = (reachable_goal_inds[1][0], reachable_goal_inds[0][0]) # just pick the first goal_state - if there are multiple, may wanna do something smarter
                action = planning_utils.bfs_backtracking_planner(bfs_parent_dict, goal_state)
            else:
                # Have seen goal, but no path to it exists yet ==> use searcher
                # print("Have seen the goal, but no path exists to it ==> Using searcher.")
                action, path = self.search_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, traversable_array)
        # self.path = path
        self.plot(obs['semantic_gridmap'])
        return action

    def bfs_planner(self, position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, traversable_array):
        action, path = self.choose_frontier_pt(position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, traversable_array)
        return action, path

    def choose_frontier_pt(self, position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, traversable_array, frontier_cost_array=None):
        # print("Looking for frontier pts in semantic map...")
        frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array = planning_utils.find_reachable_frontier_indices2(semantic_array, reachable_array, self.camera_fov, self.camera_range_x, self.camera_range_y)
        if self.save_individual_figures:
            self.saveIndivFig("frontiers", frontier_array)
            self.saveIndivFig("reachable_frontier_array", frontier_array)
            self.saveIndivFig("fov_aware_frontier_array", frontier_array)
            self.saveIndivFig("fov_aware_reachable_frontier_array", frontier_array)

        self.frontier_array = frontier_array


        if frontier_cost_array is None:
            actions_to_frontier, point_on_frontier, path = planning_utils.breadth_first_search2(reachable_array, fov_aware_reachable_frontier_array, position, theta_ind, self.env_to_coor, self.env_next_coords, self.env_to_grid, self.env_grid_resolution)
            # print("path: {}, point_on_frontier: {}".format(path, point_on_frontier))
            # print("actions_to_frontier: {}".format(actions_to_frontier))
        else:
            frontier_c2gs = np.zeros_like(frontier_cost_array)
            frontier_c2gs[np.any(fov_aware_reachable_frontier_array, axis=2) == 1] = frontier_cost_array[np.any(fov_aware_reachable_frontier_array, axis=2) == 1]

            if np.max(frontier_c2gs) == 0:
                print("none of the frontier pts have a non-zero c2g. Very bad news.")
                lowest_cost_frontier_ind = np.unravel_index(np.any(fov_aware_reachable_frontier_array, axis=2).argmax(), frontier_c2gs.shape)
            else:
                lowest_cost_frontier_ind = np.unravel_index(frontier_c2gs.argmax(), frontier_c2gs.shape)
            lowest_cost_frontier_state = (lowest_cost_frontier_ind[1], lowest_cost_frontier_ind[0])
            # print("lowest_cost_frontier_state: {}".format(lowest_cost_frontier_state))
            # print("bfs_parent_dict: {}".format(bfs_parent_dict))
            actions_to_frontier, _, path = planning_utils.construct_path(lowest_cost_frontier_state, bfs_parent_dict)

            if position[0] == lowest_cost_frontier_state[0] and position[1] == lowest_cost_frontier_state[1]:
                print("[dc2g_planner] warning: currently at the frontier position ==> spin toward nearest theta that is a frontier theta.")
                frontier_thetas = np.where(fov_aware_reachable_frontier_array[lowest_cost_frontier_ind])[0]
                closest_theta_arg = np.argmin((theta_ind - frontier_thetas) % fov_aware_reachable_frontier_array.shape[2])
                closest_theta = frontier_thetas[closest_theta_arg]
                lowest_cost_frontier_state = (lowest_cost_frontier_state[0], lowest_cost_frontier_state[1], closest_theta)
                actions_to_frontier, _, path = planning_utils.construct_path(lowest_cost_frontier_state, bfs_parent_dict)

        if len(actions_to_frontier) == 0:
            print("[dc2g_planner] warning: len(actions_to_frontier)==0 ==> somehow we are at a frontier pt, which means it shouldn't be called a frontier pt. disagreement btwn camera in env and frontier pt finder. spin?")
            actions_to_frontier = [1] # super arbitrary which dir to spin
            path = [lowest_cost_frontier_state]

        action = actions_to_frontier[0]

        self.visualize_plans(semantic_array, path, fov_aware_reachable_frontier_array, reachable_array, frontier_array, position)
        return action, path

    def visualize_plans(self, semantic_array, path, fov_aware_reachable_frontier_array, reachable_array, frontier_array, position):

        plt_colors = {}
        plt_colors["red"] = [0.8500, 0.3250, 0.0980]
        plt_colors["blue"] = [0.0, 0.4470, 0.7410]
        plt_colors["green"] = [0.4660, 0.6740, 0.1880]
        plt_colors["purple"] = [0.4940, 0.1840, 0.5560]
        plt_colors["orange"] = [0.9290, 0.6940, 0.1250]
        plt_colors["cyan"] = [0.3010, 0.7450, 0.9330]
        plt_colors["chocolate"] = [0.6350, 0.0780, 0.1840]

        # path_color = np.linspace(0.5, 0.5, len(path))
        if self.make_panels:
            path_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
            # path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
            # path_array[path_inds] = 1

            for i in range(len(path)-1):
                x_low = min(path[i][1], path[i+1][1])
                x_high = max(path[i][1], path[i+1][1])
                y_low = min(path[i][0], path[i+1][0])
                y_high = max(path[i][0], path[i+1][0])
                if x_low == x_high: x_high += 1
                if y_low == y_high: y_high += 1
                path_array[x_low:x_high, y_low:y_high] = 1


            struct2 = scipy.ndimage.generate_binary_structure(2, 2)
            for i in range(self.num_inflations_visualize_path):
                for i in range(4):
                    fov_aware_reachable_frontier_array[:,:,i] = scipy.ndimage.morphology.binary_dilation(fov_aware_reachable_frontier_array[:,:,i], struct2).astype(fov_aware_reachable_frontier_array.dtype)
                reachable_array = scipy.ndimage.morphology.binary_dilation(reachable_array, struct2).astype(reachable_array.dtype)
                path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

            planner_array = np.ones((semantic_array.shape[0], semantic_array.shape[1], 4))
            colors = {
                  'reachable': {'color': plt_colors["purple"], 'condition': reachable_array == 1, 'name': 'Fully Explored Pts'},
                  'frontier':  {'color': plt_colors["blue"], 'condition': np.any(frontier_array, axis=2) == 1, 'name': 'Frontier Pts'},
                  'reachable_frontier':  {'color': plt_colors["green"], 'condition': np.any(fov_aware_reachable_frontier_array, axis=2) == 1, 'name': 'Frontier-Extending Pts'},
                  'path':      {'color': plt_colors["red"], 'condition': path_array == 1, 'name': 'Planned Path'}
                      }

            legend_elements = []
            # keys = ['frontier']
            # keys = ['path']
            keys = ['frontier', 'reachable', 'reachable_frontier', 'path']
            # keys = ['frontier', 'reachable', 'reachable_frontier']
            for key in keys:
                param = colors[key]
                # legend_elements.append(Patch(facecolor=param["color"], label=param['name']))
                for i in range(len(param['color'])):
                    planner_array[:,:,i][param['condition']] = param['color'][i]

            planner_array[position[1], position[0], :3] = plt_colors["cyan"]
            planner_array[position[1], position[0], 3] = 1

            self.planner_array = planner_array

            # width = 1+num_inflations
            # for i in range(position[0]-width, position[0]+width):
            #     for j in range(position[1]-width, position[1]+width):
            #         planner_array[j,i,:3] = plt_colors["cyan"]
            #         planner_array[j,i,3] = 1

            if self.plot_panels:
                from matplotlib.patches import Patch
                for key in keys:
                    param = colors[key]
                    legend_elements.append(Patch(facecolor=param["color"], label=param['name']))
                plt.figure("Planner Panel")
                plt.subplot(self.subplots["planner"])
                # plt.legend(handles=legend_elements, bbox_to_anchor=(1,1.02,1,0.2), loc="lower left", mode="expand", ncol=len(colors))
                plt.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=12)

                # plt.imsave("{dir_path}/results/frontiers/step_{step_num}.png".format(dir_path=dir_path, step_num=str(step_number).zfill(3)), planner_array)
                plt.imshow(planner_array, interpolation='nearest')
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