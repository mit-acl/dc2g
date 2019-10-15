from dc2g.planners.Planner import Planner
import numpy as np
import dc2g.planning_utils as planning_utils
from dc2g.util import find_traversable_inds, find_goal_inds
from skimage.transform import resize
import scipy.misc
import matplotlib.colors as plt_colors
import tensorflow as tf
import json

import matplotlib.pyplot as plt

class DC2GPlanner(Planner):
    def __init__(self, model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, output_name="output_masked", name="DC2G"):
        super(DC2GPlanner, self).__init__(name, traversable_colors, goal_color, room_or_object_goal, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution)
        self.rescale_semantic_map = False

        self.camera_fov = camera_fov
        self.camera_range_x = camera_range_x
        self.camera_range_y = camera_range_y

        self.env_to_coor = env_to_coor
        self.env_next_coords = env_next_coords
        self.env_to_grid = env_to_grid
        self.env_grid_resolution = env_grid_resolution

        self.load_model(model_name, output_name)

    def load_model(self, model_name, output_name="output_masked"):
        # Set up the deep cost-to-go network (load network weights)
        self.tf_sess = tf.Session()
        model_dir = "{project_path}/data/trained_networks/{model_name}".format(project_path=self.project_path, model_name=model_name)
        saver = tf.train.import_meta_graph(model_dir + "/export.meta")
        saver.restore(self.tf_sess, model_dir + "/export")
        input_vars = json.loads(tf.get_collection("inputs")[0])
        output_vars = json.loads(tf.get_collection("outputs")[0])
        input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.get_default_graph().get_tensor_by_name(output_vars[output_name])
        self.tf_tensors = {'input': input, 'output': output}
        try:
            goal_rgb = tf.get_default_graph().get_tensor_by_name(input_vars["goal_rgb"])
            self.tf_tensors['goal_rgb'] = goal_rgb
        except:
            pass
        print("loaded model.")

    def plan(self, obs):
        self.step_number += 1
        traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], self.traversable_colors)
        goal_array, _, _ = find_goal_inds(obs['semantic_gridmap'], self.goal_color, self.room_or_object_goal)
        bfs_parent_dict, reachable_array = planning_utils.breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], self.env_to_coor, self.env_next_coords, self.env_to_grid, self.env_grid_resolution, exhaustive=True)
        if np.sum(goal_array) == 0:
            # Haven't seen the goal yet ==> use dc2g
            # print("Haven't seen the goal yet ==> Using DC2G.")
            # action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, self.sess, self.tf_tensors, self.bfs_parent_dict, self.step_number, self.rescale_semantic_map)
            action = self.dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict)
        else:
            goal_is_reachable, reachable_goal_inds = planning_utils.check_if_goal_reachable(goal_array, reachable_array)
            if goal_is_reachable:
                # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
                # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
                goal_states = np.where(goal_array == 1)
                goal_state = (reachable_goal_inds[1][0], reachable_goal_inds[0][0]) # just pick the first goal_state - if there are multiple, may wanna do something smarter
                action = planning_utils.bfs_backtracking_planner(bfs_parent_dict, goal_state)
            else:
                # Have seen goal, but no path to it exists yet ==> use dc2g
                # print("Have seen the goal, but no path exists to it ==> Using DC2G.")
                # action = dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, self.sess, self.tf_tensors, self.bfs_parent_dict, self.step_number, self.rescale_semantic_map)
                action = self.dc2g_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict)
        self.plot(obs['semantic_gridmap'])
        return action

    def visualize(self):
        raise NotImplementedError

    def dc2g_planner(self, position, theta_ind, semantic_array, reachable_array, bfs_parent_dict):
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
        c2g_array, input_value, raw_c2g = self.c2g_query(semantic_array)

        if self.plot_panels:
            plt.figure("DC2G")
            plt.subplot(132)
            plt.imshow(raw_c2g, cmap=plt.cm.gray)
            # plt.imshow(c2g_array, cmap=plt.cm.gray)
            # plt.figure("DC2G")
            # plt.subplot(131)
            # plt.imshow(input_value)
        if self.save_individual_figures:
            plt.imsave("{individual_figure_path}/c2g/step_{step_num}.png".format(individual_figure_path=self.individual_figure_path, step_num=str(self.step_number).zfill(3)), raw_c2g)

        # print("Looking for frontier pts in semantic map...")
        frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array = planning_utils.find_reachable_frontier_indices2(semantic_array, reachable_array, self.camera_fov, self.camera_range_x, self.camera_range_y)

        frontier_c2gs = np.zeros_like(c2g_array)
        frontier_c2gs[np.any(fov_aware_reachable_frontier_array, axis=2) == 1] = c2g_array[np.any(fov_aware_reachable_frontier_array, axis=2) == 1]

        if np.max(frontier_c2gs) == 0:
            print("none of the frontier pts have a non-zero c2g. Very bad news.")
            lowest_cost_frontier_ind = np.unravel_index(np.any(fov_aware_reachable_frontier_array, axis=2).argmax(), frontier_c2gs.shape)
        else:
            lowest_cost_frontier_ind = np.unravel_index(frontier_c2gs.argmax(), frontier_c2gs.shape)

        lowest_cost_frontier_state = (lowest_cost_frontier_ind[1], lowest_cost_frontier_ind[0])
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
        return action

    def visualize_plans(self, semantic_array, path, fov_aware_reachable_frontier_array, reachable_array, frontier_array, position):

        # path_color = np.linspace(0.5, 0.5, len(path))
        if self.plot_panels:
            path_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
            path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
            path_array[path_inds] = 1

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

    def c2g_query(self, semantic_array):

        input_data = semantic_array
        if input_data.shape[2] == 3:
            input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
        input_data = scipy.misc.imresize(input_data, self.tf_tensors['input'].shape[:2], interp='nearest')
        if np.max(input_data) > 1:
            input_data = input_data / 255.
        if input_data.shape[2] == 4:
            input_data = input_data[:,:,:3]
        feed_dict = {self.tf_tensors['input']: input_data}
        if 'goal_rgb' in self.tf_tensors:
            goal_rgb = goal_rgb_val = np.array([128., 0., 0.])/255.
            feed_dict[self.tf_tensors['goal_rgb']] = goal_rgb
        output_value = self.tf_sess.run(self.tf_tensors['output'], feed_dict=feed_dict)
        output_value_resized = scipy.misc.imresize(output_value, semantic_array.shape[:2], interp='nearest')
        hsv = plt_colors.rgb_to_hsv(output_value)
        c2g_array = hsv[:, :, 2]
        c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map
        c2g_array = scipy.misc.imresize(c2g_array, semantic_array.shape[:2], interp='nearest')
        return c2g_array, semantic_array, output_value_resized