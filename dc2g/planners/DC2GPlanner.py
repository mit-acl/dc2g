from dc2g.planners.FrontierPlanner import FrontierPlanner
import numpy as np
from skimage.transform import resize
import tensorflow as tf
import json
import os

import matplotlib.pyplot as plt


class DC2GPlanner(FrontierPlanner):
    def __init__(self, model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render, output_name="output_masked", name="DC2G"):
        super(DC2GPlanner, self).__init__(traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render, name=name)

        self.load_model(model_name, output_name)

        self.search_planner = self.dc2g_planner

    def load_model(self, model_name, output_name="output_masked"):
        # Set up the deep cost-to-go network (load network weights)
        self.tf_sess = tf.compat.v1.Session()
        model_dir = "{project_path}/data/trained_networks/{model_name}".format(project_path=self.project_path, model_name=model_name)
        saver = tf.compat.v1.train.import_meta_graph(model_dir + "/export.meta")
        saver.restore(self.tf_sess, model_dir + "/export")
        input_vars = json.loads(tf.compat.v1.get_collection("inputs")[0].decode('utf-8'))
        output_vars = json.loads(tf.compat.v1.get_collection("outputs")[0].decode('utf-8'))
        input = tf.compat.v1.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.compat.v1.get_default_graph().get_tensor_by_name(output_vars[output_name])
        self.tf_tensors = {'input': input, 'output': output}
        try:
            goal_rgb = tf.compat.v1.get_default_graph().get_tensor_by_name(input_vars["goal_rgb"])
            self.tf_tensors['goal_rgb'] = goal_rgb
        except:
            pass
        print("loaded model.")

    def visualize(self):
        raise NotImplementedError

    def dc2g_planner(self, position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, traversable_array):
        '''
        outdated doc...
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
        c2g_array, raw_c2g = self.c2g_query(semantic_array)
        c2g_array[traversable_array == 0] = 0

        # plt.imshow(c2g_array, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        self.c2g_array = c2g_array

        if self.plot_panels:
            plt.figure("Planner Panel")
            plt.subplot(self.subplots["DC2G"])
            plt.imshow(c2g_array, cmap=plt.cm.gray, interpolation='nearest')
        if self.save_individual_figures:
            plt.imsave("{individual_figure_path}/c2g/step_{step_num}.png".format(individual_figure_path=self.individual_figure_path, step_num=str(self.step_number).zfill(3)), raw_c2g)

        action, path = self.choose_frontier_pt(position, theta_ind, semantic_array, reachable_array, bfs_parent_dict, traversable_array, frontier_cost_array=c2g_array)

        return action, path

    def saveIndivFig(self, dir, arr):
        full_dir = "{individual_figure_path}/{dir}".format(dir=dir, individual_figure_path=self.individual_figure_path)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        plt.imsave("{full_dir}/step_{step_num}.png".format(full_dir=full_dir, step_num=str(self.step_number).zfill(3)), arr)

    def c2g_query(self, semantic_array):

        input_data = semantic_array
        if input_data.shape[2] == 3:
            input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
        desired_input_shape = self.tf_tensors['input'].get_shape().as_list()[:2]
        input_data = resize(input_data, desired_input_shape, order=0)
        if np.max(input_data) > 1:
            input_data = input_data / 255.
        if input_data.shape[2] == 4:
            input_data = input_data[:,:,:3]

        # input_data = input_data*255
        feed_dict = {self.tf_tensors['input']: input_data}
        if 'goal_rgb' in self.tf_tensors:
            goal_rgb = goal_rgb_val = np.array([128., 0., 0.])/255.
            feed_dict[self.tf_tensors['goal_rgb']] = goal_rgb
        output_value = self.tf_sess.run(self.tf_tensors['output'], feed_dict=feed_dict)
        output_value_resized = resize(output_value[:,:,0], semantic_array.shape[:2], order=0)
        c2g_array = output_value_resized

        # hsv = plt_colors.rgb_to_hsv(output_value)
        # c2g_array = hsv[:, :, 2]
        # c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map
        # c2g_array = scipy.misc.imresize(c2g_array, semantic_array.shape[:2], interp='nearest')
        return c2g_array, output_value_resized
