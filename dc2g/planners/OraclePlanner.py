from dc2g.planners.Planner import Planner
import matplotlib.pyplot as plt
from dc2g.util import find_traversable_inds, find_goal_inds
import dc2g.planning_utils as planning_utils
import numpy as np
import scipy.ndimage.morphology

class OraclePlanner(Planner):
    def __init__(self, traversable_colors, goal_color, room_or_object_goal, world_image_filename, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_world_array, env_render, name="Oracle"):
        super(OraclePlanner, self).__init__(name, traversable_colors, goal_color, room_or_object_goal, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render)

        # self.semantic_gridmap = env_world_array
        # self.world_image_filename = world_image_filename
        # self.semantic_gridmap = plt.imread(self.world_image_filename)

        self.env_to_coor = env_to_coor
        self.env_next_coords = env_next_coords
        self.env_to_grid = env_to_grid
        self.env_grid_resolution = env_grid_resolution

        self.actions_to_goal = None

    def plan(self, obs):
        print('--- plan ---')
        print("self.actions_to_goal: {}".format(self.actions_to_goal))
        self.step_number += 1
        self.semantic_gridmap = obs['semantic_gridmap']
        if self.actions_to_goal is None:
            traversable_array, _, _ = find_traversable_inds(self.semantic_gridmap, self.traversable_colors)
            goal_array, _, _ = find_goal_inds(self.semantic_gridmap, self.goal_color, self.room_or_object_goal)
            self.actions_to_goal, _, self.path = planning_utils.breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], self.env_to_coor, self.env_next_coords, self.env_to_grid, self.env_grid_resolution)
            print(self.actions_to_goal)
            # self.plot_oracle_path()
        # self.plot(obs['semantic_gridmap'])

        if len(self.actions_to_goal) > 0:
            action = self.actions_to_goal.pop(0)
        else:
            assert(0)
            print("no more actions in queue...")
            action = 0
        print("action: {}".format(action))
        return action

    def plot_oracle_path(self):
        if self.plot_panels:
            path_inds = (np.array([x[1] for x in self.path]), np.array([x[0] for x in self.path]))
            path_color = np.linspace(0.8, 0.2, len(self.path))
            path_array = np.zeros((self.semantic_gridmap.shape[0], self.semantic_gridmap.shape[1]))
            path_array[path_inds] = path_color

            # num_inflations = 1
            # struct2 = scipy.ndimage.generate_binary_structure(2, 2)
            # for i in range(num_inflations):
            #     path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

            planner_array = np.zeros_like(self.semantic_gridmap)
            planner_array[:,:,0] = path_array
            planner_array[:,:,1] = path_array
            planner_array[:,:,2] = path_array
            planner_array[path_array == 0] = self.semantic_gridmap[path_array == 0]
            plt.figure("DC2G")
            plt.subplot(self.subplots["planner"])
            plt.title("Oracle Path to Goal")
            plt.imshow(planner_array)

    def visualize(self):
        raise NotImplementedError