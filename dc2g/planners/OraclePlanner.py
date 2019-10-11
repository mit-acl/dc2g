from dc2g.planners.Planner import Planner
import matplotlib.pyplot as plt
from dc2g.util import find_traversable_inds, find_goal_inds
import dc2g.planning_utils as planning_utils
import numpy as np

class OraclePlanner(Planner):
    def __init__(self, traversable_colors, goal_color, room_or_object_goal, world_image_filename, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, name="Frontier"):
        self.name = name

        self.traversable_colors = traversable_colors
        self.goal_color = goal_color
        self.room_or_object_goal = room_or_object_goal

        self.world_image_filename = world_image_filename

        self.env_to_coor = env_to_coor
        self.env_next_coords = env_next_coords
        self.env_to_grid = env_to_grid
        self.env_grid_resolution = env_grid_resolution


        self.step_number = 0
        self.actions_to_goal = None

    def plan(self, obs):
        self.step_number += 1
        if self.actions_to_goal is None:
            self.full_semantic_gridmap = plt.imread(self.world_image_filename) # only used for oracle planner
            # full_size_semantic_gridmap = env.world_array
            traversable_array, _, _ = find_traversable_inds(self.full_semantic_gridmap, self.traversable_colors)
            goal_array, _, _ = find_goal_inds(self.full_semantic_gridmap, self.goal_color, self.room_or_object_goal)
            self.actions_to_goal, _, self.path = planning_utils.breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], self.env_to_coor, self.env_next_coords, self.env_to_grid, self.env_grid_resolution)
        action = self.actions_to_goal.pop(0)

        ############## auxiliary just for viz
        if self.plot_panels:
            path_inds = (np.array([x[1] for x in self.path]), np.array([x[0] for x in self.path]))
            path_color = np.linspace(0.5, 0.5, len(self.path))
            path_array = np.zeros((self.full_semantic_gridmap.shape[0], self.full_semantic_gridmap.shape[1]))
            path_array[path_inds] = path_color

            # num_inflations = 5
            # struct2 = scipy.ndimage.generate_binary_structure(2, 2)
            # for i in range(num_inflations):
            #     path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

            planner_array = np.zeros_like(self.full_semantic_gridmap)
            planner_array[:,:,0] = path_array
            planner_array[:,:,1] = path_array
            # planner_array[:,:,3] = 1
            planner_array[path_array == 0] = self.full_semantic_gridmap[path_array == 0]
            # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 0] = 1
            # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 1] = 0
            # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 2] = 0
            plt.figure("DC2G")
            plt.subplot(133)
            plt.title("Oracle Path to Goal")
            plt.imshow(planner_array)
        ###################################
        self.plot(obs['semantic_gridmap'])
        
        return action

    def visualize(self):
        raise NotImplementedError