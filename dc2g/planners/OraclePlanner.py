from dc2g.planners.Planner import Planner

class OraclePlanner(Planner):
    def __init__(self, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, name="Frontier"):
        self.name = name
        self.step_number = 0
        self.actions_to_goal = None

    def step(self, entire_semantic_array):
        if self.step_number == 0:
            full_semantic_gridmap = plt.imread(env.world_image_filename) # only used for oracle planner
            full_size_semantic_gridmap = env.world_array
            traversable_array, _, _ = find_traversable_inds(entire_semantic_array, self.traversable_colors)
            goal_array, _, _ = find_goal_inds(entire_semantic_array, self.goal_color, self.room_or_object_goal)
            self.actions_to_goal, _, path = breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'])
        action = self.actions_to_goal.pop(0)
        return action

        # ############## auxiliary just for viz
        # if plot_panels:
        #     path_inds = (np.array([x[1] for x in path]), np.array([x[0] for x in path]))
        #     path_color = np.linspace(0.5, 0.5, len(path))
        #     path_array = np.zeros((entire_semantic_array.shape[0], entire_semantic_array.shape[1]))
        #     path_array[path_inds] = path_color

        #     # num_inflations = 5
        #     # struct2 = scipy.ndimage.generate_binary_structure(2, 2)
        #     # for i in range(num_inflations):
        #     #     path_array = scipy.ndimage.morphology.binary_dilation(path_array, struct2).astype(path_array.dtype)

        #     planner_array = np.zeros_like(entire_semantic_array)
        #     planner_array[:,:,0] = path_array
        #     planner_array[:,:,1] = path_array
        #     # planner_array[:,:,3] = 1
        #     planner_array[path_array == 0] = entire_semantic_array[path_array == 0]
        #     # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 0] = 1
        #     # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 1] = 0
        #     # planner_array[obs['pos'][1]:obs['pos'][1]+5, obs['pos'][0]:obs['pos'][0]+5, 2] = 0
        #     plt.figure("DC2G")
        #     plt.subplot(133)
        #     plt.title("Oracle Path to Goal")
        #     plt.imshow(planner_array)
        # ###################################

    def visualize(self):
        raise NotImplementedError