import numpy as np
from dc2g.planners.DC2GPlanner import DC2GPlanner


class DC2GRescalePlanner(DC2GPlanner):
    def __init__(self, model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render, output_name="output", name="DC2G_Rescale"):
        super(DC2GRescalePlanner, self).__init__(model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, env_render, output_name="output", name=name)
        self.rescale_semantic_map = True

    def c2g_query(self, semantic_array):
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

        c2g_array, output_value_resized = super(DC2GRescalePlanner, self).c2g_query(cropped_padded_semantic_array)

        # Remove padding
        c2g_array = c2g_array[padding:-padding, padding:-padding]

        # Stick remaining square back into right slot within full semantic map
        final_c2g_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        final_c2g_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff] = c2g_array

        return final_c2g_array, output_value_resized