import numpy as np
from dc2g.planners.DC2GPlanner import DC2GPlanner


class DC2GRescalePlanner(DC2GPlanner):
    def __init__(self, model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, output_name="output", name="DC2G_Rescale"):
        super(DC2GRescalePlanner, self).__init__(model_name, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, output_name="output", name=name)
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
        # Scale image up to translator's desired size (256x256)
        # resized_semantic_array = scipy.misc.imresize(cropped_padded_semantic_array, self.tf_tensors['input'].shape, interp='nearest')
        # resized_semantic_array = resize(cropped_padded_semantic_array, self.tf_tensors['input'].shape, order=0)

        c2g_array, semantic_array, output_value_resized = super(DC2GRescalePlanner, self).c2g_query(cropped_padded_semantic_array)
        # input_data = cropped_padded_semantic_array
        # if input_data.shape[2] == 3:
        #     input_data = np.dstack( ( input_data, np.ones(input_data.shape[:2]) ) )
        # input_data = scipy.misc.imresize(input_data, self.tf_tensors['input'].shape[:2], interp='nearest')
        # if np.max(input_data) > 1:
        #     input_data = input_data / 255.
        # if input_data.shape[2] == 4:
        #     input_data = input_data[:,:,:3]

        # # Query the encoder-decoder network for image translation
        # feed_dict = {self.tf_tensors['input']: input_data}
        # if 'goal_rgb' in self.tf_tensors:
        #     goal_rgb = goal_rgb_val = np.array([128., 0., 0.])/255.
        #     feed_dict[self.tf_tensors['goal_rgb']] = goal_rgb
        # output_value = self.tf_sess.run(self.tf_tensors['output'], feed_dict=feed_dict)

        # # Remove red from image, convert to grayscale via HSV saturation
        # hsv = plt_colors.rgb_to_hsv(output_value)
        # c2g_array = hsv[:, :, 2]
        # c2g_array[(hsv[:, :, 1] > 0.3)] = 0 # remove all "red" (non-traversable pixels) from c2g map

        # # Scale down to padded, cropped size
        # c2g_array = scipy.misc.imresize(c2g_array, (cropped_padded_semantic_array.shape[0], cropped_padded_semantic_array.shape[1]), interp='nearest')

        # Remove padding
        c2g_array = c2g_array[padding:-padding, padding:-padding]

        # Stick remaining square back into right slot within full semantic map
        final_c2g_array = np.zeros((semantic_array.shape[0], semantic_array.shape[1]))
        final_c2g_array[min_x_ind:min_x_ind+diff, min_y_ind:min_y_ind+diff] = c2g_array

        return final_c2g_array, semantic_array, output_value