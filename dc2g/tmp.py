import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from run_episode import find_reachable_frontier_indices2
# from util import get_traversable_colors

# def get_fov_aware_goal_array(raw_goal_array):
#     # global ENVIRONMENT
#     # If the raw goal_array contains an axis that defines what theta_ind will be able to see that goal, that info can be ignored with this function.
#     if raw_goal_array.ndim > 2:
#         goal_array = np.any(raw_goal_array, axis=2)
#     else:
#         goal_array = raw_goal_array.copy()

#     # plt.figure("goal_array")
#     # plt.imshow(goal_array)
#     # plt.pause(1)
#     goal_inds = np.where(goal_array == 1)
#     camera_fov = np.pi/12 # full FOV in radians
#     camera_range_in_meters = 2
#     num_theta_inds = 4
#     fov_aware_goal_array = np.repeat(goal_array[:, :, np.newaxis], num_theta_inds, axis=2)
#     # fov_aware_goal_array = goal_array.copy()
#     # tx1, ty1, tx2, ty2 = ENVIRONMENT.rescale(0, 0, camera_range_in_meters, camera_range_in_meters)
#     # camera_range_x = tx2 - tx1; camera_range_y = ty2 - ty1
    
#     camera_range_x = 20; camera_range_y = 10;
#     grid_inds = np.indices(goal_array.shape)
#     grid_array = np.dstack([grid_inds[1], grid_inds[0]])
#     for i in range(len(goal_inds[0])):
#         print("{}/{}".format(i, len(goal_inds[0])))
#         goal_pos = np.array([goal_inds[1][i], goal_inds[0][i]])
#         rel_pos = goal_pos - grid_array
#         ellipse_r = rel_pos**2 / np.array([camera_range_x, camera_range_y])**2
#         r_arr = np.sum(ellipse_r, axis=2) <= 1

#         rel_angle = np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) # angle from a particular grid cell to the current cam pos
#         for theta_ind in range(num_theta_inds):
#             cam_angle = wrap(np.radians((theta_ind+2) * 90))
#             angle_offset = wrap(rel_angle + cam_angle)
#             angle_arr = abs(angle_offset) < (camera_fov/2)
#             observable_arr = np.bitwise_and(r_arr, angle_arr)
#             fov_aware_goal_array[:,:,theta_ind][observable_arr == 1] = 1
#     # plt.figure("fov_aware_goal_array")
#     # plt.imshow(fov_aware_goal_array[:,:,2])
#     # plt.imshow(np.any(fov_aware_goal_array, axis=2))
#     # plt.pause(1)

#     # TODO!!!!
#     return fov_aware_goal_array

# def get_fov_aware_goal_array2(raw_goal_array):
#     num_theta_inds = 4
#     if np.sum(raw_goal_array) == 0:
#         # none of the points are goals, so no point in using the FOV to see which points see the non-existent goal
#         return raw_goal_array
#         # if raw_goal_array.ndim > 2:
#         #     return raw_goal_array
#         # else:
#         #     return np.repeat(raw_goal_array[:, :, np.newaxis], num_theta_inds, axis=2)
#     global ENVIRONMENT
#     # If the raw goal_array contains an axis that defines what theta_ind will be able to see that goal, that info can be ignored with this function.
#     if raw_goal_array.ndim > 2:
#         goal_array = np.any(raw_goal_array, axis=2)
#     else:
#         goal_array = raw_goal_array.copy()

#     goal_inds = np.where(goal_array == 1)

#     camera_fov = ENVIRONMENT.camera_fov # full FOV in radians
#     camera_range_x = ENVIRONMENT.camera_range_x-1; camera_range_y = ENVIRONMENT.camera_range_y-1;

#     padded_goal_array = np.pad(goal_array,((camera_range_y,camera_range_y),(camera_range_x,camera_range_x)), 'constant',constant_values=0)
#     fov_aware_goal_array = np.repeat(padded_goal_array[:, :, np.newaxis], num_theta_inds, axis=2)

#     window = np.empty((2*camera_range_y+1, 2*camera_range_x+1))
#     grid_inds = np.indices(window.shape)
#     grid_array = np.dstack([grid_inds[1], grid_inds[0]])
#     goal_pos = np.array([camera_range_x, camera_range_y])
#     rel_pos = goal_pos.astype(np.float32) - grid_array
#     ellipse_r = rel_pos**2 / np.array([camera_range_x, camera_range_y])**2
#     r_arr = np.sum(ellipse_r, axis=2) < 1

#     rel_angle = np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) # angle from a particular grid cell to the current cam pos
#     observable_arr = np.repeat(r_arr[:, :, np.newaxis], num_theta_inds, axis=2)
#     for theta_ind in range(num_theta_inds):
#         cam_angle = wrap(np.radians((theta_ind+2) * 90))
#         angle_offset = wrap(rel_angle + cam_angle)
#         angle_arr = abs(angle_offset) < (camera_fov/2)
#         observable_arr[:,:,theta_ind] = np.bitwise_and(r_arr, angle_arr)

#     for i in range(len(goal_inds[0])):
#         gy = goal_inds[0][i]; gx = goal_inds[1][i]
#         fov_aware_goal_array[gy:gy+2*camera_range_y+1,gx:gx+2*camera_range_x+1, :] += observable_arr
#         if gy == 83 and gx == 42:
#             fov_aware_goal_array[gy, gx] = 

#     # fov_aware_goal_array = fov_aware_goal_array > 0
#     fov_aware_goal_array = (fov_aware_goal_array > 0).astype(int)

#     for i in range(len(goal_inds[0])):
#         gy = goal_inds[0][i]; gx = goal_inds[1][i]
#         fov_aware_goal_array[gy+camera_range_y:gy+camera_range_y+1,gx+camera_range_x:gx+camera_range_x+1, :] = 2

#     plt.figure("fov_goal_array")
#     plt.subplot(2,2,1)
#     plt.imshow(fov_aware_goal_array[:,:,0])
#     plt.subplot(2,2,2)
#     plt.imshow(fov_aware_goal_array[:,:,1])
#     plt.subplot(2,2,3)
#     plt.imshow(fov_aware_goal_array[:,:,2])
#     plt.subplot(2,2,4)
#     plt.imshow(fov_aware_goal_array[:,:,3])
#     plt.pause(0.01)

#     unpadded_fov_aware_goal_array = fov_aware_goal_array[camera_range_y:-camera_range_y, camera_range_x:-camera_range_x]
#     return unpadded_fov_aware_goal_array

# # keep angle between [-pi, pi]
# def wrap(angle):
#     return (angle + np.pi) % (2*np.pi) - np.pi


if __name__ == '__main__':
    # raw_goal_array = np.zeros((100,100))
    # raw_goal_array[10,60] = 1
    # raw_goal_array[70,40] = 1

    # t_start = time.time()
    # get_fov_aware_goal_array(raw_goal_array)
    # t_end = time.time()
    # t1 = t_end - t_start

    # t_start = time.time()
    # get_fov_aware_goal_array2(raw_goal_array)
    # t_end = time.time()
    # t2 = t_end - t_start

    # print("t1: {}, t2: {}".format(t1, t2))

    with open("/home/mfe/code/dc2g/semantic_array.p",'rb') as f:
        semantic_array = pickle.load(f)
    with open("/home/mfe/code/dc2g/reachable_array.p",'rb') as f:
        reachable_array = pickle.load(f)
    # semantic_array[48,77] = semantic_array[48,76]
    # semantic_array[47,78] = semantic_array[48,76]
    frontier_array, reachable_frontier_array, fov_aware_frontier_array, fov_aware_reachable_frontier_array = find_reachable_frontier_indices2(semantic_array, reachable_array)
    # lowest_cost_frontier_state = (83, 42)
    lowest_cost_frontier_state = (42, 83)
    print(fov_aware_reachable_frontier_array[lowest_cost_frontier_state])
    plt.figure('semantic')
    plt.imshow(semantic_array)
    plt.figure('reachable')
    plt.imshow(reachable_array)
    plt.figure('front')
    plt.imshow(np.any(frontier_array, axis=2))
    plt.figure('fov')
    # plt.imshow(np.any(fov_aware_frontier_array, axis=2))
    plt.imshow(fov_aware_frontier_array[:,:,2])
    plt.show()