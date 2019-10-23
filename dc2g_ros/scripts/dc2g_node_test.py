from dc2g_node import DC2G
import os
import pickle
from dc2g.util import find_traversable_inds
import numpy as np
import matplotlib.pyplot as plt

def main():
    n = DC2G(non_ros_mode=True)

    with open(os.path.dirname(os.path.realpath(__file__))+"/obs_273.pkl", "rb") as f:
    # with open(os.path.dirname(os.path.realpath(__file__))+"/obs_50.pkl", "rb") as f:
        obs = pickle.load(f)


    # started at gridmap_x, gridmap_y, theta_ind: (36,36,0)
    # real_x, real_y, theta_ind: (3.6,3.6,0)
    # next_states: [[4.1 3.6 0. ]
    #  [3.6 3.6 3. ]
    #  [3.6 3.6 1. ]]
    # next_gridmap_x, next_gridmap_y: ([40 36 36],[36 36 36])
    # num_jumps_x, next_gridmap_x: ([0 0 0],[36 36 36])
    # num_jumps_y, next_gridmap_y: ([0 0 0],[36 36 36])

    # px, py = n.to_coor(36,36)
    # print(px, py)
    # next_states, _, _ = n.next_coords(px, py, 0)
    # print(next_states)
    # x = next_states[0,0]
    # print(x - n.lower_grid_x_min)
    # print(np.around((x - n.lower_grid_x_min) / n.grid_resolution, 7))
    # print(np.floor((x - n.lower_grid_x_min) / n.grid_resolution))
    # print(np.floor((x - n.lower_grid_x_min) / n.grid_resolution).astype(int))
    # # gx = np.floor((x - n.lower_grid_x_min) / n.grid_resolution).astype(int)
    # # print(gx)
    # gx, gy = n.to_grid(next_states[0,0],next_states[0,1])
    # print(gx, gy)

    # assert(0)

    obs['pos'] = np.array([6, 31])
    obs['theta'] = 3

    # obs['semantic_gridmap'][obs['semantic_gridmap'] == 1] = 1

    obs['semantic_gridmap'][:12,:,:] = 0
    obs['semantic_gridmap'][11:14,17:20,:] = [0., 0., 128./255.]
    # obs['semantic_gridmap'][14,21,:] = [0., 0., 1.]

    # print(obs['semantic_gridmap'])
    # traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], n.planner.traversable_colors)
    # traversable_gridmap = np.zeros_like(obs['semantic_gridmap'])
    # traversable_gridmap[:,:,0] = 1.
    # traversable_gridmap[traversable_array == 1] = 1.
    # obs['semantic_gridmap'] = traversable_gridmap

    # plt.imshow(traversable_gridmap, interpolation='nearest')
    # plt.show()

    action = n.planner.plan(obs)
    
    print(action)

    print('---\nDone.')

if __name__ == "__main__":
    main()

