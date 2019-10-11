from dc2g.planners.Planner import Planner

class FrontierPlanner(Planner):
    def __init__(self, traversable_colors, goal_color, room_or_object_goal, camera_fov, camera_range_x, camera_range_y, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution, name="Frontier"):
        self.name = name
        self.step_number = 0

    def step(self, obs):
        traversable_array, _, _ = find_traversable_inds(obs['semantic_gridmap'], self.traversable_colors)
        goal_array, _, _ = find_goal_inds(obs['semantic_gridmap'], self.goal_color, self.room_or_object_goal)
        bfs_parent_dict, reachable_array = breadth_first_search2(traversable_array, goal_array, obs['pos'], obs['theta_ind'], exhaustive=True)    

        if np.sum(goal_array) == 0:
            # Haven't seen the goal yet ==> use frontier_nav
            # print("Haven't seen the goal yet ==> Using frontier_nav.")
            action = bfs_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, step_number)
        else:
            goal_is_reachable, reachable_goal_inds = check_if_goal_reachable(goal_array, reachable_array)
            if goal_is_reachable:
                # Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it
                # print("Have seen the goal & a path to it exists from exhaustive BFS ==> plan to it.")
                goal_states = np.where(goal_array == 1)
                goal_state = (reachable_goal_inds[1][0], reachable_goal_inds[0][0]) # just pick the first goal_state - if there are multiple, may wanna do something smarter
                action = bfs_backtracking_planner(bfs_parent_dict, goal_state)
            else:
                # Have seen goal, but no path to it exists yet ==> use frontier_nav
                # print("Have seen the goal, but no path exists to it ==> Using frontier_nav.")
                action = bfs_planner(obs['pos'], obs['theta_ind'], obs['semantic_gridmap'], reachable_array, bfs_parent_dict, step_number)
        if make_individual_figures:
            plt.figure("Observation")
            plt.imshow(obs['semantic_gridmap'])
        return action

    def visualize(self):
        raise NotImplementedError