from dc2g.planners.DC2GPlanner import DC2GPlanner
from dc2g.planners.DC2GRescalePlanner import DC2GRescalePlanner
from dc2g.planners.OraclePlanner import OraclePlanner
from dc2g.planners.FrontierPlanner import FrontierPlanner

from dc2g.util import get_traversable_colors, get_goal_colors, find_traversable_inds, find_goal_inds, inflate, wrap, round_base_down, round_base

planner_args = {
    'dc2g': {
        'model': "driveways_bing_iros19_full_test_works",
        'use_semantic_coloring': True,
        'output_name': "output",
        'load_nn': True
        },
    'dc2g_rescale': {
        'model': "driveways_bing_iros19_full_test_works",
        'use_semantic_coloring': True,
        'output_name': "output",
        'load_nn': True
        },
    'dc2g_without_semantics': {
        'model': "driveways_bing_iros19_masked2",
        'use_semantic_coloring': False,
        'output_name': "output_masked",
        'load_nn': True
        },
    'frontier': {
        'model': None,
        'use_semantic_coloring': True,
        'load_nn': False
        },
    'oracle': {
        'model': None,
        'use_semantic_coloring': True,
        'load_nn': False
        },
}

def instantiate_planner(planner_name, env, env_type,
    env_camera_fov=None, env_camera_range_x=None, env_camera_range_y=None, env_to_coor=None, env_next_coords=None, env_to_grid=None, env_grid_resolution=None,
    env_render=None, env_world_image_filename=None, env_world_array=None,
    make_individual_figures=False, save_individual_figures=False, save_panel_figures=False, make_panels=True, plot_panels=True, make_video=False):

    env_camera_fov = env.camera_fov if env_camera_fov is None else env_camera_fov
    env_camera_range_x = env.camera_range_x if env_camera_range_x is None else env_camera_range_x
    env_camera_range_y = env.camera_range_y if env_camera_range_y is None else env_camera_range_y
    env_to_coor = env.to_coor if env_to_coor is None else env_to_coor
    env_next_coords = env.next_coords if env_next_coords is None else env_next_coords
    env_to_grid = env.to_grid if env_to_grid is None else env_to_grid
    env_grid_resolution = env.grid_resolution if env_grid_resolution is None else env_grid_resolution
    env_render = env.render if env_render is None else env_render
    env_world_image_filename = env.world_image_filename if env_world_image_filename is None else env_world_image_filename
    env_world_array = env.world_array if env_world_array is None else env_world_array

    # for attr in ['camera_fov', 'camera_range_x', 'camera_range_y', 'to_coor', 'next_coords', 'to_grid', 'grid_resolution', 'render', 'world_image_filename']:
    #     exec('env_{} = env.{} if env_{} is None else env_{}'.format(attr, attr, attr, attr, attr))

    dataset, render_mode, target, target_str, object_goal_names, room_goal_names, room_or_object_goal = setup_goal(env_type)
    traversable_colors = get_traversable_colors(dataset)
    goal_color = get_goal_colors(dataset, [target], room_or_object_goal=room_or_object_goal)[target]

    if planner_name == 'dc2g' or planner_name == 'dc2g_without_semantics':
        kwargs = {
            'model_name':           planner_args[planner_name]['model'],
            'traversable_colors':   traversable_colors,
            'goal_color':           goal_color,
            'room_or_object_goal':  room_or_object_goal,
            'camera_fov':           env_camera_fov,
            'camera_range_x':       env_camera_range_x,
            'camera_range_y':       env_camera_range_y,
            'env_to_coor':          env_to_coor,
            'env_next_coords':      env_next_coords,
            'env_to_grid':          env_to_grid,
            'env_grid_resolution':  env_grid_resolution,
            'output_name':          planner_args[planner_name]['output_name'],
            'env_render':           env_render
        }
        planner = DC2GPlanner(**kwargs)
    elif planner_name == 'dc2g_rescale':
        kwargs = {
            'model_name':           planner_args[planner_name]['model'],
            'traversable_colors':   traversable_colors,
            'goal_color':           goal_color,
            'room_or_object_goal':  room_or_object_goal,
            'camera_fov':           env_camera_fov,
            'camera_range_x':       env_camera_range_x,
            'camera_range_y':       env_camera_range_y,
            'env_to_coor':          env_to_coor,
            'env_next_coords':      env_next_coords,
            'env_to_grid':          env_to_grid,
            'env_grid_resolution':  env_grid_resolution,
            'output_name':          planner_args[planner_name]['output_name'],
            'env_render':           env_render
        }
        planner = DC2GRescalePlanner(**kwargs)
    elif planner_name == 'frontier':
        kwargs = {
            'traversable_colors':   traversable_colors,
            'goal_color':           goal_color,
            'room_or_object_goal':  room_or_object_goal,
            'camera_fov':           env_camera_fov,
            'camera_range_x':       env_camera_range_x,
            'camera_range_y':       env_camera_range_y,
            'env_to_coor':          env_to_coor,
            'env_next_coords':      env_next_coords,
            'env_to_grid':          env_to_grid,
            'env_grid_resolution':  env_grid_resolution,
            'env_render':           env_render
        }
        planner = FrontierPlanner(**kwargs)
    elif planner_name == 'oracle':
        kwargs = {
            'traversable_colors':   traversable_colors,
            'goal_color':           goal_color,
            'room_or_object_goal':  room_or_object_goal,
            'env_to_coor':          env_to_coor,
            'env_next_coords':      env_next_coords,
            'env_to_grid':          env_to_grid,
            'env_grid_resolution':  env_grid_resolution,
            'env_world_array':      env_world_array,
            'world_image_filename': env_world_image_filename,
            'env_render':           env_render
        }
        planner = OraclePlanner(**kwargs)
    else:
        print("That planner wasn't implemented yet.")
        raise NotImplementedError

    planner.setup_plots(make_individual_figures, make_panels, plot_panels, save_panel_figures, save_individual_figures, make_video)

    return planner

def setup_goal(env_type):
    if env_type == "MiniGrid" or "AirSim" or "Jackal":
        dataset = "driveways_bing_iros19"
        render_mode = "rgb_array"
        target = "front_door"
        target_str = ""
        object_goal_names = ["front_door"]
        room_goal_names = []
        room_or_object_goal = "object"
    elif env_type == "House3D":
        dataset = "house3d"
        render_mode = "rgb"
        target = env.info['target_room']
        target_str = "-{target}".format(target=target)
        object_goal_names = ["desk", "television", "table", "household_appliance", "sofa"]
        room_goal_names = ["bedroom", "dining_room", "kitchen", "office"]
        if target in room_goal_names:
            room_or_object_goal = "room"
        elif target in object_goal_names:
            room_or_object_goal = "object"
        else:
            print("--- Error: goal type ({}) is invalid!! ---".format(target))
    return dataset, render_mode, target, target_str, object_goal_names, room_goal_names, room_or_object_goal