from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import glob
import scipy.ndimage.morphology
from dc2g.util import find_goal_inds, wrap, get_colormap_dict
from PIL import Image
import numpy as np

dir_path, _ = os.path.split(os.path.dirname(os.path.realpath(__file__)))

class DrivewayGrid(Grid):
    def __init__(self, width, height, remember_seen_cells, use_semantic_coloring):
        Grid.__init__(self, width, height)
        self.remember_seen_cells = remember_seen_cells
        self.use_semantic_coloring = use_semantic_coloring

class DrivewayActions(IntEnum):
    # These are diff from the MiniGridEnv. TODO: Make them consistent
    forward = 0
    left = 1
    right = 2

class DrivewayEnv(MiniGridEnv):
    # Enumeration of possible actions

    """
    Empty grid environment, no obstacles, sparse reward
    Agent observes image of *entire* world, but only can "see" cells it has
    seen at some point ==> we maintain which cells have been seen.
    """

    def __init__(self, size=8):
        self.finished_init = False
        self.reset_on_init = False
        self.grid_size = size
        self.reset_on_init = False
        self.remember_seen_cells = True
        self.use_semantic_coloring = True
        self.visited_cells = []
        MiniGridEnv.__init__(
            self,
            grid_size=size,
            max_steps=500,
            # max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            # reset_on_init=False,
            # remember_seen_cells=True,
            # use_semantic_coloring=True
        )

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size, 3),
            dtype='uint8'
        )
        position_observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype='uint8')
        theta_observation_space = spaces.Box(low=0, high=3, shape=(1,), dtype='uint8')
        goal_observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype='uint8')
        # self.observation_space = spaces.Tuple([position_observation_space, goal_observation_space])
        # self.observation_space = image_observation_space
        # self.observation_space = spaces.Box(low=0,high=self.grid_size, shape=(6,), dtype='uint8')
        self.observation_space = spaces.Dict({
            'semantic_gridmap': image_observation_space,
            'pos': position_observation_space,
            'theta_ind': theta_observation_space,
            'goal_pos': goal_observation_space,
        })

        # Actions are discrete integer values: only allow forward/left/right/stop for now (no picking up/done)
        self.action_space = spaces.Discrete(3)

        # self.world_id = 30
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_icra19/full_semantic/test/world' + str(self.world_id).zfill(3) + '.png'
        # self.world_image_filename = '/home/mfe/code/dc2g/training_data/driveways_icra19/full_semantic/test/world' + str(self.world_id).zfill(3) + '.png'

        self.set_camera_params()

        # Probably don't need to call self.reset(), since the process running
        # the env will need to capture the 1st obs anyway.
        self.reset()
        self.finished_init = True
        self.actions = DrivewayActions

    def reset(self):
        if not self.reset_on_init and not self.finished_init:
            return
        return super().reset()

    def _rand_choice(self, choices):
        """
        Generate random choice btwn list of options
        """

        index = self.np_random.randint(0, len(choices))
        return choices[index]

    def show_overlay(self):
        # plt.figure(0)
        traj_array = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.uint8)
        for pos in self.visited_cells:
            traj_array[pos[1], pos[0], :] = 255

        img = Image.fromarray(np.uint8(traj_array))
        img = img.resize(self.render_background.size)
        self.render_background.paste(img, (0, 0), img)
        return self.render_background
        # plt.imshow(self.render_background)
        # plt.pause(1)

    def set_difficulty_level(self, difficulty_level):
        dataset = "driveways_bing_iros19"
        image_type = "full_semantic"

        image_filename = "{dir_path}/data/datasets/{dataset}/{image_type}/{mode}/{world_id}{goal}.{extension}"
        worlds = {'training': {'mode': 'train', 'worlds': "world*"},
                  'same_neighborhood': {'mode': 'test', 'worlds': "worldn001*"},
                  'new_neighborhood': {'mode': 'test', 'worlds': "worldn002*"},
                  'urban': {'mode': 'test', 'worlds': "worldn004*"},
                  'test_scenario': {'mode': 'test', 'worlds': 'worldn002m002h011'}}
        if difficulty_level == 'easy':
            if dataset == "driveways_bing_iros19":
                category = 'training'
            # self.world_id = self._rand_int(25, 31)
        elif difficulty_level == 'medium':
            if dataset == "driveways_bing_iros19":
                category = 'same_neighborhood'
            # self.world_id = self._rand_int(31, 34)
        elif difficulty_level == 'hard':
            if dataset == "driveways_bing_iros19":
                category = 'new_neighborhood'
            # self.world_id = self._rand_int(34, 41)
        elif difficulty_level == 'very_hard':
            if dataset == "driveways_bing_iros19":
                category = 'urban'
        elif difficulty_level == 'test_scenario':
            if dataset == "driveways_bing_iros19":
                category = 'test_scenario'

        image_filename_ = image_filename.format(
            dir_path=dir_path,
            dataset=dataset,
            image_type="full_semantic",
            goal="",
            world_id=worlds[category]['worlds'],
            mode=worlds[category]['mode'],
            extension="png")

        world_id_filenames = glob.glob(image_filename_)

        self.world_id = None
        while self.world_id in [None, "worldn001m001h002", "worldn002m001h003"]:
            self.world_image_filename = self._rand_choice(world_id_filenames)
            self.world_id = self.world_image_filename.split('/')[-1].split('.')[0]

        self.difficulty_level = difficulty_level

        self.satellite_img_filename = image_filename.format(
            dir_path=dir_path,
            dataset=dataset,
            image_type="raw",
            goal="",
            world_id=worlds[category]['worlds'],
            mode=worlds[category]['mode'],
            extension="jpg"
            )
        # self.satellite_img = plt.imread(self.satellite_img_filename)

        # self.render_background = Image.open(self.satellite_img_filename)
        # self.render_background = self.render_background.resize([max(self.render_background.size), max(self.render_background.size)])

    def _get_c2g_image(self, width, height):
        # Construct path
        path = self.world_image_filename.split("full_semantic")[0]
        mode = self.world_image_filename.split("/")[-2]  # train, val, test
        filename = self.world_image_filename.split("/")[-1]  # e.g., worldn000m001h006.png
        c2g_path = path + "full_c2g/" + mode + "/" + filename[:-4] + "-front_door.png"

        # Post process image
        image = plt.imread(c2g_path)  # shape: (row, column, 4)
        image = image[:, :, 0:3]  # shape: (row, column, 3)

        return resize(image, (height, width, 3), order=0)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = DrivewayGrid(width, height, remember_seen_cells=self.remember_seen_cells, use_semantic_coloring=self.use_semantic_coloring)

        self.orig_world_array = resize(plt.imread(self.world_image_filename), (height,width,3), order=0)
        assert height == self.orig_world_array.shape[0], "height of loaded world doesn't match gridworld size"
        assert width == self.orig_world_array.shape[1], "width of loaded world doesn't match gridworld size"

        # Get cost2distance image
        self.c2g_image = self._get_c2g_image(width, height)

        # Place a goal square
        dataset = "driveways_bing_iros19"
        goal_color_dict = get_colormap_dict(dataset)

        gym_terrain_class_name_dict = {
            # labels_colors name: regex for airsim object
            "road": Road,
            "driveway": Driveway,
            "path": Path,
            "sidewalk": Sidewalk,
            "front_door": FrontDoor,
            "house": House,
            "grass": Grass,
            "car": Car,
            # "__ignore__": "(?:fence|garden_chair|street_sign|stop_sign)[\w]*",
        }

        # # Initialize with an blank class
        # for row in range(height):
        #     for col in range(width):
        #         self.grid.set(row, col, Blank(np.array([255, 255, 255])))

        self.world_array = self.orig_world_array.copy()
        for key in gym_terrain_class_name_dict:
            try:
                color = goal_color_dict[key]
            except:
                print("{key}'s color isn't defined in the labels_colors.csv file for this dataset.".format(key=key))
                continue
            obj_name = gym_terrain_class_name_dict[key]
            goal_array, inds, goal_inds_arr = find_goal_inds(self.world_array, color, "object")
            for i in range(len(inds[0])):
                self.grid.set(inds[1][i], inds[0][i], obj_name(255.*np.array(color)))
            if obj_name == Path:
                struct2 = scipy.ndimage.generate_binary_structure(2, 2)
                goal_array = scipy.ndimage.morphology.binary_dilation(goal_array, struct2).astype(goal_array.dtype)
                self.world_array[goal_array == 1] = color
                inflated_inds = np.where(goal_array == 1)
                for i in range(len(inflated_inds[0])):
                    self.grid.set(inflated_inds[1][i], inflated_inds[0][i], obj_name(255.*np.array(color)))
        
        key = "front_door"
        color = goal_color_dict[key]
        obj_name = gym_terrain_class_name_dict[key]
        goal_array, inds, goal_inds_arr = find_goal_inds(self.orig_world_array, color, "object")
        self.world_array[goal_array == 1] = color
        # for i in range(len(inds[0])):
        i = 0
        goal_worldobj = obj_name(255.*np.array(color))
        self.grid.set(inds[1][i], inds[0][i], goal_worldobj)
        self.goal_pos = [inds[1][i], inds[0][i]]

        self.goal_worldobj_type = goal_worldobj.type

        # self.start_pos = np.array([inds[1][i]+1, inds[0][i]+1])
        # self.start_dir = 0

        self.place_agent(max_tries=500, reject_fn=reject_untraversable)
        
        # self.place_agent(top=[9, 4], size=[5, 5], max_tries=500, reject_fn=reject_untraversable)
        # print("setting agent at {}, {}".format(self.start_pos, self.start_dir))
        # self.start_pos = (9, 10)
        # self.start_pos = (np.random.randint(7, 10), np.random.randint(5, 10))
        # self.start_pos = (np.random.randint(2,15), np.random.randint(5, 15))
        # self.start_dir = 0
        # self.grid.wall_rect(0, 0, width, height)

        self.mission = "get to the yellow goal square"

    def set_camera_params(self):
        self.camera_fov = np.pi/2 # full FOV in radians
        self.camera_range_meters = 10. # range of sensor horizon in meters
        self.world_size_m = np.array([100., 100.]) # meters of true world
        grid_width = self.grid.width if hasattr(self, "grid") else self.grid_size
        grid_height = self.grid.height if hasattr(self, "grid") else self.grid_size
        self.grid_resolution_array = self.world_size_m / np.array([grid_width, grid_height]) # meters/gridcell
        self.grid_resolution_array = self.world_size_m / np.array([grid_width, grid_height]) # meters/gridcell
        self.camera_range_x = int(self.camera_range_meters / self.grid_resolution_array[0])  # range of sensor horizon in cells
        self.camera_range_y = int(self.camera_range_meters / self.grid_resolution_array[1]) # range of sensor horizon in cells

        self.grid_resolution = 1

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        self.set_camera_params()

        grid_inds = np.indices((self.grid.height,self.grid.width))
        grid_array = np.dstack([grid_inds[1], grid_inds[0]])
        rel_pos = self.agent_pos.astype(np.float32) - grid_array
        ellipse_r = rel_pos**2 / np.array([self.camera_range_x, self.camera_range_y])**2
        r_inds = np.where(np.sum(ellipse_r, axis=2) <= 1)
        angle_offset = wrap(np.arctan2(rel_pos[:,:,1], -rel_pos[:,:,0]) + wrap(np.pi * self.agent_dir / 2.0))
        angle_offset_inds = np.where(abs(angle_offset) < (self.camera_fov/2))
        r_inds_arr = np.dstack([r_inds[1], r_inds[0]])[0]
        angle_offset_inds_arr = np.dstack([angle_offset_inds[1], angle_offset_inds[0]])[0]
        observable_inds_arr = np.array([x for x in set(tuple(x) for x in r_inds_arr) & set(tuple(x) for x in angle_offset_inds_arr)])

        for i in range(observable_inds_arr.shape[0]):
            cell = self.grid.get(observable_inds_arr[i, 0], observable_inds_arr[i, 1])
            if cell:
                cell.has_been_seen = True
        self.grid.get(*self.agent_pos).has_been_seen = True

        return None, None

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        # grid: the small region the agent can currently see
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = np.transpose(self.grid.encode(), axes=(1, 0, 2)) / 255.0

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        # obs = [self.agent_pos[0], self.agent_pos[1], self.agent_dir, self.goal_pos[0], self.goal_pos[1], self.max_steps - self.step_count]
        # obs = image
        obs = {
            'semantic_gridmap': image,
            'pos': self.agent_pos,
            # 'goal_pos': self.goal_pos,
            'theta': wrap(np.pi * self.agent_dir / 2.0),
            'theta_ind': self.agent_dir,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def to_grid(self, x, y):
        return x, y
        # """
        # Convert continuous coordinate to grid location
        # """
        # # print("x,y: ({},{})".format(x,y))
        # # print("self.lower_grid_x_min: {}".format(self.lower_grid_x_min))
        # # print("self.lower_grid_y_min: {}".format(self.lower_grid_y_min))
        # # print("self.grid_resolution: {}".format(self.grid_resolution))
        # gx = np.floor((x - self.lower_grid_x_min) / self.grid_resolution).astype(int)
        # gy = np.floor((y - self.lower_grid_y_min) / self.grid_resolution).astype(int)
        # return gx, gy
        # # print("gx,gy: ({},{})".format(gx,gy))
        # # return int(gx), int(gy)

    def to_coor(self, x, y):
        return x, y
        # """
        # Convert grid location to continuous coordinate
        # """
        # wx = x * self.grid_resolution + self.lower_grid_x_min
        # wy = y * self.grid_resolution + self.lower_grid_y_min
        # return wx, wy

    def rescale(self,x1,y1,x2,y2,n_row=None):
        return x1, y1, x2, y2
        # """
        # convert the continuous rectangle region in the SUNCG dataset to the grid region in the house
        # """
        # gx1, gy1 = self.to_grid(x1, y1)
        # gx2, gy2 = self.to_grid(x2, y2)
        # return gx1, gy1, gx2, gy2
        # # if n_row is None: n_row = self.n_row
        # # tiny = 1e-9
        # # tx1 = np.floor((x1 - self.L_lo) / self.L_det * n_row+tiny)
        # # ty1 = np.floor((y1 - self.L_lo) / self.L_det * n_row+tiny)
        # # tx2 = np.floor((x2 - self.L_lo) / self.L_det * n_row+tiny)
        # # ty2 = np.floor((y2 - self.L_lo) / self.L_det * n_row+tiny)
        # # return int(tx1),int(ty1),int(tx2),int(ty2)

    def next_coords(self, start_x, start_y, start_theta_ind):
        # action_dict = { 0: (1, 0),
        #                 1: (0, np.pi/2),
        #                 2: (0, -np.pi/2)}
        action_dict = { 0: (1, 0),
                        1: (0, -np.pi/2),
                        2: (0, np.pi/2)}
        start_theta = theta_ind_to_theta(start_theta_ind)
        num_actions = len(action_dict.keys())
        state_dim = 3 # (x,y,theta)
        next_states = np.empty((num_actions, state_dim))
        actions = [None for i in range(num_actions)]
        for i in range(num_actions):
            action = list(action_dict)[i]
            actions[i] = action
            cmd_vel = action_dict[action]

            # Compute displacement
            num_steps = 10
            dt = 0.1 * num_steps
            dx = dt * cmd_vel[0]
            dy = 0
            dtheta = dt * cmd_vel[1]

            x = start_x + dx * np.cos(start_theta) - dy * np.sin(start_theta)
            y = start_y + dx * np.sin(start_theta) + dy * np.cos(start_theta)
            theta = start_theta + dtheta

            next_states[i,0] = x
            next_states[i,1] = y
            next_states[i,2] = theta_to_theta_ind(theta)
        return next_states, actions, 1

    def get_reward(self, sparse):
        """Get reward for RL agent. Two options are possible:
        * Sparse: reward = 10 iff goal reached. O/w step cost of -0.01
        * Dense: If traversable, a reward based on distance. 
        If non-traversable, a fixed penalty cost
        """
        if sparse:
            new_cell = self.grid.get(*self.agent_pos)
            if new_cell.type == self.goal_worldobj_type:
                reward = 10
            else:
                reward = -0.01
        else:
            pixel = self.c2g_image[self.agent_pos[1], self.agent_pos[0], :]

            if pixel[0] == 1. and pixel[1] == 0. and pixel[2] == 0.:
                reward = -1.  # Non-traversable road
            else:
                reward = pixel[0] - 1.  # Traversable road

            assert reward <= 0.

        return reward
        
    def step(self, action):
        self.step_count += 1

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Record position of agent prior to step as seen
        current_cell = self.grid.get(*self.agent_pos)
        current_cell.has_been_visited = True
        self.visited_cells.append(self.agent_pos)

        # Take action
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
        else:
            raise ValueError()

        # Get new obs
        obs = self.gen_obs()

        # Get reward
        reward = self.get_reward(sparse=False)

        # Get done
        done = False
        if self.step_count >= self.max_steps:
            done = True

        new_cell = self.grid.get(*self.agent_pos)
        if new_cell.type == self.goal_worldobj_type:
            print("Goal reached!")
            done = True

        return obs, reward, done, {}

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=np.inf,
        reject_fn=None
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries, reject_fn=reject_fn, ok_to_overlap=True)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=np.inf,
        ok_to_overlap=False
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], top[0] + size[0]),
                self._rand_int(top[1], top[1] + size[1])
            ))

            # Don't place the object on top of another object
            if not ok_to_overlap and self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        # self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

# keep angle between [-pi, pi]
def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def theta_to_theta_ind(yaw):
    return int(round(yaw / (np.pi/2))) % 4

def theta_ind_to_theta(theta_ind):
    return wrap(theta_ind * np.pi/2)

def reject_untraversable(env, pos):
    pi, pj = pos
    # return env.grid.get(pi, pj).type not in ['driveway']
    return env.grid.get(pi, pj).type not in ['road']
    # return env.grid.get(pi, pj).type not in ['path']

class DrivewayEnv6x6(DrivewayEnv):
    def __init__(self):
        DrivewayEnv.__init__(self, size=6)

class DrivewayEnv16x16(DrivewayEnv):
    def __init__(self):
        DrivewayEnv.__init__(self, size=16)

class DrivewayEnv32x32(DrivewayEnv):
    def __init__(self):
        DrivewayEnv.__init__(self, size=50)

register(
    id='MiniGrid-DrivewayEnv-6x6-v0',
    entry_point=DrivewayEnv6x6
)

register(
    id='MiniGrid-DrivewayEnv-16x16-v0',
    entry_point=DrivewayEnv16x16
)

register(
    id='MiniGrid-DrivewayEnv-32x32-v0',
    entry_point=DrivewayEnv32x32
)

# Number of cells (width and height) in the agent view
AGENT_VIEW_SIZE = 5

# Size of the array given as an observation to the agent
OBS_ARRAY_SIZE = (AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3)

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'magenta'  : np.array([255, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([255, 255, 255])
}

TRAVERSABLE_COLORS = {
    True: 'white',
    False: 'red'
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'magenta': 3,
    'purple': 4,
    'yellow': 5,
    'grey'  : 6
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty':        0,
    'wall':         1,
    'floor':        2,
    'door':         3,
    'locked_door':  4,
    'key':          5,
    'ball':         6,
    'box':          7,
    'frontdoor':    8,
    'sidewalk':     9,
    'house':        10,
    'road':         11,
    'grass':        12,
    'driveway':     13,
    'path':         14,
    'car':          15,
    'blank':        16
}

# Map of object type to color names
OBJECT_TO_COLOR = {
    'front_door': 'yellow',
    'sidewalk': 'blue',
    'house':    'magenta',
    'road':     'red',
    'grass':    'green'
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

class GenericTerrain(WorldObj):
    """
    Colored grass tile the agent can walk over
    """

    def __init__(self, terrain_type, color, is_traversable=True):
        self.is_traversable = is_traversable
        self.type = terrain_type
        assert terrain_type in OBJECT_TO_IDX, terrain_type
        if type(color) == np.ndarray:
            self.color = color
        elif obj_type in OBJECT_TO_COLOR:
            self.color = OBJECT_TO_COLOR[obj_type]
            assert self.color in COLOR_TO_IDX, self.color
        else:
            self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

        self.has_been_seen = False
        self.has_been_visited = False

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        if type(self.color) == np.ndarray:
            c = self.color
        else:
            c = COLORS[self.color]
        # r.setLineColor(100, 100, 100, 0)
        r.setLineColor(*c/2)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           1),
            (1          ,           1)
        ])

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        if not self.has_been_seen:
            return (0,0,0)
        else:
            if type(self.color) == np.ndarray:
                return self.color
            else:
                return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

class Grass(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=False)

class Driveway(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=True)

class Sidewalk(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=True)

class Road(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=True)

class Car(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=False)

class Blank(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=False)

class Path(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=True)

class House(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=False)

class FrontDoor(GenericTerrain):
    def __init__(self, color=None):
        GenericTerrain.__init__(self,
            self.__class__.__name__.lower(), 
            color=color,
            is_traversable=False)
