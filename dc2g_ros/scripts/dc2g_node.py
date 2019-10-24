#!/usr/bin/env python
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from visualization_msgs.msg import Marker
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf_conversions
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import message_filters
import pickle
import os

from dc2g.planners.util import instantiate_planner

class DC2G:
    def __init__(self, non_ros_mode=False):
        self.non_ros_mode = non_ros_mode

        self.odom = None

        self.bridge = CvBridge()

        self.semantic_gridmap = None
        self.grid_resolution = 0.1
        self.lower_grid_x_min = 0
        self.lower_grid_y_min = 0

        if self.non_ros_mode:
            make_panels = True
            plot_panels = True
        else:
            make_panels = True
            plot_panels = False

        # square_half_side_length = 10 # meters
        # self.map_x_min = self.init_px - square_half_side_length
        # self.map_x_max = self.init_px + square_half_side_length
        # self.map_y_min = self.init_py - square_half_side_length
        # self.map_y_max = self.init_py + square_half_side_length

        self.camera_fov = np.pi/2
        self.camera_range_x = 10; self.camera_range_y = 10;

        self.planner_name = rospy.get_param("~planner_name", "dc2g")
        self.env_type = "Jackal"
        self.planner = instantiate_planner(self.planner_name, None, self.env_type,
            env_camera_fov=self.camera_fov,
            env_camera_range_x=self.camera_range_x,
            env_camera_range_y=self.camera_range_y,
            env_to_coor=self.to_coor,
            env_next_coords=self.next_coords,
            env_to_grid=self.to_grid,
            env_grid_resolution=self.grid_resolution,
            env_render=render,
            env_world_image_filename="tmp.png",
            make_panels=make_panels,
            plot_panels=plot_panels)

        if self.non_ros_mode:
            return

        self.tf_listener = tf.TransformListener()
        self.base_link_frame_id = rospy.get_param("~base_link_frame_id", "/base_link")
        self.robot_linear_speed = rospy.get_param("~robot_linear_speed", 0.2)
        self.robot_angular_speed = rospy.get_param("~robot_angular_speed", 0.3)

        self.sub_pose = rospy.Subscriber("~pose", Odometry, self.cbPose)
        self.pub_cmd_vel = rospy.Publisher("~cmd_vel", Twist, queue_size=1)
        self.pub_map_debug = rospy.Publisher("~map_debug", Image, queue_size=1)
        self.pub_map_debug2 = rospy.Publisher("~map_debug2", Image, queue_size=1)
        self.pub_planned_path_marker = rospy.Publisher("~planned_path_marker", Marker, queue_size=1)

        self.semantic_map_image_sub = message_filters.Subscriber("/octomap_map2d_image", Image,
                                            queue_size=1, buff_size=20*500*500)
        self.semantic_map_info_sub = message_filters.Subscriber("/octomap_map2d_info", Float32MultiArray,
                                            queue_size=1, buff_size=20*100)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.semantic_map_image_sub, self.semantic_map_info_sub], queue_size=5, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.cbSemanticMap)

        self.planner_hz = rospy.get_param("~planner_hz", 10)
        self.planner_period = 1./self.planner_hz
        self.timer = rospy.Timer(rospy.Duration(self.planner_period), self.cbTimer)

    def cbTimer(self, event):
        rospy.loginfo("[cbTimer]")
        if self.semantic_gridmap is None:
            rospy.loginfo("Haven't received a map yet.")
            return
        if self.odom is None:
            rospy.loginfo("Haven't received a pose yet.")
            return
        rospy.loginfo("[cbTimer] continuing...")

        px = self.odom.pose.pose.position.x; py = self.odom.pose.pose.position.y
        gx, gy = self.to_grid(px, py)
        px2, py2 = self.to_coor(gx, gy)

        print("True: ({:.2f}, {:.2f}), Est: ({:.2f}, {:.2f})".format(px, py, px2, py2))

        obs = self.make_obs()
        action = self.planner.plan(obs)

        if hasattr(self.planner, 'path'):
            self.visualizePlannedPath(self.planner.path)

        # if hasattr(self.planner, 'c2g_array'):
        #     map_img = self.bridge.cv2_to_imgmsg(self.planner.c2g_array)
        #     self.pub_map_debug.publish(map_img)
        # if hasattr(self.planner, 'planner_array'):
        #     with open("{dir}/obs_{step_number}.pkl".format(dir=os.path.dirname(os.path.realpath(__file__)), step_number=str(self.planner.step_number)).zfill(3), "wb") as f:
        #         pickle.dump(obs, f)
        #     rgbImage = cv2.cvtColor((self.planner.planner_array*255.).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        #     map_img = self.bridge.cv2_to_imgmsg(rgbImage)
        #     self.pub_map_debug2.publish(map_img)

        # twist_msg = self.actionToTwist(action)
        # self.pubCmdVel(twist_msg)

    def visualizePlannedPath(self, path):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'odom_ekf'
        marker.ns = 'planned_path'
        marker.id = 0
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD
        for coord in path:
            px, py = self.to_coor(coord[0], coord[1])
            marker.points.append(Point(x=px, y=py, z=1.0))
        marker.scale = Vector3(x=0.1,y=0.2,z=0.2)
        marker.color = ColorRGBA(b=1.0,a=1.0)
        marker.lifetime = rospy.Duration(0.5)
        self.pub_planned_path_marker.publish(marker)


    def pubCmdVel(self, twist_msg):
        self.pub_cmd_vel.publish(twist_msg)

    def actionToTwist(self, action):
        twist_msg = Twist()

        if action == 0:
            twist_msg.linear.x = self.robot_linear_speed
        elif action == 1:
            twist_msg.angular.z = self.robot_angular_speed
        elif action == 2:
            twist_msg.angular.z = -self.robot_angular_speed
        else:
            rospy.logwarn("Action: {} not supported. Stopping.".format(action))

        return twist_msg

    # def cbMap(self, msg):
    #     self.map = msg

    #     try:
    #         self.map_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #     except CvBridgeError as e:
    #         rospy.logwarn(e)

    def cbPose(self, msg):
        self.odom = msg
        quat = msg.pose.pose.orientation
        _, _, self.yaw = tf_conversions.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

    def cbSemanticMap(self, image_msg, info_msg):
        '''
         Gridmap is an image with width==>x, height==>y
        
             width
        a--------------b
        |              |  height
        |              |
        c--------------d
        
          ^ +y
          |
          --> +x
        point     cont. coord                                     map index
        a         (self.lower_grid_x_min, self.lower_grid_y_max)  (0,0)
        b         (self.lower_grid_x_max, self.lower_grid_y_max)  (0,width-1)
        c         (self.lower_grid_x_min, self.lower_grid_y_min)  (height,0)
        d         (self.lower_grid_x_max, self.lower_grid_y_min)  (height,width-1)
        
        To convert cont coord (x,y) ==> map index (gx, gy):
        gx = int((x - self.lower_grid_x_min) / grid_resolution)
        
        To convert map index (gx, gy) ==> cont coord (x,y):
        x = (gx * grid_resolution) + self.lower_grid_x_min

        '''
        print("Got semantic map pair.")
        self.lower_grid_x_min = info_msg.data[-3]
        self.lower_grid_y_min = info_msg.data[-2]
        self.grid_resolution = info_msg.data[-1]

        print('====')
        print('====')
        print("xmin, ymin, res: {:.2f}, {:.2f}, {:.2f}".format(self.lower_grid_x_min, self.lower_grid_y_min, self.grid_resolution))
        print('====')
        print('====')

        self.semantic_gridmap = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        self.waiting_on_semantic_gridmap = False

    def make_obs(self):
        print("[make_obs]")
        # (p,q) = self.tf_listener.lookupTransform('/odom_ekf', '/realsense_color_optical_frame', rospy.Time(0))
        # px, py, pz = p
        px = self.odom.pose.pose.position.y; py = self.odom.pose.pose.position.x
        print("px, py: {}, {}.".format(px, py))
        gx, gy = self.to_grid(px, py)
        print("gx, gy: {}, {}.".format(gx, gy))
        q = self.odom.pose.pose.orientation
        q = (q.x, q.y, q.z, q.w)
        _, _, yaw = euler_from_quaternion(q)
        theta_ind = theta_to_theta_ind(yaw)
        print("theta_ind: {}".format(theta_ind))

        if self.semantic_gridmap is not None:
            semantic_gridmap = self.semantic_gridmap

            # semantic_gridmap[gx-1:gx+1, gy-1:gy+1] = [255, 0, 0]
            # map_img = self.bridge.cv2_to_imgmsg(semantic_gridmap)
            # self.pub_map_debug.publish(map_img)

            # semantic_gridmap = np.flip(semantic_gridmap, axis=0)
            # semantic_gridmap[semantic_gridmap == 55] = 128
            # semantic_gridmap[semantic_gridmap < 55] = 0
            semantic_gridmap = semantic_gridmap / 255.
            pos = np.array([gx, gy])
        else:
            semantic_gridmap = None

        print("semantic_gridmap.shape: {}".format(semantic_gridmap.shape))

        obs = {
            'semantic_gridmap': semantic_gridmap,
            'pos': pos,
            'theta_ind': theta_ind,
            'mission': "todo"
        }
        return obs

    def to_grid(self, x, y):
        """
        Convert continuous coordinate to grid location
        """
        gx = np.floor(np.around((x - self.lower_grid_x_min) / self.grid_resolution, 8)).astype(int)
        gy = np.floor(np.around((y - self.lower_grid_y_min) / self.grid_resolution, 8)).astype(int)
        # gy = -(self.semantic_gridmap.shape[1] - np.floor((y - self.lower_grid_y_min) / self.grid_resolution).astype(int))
        # gy = self.semantic_gridmap.shape[1] - np.floor((y - self.lower_grid_y_min) / self.grid_resolution).astype(int)
        return gx, gy

    def to_coor(self, x, y):
        """
        Convert grid location to continuous coordinate
        """
        wx = x * self.grid_resolution + self.lower_grid_x_min
        wy = y * self.grid_resolution + self.lower_grid_y_min
        # wy = (self.semantic_gridmap.shape[1] + y) * self.grid_resolution + self.lower_grid_y_min
        # wy = (self.semantic_gridmap.shape[1] - y) * self.grid_resolution + self.lower_grid_y_min
        return wx, wy

    def rescale(self,x1,y1,x2,y2,n_row=None):
        """
        convert the continuous rectangle region in the SUNCG dataset to the grid region in the house
        """
        gx1, gy1 = self.to_grid(x1, y1)
        gx2, gy2 = self.to_grid(x2, y2)
        return gx1, gy1, gx2, gy2

    def next_coords(self, start_x, start_y, start_theta_ind):
        action_dict = { 0: (1, 0),
                        1: (0, -np.pi/2),
                        2: (0, np.pi/2)}
        start_theta = theta_ind_to_theta(start_theta_ind)
        num_actions = len(action_dict.keys())
        state_dim = 3 # (x,y,theta)
        next_states = np.empty((num_actions, state_dim))
        actions = [None for i in range(num_actions)]
        
        smallest_forward_action = 1.0 # meters
        gridmap_discretization = int(smallest_forward_action/self.grid_resolution)

        for i in range(num_actions):
            action = action_dict.keys()[i]
            actions[i] = action
            cmd_vel = action_dict[action]

            dx = smallest_forward_action * cmd_vel[0]
            dy = 0
            dtheta = 1 * cmd_vel[1]
            # Compute displacement
            # num_steps = 5
            # dt = 0.1 * num_steps
            # dx = dt * cmd_vel[0]
            # dy = 0
            # dtheta = dt * cmd_vel[1]

            x = start_x + dx * np.cos(start_theta) - dy * np.sin(start_theta)
            y = start_y + dx * np.sin(start_theta) + dy * np.cos(start_theta)
            theta = start_theta + dtheta

            next_states[i,0] = x
            next_states[i,1] = y
            next_states[i,2] = theta_to_theta_ind(theta)

        return next_states, actions, gridmap_discretization

# keep angle between [-pi, pi]
def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def theta_to_theta_ind(yaw):
    return int(round(yaw / (np.pi/2))) % 4

def theta_ind_to_theta(theta_ind):
    return wrap(theta_ind * np.pi/2)

def render():
    return

if __name__ == "__main__":
    rospy.init_node("dc2g_node", anonymous=True)
    n = DC2G()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")