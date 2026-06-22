import time
import math
import random
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
N_LIDAR        = 50
RANGE_MAX      = 5.0
N_ACTIONS      = 11
LINEAR_VEL     = 0.2
ANGULAR_VELS   = [-1.5 + 0.3 * i for i in range(11)]
COLLISION_DIST = 0.20  

GOAL_X      = 5.00
GOAL_Y      = 4.50
GOAL_RADIUS = 0.6

SPAWN_POSITIONS = [
    (2.20, -0.09, 0.00),
    (-1.80, 4.00, 0.00),
    (-3.30, 1.65, 2.90),
]
SPAWN_Z = 0.5

WORLD_NAME = 'test_map_2'


# ---------------------------------------------------------------------------
# Nodo ROS2
# ---------------------------------------------------------------------------
class _RosNode(Node):
    def __init__(self):
        super().__init__('storm_dqn')

        self.pub_vel = self.create_publisher(Twist, '/model/storm/cmd_vel', 10)

        self._scan      = None
        self._scan_lock = threading.Lock()
        self._scan_seq  = 0
        self.create_subscription(LaserScan, '/model/storm/scan', self._cb_scan, 10)

        self._odom      = None
        self._odom_lock = threading.Lock()
        self.create_subscription(Odometry, '/model/storm/odometry', self._cb_odom, 10)

        self.cli_pose = self.create_client(
            SetEntityPose, f'/world/{WORLD_NAME}/set_pose')

    def _cb_scan(self, msg):
        with self._scan_lock:
            self._scan = msg
            self._scan_seq += 1

    def _cb_odom(self, msg):
        with self._odom_lock:
            self._odom = msg

    def get_scan(self):
        with self._scan_lock:
            return self._scan

    def get_scan_seq(self):
        with self._scan_lock:
            return self._scan_seq

    def get_odom(self):
        with self._odom_lock:
            return self._odom

    def pub_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self.pub_vel.publish(msg)

    def set_pose(self, x, y, z, qx, qy, qz, qw):
        req = SetEntityPose.Request()
        req.entity.name        = 'storm'
        req.entity.type        = 1
        req.pose.position.x    = float(x)
        req.pose.position.y    = float(y)
        req.pose.position.z    = float(z)
        req.pose.orientation.x = float(qx)
        req.pose.orientation.y = float(qy)
        req.pose.orientation.z = float(qz)
        req.pose.orientation.w = float(qw)

        for attempt in range(5):
            if not self.cli_pose.wait_for_service(timeout_sec=2.0):
                time.sleep(0.5 * (attempt + 1))
                continue
            future   = self.cli_pose.call_async(req)
            deadline = time.time() + 5.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.05)
            if future.done():
                return
            time.sleep(0.5 * (attempt + 1))


# ---------------------------------------------------------------------------
# StormEnv
# ---------------------------------------------------------------------------
class StormEnv(gym.Env):
    def __init__(self, max_steps=1000):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N_LIDAR + 2,), dtype=np.float32)

        self.action_space = spaces.Discrete(N_ACTIONS)

        self.max_steps      = max_steps
        self.step_count     = 0
        self.dist_prec_goal = 0.0
        self.spawn_x        = 0.0
        self.spawn_y        = 0.0
        self.odom_x0        = 0.0
        self.odom_y0        = 0.0

        if not rclpy.ok():
            rclpy.init()
        self._node = _RosNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        print('StormEnv: in attesa dei sensori...')
        t0 = time.time()
        while self._node.get_scan() is None or self._node.get_odom() is None:
            time.sleep(0.1)
            if time.time() - t0 > 10.0:
                raise RuntimeError('Timeout sensori. Gazebo avviato?')
        print('StormEnv: connesso a ROS2.')

    def step(self, action):
        self._send_action(action)
        time.sleep(0.05)

        obs          = self._get_obs()
        lidar_data   = obs[:-2]
        dist_attuale = obs[-2]
        theta_goal   = obs[-1]

        if self._check_collision(lidar_data):
            print(f'  [COLLISION] min_lidar={np.min(lidar_data):.3f}')
            reward     = -1000.0
            terminated = True
            self._stop()

        elif dist_attuale < GOAL_RADIUS:
            reward     = +1000.0
            terminated = True
            self._stop()

        else:
            # progress reward dominante
            r_avvicinamento = (self.dist_prec_goal - dist_attuale) * 20.0
            # allineamento più forte per guidare la direzione
            r_allineamento  = math.cos(theta_goal) * 1.0
            # step cost ridotto per non penalizzare troppo
            reward          = -0.1 + r_avvicinamento + r_allineamento
            terminated      = False

        self.dist_prec_goal  = dist_attuale
        self.step_count     += 1
        truncated = (self.step_count >= self.max_steps)

        if truncated:
            self._stop()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Ferma movimenti pregressi
        self._stop()

        # 2. Teletrasporto — spawn casuale tra tutti e 3
        self._reset_pose()
        self.step_count = 0

        # 3. Ferma di nuovo per uccidere inerzia
        self._stop()

        # 4. Pausa stabilizzazione fisica e odometria
        time.sleep(0.5)

        # 5. Aspetta scan fresco
        seq_before = self._node.get_scan_seq()
        t0 = time.time()
        while self._node.get_scan_seq() == seq_before:
            time.sleep(0.05)
            if time.time() - t0 > 3.0:
                break

        # 6. Salva odometria come riferimento zero per questo episodio
        odom = self._node.get_odom()
        self.odom_x0 = odom.pose.pose.position.x if odom else 0.0
        self.odom_y0 = odom.pose.pose.position.y if odom else 0.0

        obs = self._get_obs()
        self.dist_prec_goal = obs[-2]

        print(f'  [RESET] spawn=({self.spawn_x:.2f},{self.spawn_y:.2f}) '
              f'dist_goal={obs[-2]:.2f}')

        return obs, {}

    def close(self):
        self._stop()
        self._node.destroy_node()

    def _get_goal_data(self):
        odom = self._node.get_odom()
        if odom is None:
            return 10.0, 0.0

        dx = odom.pose.pose.position.x - self.odom_x0
        dy = odom.pose.pose.position.y - self.odom_y0

        x_robot = self.spawn_x + dx
        y_robot = self.spawn_y + dy

        q = odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw_robot = math.atan2(siny_cosp, cosy_cosp)

        dist         = math.sqrt((GOAL_X - x_robot)**2 + (GOAL_Y - y_robot)**2)
        target_angle = math.atan2(GOAL_Y - y_robot, GOAL_X - x_robot)
        theta_goal   = math.atan2(
            math.sin(target_angle - yaw_robot),
            math.cos(target_angle - yaw_robot))

        return dist, theta_goal

    def _get_obs(self):
        msg = self._node.get_scan()
        if msg is None:
            lidar_raw = np.full(N_LIDAR, RANGE_MAX, dtype=np.float32)
        else:
            raw = np.array(msg.ranges, dtype=np.float32)
            raw[np.isinf(raw) | np.isnan(raw)] = RANGE_MAX
            raw = np.clip(raw, 0.0, RANGE_MAX)
            idx = np.round(np.linspace(0, len(raw) - 1, N_LIDAR)).astype(int)
            lidar_raw = raw[idx]

        dist, theta = self._get_goal_data()
        return np.concatenate((lidar_raw, [dist, theta]), dtype=np.float32)

    def _check_collision(self, lidar_data):
        return bool(np.any(lidar_data < COLLISION_DIST))

    def _send_action(self, action_idx):
        self._node.pub_cmd(LINEAR_VEL, ANGULAR_VELS[int(action_idx)])

    def _stop(self):
        self._node.pub_cmd(0.0, 0.0)

    def _reset_pose(self):
        # Spawn casuale
        base_x, base_y, base_yaw = random.choice(SPAWN_POSITIONS)

        # Randomizzazione ulteriore per evitare spawn sempre identici, che potrebbero
       # X e Y molto lievi (giusto per non avere pixel identici)
        x   = base_x   + random.uniform(-0.05, 0.05)
        y   = base_y   + random.uniform(-0.05, 0.05)

        # YAW (Rotazione): +/- 0.20 radianti
        yaw = base_yaw + random.uniform(-0.20, 0.20)

        self.spawn_x = x
        self.spawn_y = y

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, qz, qw)
        time.sleep(0.3)