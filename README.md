
# STORM — Collision Avoidance with Deep Reinforcement Learning

A map-free collision-avoidance policy for a mobile robot, trained with a
**Double Deep Q-Network (DDQN)** in **Ignition Gazebo** under **ROS 2**.
This repository replicates the obstacle-avoidance method of Feng *et al.* (2021)
and validates it on multiple simulated maps.

> Course of Robotics for Mechatronic Engineering 2025/2026 — Team 28
> Michele Gallizzi, Riccardo Lo Nigro, Claudia Mangano, Pierluigi Saccone, Roberto Vigna Cit

---

## Overview

The agent learns to drive a skid-steer robot through unknown environments using
**only LiDAR readings** — no map, no costmap, no global planner. The robot
observes 50 range measurements sampled from a 270° scan and selects one of 11
discrete steering actions. Training is fully reactive and rewards the agent for
staying alive while heavily penalizing collisions.

- **State:** 50 LiDAR ranges, clipped to `[0, 5] m`
- **Actions:** 11 discrete `(v, ω)` pairs at fixed `v = 0.2 m/s`, `ω ∈ [-0.8, 0.8] rad/s`
- **Reward:** `+5` per safe step, `-1000` on collision; episode ends on collision or after 1000 steps
- **Network:** `50 → Dense(300, ReLU) → Dense(300, ReLU) → 11` (linear Q-values)
- **Algorithm:** DDQN with experience replay and a periodically updated target network

---

## Repository structure

```
.
├── README.md
├── .gitignore
├── storm_robot_collision_avoidance/     # ROS 2 robot description (URDF/Xacro, worlds, launch)
│   └── ...                              # e.g. sim_launch.py, robot model, Gazebo worlds
└── storm_env_collision_avoidance/       # DRL environment, training & evaluation
    ├── storm_env.py                     # Gymnasium environment (ROS 2 ↔ Gazebo bridge)
    ├── train_dqn.py                     # DDQN training loop + TensorBoard logging
    ├── test_dqn.py                      # Greedy evaluation (collisions over a 5-min run)
    ├── load_storm_ddqn.py               # Loads Keras-3 .h5 weights into a Keras-2/3 model
    ├── random_spawn_generator.py        # Generates valid spawn points from world geometry
    └── semi_corridoio.npy               # Precomputed valid spawn points
```

> **Note on ROS 2 package names.** A ROS 2 package is identified by its
> `package.xml`, not by the folder name. If you rename the description folder,
> keep the `<name>` field in `package.xml` (and the `ros2 launch` command) consistent.

---

## Requirements

- **Python 3.10**
- **ROS 2 Humble**
- **Ignition / Gazebo** with the `ros_gz` bridge (`ros_gz_interfaces`)
- Python packages: `tensorflow`, `numpy`, `gymnasium`, `h5py`, `matplotlib` (optional, for spawn plots)

ROS 2 message/service types used: `geometry_msgs/Twist`, `sensor_msgs/LaserScan`,
`tf2_msgs/TFMessage`, `ros_gz_interfaces/srv/SetEntityPose`.

### Setup

```bash
# 1. Source ROS 2
source /opt/ros/humble/setup.bash

# 2. Create and activate a Python virtual environment
python3 -m venv ~/storm_env
source ~/storm_env/bin/activate

# 3. Install Python dependencies
pip install tensorflow numpy gymnasium h5py matplotlib
```

---

## Usage

### 1. Train

```bash
# Terminal 1 — launch Gazebo + robot
ros2 launch storm_robot_collision_avoidance sim_launch.py

# Terminal 2 — train
source ~/storm_env/bin/activate
cd storm_env_collision_avoidance
python3 train_dqn.py
```

Checkpoints and the best policy are saved under `storm_agents/`
(`storm_ddqn_best.weights.h5`, periodic checkpoints, and a final model).

### 2. Monitor (TensorBoard)

```bash
source ~/storm_env/bin/activate
tensorboard --logdir logs
# open http://localhost:6006
```

Logged scalars: `reward/episode`, `reward/avg50`, `train/loss`,
`train/epsilon`, `train/total_steps`.

### 3. Evaluate

```bash
# Terminal 1 — launch the simulator
ros2 launch storm_robot_collision_avoidance sim_launch.py

# Terminal 2 — greedy test (5-minute run, collisions as metric)
source ~/storm_env/bin/activate
cd storm_env_collision_avoidance
python3 test_dqn.py
```

> **Before running the test:** weight loading in `test_dqn.py` is commented out by
> default. Set `WEIGHTS_PATH` and uncomment **one** loading line — use
> `load_keras3_weights_into(model, WEIGHTS_PATH)` if the `.h5` was saved with
> Keras 3, or `model.load_weights(WEIGHTS_PATH)` otherwise (see *Loading weights* below).

---

## Configuration

Key hyperparameters (defined at the top of `train_dqn.py`):

| Parameter | Value | Description |
|---|---|---|
| `MAX_EPISODES` | 3000 | Training episodes |
| `MAX_STEPS_PER_EPISODE` | 1000 | Max steps per episode |
| `BUFFER_SIZE` | 50000 | Replay buffer capacity |
| `BATCH_SIZE` | 64 | Minibatch size |
| `GAMMA` | 0.99 | Discount factor |
| `LR` | 0.0005 | Adam learning rate |
| `TARGET_UPDATE_FREQ` | 200 | Target-network hard-update interval (steps) |
| `BETA` | 0.997 | ε-decay rate per episode (`ε_{k+1} = β·ε_k`) |
| `EPSILON_START → EPSILON_MIN` | 1.0 → 0.05 | Exploration schedule |

> **Decay-rate comparison.** The target-update frequency is **not** fixed by the
> original paper and is treated here as a free design choice. A slower update
> (`β = 0.995`, every 2000 steps) gives a markedly more stable learning process
> and fewer collisions than the faster schedules (`β = 0.997` / `0.999`, every
> 200 steps).

### Choosing the map

`storm_env.py` switches map by selecting the matching `set_pose` service and
spawn points (`/world/<map>/set_pose`). Comment/uncomment the relevant block to
target `test_map_1`, `test_map_2`, `test_map_3`, or `training_map`.

---

## Notes

**Spawn generation.** `random_spawn_generator.py` expands a set of known-good
anchor points into many valid spawns by sampling discs around them and rejecting
points too close to walls or cylinders (geometry parsed from the `.world` file).
It writes `semi_corridoio.npy` and an optional verification plot.

**Loading weights (Keras 2 vs 3).** Weights saved with Keras 3 are not directly
loadable by Keras 2 (`Layer count mismatch ... found 0 saved layers`).
`load_storm_ddqn.py` works around this by reading the `.h5` with `h5py` and
assigning weights layer-by-layer with `set_weights`, so the same `.h5` works with
any Keras version.

---

## Reference

[1] S. Feng, B. Sebastian, and P. Ben-Tzvi,
*"A Collision Avoidance Method Based on Deep Reinforcement Learning,"*
Robotics, vol. 10, no. 2, p. 73, 2021.

## Acknowledgment of AI assistance

Claude (Sonnet 4.6) supported concept clarification (DRL, network architecture,
hyperparameter choices) and code writing/debugging (URDF/Xacro description, DDQN
training loop, Gazebo/ROS 2 wrappers, odometry handling). All design choices,
analysis, and final content remain the responsibility of the authors.
