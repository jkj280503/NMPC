# linorobot2 Documentation Overview

## Introduction

Welcome to the linorobot2 documentation. This set of guides walks you through everything you need to go from a bare robot with wheels and motors to a fully autonomous robot that can build maps of its environment and navigate them on its own.

These docs are structured as a tutorial sequence, with each section building on the previous one, explaining not just *how* to configure something but *why* it exists and what role it plays in the larger system. The approach is inspired by the [Nav2 setup guides](https://docs.nav2.org/setup_guides/index.html) and the original [linorobot](https://github.com/linorobot/linorobot) ROS1 wiki.

linorobot2 supports three robot base configurations (2WD, 4WD, Mecanum drive) and integrates with ROS2 Jazzy, Nav2, SLAM Toolbox, and robot_localization out of the box.

## Architecture

The diagram below shows the launch file structure and how the different components connect:

![linorobot2 architecture](assets/linorobot2_launchfiles.png)

## The Journey: Bare Robot → Autonomous Robot

Here's the full path from hardware to autonomous navigation, in order:

---

### 1. [Installation](installation/)

Get linorobot2 installed on both your robot computer and your host/development machine. Covers the one-command `install.bash` setup script, the supported robot types and sensors, points to documentation on how to install the on-robot firmware for motor control, and how to install the RViz visualization package for remote debugging. Also covers the Gazebo simulation setup and the Docker option for host machines not running Ubuntu 24.04.

---

### 2. [Base Controller](base_controller/)

The base controller is the bridge between ROS2 and your robot's motors. A microcontroller (e.g., Teensy) running micro-ROS handles the low-level work: receiving `/cmd_vel` velocity commands, driving the motor controllers, reading wheel encoders, and publishing raw odometry and IMU data. This section explains the hardware architecture, the published/subscribed topics, and how to bring up the microcontroller agent.

---

### 3. [Odometry](odometry/)

Odometry is the robot's estimate of its position based on how far its wheels have turned. Wheel odometry alone drifts over time, as slip, uneven terrain, and mechanical tolerances all accumulate errors. This section explains how the `robot_localization` package uses an Extended Kalman Filter (EKF) to fuse wheel odometry with IMU data, producing a much more reliable filtered odometry on `/odometry/filtered`. It also covers how to verify your odometry before running SLAM.

---

### 4. [Setting Up Sensors](sensors/)

Sensors are what let the robot perceive its environment. linorobot2 uses 2D lidar (LaserScan) as the primary sensor for mapping and obstacle detection, with optional depth cameras (RGBD) for detecting obstacles outside the lidar's scan plane. This section covers the supported real hardware sensors and how to configure them via the install script, as well as how simulation sensors are defined in URDF/xacro files and how to adjust their positions on the robot.

---

### 5. [Setting Up Transforms (TF)](transforms/)

TF2 is the ROS2 system that tracks the spatial relationship between every coordinate frame on the robot, from the global map frame down to each individual sensor. This section explains the full frame hierarchy (`map → odom → base_footprint → base_link → sensors`), what publishes each transform, and how to configure sensor positions in the URDF properties files. Includes verification steps using `view_frames` and RViz.

---

### 6. [Mapping](mapping/)

Before autonomous navigation is possible, the robot needs a map. This section covers SLAM (Simultaneous Localization and Mapping) using SLAM Toolbox, including how it works conceptually (scan matching and loop closure), the key configuration parameters in `slam.yaml`, and the step-by-step process for driving the robot through an environment, building the map, and saving it to disk.

---

### 7. [Navigation](navigation/)

With a map in hand, Nav2 can navigate the robot autonomously. This section gives an overview of Nav2's components (AMCL, global planner, local controller, costmaps, behavior tree), walks through launching navigation with a pre-built map, and explains how to set the robot's initial pose and send navigation goals in RViz. Importantly, it covers the **robot footprint**, the one parameter you must configure to match your robot's actual size.

---

### 8. [Docker](docker/)

Covers running linorobot2 without a native ROS2 installation using Docker. Includes configuring the Docker environment for hardware, simulation, or CUDA-accelerated simulation; building the image; web-based visualization via KasmVNC (accessible in any browser at `http://<host_ip>:3000`); running the Physical Robot or Simulated Robot via Tmuxinator profiles; and creating custom profiles for your own workflow.

---

### 9. [Simulation Tools](tools/)

Documents the two Gazebo world generation tools in `linorobot2_gazebo`. `image_to_gazebo` is a GUI tool that converts any floor plan image into a Gazebo world by calibrating its scale and origin interactively. `create_worlds_from_maps` is a batch CLI tool that converts all saved SLAM maps into Gazebo worlds in one command. Both let you simulate in the exact same environment your robot operates in physically.

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `install.bash` | One-command robot computer setup |
| `linorobot2_base/config/ekf.yaml` | EKF sensor fusion configuration |
| `linorobot2_description/urdf/<robot>_properties.urdf.xacro` | Robot dimensions and sensor poses |
| `linorobot2_navigation/config/slam.yaml` | SLAM Toolbox configuration |
| `linorobot2_navigation/config/navigation.yaml` | Nav2 configuration (including footprint) |
| `linorobot2_navigation/maps/` | Where to save maps |

## Quick Reference: Key Launch Commands

| Command | Purpose |
|---------|---------|
| `ros2 launch linorobot2_bringup bringup.launch.py` | Boot the Physical Robot |
| `ros2 launch linorobot2_gazebo gazebo.launch.py` | Start Simulated Robot |
| `ros2 launch linorobot2_navigation slam.launch.py` | Start SLAM (mapping) |
| `ros2 launch linorobot2_navigation navigation.launch.py map:=<map>.yaml` | Start Nav2 (navigation) |
| `ros2 launch linorobot2_viz slam.launch.py` | Visualize SLAM from host machine |
| `ros2 launch linorobot2_viz navigation.launch.py` | Visualize navigation from host machine |

## External Resources

- [Nav2 Setup Guides](https://docs.nav2.org/setup_guides/index.html)
- [Nav2 Configuration Guide](https://docs.nav2.org/configuration/index.html)
- [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox)
- [robot_localization](https://docs.ros.org/en/ros2_packages/rolling/api/robot_localization/index.html)
- [linorobot2_hardware](https://github.com/linorobot/linorobot2_hardware): microcontroller firmware and hardware build guide
- [Gazebo ROS2 Overview](https://gazebosim.org/docs/latest/ros2_overview)
