# Installation

This guide walks you through installing linorobot2, whether you're setting up a Physical Robot or running the Gazebo simulation on your desktop.

## Physical Robot

### Prerequisites

The Robot Computer needs ROS2 Jazzy. If you haven't installed it yet, use the [ros2me](https://github.com/linorobot/ros2me) installer script, which has been tested on x86 and ARM boards including Raspberry Pi 4 and Nvidia Jetson series.

### Hardware and Robot Firmware

Before diving into software, you'll need to build the robot and flash the microcontroller firmware. All hardware documentation (schematics, wiring, firmware) is at [linorobot2_hardware](https://github.com/linorobot/linorobot2_hardware).

### Robot Computer Installation

The easiest way to get everything installed on the robot computer is to run the provided bash script from the root of this repository. It installs all dependencies, sets the required environment variables, and creates a `linorobot2_ws` in your `$HOME` directory.

```bash
source /opt/ros/<ros_distro>/setup.bash
cd /tmp
wget https://raw.githubusercontent.com/linorobot/linorobot2/${ROS_DISTRO}/install.bash
bash install.bash --base <robot_type> [--laser <laser_sensor>] [--depth <depth_sensor>] [--workspace <path>]
source ~/.bashrc
```

Passing `--base` runs the full installation: workspace setup, sensor drivers, micro-ROS, and the linorobot2 package, and exports the required env variables to `~/.bashrc`. Omitting `--base` installs only the specified sensor drivers, which is useful when integrating into an existing workspace.

**robot_type:** choose the base that matches your hardware:

| Value | Description |
|-------|-------------|
| `2wd` | 2 wheel drive robot |
| `4wd` | 4 wheel drive robot |
| `mecanum` | Mecanum drive robot |

**laser_sensor:** choose your 2D lidar or depth-camera-as-lidar:

| Value | Sensor |
|-------|--------|
| `a1` | [RPLIDAR A1](https://www.slamtec.com/en/Lidar/A1) |
| `a2` | [RPLIDAR A2](https://www.slamtec.ai/product/slamtec-rplidar-a2/) |
| `a3` | [RPLIDAR A3](https://www.slamtec.ai/product/slamtec-rplidar-a3/) |
| `s1` | [RPLIDAR S1](https://www.slamtec.com/en/Lidar/S1) |
| `s2` | [RPLIDAR S2](https://www.slamtec.com/en/Lidar/S2) |
| `s3` | [RPLIDAR S3](https://www.slamtec.com/en/Lidar/S3) |
| `c1` | [RPLIDAR C1](https://www.slamtec.ai/product/slamtec-rplidar-a3/) |
| `ld06` | [LD06 LIDAR](https://www.ldrobot.com/ProductDetails?sensor_name=STL-06P) |
| `ld19` | [LD19/LD300 LIDAR](https://www.ldrobot.com/ProductDetails?sensor_name=STL-19P) |
| `stl27l` | [STL27L LIDAR](https://www.ldrobot.com/ProductDetails?sensor_name=STL-27L) |
| `ydlidar` | [YDLIDAR](https://www.ydlidar.com/lidars.html) |
| `xv11` | [XV11](http://xv11hacking.rohbotics.com/mainSpace/home.html) |
| `realsense` * | [Intel RealSense](https://www.intelrealsense.com/stereo-depth/) D435, D435i |
| `zed` * | [Zed](https://www.stereolabs.com/zed) |
| `zed2` * | [Zed 2](https://www.stereolabs.com/zed-2) |
| `zed2i` * | [Zed 2i](https://www.stereolabs.com/zed-2i) |
| `zedm` * | [Zed Mini](https://www.stereolabs.com/zed-mini) |

Sensors marked with `*` are depth cameras. When used as a laser sensor, the launch files will run [depthimage_to_laserscan](https://github.com/ros-perception/depthimage_to_laserscan) to convert the depth image to a laser scan. Omit `--laser` if you have no laser sensor.

**depth_sensor:** choose your RGBD camera (optional, separate from the laser sensor):

| Value | Sensor |
|-------|--------|
| `realsense` | [Intel RealSense](https://www.intelrealsense.com/stereo-depth/) D435, D435i |
| `zed` | [Zed](https://www.stereolabs.com/zed) |
| `zed2` | [Zed 2](https://www.stereolabs.com/zed-2) |
| `zed2i` | [Zed 2i](https://www.stereolabs.com/zed-2i) |
| `zedm` | [Zed Mini](https://www.stereolabs.com/zed-mini) |
| `oakd` | [OAK-D](https://shop.luxonis.com/collections/oak-cameras-1/products/oak-d) |
| `oakdlite` | [OAK-D Lite](https://shop.luxonis.com/collections/oak-cameras-1/products/oak-d-lite-1) |
| `oakdpro` | [OAK-D Pro](https://shop.luxonis.com/collections/oak-cameras-1/products/oak-d-pro) |

### Host Machine: RVIZ (Remote Visualization)

When working with a Physical Robot, you'll want to visualize what's happening from a separate machine (the laser scan, the map being built, and the robot's pose) without running a full simulation. Install [linorobot2_viz](https://github.com/linorobot/linorobot2_viz) on your host machine for this purpose:

```bash
cd <host_machine_ws>
git clone https://github.com/linorobot/linorobot2_viz src/linorobot2_viz
rosdep update && rosdep install --from-path src --ignore-src -y
colcon build
source install/setup.bash
```

This package is kept separate from the main installation to keep the Robot Computer lean. Both machines must be on the same ROS2 network (same `ROS_DOMAIN_ID`).

---

## Simulated Robot

The simulation runs on your host/development machine and does not require a Physical Robot at all. It's a great way to tune Nav2 parameters and test your configuration before deploying to hardware.

### 2.1 Install linorobot2 Package

```bash
cd <host_machine_ws>
git clone -b $ROS_DISTRO https://github.com/linorobot/linorobot2 src/linorobot2
rosdep update && rosdep install --from-path src --ignore-src -y --skip-keys microxrcedds_agent --skip-keys micro_ros_agent
colcon build
source install/setup.bash
```

> The `--skip-keys` flags prevent a known [rosdep issue](https://github.com/micro-ROS/micro_ros_setup/issues/138) with micro-ROS keys. Always include them when running `rosdep install` on this workspace.

### 2.2 Define Robot Type

Set the `LINOROBOT2_BASE` environment variable to match the robot base you want to simulate:

```bash
echo "export LINOROBOT2_BASE=2wd" >> ~/.bashrc
source ~/.bashrc
```

Available values: `2wd`, `4wd`, `mecanum`.

The simulation package already includes RVIZ configurations, so you do not need to install `linorobot2_viz` separately when using the simulation.

### Docker Option

If you're not running Ubuntu 24.04 with ROS2 Jazzy, Docker is a convenient alternative for running the simulation.

Install Docker using the [official instructions](https://docs.docker.com/engine/install/) and follow the [post-install steps](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker without `sudo`.

Customize `docker/.env` if needed (e.g., to change the robot type), then build the image:

```bash
git clone https://github.com/linorobot/linorobot2.git
cd linorobot2/docker
docker compose build
```

To start the simulation in Docker:

```bash
# Start Gazebo
docker compose up gazebo

# In another terminal, start navigation
export SIM=true
docker compose up navigate

# In another terminal, open RViz
docker compose up rviz-nav
```

If you see "Unable to create rendering window" errors with Gazebo, run `xhost +` first.

You can also run everything in daemon mode:

```bash
export SIM=true
export RVIZ=true
docker compose up -d gazebo navigate
```

Shut everything down with `docker compose down`. View logs with `docker compose logs`.

> Note: you cannot mix Docker containers with native ROS2 nodes on the same machine.
