# YOLO TensorRT ROS2 Package

This ROS2 package provides an integration of YOLONAS with TensorRT for high-performance object detection. The package allows flexibility in specifying input topics and model names through launch parameters.

## Table of Contents
- [Demonstration](#demonstration)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Launch Parameters](#launch-parameters)
- [Nodes](#nodes)
  - [YOLO Node](#yolo-node)
- [Example Launch Commands](#example-launch-commands)
- [License](#license)

## Demonstration

Check out the following videos and images demonstrating the YOLONAS TensorRT ROS2 package in action:

### Video Demos

1. **YOLO Detection in Real-Time:**
   ![YOLO Detection Video](path_to_your_video/demo1.gif)

2. **Object Detection with Custom Model:**
   ![Custom Model Detection Video](path_to_your_video/demo2.gif)

### Image Results

1. **Sample Detection Output:**
   ![Sample Detection Image](path_to_your_image/sample_detection.png)

2. **Custom Model Detection Output:**
   ![Custom Model Detection Image](path_to_your_image/custom_model_detection.png)

## Prerequisites

Before using this package, ensure you have the following dependencies installed:

- ROS2 Humble
- TensorRT `tensorrt==8.6.1.post1`
- Pycuda `pycuda==2024.1`
- OpenCV
- PIL (Python Imaging Library)
- `vision_msgs`
- `cv_bridge`

```sh
pip install tensorrt==8.6.1.post1 --extra-index-url https://pypi.nvidia.com
pip install pycuda==2024.1
pip install opencv-python
pip install pillow
sudo apt-get install ros-humble-vision-msgs
sudo apt-get install ros-humble-cv-bridge
```
## Installation

1. **Clone the repository:**

    ```sh
    cd ~/ros2_ws/src
    git clone https://github.com/djetshu/yolonas_trt_ros2.git
    ```

2. **Install dependencies:**

    Make sure you have the required dependencies installed. You can use `rosdep` to install ROS2 package dependencies:

    ```sh
    cd ~/ros2_ws
    rosdep install --from-paths src --ignore-src -r -y
    ```

3. **Build the package:**

    ```sh
    colcon build --packages-select yolo_py
    ```

4. **Source the workspace:**

    ```sh
    source ~/ros2_ws/install/setup.bash
    ```

## Usage

To use this package, you need to launch the `yolo_trt` node. The package includes a launch file that allows you to specify the input topic and the model name.

### Launch Parameters

- **`rviz_config`**: Path to the RViz config file. Default is set to the RViz config provided in the package.
- **`rviz_en`**: Enable or disable RViz. Default is `true`.
- **`input_topic`**: The input topic for YOLO. Default is `/camera/image`.
- **`model_name`**: The name of the YOLO model to use. Default is `yolo_nas_m.trt`.

### Nodes

#### YOLO Node

This node subscribes to an image topic, processes the image with YOLO using TensorRT, and publishes the detection results.

- **Subscribed Topics:**
  - `/image` (sensor_msgs/Image): Input image topic.

- **Published Topics:**
  - `/detection` (vision_msgs/Detection2DArray): Detection results.
  - `/detection_image` (sensor_msgs/Image): Image with detection results.

### Example Launch Commands

To launch the YOLO node:

```sh
ros2 launch yolo_py yolo_trt.launch.py input_topic:=/your_custom_topic model_name:=your_custom_model.trt
```
