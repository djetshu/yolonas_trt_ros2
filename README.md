# YOLO TensorRT ROS2 Package

This ROS2 package provides an integration of YOLONAS with TensorRT for high-performance object detection. The package allows flexibility in specifying input topics and model names through launch parameters.

## Demonstration - Object Detection with Custom Model

Check out the following gifs demonstrating the YOLONAS TensorRT ROS2 package in action:
### YOLO NAS TRT in Real-Time: 
1. **Flower Detection in Lab**
   ![Flower Detection in Lab](docs/gif/lab_test.gif)

2. **Flower Detection in the Field**
   ![Flower Detection in the Field](docs/gif/field_test.gif)

## Quick Start

To quickly get started with the YOLO TensorRT ROS2 package, follow these steps:

1. **Install prerequisites:**

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

2. **Clone the repository and install dependencies:**

    ```sh
    cd ~/ros2_ws/src
    git clone https://github.com/djetshu/yolonas_trt_ros2.git
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

5. **Download the TensorRT model and ROSBAG files from [this link](https://1drv.ms/f/s!AsRwIEcpFAcigap3rsBEBVt1iJpW5g).**

6. **Move the YOLO NAS TRT model to the following directory:**

    ```sh
    mv /path/to/downloaded/yolo_nas_m_flower.trt ~/ros2_ws/src/yolonas_trt_ros2/models/
    ```

7. **Modify the class names to your custom classes' names in `ros2_ws/src/yolonas_trt_ros2/yolo_py/dataset.py`.**

8. **Launch the YOLO node with the custom model:**

    ```sh
    ros2 launch yolo_py yolo_trt.launch.py input_topic:=/oak/rgb/image_raw model_name:=yolo_nas_m_flower.trt
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
## Contact Information

For inquiries, collaboration opportunities, or questions feel free to contact:

- **Email:** daffer.queque@outlook.com

## Previous Step: Training YoloNAS and Conversion to TensorRT

Consult this [previous repository](https://github.com/djetshu/yolo_nas_trt_training/tree/main) to train YoloNAS and convert to TensorRT model.
