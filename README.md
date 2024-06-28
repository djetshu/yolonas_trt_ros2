# yolonas_trt_ros2
ros2 launch yolo_py yolo_trt.launch.py input_topic:=/oak/rgb/image_raw
ros2 launch yolo_py yolo_trt.launch.py input_topic:=/oak/rgb/image_raw model_name:=yolo_nas_m_weeds_v5.trt

colcon build --packages-select yolo_py
