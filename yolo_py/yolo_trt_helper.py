# ROS2 imports 
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
import cv2
from cv_bridge import CvBridge, CvBridgeError
from yolo_py.inference import InferenceSession
import tensorrt as trt
import os
from PIL import Image as PILImage
import time

class YoloTRTNode(Node):
    def __init__(self):
        super().__init__('yolo_trt_node')

        #QOS: quality of service 
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        
        # Subscription: Recieve image from topic /image and process in listener_callback
        self.subscription = self.create_subscription(Image, 'image', self.listener_callback, qos_policy) 
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Publisher: Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection2DArray, 'detection', 10)

        # Publisher: Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'detection_image',10)

        # Get the model name parameter
        self.declare_parameter('model_name', 'yolo_nas_m.trt') # Default name of the model
        model_name = self.get_parameter('model_name').get_parameter_value().string_value

        self.get_logger().info(f"MODEL NAME: {model_name}")

        # Initializing YOLO NAS with TensorRT
        base_directory = os.getcwd() # From workspace path
        subdirectory = "src/yolonas_trt_ros2/models"
        subdirectory_path = os.path.join(base_directory, subdirectory, model_name)
        self.get_logger().info(f"MODEL PATH: {subdirectory_path}")
        self.yolo_nas = InferenceSession(subdirectory_path, (640, 640), trt_logger = trt.Logger(trt.Logger.VERBOSE)) 

        # Measure FPS variables
        self.counter = 0
        self.start_time = time.time()
        self.INTERVAL = 1

    def listener_callback(self, data):
        # self.get_logger().info("Recieved an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        # Conversion CV to PIL image
        im_pil = PILImage.fromarray(cv_image)

        # Process the image with YOLO tensorRT
        result = self.yolo_nas(im_pil)

        # Get the FPS after processing
        self.counter+=1
        if (time.time() - self.start_time) > self.INTERVAL :
            self.get_logger().info(f"FPS: {self.counter / (time.time() - self.start_time)}")
            self.counter = 0
            self.start_time = time.time()

        # Creating Detection2D array instance
        detection_array = Detection2DArray()
        # Show the Image and results of YOLO TRT
        cv_image, pred_boxes, CLASSES_LIST = self.yolo_nas.show_predictions_from_batch_format(im_pil, result)
        # pred_boxes = [[x1, y1, x2, y2, scores, label_id], ...]
        for box in pred_boxes:
            # Definition of 2D array message and ading all object stored in it.
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis_with_pose.hypothesis.class_id = str(CLASSES_LIST[int(box[5])])
            object_hypothesis_with_pose.hypothesis.score = float(box[4])

            bounding_box = BoundingBox2D()
            bounding_box.center.position.x = float((box[0] + box[2])/2)
            bounding_box.center.position.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            bounding_box.size_x = float(2*(bounding_box.center.position.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.position.y - box[1]))

            detection = Detection2D()
            detection.header = data.header
            detection.results.append(object_hypothesis_with_pose)
            detection.bbox = bounding_box
            
            detection_array.header = data.header
            detection_array.detections.append(detection)

        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.detection_publisher.publish(detection_array)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        ros_image.header.frame_id = 'camera_frame'
        self.result_publisher.publish(ros_image)
        cv2.waitKey(1)

        
    pass
